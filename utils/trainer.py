import torch
import constants

from torch.utils.data import DataLoader
from model.sync_batchnorm.replicate import patch_replication_callback
from utils.metrics import Evaluator
from utils.misc import AverageMeter, get_learning_rate
from utils.lr_scheduler import LR_Scheduler
from utils.loss import CELoss, MaskLoss
from tqdm import tqdm


class Trainer:

    def __init__(self, args, model, train_set, valid_set, test_set, saver, writer):
        self.args = args
        self.saver = saver
        self.saver.save_experiment_config()  # save cfgs
        self.writer = writer

        self.num_classes = train_set.num_classes

        # dataloaders
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        self.valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        self.dataset_size = {'train': len(train_set), 'val': len(valid_set), 'test': len(test_set)}
        print('dataset size:', self.dataset_size)

        # iters_per_epoch
        self.iters_per_epoch = args.iters_per_epoch if args.iters_per_epoch else len(self.train_loader)

        # optimizer & lr_scheduler
        train_params = [
            {'params': model.get_1x_lr_params(), 'lr': args.lr},  # backbone
            {'params': model.get_10x_lr_params(), 'lr': args.lr * 10},  # aspp,decoder
        ]
        if args.with_mask and args.with_pam:  # make gamma learnable
            train_params.append({'params': model.mask_head.pam.gamma, 'lr': args.lr * 10})

        self.optimizer = torch.optim.SGD(train_params,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay,
                                         nesterov=args.nesterov)
        self.lr_scheduler = LR_Scheduler(mode=args.lr_scheduler, base_lr=args.lr,
                                         lr_step=args.lr_step,
                                         num_epochs=args.epochs,
                                         warmup_epochs=args.warmup_epochs,
                                         iters_per_epoch=self.iters_per_epoch)
        num_gpu = len(args.gpu_ids.split(','))
        if num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))
            patch_replication_callback(model)
            print(args.gpu_ids)

        self.model = model.cuda()

        # loss
        self.criterion = CELoss()  # naive ce loss
        self.criterion_mask = MaskLoss(mode=self.args.mask_loss)

        # evaluator
        self.evaluator = Evaluator(self.num_classes)
        self.best_mIoU = 0.
        self.best_Acc = 0.

    def training(self, epoch, prefix='Train', evaluation=False):
        self.model.train()
        self.evaluator.reset()

        train_losses, seg_losses, mask_losses = AverageMeter(), AverageMeter(), AverageMeter()
        tbar = tqdm(self.train_loader, total=self.iters_per_epoch)

        for i, sample in enumerate(tbar):
            if i == self.iters_per_epoch:
                break

            # update lr each iteration
            self.lr_scheduler(self.optimizer, i, epoch)
            self.optimizer.zero_grad()

            image, target = sample['img'].cuda(), sample['target'].cuda()

            if not self.args.with_mask:  # ori
                output = self.model(image)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                train_losses.update(loss.item())
                tbar.set_description('Epoch {}, train loss: {:.3f}'.format(epoch, train_losses.avg))
            else:
                output, soft_mask = self.model(image)
                seg_loss = self.criterion(output, target)
                # mask
                target_error_mask = self.generate_target_error_mask(output, target)  # B,H,W
                mask_loss = self.criterion_mask(soft_mask, target_error_mask)

                loss = seg_loss + mask_loss
                loss.backward()
                self.optimizer.step()

                # loss
                train_losses.update(loss.item())
                seg_losses.update(seg_loss.item())
                mask_losses.update(mask_loss.item())

                tbar.set_description('Epoch {}, train loss: {:.3f}, seg: {:.3f}, mask: {:.3f}'.format(
                    epoch, train_losses.avg, seg_losses.avg, mask_losses.avg))

            if evaluation:
                pred = torch.argmax(output, dim=1)
                self.evaluator.add_batch(target.cpu().numpy(), pred.cpu().numpy())  # B,H,W

        if not self.args.with_mask:
            self.writer.add_scalars(f'{prefix}/loss', {
                'train': train_losses.avg,
            }, epoch)
        else:
            self.writer.add_scalars(f'{prefix}/loss', {
                'train': train_losses.avg,
                'segment': seg_losses.avg,
                'mask': mask_losses.avg
            }, epoch)
            # attention coefficient
            self.writer.add_scalar(f'{prefix}/gamma', self.model.mask_head.pam.gamma.item(), epoch)

        if evaluation:
            Acc = self.evaluator.Pixel_Accuracy()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            self.writer.add_scalar(f'{prefix}/mIoU', mIoU, epoch)
            self.writer.add_scalar(f'{prefix}/Acc', Acc, epoch)
            print('Epoch: {}, Acc: {:.3f}, mIoU: {:.3f}'.format(epoch, Acc, mIoU))

    @torch.no_grad()
    def validation(self, epoch, test=False):
        self.model.eval()
        self.evaluator.reset()

        tbar, prefix = (tqdm(self.test_loader), 'Test') if test else (tqdm(self.valid_loader), 'Valid')

        # loss
        seg_losses, mask_losses = AverageMeter(), AverageMeter()

        for i, sample in enumerate(tbar):
            image, target = sample['img'].cuda(), sample['target'].cuda()

            if not self.args.with_mask:
                output = self.model(image)
                seg_loss = self.criterion(output, target)
                seg_losses.update(seg_loss.item())
                tbar.set_description(f'{prefix} loss: %.3f' % seg_losses.avg)
            else:
                output, soft_mask = self.model(image)
                seg_loss = self.criterion(output, target)
                # mask
                target_error_mask = self.generate_target_error_mask(output, target)  # B,H,W
                mask_loss = self.criterion_mask(soft_mask, target_error_mask)

                # loss
                seg_losses.update(seg_loss.item())
                mask_losses.update(mask_loss.item())

                tbar.set_description('{} segment loss: {:.3f}, mask loss: {:.3f}'.format(prefix, seg_losses.avg, mask_losses.avg))

            pred = torch.argmax(output, dim=1)
            self.evaluator.add_batch(target.cpu().numpy(), pred.cpu().numpy())  # B,H,W

        Acc = self.evaluator.Pixel_Accuracy()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        self.writer.add_scalar(f'{prefix}/mIoU', mIoU, epoch)
        self.writer.add_scalar(f'{prefix}/Acc', Acc, epoch)
        print('Epoch: {}, Acc: {:.3f}, mIoU: {:.3f}'.format(epoch, Acc, mIoU))

        if not self.args.with_mask:
            self.writer.add_scalars(f'{prefix}/loss', {
                'segment': seg_losses.avg,
            }, epoch)
        else:
            self.writer.add_scalars(f'{prefix}/loss', {
                'segment': seg_losses.avg,
                'mask': mask_losses.avg
            }, epoch)

        if not test and mIoU > self.best_mIoU:
            print('saving model...')
            self.best_mIoU = mIoU
            self.best_Acc = Acc
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),  # 方便 test 保持同样结构?
                'optimizer': self.optimizer.state_dict(),
                'mIoU': mIoU,
                'Acc': Acc
            }
            self.saver.save_checkpoint(state)
            print('save model at epoch', epoch)

        return mIoU, Acc

    def load_best_checkpoint(self, file_path=None, load_optimizer=False):
        checkpoint = self.saver.load_checkpoint(file_path=file_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if file_path:
            print('load', file_path)
        print(f'=> loaded checkpoint - epoch {checkpoint["epoch"]}')
        return checkpoint["epoch"]

    @staticmethod
    def generate_target_error_mask(output, target):
        pred = torch.argmax(output, dim=1)
        target_error_mask = (pred != target).float()  # error=1
        target_error_mask[target == constants.BG_INDEX] = 0.  # ingore bg
        return target_error_mask
