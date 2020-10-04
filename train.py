import argument_parser
from pprint import pprint

args = argument_parser.parse_args()
pprint(vars(args))

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

if len(args.gpu_ids) > 1:
    args.sync_bn = True

import random

from torch.utils.tensorboard import SummaryWriter
from datasets.build_datasets import build_datasets
from model.deeplab import DeepLab
from utils.saver import Saver
from utils.trainer import Trainer
from utils.misc import get_curtime


def is_interval(epoch):
    return epoch % args.eval_interval == (args.eval_interval - 1)


def main():
    random.seed(args.seed)
    trainset, validset, testset = build_datasets(args.dataset, args.base_size, args.crop_size)

    model = DeepLab(args.backbone, args.out_stride, trainset.num_classes, args.sync_bn)

    saver = Saver(args, timestamp=get_curtime())
    writer = SummaryWriter(saver.exp_dir)
    trainer = Trainer(args, model, trainset, validset, testset, saver, writer)

    # train/valid
    for epoch in range(args.epochs):
        trainer.training(epoch)
        if is_interval(epoch):
            trainer.validation(epoch)
    print('Valid: best mIoU:', trainer.best_mIoU, 'Acc:', trainer.best_Acc)

    # test
    epoch = trainer.load_best_checkpoint()
    test_mIoU, test_Acc = trainer.validation(epoch, test=True)
    print('Test: best mIoU:', test_mIoU, 'pixelAcc:', test_Acc)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
