import argument_parser
from pprint import pprint

args = argument_parser.parse_args()
pprint(vars(args))

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids  # 设置可见 gpus
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"

if len(args.gpu_ids) > 1:
    args.sync_bn = True

import torch
from torch.utils.tensorboard import SummaryWriter
import argument_parser
import constants
from datasets.build_datasets import build_datasets, data_cfg
from model.deeplab import DeepLab
from utils.calculate_weights import calculate_class_weights
from utils.saver import Saver
from utils.trainer import Trainer
from utils.misc import AccCaches, get_curtime, write_list_to_txt, read_txt_as_list
from pprint import pprint
from active_selection import get_active_selector
import shutil
import random
import math
import glob
import numpy as np


def is_interval(epoch):
    return epoch % args.eval_interval == (args.eval_interval - 1)


def main():
    random.seed(args.seed)  # active trainset
    active_trainset, validset, testset = build_datasets(args.dataset, args.base_size, args.crop_size, args.init_percent)

    if args.resume_dir and args.resume_percent:  # 此 iteration 已选数据，但还未训练模型
        iter_dir = f'runs/{args.dataset}/{args.resume_dir}/runs_0{args.resume_percent}'
        active_trainset.add_preselect_data(iter_dir)  # add preselect data, and update label/unlabel data

    # global writer
    timestamp = get_curtime()
    global_saver = Saver(args, exp_dir=args.resume_dir, timestamp=timestamp)
    global_writer = SummaryWriter(global_saver.exp_dir)

    # 设置样本选择器
    active_selector = get_active_selector(args)

    # budget 选样本数目
    select_num = args.select_num
    if select_num is None:
        if args.percent_step:  # 转换 percent_step 为全局每次 引入的图片数量
            select_num = math.ceil(active_trainset.len_total_dataset * args.percent_step / 100)
        else:
            raise ValueError('must set select_num or percent_step')

    start_percent = args.resume_percent if args.resume_percent else args.init_percent

    for percent in range(start_percent, args.max_percent + 1, args.percent_step):
        run_id = f'runs_{percent:03d}'
        print(run_id)

        # global: len(dataset) of current percent data
        global_writer.add_scalar('Active/global_trainset', len(active_trainset), global_step=percent)

        ## ------------ begin training with current percent data ------------

        # saver/writer of each iteration
        saver = Saver(args, exp_dir=args.resume_dir, timestamp=timestamp, suffix=run_id)
        writer = SummaryWriter(saver.exp_dir)
        # save current data path -> train model -> select new data -> 下一轮再 save data path
        write_list_to_txt(active_trainset.label_img_paths, txt_path=os.path.join(saver.exp_dir, 'label_imgs.txt'))
        write_list_to_txt(active_trainset.label_target_paths, txt_path=os.path.join(saver.exp_dir, 'label_targets.txt'))

        # create model from scratch
        model = DeepLab(args.backbone, args.out_stride, active_trainset.num_classes, args.sync_bn,
                        with_mask=args.with_mask,
                        with_pam=args.with_pam, branch_early=args.branch_early)

        trainer = Trainer(args, model, active_trainset, validset, testset, saver, writer)

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

        ## ------------ end training with current percent data ------------

        # global: eval metrics of current percent data
        global_writer.add_scalar('Active/global_mIoU', test_mIoU, global_step=percent)
        global_writer.add_scalar('Active/global_Acc', test_Acc, global_step=percent)

        # end active training
        if percent == args.max_percent:
            global_writer.flush()
            global_writer.close()
            print('end active training')
            break

        # select samples
        active_selector.select_next_batch(trainer.model, active_trainset, select_num)


if __name__ == '__main__':
    main()
