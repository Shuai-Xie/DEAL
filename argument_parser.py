import argparse
import random


def parse_args(params=None):
    parser = argparse.ArgumentParser(description="DEAL")

    # model
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet50', 'resnet101', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: mobilenet)')
    parser.add_argument('--out-stride', type=int, default=16, help='network output stride (default: 16)')
    parser.add_argument('--with-mask', action='store_true', default=False, help='whether to predict error mask')

    # dataset
    parser.add_argument('--dataset', type=str, default='CamVid', help='dataset name (default: CamVid)')
    parser.add_argument('--base-size', type=str, default="513,513", help='base image size')
    parser.add_argument('--crop-size', type=str, default="513,513", help='crop image size')

    # gpu
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--sync-bn', type=bool, default=False, help='whether to use sync bn (default: False)')

    # train
    parser.add_argument('--batch-size', type=int, default=4, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--epochs', type=int, default=None, metavar='N', help='number of epochs to train (default: auto)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N', help='number of epochs to train (default: auto)')
    parser.add_argument('--eval-interval', type=int, default=5, help='evaluation interval (default: 5) - record metrics every Nth iteration')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
    parser.add_argument('--iters-per-epoch', type=int, default=None, help='iterations per epoch')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default='False', help='whether use nesterov (default: False)')
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['step', 'poly', 'cos'], help='lr scheduler mode: (default: poly)')
    parser.add_argument('--lr-step', type=str, default='35', help='step size for lr-step-scheduler')

    # loss
    parser.add_argument('--seg-loss', type=str, default='ce', help='loss func type (default: ce)')
    parser.add_argument('--mask-loss', type=str, default='bce', choices=['bce', 'wce'], help='mask loss in object function (default: bce)')

    # ablation study
    parser.add_argument('--with-pam', action='store_true', default=False, help='use Probability Attention Module')
    parser.add_argument('--branch-early', action='store_true', default=False, help='branch at the boarder of encoder and decoder')

    # AL options
    parser.add_argument('--active_selection_mode', type=str, default='random',
                        choices=['coreset', 'deal', 'dropout', 'entropy', 'random'])
    parser.add_argument('--max-percent', type=int, default=40, help='max active iterations')  # 15,20,25,30,35,40
    parser.add_argument('--init-percent', type=int, default=10, help='init label data percent')
    parser.add_argument('--percent-step', type=int, default=5, help='incremental label data percent')
    parser.add_argument('--select-num', type=int, help='image num of 5% data')
    parser.add_argument('--strategy', type=str, default='DS', choices=['DS', 'DE'], help='two strategies')
    parser.add_argument('--hard-levels', type=int, default=None, help='quantified difficulty levels for DE strategy')
    # seed
    parser.add_argument('--seed', type=int, default=-1, metavar='S', help='random seed (default: -1)')

    # resume
    parser.add_argument('--resume-dir', type=str, default=None, help='resume exp dir')
    parser.add_argument('--resume-percent', type=int, default=None, help='resume active iteration')

    args = parser.parse_args(params)
    args.base_size = [int(s) for s in args.base_size.split(',')]
    args.crop_size = [int(s) for s in args.crop_size.split(',')]

    # manual seeding
    if args.seed == -1:
        args.seed = int(random.random() * 2000)
    print('Using random seed =', args.seed)
    print('ActiveSelector:', args.active_selection_mode)

    return args
