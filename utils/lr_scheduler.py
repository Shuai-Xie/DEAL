##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math


class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`
        iters_per_epoch: number of iterations per epoch
    """

    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0, warmup_start_lr=1e-5):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        if isinstance(self.lr_step, list):  # 添加首尾
            self.lr_step = [0] + self.lr_step + [num_epochs]

        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1

        self.warmup_iters = warmup_epochs * iters_per_epoch
        if self.warmup_iters > 0:
            self.warmup_start_lr = warmup_start_lr
            self.warmup_factor = (self.lr / self.warmup_start_lr) ** (1. / self.warmup_iters)

    def __call__(self, optimizer, i, epoch, best_pred=None):
        T = epoch * self.iters_per_epoch + i

        # warm up lr schedule, 从1个小小的 lr 缓慢增加到 base_lr
        if self.warmup_iters > 0 and T <= self.warmup_iters:
            lr = self.warmup_start_lr * (self.warmup_factor ** T)
        else:
            if self.mode == 'cos':
                lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
            elif self.mode == 'poly':
                lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
            elif self.mode == 'step':
                if isinstance(self.lr_step, int):
                    lr = self.lr * (0.1 ** (epoch // self.lr_step))  # lr reduce step
                elif isinstance(self.lr_step, list):
                    for idx in range(len(self.lr_step) - 1):
                        if self.lr_step[idx] <= epoch < self.lr_step[idx + 1]:
                            lr = self.lr * (0.1 ** idx)
                            break
                else:
                    raise TypeError
            else:
                raise NotImplemented

        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):  # 认为后面层为自定义，设置更大 lr
                optimizer.param_groups[i]['lr'] = lr * 10
