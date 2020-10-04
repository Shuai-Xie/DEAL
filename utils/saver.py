import os
import shutil
import torch
import json
import imageio
import numpy as np
import constants

"""
exp_dir: tensorboard
    save/load checkpoint
    save_experiment_config
    save_active_selections, mask of select parts
"""


class Saver:

    def __init__(self, args, suffix='', timestamp='',
                 exp_group=None, exp_dir=None,
                 remove_existing=False):

        self.args = args

        if exp_group is None:
            exp_group = args.dataset

        # runs/ tensorboard
        if not exp_dir:
            exp_dir = f'{args.checkname}_{timestamp}'

        self.exp_dir = os.path.join(constants.RUNS, exp_group, exp_dir, suffix)

        if remove_existing and os.path.exists(self.exp_dir):
            shutil.rmtree(self.exp_dir)

        if not os.path.exists(self.exp_dir):
            print(f'Creating dir {self.exp_dir}')
            os.makedirs(self.exp_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.exp_dir, filename)
        torch.save(state, filename)

    def load_checkpoint(self, filename='checkpoint.pth.tar', file_path=None):
        filename = os.path.join(self.exp_dir, filename) if not file_path else file_path
        return torch.load(filename)

    def save_experiment_config(self):
        logfile = os.path.join(self.exp_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        arg_dictionary = vars(self.args)
        log_file.write(json.dumps(arg_dictionary, indent=4, sort_keys=True))  # 按 key 排序
        log_file.close()

    def save_active_selections(self, paths, regional=False):
        if regional:
            Saver.save_masks(os.path.join(self.exp_dir, "selections"), paths)
        else:
            filename = os.path.join(self.exp_dir, 'selections.txt')
            with open(filename, 'w') as fptr:
                for p in paths:
                    fptr.write(p.decode('utf-8') + '\n')

    @staticmethod
    def save_masks(directory, paths):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for p in paths:
            imageio.imwrite(os.path.join(directory, p.decode('utf-8') + '.png'),
                            (paths[p] * 255).astype(np.uint8))
