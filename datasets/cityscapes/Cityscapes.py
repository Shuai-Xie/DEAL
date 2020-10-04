import os
import random
import numpy as np
from PIL import Image

from datasets.base_dataset import BaseDataset, ActiveBaseDataset
from datasets.transforms import get_transform
from datasets.cityscapes.helper import remap_19idxs

from utils.misc import read_txt_as_list, get_select_paths_by_idxs


class Cityscapes(BaseDataset):
    def __init__(self, root, split, base_size, crop_size):
        img_paths = read_txt_as_list(os.path.join(root, f'{split}_img_paths.txt'))
        target_paths = read_txt_as_list(os.path.join(root, f'{split}_target_paths.txt'))

        super(Cityscapes, self).__init__(
            img_paths, target_paths, split, base_size, crop_size,
            num_classes=19,
            bg_idx=0,
            csv_path=os.path.join(os.path.dirname(__file__), 'cityscapes19.csv')
        )

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = img.resize(self.base_size, Image.BILINEAR)  # 原始图像太大

        if self.target_paths[index].endswith('.npy'):
            target = np.load(self.target_paths[index]).astype(int)
        else:
            target = np.asarray(Image.open(self.target_paths[index]), dtype=int)

        target = remap_19idxs(target)
        target = self.mapbg_fn(target)  # bg -> constant_bg 255
        target = Image.fromarray(target)
        target = target.resize(img.size, Image.NEAREST)  # 统一 target 和 img 尺寸

        sample = {
            'img': img,
            'target': target,
            'img_path': self.img_paths[index],
            'target_path': self.target_paths[index]
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_transform(self, split):
        return get_transform(self.split, self.base_size, self.crop_size)


class ActiveCityscapes(ActiveBaseDataset):
    def __init__(self, root, split, base_size, crop_size, init_percent=10):
        if split != 'train':
            raise ValueError('use class `Cityscapes` to instantiate val/test dataset')

        train_img_paths = read_txt_as_list(os.path.join(root, 'train_img_paths.txt'))
        train_target_paths = read_txt_as_list(os.path.join(root, 'train_target_paths.txt'))

        # split data
        label_img_paths, label_target_paths, unlabel_img_paths, unlabel_target_paths = \
            self.random_split_train_data(train_img_paths, train_target_paths, init_percent)

        super(ActiveCityscapes, self).__init__(
            label_img_paths, label_target_paths, unlabel_img_paths, unlabel_target_paths,
            split, base_size, crop_size,
            num_classes=19,
            bg_idx=0,
            csv_path=os.path.join(os.path.dirname(__file__), 'cityscapes19.csv')
        )

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = img.resize(self.base_size, Image.BILINEAR)  # 原始图像太大

        if self.target_paths[index].endswith('.npy'):
            target = np.load(self.target_paths[index]).astype(int)
        else:
            target = np.asarray(Image.open(self.target_paths[index]), dtype=int)

        target = remap_19idxs(target)
        target = self.mapbg_fn(target)  # bg -> constant_bg 255
        target = Image.fromarray(target)
        target = target.resize(img.size, Image.NEAREST)  # 统一 target 和 img 尺寸

        sample = {
            'img': img,
            'target': target,
            'img_path': self.img_paths[index],
            'target_path': self.target_paths[index]
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_transform(self, split):
        return get_transform(self.split, self.base_size, self.crop_size)

    @staticmethod
    def random_split_train_data(img_paths, target_paths, percent):
        img_idxs = list(range(len(img_paths)))
        random.shuffle(img_idxs)

        # 遵从 VAAL 实验配置
        init_select_num = 300 if percent == 10 else round(len(img_paths) * percent / 100)

        label_idxs, unlabel_idxs = img_idxs[:init_select_num], img_idxs[init_select_num:]
        label_img_paths, label_target_paths = get_select_paths_by_idxs(img_paths, target_paths, label_idxs)
        unlabel_img_paths, unlabel_target_paths = get_select_paths_by_idxs(img_paths, target_paths, unlabel_idxs)

        return label_img_paths, label_target_paths, unlabel_img_paths, unlabel_target_paths
