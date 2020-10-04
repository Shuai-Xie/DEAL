from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import datasets.transforms as tr
from utils.vis import get_label_name_colors
from utils.misc import read_txt_as_list
import os


class BaseDataset(Dataset):
    def __init__(self,
                 img_paths, target_paths,
                 split=None, base_size=None, crop_size=None, **kwargs):
        """
        base dataset, with img/target paths
        """
        self.img_paths = img_paths
        self.target_paths = target_paths
        self.len_dataset = len(self.img_paths)

        self.base_size = base_size  # train 基准 size
        self.crop_size = crop_size  # train, valid, test

        self.split = split
        self.transform = self.get_transform(split)

        self.num_classes = kwargs.get('num_classes', -1)
        self.bg_idx = kwargs.get('bg_idx', -1)
        self.mapbg_fn = tr.mapbg(self.bg_idx)
        self.remap_fn = tr.remap(self.bg_idx)

        if 'csv_path' in kwargs:
            self.label_names, self.label_colors = get_label_name_colors(kwargs['csv_path'])

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')

        if self.target_paths[index].endswith('.npy'):
            target = np.load(self.target_paths[index]).astype(int)
        else:
            target = np.asarray(Image.open(self.target_paths[index]), dtype=int)

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

    def __len__(self):
        return len(self.img_paths)

    def get_transform(self, split):
        return None

    def make_dataset_multiple_of_batchsize(self, batch_size):
        remainder = self.len_dataset % batch_size
        if remainder > 0:
            num_new_entries = batch_size - remainder
            self.img_paths.extend(self.img_paths[:num_new_entries])
            self.target_paths.extend(self.target_paths[:num_new_entries])

    def reset_dataset(self):
        self.img_paths = self.img_paths[:self.len_dataset]
        self.target_paths = self.target_paths[:self.len_dataset]


class ActiveBaseDataset(BaseDataset):
    def __init__(self,
                 label_img_paths, label_target_paths,
                 unlabel_img_paths, unlabel_target_paths,
                 split,
                 base_size, crop_size, **kwargs):
        """
        Active base dataset, with label/unlabel img/target paths
        """
        super().__init__(label_img_paths, label_target_paths,
                         split,
                         base_size, crop_size, **kwargs)
        self.base_img_paths = label_img_paths
        self.base_target_paths = label_target_paths

        self.label_img_paths = label_img_paths[:]  # 复制一份, 防止改动 base_img_paths
        self.label_target_paths = label_target_paths[:]

        self.unlabel_img_paths = unlabel_img_paths
        self.unlabel_target_paths = unlabel_target_paths

    def add_by_select_remain_paths(self,
                                   select_img_paths, select_target_paths,
                                   remain_img_paths, remain_target_paths):
        """
            label_data  += select data
            unlabel_data = remain data
        """
        self.label_img_paths += select_img_paths
        self.label_target_paths += select_target_paths

        self.unlabel_img_paths = remain_img_paths
        self.unlabel_target_paths = remain_target_paths

        self.update_iter_img_paths()

    def add_by_select_unlabel_idxs(self, select_idxs):
        """
        :param select_idxs: 从 unlabel idx 划分样本
        :return:
        """
        remain_idxs = set(range(len(self.unlabel_img_paths))) - set(select_idxs)

        select_img_paths = [self.unlabel_img_paths[i] for i in select_idxs]
        select_target_paths = [self.unlabel_target_paths[i] for i in select_idxs]

        self.label_img_paths += select_img_paths
        self.label_target_paths += select_target_paths

        self.unlabel_img_paths = [self.unlabel_img_paths[i] for i in remain_idxs]
        self.unlabel_target_paths = [self.unlabel_target_paths[i] for i in remain_idxs]

        self.update_iter_img_paths()

    def add_preselect_data(self, iter_dir):
        """
        :param iter_dir: read preselect data paths from iter_dir, and update label/unlabel data
        """
        # preselect label data
        label_img_paths = read_txt_as_list(os.path.join(iter_dir, 'label_imgs.txt'))
        label_target_paths = read_txt_as_list(os.path.join(iter_dir, 'label_targets.txt'))
        label_data = set(label_img_paths)

        # remain_unlabel = ori_unlabel - preselect_label
        remain_img_paths, remain_target_paths = [], []
        for i in range(len(self.unlabel_img_paths)):
            if self.unlabel_img_paths[i] not in label_data:
                remain_img_paths.append(self.unlabel_img_paths[i])
                remain_target_paths.append(self.unlabel_target_paths[i])

        # update active_trainset
        self.update_label_unlabel_paths(label_img_paths, label_target_paths,
                                        remain_img_paths, remain_target_paths)

    def update_label_unlabel_paths(self,
                                   label_img_paths, label_target_paths,
                                   unlabel_img_paths, unlabel_target_paths):
        """
            reset label/unlabel path, for resume training
        """
        self.label_img_paths = label_img_paths
        self.label_target_paths = label_target_paths

        self.unlabel_img_paths = unlabel_img_paths
        self.unlabel_target_paths = unlabel_target_paths

        self.update_iter_img_paths()

    def update_iter_img_paths(self):  # train get_item
        self.img_paths = self.label_img_paths
        self.target_paths = self.label_target_paths
