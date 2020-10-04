import random
from utils.misc import *


class RandomSelector:

    def __init__(self, dataset, img_size):
        self.dataset = dataset
        self.img_size = img_size

    def select_next_batch(self, model, active_trainset, select_num):
        # get a subset
        subset_img_paths, subset_target_paths, remset_img_paths, remset_target_paths = get_subset_paths(
            active_trainset.unlabel_img_paths, active_trainset.unlabel_target_paths,
        )
        img_idxs = list(range(len(subset_img_paths)))
        random.shuffle(img_idxs)

        select_idxs = img_idxs[:select_num]
        select_img_paths, select_target_paths, remain_img_paths, remain_target_paths = get_select_remain_paths(
            subset_img_paths, subset_target_paths, select_idxs
        )
        # remset 补充回去
        remain_img_paths += remset_img_paths
        remain_target_paths += remset_target_paths

        # 更新 DL, DU
        active_trainset.add_by_select_remain_paths(select_img_paths, select_target_paths,
                                                   remain_img_paths, remain_target_paths)
