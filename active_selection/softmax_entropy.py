import torch
from datasets.base_dataset import BaseDataset
from datasets.transforms import get_transform
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils.misc import *


class SoftmaxEntropySelector:

    def __init__(self, dataset, img_size):
        self.dataset = dataset
        self.img_size = img_size
        self.softmax = torch.nn.Softmax2d()

    @torch.no_grad()
    def select_next_batch(self, model, active_trainset, select_num):
        model.eval()

        # get a subset from the whole unlabelset
        subset_img_paths, subset_target_paths, remset_img_paths, remset_target_paths = get_subset_paths(
            active_trainset.unlabel_img_paths, active_trainset.unlabel_target_paths,
        )
        print('subset_img_paths', len(subset_img_paths))
        print('remset_img_paths', len(remset_img_paths))
        unlabelset = BaseDataset(subset_img_paths, subset_target_paths)
        unlabelset.transform = get_transform('test', base_size=self.img_size)

        dataloader = DataLoader(unlabelset,
                                batch_size=8, shuffle=False,
                                pin_memory=True, num_workers=4)

        scores = []
        tbar = tqdm(dataloader, desc='\r')
        tbar.set_description(f'cal_entropy_score')

        for sample in tbar:
            img = sample['img'].cuda()
            probs = self.softmax(model(img))  # B,C,H,W
            probs = probs.detach().cpu().numpy()
            scores += self.cal_entropy_score(probs)

        select_idxs = get_topk_idxs(scores, select_num)

        # 从 subset 中选出样本
        select_img_paths, select_target_paths, remain_img_paths, remain_target_paths = get_select_remain_paths(
            subset_img_paths, subset_target_paths, select_idxs
        )
        # remset 补充回去
        remain_img_paths += remset_img_paths
        remain_target_paths += remset_target_paths
        print('select_img_paths', len(select_img_paths))
        print('remain_img_paths', len(remain_img_paths))

        # 更新 DL, DU
        active_trainset.add_by_select_remain_paths(select_img_paths, select_target_paths,
                                                   remain_img_paths, remain_target_paths)

    @staticmethod
    def cal_entropy_score(probs):  # C,H,W  熵越大，越难分
        batch_scores = []
        for i in range(len(probs)):  # prob img
            entropy = np.mean(-np.nansum(np.multiply(probs[i], np.log(probs[i] + 1e-12)), axis=0))  # 表示沿着第1维计算
            batch_scores.append(entropy)
        return batch_scores
