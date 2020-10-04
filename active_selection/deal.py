import torch
from datasets.base_dataset import BaseDataset
from datasets.transforms import get_transform
from utils.misc import get_topk_idxs, get_subset_paths, get_select_remain_paths
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from active_selection.uc_criterion import entropy, least_confidence, margin_sampling


class DEALSelector:

    def __init__(self, dataset, img_size, strategy, hard_levels):
        self.dataset = dataset
        self.img_size = img_size

        self.strategy = strategy
        self.hard_levels = hard_levels

        self.softmax = torch.nn.Softmax2d()

    @torch.no_grad()
    def select_next_batch(self, model, trainset, select_num):
        model.cuda()
        model.eval()

        # subset: 参与样本选择
        # remset: 在 subset 选出之后，再补充到 remain path 中
        subset_img_paths, subset_target_paths, remset_img_paths, remset_target_paths = get_subset_paths(
            trainset.unlabel_img_paths, trainset.unlabel_target_paths, sub_ratio=0.5,
        )
        print('subset_img_paths', len(subset_img_paths))
        print('remset_img_paths', len(remset_img_paths))
        unlabelset = BaseDataset(subset_img_paths, subset_target_paths)  # load 时已将 bg_idx 统一为 255
        unlabelset.transform = get_transform('test', base_size=self.img_size)

        dataloader = DataLoader(unlabelset,
                                batch_size=8, shuffle=False,
                                pin_memory=False, num_workers=4)

        scores = []
        tbar = tqdm(dataloader, desc='\r')
        tbar.set_description(f'{self.strategy}')
        for sample in tbar:
            img = sample['img'].cuda()
            output, diff_map = model(img)

            diff_map = diff_map.detach().cpu().numpy().squeeze(1)  # remove C=1, B,H,W

            if self.strategy == 'diff_score':
                probs = self.softmax(output)  # B,C,H,W
                probs = np.transpose(probs.detach().cpu().numpy(), (0, 2, 3, 1))  # B,H,W,C
                scores += self.batch_diff_score(probs, diff_map)
            elif self.strategy == 'diff_entropy':
                scores += self.batch_diff_entropy(diff_map, hard_levels=self.hard_levels)

        select_idxs = get_topk_idxs(scores, select_num)

        # 从 subset 中选出样本
        select_img_paths, select_target_paths, remain_img_paths, remain_target_paths = get_select_remain_paths(
            subset_img_paths, subset_target_paths, select_idxs
        )
        # remain set 补充回去
        remain_img_paths += remset_img_paths
        remain_target_paths += remset_target_paths
        print('select_img_paths', len(select_img_paths))
        print('remain_img_paths', len(remain_img_paths))

        # 更新 DL, DU
        trainset.add_by_select_remain_paths(select_img_paths, select_target_paths,
                                            remain_img_paths, remain_target_paths)

    def batch_diff_score(self, probs, diff_maps, uc_criterion='none'):
        batch_scores = []
        for i in range(len(probs)):
            if uc_criterion == 'en':
                uc_map = entropy(probs[i])
            elif uc_criterion == 'ms':
                uc_map = margin_sampling(probs[i])
            elif uc_criterion == 'lc':
                uc_map = least_confidence(probs[i])
            elif uc_criterion == 'none':
                uc_map = 1.
            else:
                raise NotImplementedError
            batch_scores.append(np.mean(uc_map * diff_maps[i]))

        return batch_scores

    def batch_diff_entropy(self, diff_map, hard_levels):
        batch_scores = []
        for i in range(len(diff_map)):
            region_areas, score_ticks = np.histogram(diff_map[i], bins=hard_levels)
            probs = region_areas / region_areas.sum()
            entropy = -np.nansum(np.multiply(probs, np.log(probs + 1e-12)))
            batch_scores.append(entropy)

        return batch_scores
