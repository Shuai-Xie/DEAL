import os
import time
import numpy as np
import torch
import constants
import random

epsilon = 1e-5


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def write_list_to_txt(a_list, txt_path):
    with open(txt_path, 'w') as f:
        for p in a_list:
            f.write(p + '\n')


def read_txt_as_list(f):
    with open(f, 'r') as f:
        return [p.replace('\n', '') for p in f.readlines()]


def one_hot_target(target, num_classes):  # 360,480, 划分到每个类
    if isinstance(target, torch.Tensor):
        target = to_numpy(target)
    h, w = target.shape
    res = np.zeros((num_classes, h, w)).astype(int)
    for i in range(num_classes):  # [0,10]
        res[i][target == i] = 1

    return res


def generate_target_error_mask(pred, target, class_aware=False, num_classes=0):
    """
    :param pred: H,W
    :param target: H,W
    :param class_aware:
    :param num_classes: use with class_aware
    :return:
    """

    if isinstance(target, torch.Tensor):
        pred, target = to_numpy(pred), to_numpy(target)
    target_error_mask = (pred != target).astype('uint8')  # 0,1
    target_error_mask[target == constants.BG_INDEX] = 0

    if class_aware:
        # 不受类别数量影响
        error_mask = target_error_mask == 1
        target_error_mask[~error_mask] = constants.BG_INDEX  # bg

        for c in range(num_classes):  # C
            cls_error = error_mask & (target == c)
            target_error_mask[cls_error] = c

    return target_error_mask


def mkdir(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def cal_nobg_mean_score(score_map, target):
    score_map, target = to_numpy(score_map), to_numpy(target)
    mask = target != constants.BG_INDEX
    score = np.sum(score_map[mask]) / mask.sum()
    return score


def turn_on_dropout(module):
    if type(module) == torch.nn.Dropout:
        module.train()


def turn_off_dropout(module):
    if type(module) == torch.nn.Dropout:
        module.eval()


def get_topk_idxs(a, k):
    if isinstance(a, list):
        a = np.array(a)
    return a.argsort()[::-1][:k]


def get_subset_paths(unlabel_img_paths, unlabel_target_paths, sub_ratio=0.5):
    """
    random choose a subset from the whole unlabel data pool,
    简单缓解从全集中选到太多相似数据
    """
    total_num = len(unlabel_target_paths)
    sub_num = int(total_num * sub_ratio)

    img_idxs = list(range(total_num))
    random.shuffle(img_idxs)

    sub_idxs, remain_idxs = img_idxs[:sub_num], img_idxs[sub_num:]

    subset_img_paths, subset_target_paths = get_select_paths_by_idxs(
        unlabel_img_paths, unlabel_target_paths, sub_idxs
    )
    remset_img_paths, remset_target_paths = get_select_paths_by_idxs(
        unlabel_img_paths, unlabel_target_paths, remain_idxs
    )
    return subset_img_paths, subset_target_paths, remset_img_paths, remset_target_paths


def get_select_paths_by_idxs(img_paths, target_paths, select_idxs):
    select_img_paths, select_target_paths = [], []
    for idx in select_idxs:
        select_img_paths.append(img_paths[idx])
        select_target_paths.append(target_paths[idx])
    return select_img_paths, select_target_paths


def get_select_remain_paths(unlabel_img_paths, unlabel_target_paths, select_idxs):
    remain_idxs = list(set(range(len(unlabel_img_paths))) - set(select_idxs))

    select_img_paths = [unlabel_img_paths[i] for i in select_idxs]
    select_target_paths = [unlabel_target_paths[i] for i in select_idxs]

    remain_img_paths = [unlabel_img_paths[i] for i in remain_idxs]
    remain_target_paths = [unlabel_target_paths[i] for i in remain_idxs]

    return select_img_paths, select_target_paths, remain_img_paths, remain_target_paths


def get_group_topk_idxs(scores, groups=5, select_num=10):
    total_num = len(scores)
    base = total_num // groups
    remain = total_num % groups
    per_select = select_num // groups
    if remain > groups / 2:
        base += 1
        per_select += 1  # 多组多选
    last_select = select_num - per_select * (groups - 1)

    begin_idxs = [0] + [base * (i + 1) for i in range(groups - 1)] + [total_num]
    total_idxs = list(range(total_num))
    random.shuffle(total_idxs)

    select_idxs = []
    for i in range(groups):
        begin, end = begin_idxs[i], begin_idxs[i + 1]
        group_rand_idxs = total_idxs[begin:end]
        group_scores = [scores[s] for s in group_rand_idxs]
        if i == groups - 1:  # 最后一组
            per_select = last_select
        group_select_idxs = get_topk_idxs(group_scores, k=per_select).tolist()
        group_select_idxs = [group_rand_idxs[s] for s in group_select_idxs]  # 转成全局 idx

        select_idxs += group_select_idxs

    return select_idxs


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_curtime():
    current_time = time.strftime('%b%d_%H%M%S', time.localtime())
    return current_time


def get_percent_thre(soft_mask, percents):
    thres = soft_mask.detach().cpu().numpy()
    p_thre = np.percentile(thres, percents)
    return p_thre


def to_numpy(var, toint=False):
    #  Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
    if isinstance(var, torch.Tensor):
        var = var.squeeze().detach().cpu().numpy()
    if toint:
        var = var.astype('uint8')
    return var


def minmax_normalize(a):  # min/max -> [0,1]
    min_a, max_a = np.min(a), np.max(a)
    return (a - min_a) / (max_a - min_a)


def ajust_upper_imb_weights(w_one, w_zero, upper=4):
    if w_zero / w_one > upper:
        w_one = 1. / (upper + 1)
        w_zero = 1 - w_one
    imb_weights = torch.tensor([w_one, w_zero])
    return imb_weights


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccCaches:
    """acc cache queue"""

    def __init__(self, patience):
        self.accs = []  # [(epoch, acc), ...]
        self.patience = patience

    def reset(self):
        self.accs = []

    def add(self, epoch, acc):
        if len(self.accs) >= self.patience:  # 先满足 =
            self.accs = self.accs[1:]  # 队头出队列
        self.accs.append((epoch, acc))  # 队尾添加

    def full(self):
        return len(self.accs) == self.patience

    def max_cache_acc(self):
        max_id = int(np.argmax([t[1] for t in self.accs]))  # t[1]=acc
        max_epoch, max_acc = self.accs[max_id]
        return max_epoch, max_acc


class MinRecoder:
    def __init__(self, mask_factor):
        self.min = 100.
        self.mask_factor = mask_factor

    # 始终返回训练过程中 最小的 loss
    # 这样会使 loss 从 graph 中取出来，无法训练，还是 mask_factor 方便一些
    def restrict_loss(self, loss):
        if self.min > loss:
            self.min = loss
        return self.min

    def update_mask_factor(self, loss):
        # 如果出现了更小的 loss， mask_factor=1
        # 如果 loss 变大了，通过控制 mask_factor 限制 mask_loss 在合理范围内
        if self.min > loss.item():  # 初始训练阶段更新 loss 过程
            self.min = loss.item()
        else:  # loss 开始增加了, 注意只返回标量数值
            self.mask_factor = self.min / loss.item()

        return self.mask_factor
