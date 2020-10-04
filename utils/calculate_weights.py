import sys

sys.path.insert(0, '/nfs/xs/tmp/SegPAM')

import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from datasets.transforms import get_test_transforms
import copy
import matplotlib.pyplot as plt
from datasets.base_dataset import BaseDataset
from utils.misc import read_txt_as_list


def calculate_class_freqs(dataset, num_classes):
    z = np.zeros((num_classes,))
    dataset = copy.deepcopy(dataset)  # 替换 transforms，计算 class weights

    if num_classes == 19:
        img_size = (512, 1024)  # 原尺度太大了
    elif num_classes == 11:
        img_size = (360, 480)
    elif num_classes == 37:
        img_size = (480, 640)
    else:
        raise NotImplementedError

    dataset.transforms = get_test_transforms(img_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print("Calculating class weights..")

    # 计算每个像素点数目
    for (image, target) in tqdm(dataloader):  # 如果 trainset 有 random crop 这样就不太好了
        y = target.cpu().numpy()
        mask = np.logical_and((y >= 0), (y < num_classes))  # 逻辑或，合理区域
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)  # 19D vec
        z += count_l

    class_ratios = z / z.sum()
    return z, class_ratios


def calculate_class_weights(dataset, num_classes):
    """
    weight = 1/√num, 再归一化
    """
    z, _ = calculate_class_freqs(dataset, num_classes)
    z = np.nan_to_num(np.sqrt(1 + z))  # smooth num, 防止下文分母 frequency=0
    class_weights = [1 / f for f in z]  # frequency

    ret = np.nan_to_num(np.array(class_weights))  # NaN is replaced by zero
    # ret[ret > 2 * np.median(ret)] = 2 * np.median(ret)  # > 2倍中位数的 用后者代替，避免过大 weight [Pedestrian, Bicyclist]
    ret = ret / ret.sum()
    print('Class weights: ')
    print(ret)

    return ret


def plt_freq_bar(frequency, xlabels):
    print(frequency)
    x, y = range(len(frequency)), frequency
    plt.bar(x, y)
    # x 轴标签
    plt.xticks(x, xlabels, size='small', rotation=30)
    # y 轴数字标签
    for a, b in zip(x, y):
        plt.text(a, b + 0.002, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    plt.title('CamVid class frequency')
    plt.show()


labels = {
    'CamVid': ['Sky', 'Building', 'Pole', 'Road', 'Pavement',
               'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
}


def cal_full_ratios():
    num_classes = 11

    full_ratios = np.zeros((0, num_classes))

    root = '/nfs/xs/tmp/SegPAM/runs/CamVid/alpha1_Jun14_164450'
    for split in range(10, 41, 5):
        print(split)
        run_dir = f'{root}/runs_0{split}'
        img_paths = read_txt_as_list(f'{run_dir}/label_imgs.txt')
        target_paths = read_txt_as_list(f'{run_dir}/label_targets.txt')

        trainset = BaseDataset(img_paths, target_paths)
        _, class_ratios = calculate_class_freqs(trainset, num_classes)
        full_ratios = np.vstack((full_ratios, class_ratios))

    np.save(os.path.join(root, 'full_class_ratios.npy'), full_ratios)


def plt_results(full_ratios):
    dataset = 'CamVid'
    lbl = labels[dataset]
    x = list(range(10, 41, 5))
    for i in range(len(lbl)):
        ratios = full_ratios[:, i]  # class i
        plt.plot(x, ratios, '-', label=lbl[i])

    plt.legend(loc='best')  # 两列
    plt.xlabel('% of Labeled Data')
    plt.ylabel('class ratio')
    plt.show()


def cal_entropy(probs):
    return -np.nansum(np.multiply(probs, np.log(probs)))


"""
active_coreset_701_Jun15_110606             1.7794090569696586
active_vaal_701_Jun24_161747                1.783065693068772
active_dropout_701_Jun15_110938             1.8055311855646556
active_entropy_701_Jun15_160105             1.8174121848637033
active_region_entropy8_701_Jun21_191044     1.82820586931263
"""


def cal_class_entropy():
    exp = 'active_coreset_400_Jun23_164255'
    root = os.path.join('runs/CamVid', exp)

    print(exp)
    ens = []
    for step in range(15, 41, 5):
        run_dir = f'{root}/runs_0{step}'
        img_paths = read_txt_as_list(f'{run_dir}/label_imgs.txt')
        target_paths = read_txt_as_list(f'{run_dir}/label_targets.txt')

        trainset = BaseDataset(img_paths, target_paths)
        _, class_ratios = calculate_class_freqs(trainset, num_classes=11)
        en = cal_entropy(class_ratios)
        ens.append(en)
    print(','.join(map(str, ens)))
    print(np.mean(ens))


if __name__ == '__main__':
    exp = 'active_region_entropy8_701_Jun21_191044'
    root = os.path.join('runs/CamVid', exp)
    num_classes = 11

    print(exp)
    al_class_freqs = []

    # steps = range(15, 21, 5)
    # steps = [15]
    for step in range(15, 41, 5):
        run_dir = f'{root}/runs_0{step}'
        img_paths = read_txt_as_list(f'{run_dir}/label_imgs.txt')
        target_paths = read_txt_as_list(f'{run_dir}/label_targets.txt')

        trainset = BaseDataset(img_paths, target_paths)
        _, class_freq = calculate_class_freqs(trainset, num_classes)
        al_class_freqs.append(class_freq)
        print(class_freq)

    al_class_freqs = np.array(al_class_freqs)
    print(al_class_freqs)
