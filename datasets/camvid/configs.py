import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


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


def get_stats():
    stats = {
        'label': [
            'Sky', 'Building', 'Pole', 'Road', 'Pavement',
            'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'
        ],
        'pixel_count': np.array([
            7.6801e+07, 1.1737e+08, 4.7987e+06, 1.4054e+08, 3.3614e+07,
            5.4259e+07, 5.2242e+06, 6.9211e+06, 6.9211e+06, 3.4029e+06, 2.5912e+06
        ]),
        'img_pixel_count': np.array([
            4.8315e+08, 4.8315e+08, 4.8315e+08, 4.8453e+08, 4.7209e+08,
            4.4790e+08, 4.6863e+08, 2.5160e+08, 4.8315e+08, 4.4444e+08, 2.6196e+08
        ])
    }

    # pixel 总数 / 包含此类的 image pixel 总数 [更科学]
    pixel_freq = stats['pixel_count'] / np.sum(stats['pixel_count'])
    # plt_freq_bar(pixel_freq, xlabels=stats['label'])

    img_freq = stats['pixel_count'] / stats['img_pixel_count']
    class_weights = np.median(img_freq) / img_freq

    stats['pixel_freq'] = pixel_freq
    stats['img_freq'] = img_freq
    stats['class_weights'] = class_weights
    # pprint(stats)

    return stats


# get_stats()

from datasets.build_datasets import data_cfg

label_names, _ = data_cfg['Cityscapes']['label_colors']

label_names = [s.title() for s in label_names]
print(label_names)
