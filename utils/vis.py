import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
from utils.misc import minmax_normalize
import torch


def get_label_name_colors(csv_path):
    """
    read csv_file and save as label names and colors list
    :param csv_path: csv color file path
    :return: lable name list, label color list
    """
    label_names, label_colors = [], []
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader):
            label_names.append(row[0])
            label_colors.append([int(row[1]), int(row[2]), int(row[3])])

    return label_names, label_colors


def recover_color_img(img):
    """
    cvt tensor image to RGB [note: not BGR]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().squeeze().cpu().numpy()

    img = np.transpose(img, axes=[1, 2, 0])  # h,w,c
    img = img * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)  # 直接通道相成?
    img = (img * 255).astype('uint8')
    return img


def color_code_target(target, label_colors):
    return np.array(label_colors, dtype='uint8')[target.astype(int)]


def get_legends(class_set, label_names, label_colors):
    legend_names, legend_lines = [], []
    for i in class_set:
        legend_names.append(label_names[i])  # 图例
        legend_lines.append(Line2D([0], [0], color=map_color(label_colors[i]), lw=2))  # 颜色线
    return legend_names, legend_lines


def map_color(rgb):
    return [v / 255 for v in rgb]


def plt_img_target_pred(img, target, pred, label_names, label_colors, vertial=False):
    # target_class_set = set(target.astype('int').flatten().tolist())
    # pred_class_set = set(pred.astype('int').flatten().tolist())
    # target_leg_names, target_leg_lines = get_legends(target_class_set, label_names, label_colors)
    # pred_leg_names, pred_leg_lines = get_legends(pred_class_set, label_names, label_colors)

    if vertial:
        f, axs = plt.subplots(nrows=3, ncols=1)
        f.set_size_inches((4, 9))
    else:
        f, axs = plt.subplots(nrows=1, ncols=3)
        f.set_size_inches((10, 3))

    ax1, ax2, ax3 = axs.flat[0], axs.flat[1], axs.flat[2]

    ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('img')

    ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    # ax2.legend(target_leg_lines, target_leg_names, loc=1)
    ax2.set_title('target')

    ax3.axis('off')
    ax3.imshow(color_code_target(pred, label_colors))
    # ax3.legend(pred_leg_lines, pred_leg_names, loc=1)
    ax3.set_title('predict')

    plt.show()


def plt_img_target(img, target, label_colors):
    f, axs = plt.subplots(nrows=1, ncols=2)
    f.set_size_inches((8, 3))
    ax1, ax2 = axs.flat[0], axs.flat[1]

    ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('img')

    ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    # ax2.legend(target_leg_lines, target_leg_names, loc=1)
    ax2.set_title('target')

    plt.show()


def plt_img_target_diff_error(img, target, target_error_mask, diff_map, label_colors):
    f, axs = plt.subplots(nrows=2, ncols=2)
    f.set_size_inches((12, 6))  # 800, 600
    ax1, ax2, ax3, ax4 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3]

    ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('Image')

    ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    ax2.set_title('GT')

    ax3.axis('off')
    ax3.imshow(diff_map, cmap='jet')
    ax3.set_title('Semantic Difficulty Map')

    ax4.axis('off')
    ax4.imshow(color_code_target(target_error_mask, label_colors))
    ax4.set_title('Error Mask')

    plt.show()


def get_plt_img_target_gt_ceal(img, target, gt, ceal, label_colors, figsize=(8, 6), title=None):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(figsize)
    ax1, ax2, ax3, ax4 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3]

    ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('img')

    ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))
    ax2.set_title('target')

    ax3.axis('off')
    ax3.imshow(color_code_target(gt, label_colors))
    ax3.set_title('gt')

    ax4.axis('off')
    ax4.imshow(color_code_target(ceal, label_colors))
    ax4.set_title('ceal')

    if title:
        plt.suptitle(title)

    # cvt plt result to np img
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape((h, w, 3))  # 转成 img 实际大小
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.cla()
    plt.close("all")

    return img


def plt_img_target_error(img, target, error_mask, label_colors, save_path=None):
    f, axs = plt.subplots(nrows=1, ncols=3, dpi=200)
    f.set_size_inches((10, 2))

    ax1, ax2, ax3 = axs.flat[0], axs.flat[1], axs.flat[2]
    ax1.axis('off')
    ax1.imshow(img)

    ax2.axis('off')
    ax2.imshow(color_code_target(target, label_colors))

    ax3.axis('off')
    ax3.imshow(error_mask, cmap='jet')

    f.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.02)  # 调整子图间距(inch)，存储时能看到调节了间距

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)

    plt.show()


def plt_att(target, pred, label_colors, atten,
            target_error_mask, pred_error_mask, title=None):
    f, axs = plt.subplots(nrows=2, ncols=3)
    f.set_size_inches((10, 6))

    ax1, ax2, ax3 = axs.flat[0], axs.flat[1], axs.flat[2]
    ax4, ax5, ax6 = axs.flat[3], axs.flat[4], axs.flat[5]

    # semantic
    ax1.axis('off')
    ax1.imshow(color_code_target(target, label_colors))
    ax1.set_title('target')

    # mask
    ax2.axis('off')
    ax2.imshow(color_code_target(target_error_mask, label_colors))
    ax2.set_title('target_error_mask')

    # att
    if atten is not None:
        ax3.axis('off')
        ax3.imshow(atten, cmap='jet')
        ax3.set_title('attention')

    # predict
    ax4.axis('off')
    ax4.imshow(color_code_target(pred, label_colors))
    ax4.set_title('predict')

    # error mask
    ax5.axis('off')
    ax5.imshow(pred_error_mask, cmap='jet')
    ax5.set_title('pred_error_mask')

    ax6.axis('off')

    if title:
        plt.suptitle(title)

    plt.show()


def plt_cmp(img, lc, ms, entro, dropout, error_mask, save_path=None):
    f, axs = plt.subplots(nrows=1, ncols=6, dpi=200)
    f.set_size_inches((20, 3))

    maps = [img, lc, ms, entro, dropout, error_mask]
    titles = ['Image', 'LC', 'MS', 'Entropy', 'Dropout', 'Ours']

    ax = axs.flat[0]
    ax.axis('off')
    ax.imshow(maps[0])
    # ax.set_title(titles[0])

    for i in range(1, 6):
        ax = axs.flat[i]
        ax.axis('off')
        if maps[i] is not None:
            ax.imshow(maps[i], cmap='jet')
        # ax.set_title(titles[i])

    f.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.04)  # 调整子图间距(inch)，存储时能看到调节了间距

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)

    plt.show()


def plt_cmp_v2(img, lc, ms, en, dr, pred_error_mask, save_path=None):
    f, axs = plt.subplots(nrows=2, ncols=5, dpi=200)
    f.set_size_inches((16, 5))

    # img
    ax = axs.flat[0]
    ax.axis('off')
    ax.imshow(img)

    # norm uncer map
    lc, ms, en = minmax_normalize(lc), minmax_normalize(ms), minmax_normalize(en)
    dr = minmax_normalize(dr)

    maps = [lc, ms, en, dr]

    for i in range(len(maps)):
        ax = axs.flat[i + 1]
        ax.axis('off')
        ax.imshow(maps[i], cmap='jet')

    # pred error mask
    ax = axs.flat[5]
    ax.axis('off')
    ax.imshow(pred_error_mask, cmap='jet')

    # semantic attention uncer_map, and normlize
    for i in range(len(maps)):
        ax = axs.flat[i + 6]
        ax.axis('off')
        att_uncer_map = minmax_normalize(maps[i] * pred_error_mask)
        # att_uncer_map = maps[i] * pred_error_mask
        ax.imshow(att_uncer_map, cmap='jet')

    f.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.04)  # 调整子图间距(inch)，存储时能看到调节了间距

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)

    plt.show()


def plt_compare(target, pred, label_colors,
                target_error_mask, pred_error_mask, lc, ms, en, mc_droput=None):
    # 不同方法得到的 uncertain map 对比
    f, axs = plt.subplots(nrows=2, ncols=4)
    f.set_size_inches((12, 6))

    ax1, ax2, ax3, ax4 = axs.flat[0], axs.flat[1], axs.flat[2], axs.flat[3]
    ax5, ax6, ax7, ax8 = axs.flat[4], axs.flat[5], axs.flat[6], axs.flat[7]

    # semantic
    ax1.axis('off')
    ax1.imshow(color_code_target(target, label_colors))
    ax1.set_title('target')

    ax5.axis('off')
    ax5.imshow(color_code_target(pred, label_colors))
    ax5.set_title('predict')

    # mask
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(target_error_mask, cmap='gray')
    ax2.set_title('target_error_mask')

    ax6.axis('off')
    ax6.imshow(pred_error_mask, cmap='jet')
    ax6.set_title('pred_error_mask')

    # compare
    ax3.axis('off')
    ax3.imshow(lc, cmap='jet')
    ax3.set_title('least confidence')

    ax4.axis('off')
    ax4.imshow(ms, cmap='jet')
    ax4.set_title('margin sampling')

    ax7.axis('off')
    ax7.imshow(en, cmap='jet')
    ax7.set_title('entropy')

    ax8.axis('off')
    ax8.set_title('mc droput')

    plt.show()
