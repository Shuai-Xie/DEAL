import os
import glob
import random
from utils.misc import write_list_to_txt, read_txt_as_list, get_select_paths_by_idxs

root = '/nfs/xs/Datasets/Segment/Cityscapes'

"""
5000 = train: 2975, val: 500, test: 1525 (no label, only on server)
"""

void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]  # 按正序，不会重复
class_map = {valid_classes[i]: i + 1 for i in range(len(valid_classes))}


def remap_19idxs(mask):
    for _voidc in void_classes:
        mask[mask == _voidc] = 0  # set bg_idx=0
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask


def get_city_sample_paths(city, split='train'):
    img_dir = f'{root}/leftImg8bit/{split}/{city}'
    target_dir = f'{root}/gtFine/{split}/{city}'

    imgs = sorted(os.listdir(img_dir))
    img_paths, target_paths = [], []

    for img_name in imgs:
        img_paths.append(os.path.join(img_dir, img_name))
        target_paths.append(os.path.join(target_dir, img_name.replace('leftImg8bit', 'gtFine_labelIds')))

    return img_paths, target_paths


def get_img_target_paths(split):
    img_dir = os.path.join(root, 'leftImg8bit', split)  # image 所在目录
    target_dir = os.path.join(root, 'gtFine', split)

    img_paths = glob.glob(os.path.join(img_dir, '*', '*_leftImg8bit.png'))  # * are citys
    target_paths = [
        os.path.join(target_dir, p.split('/')[-2], p.split('/')[-1].replace('leftImg8bit', 'gtFine_labelIds'))
        for p in img_paths]
    return img_paths, target_paths


def save_data_path_to_txt(val_num=300):
    # train/val, split 300 val from train
    img_paths, target_paths = get_img_target_paths('train')
    img_idxs = list(range(len(img_paths)))

    random.seed(100)  # 只用1次，分好数据后就不用了
    random.shuffle(img_idxs)

    val_idxs, train_idxs = img_idxs[:val_num], img_idxs[val_num:]

    # val
    val_img_paths, val_target_paths = get_select_paths_by_idxs(img_paths, target_paths, val_idxs)
    write_list_to_txt(val_img_paths, txt_path=os.path.join(root, 'val_img_paths.txt'))
    write_list_to_txt(val_target_paths, txt_path=os.path.join(root, 'val_target_paths.txt'))

    # train
    train_img_paths, train_target_paths = get_select_paths_by_idxs(img_paths, target_paths, train_idxs)
    write_list_to_txt(train_img_paths, txt_path=os.path.join(root, 'train_img_paths.txt'))
    write_list_to_txt(train_target_paths, txt_path=os.path.join(root, 'train_target_paths.txt'))

    # test
    test_img_paths, test_target_paths = get_img_target_paths('val')  # 原始 val 作为 test
    write_list_to_txt(test_img_paths, txt_path=os.path.join(root, 'test_img_paths.txt'))
    write_list_to_txt(test_target_paths, txt_path=os.path.join(root, 'test_target_paths.txt'))


def check_txt():
    val_img_paths = read_txt_as_list(os.path.join(root, 'val_img_paths.txt'))
    val_target_paths = read_txt_as_list(os.path.join(root, 'val_target_paths.txt'))

    for ip, tp in zip(val_img_paths, val_target_paths):
        print(ip)
        print(tp)
        print()


if __name__ == '__main__':
    check_txt()
