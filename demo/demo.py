import os
import sys

sys.path.insert(0, '/nfs/xs/codes/DEAL')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"

import torch
import numpy as np
import random
import cv2

from model.deeplab import DeepLab

from utils.vis import *
from utils.misc import to_numpy, mkdir, generate_target_error_mask

from datasets.build_datasets import data_cfg
from datasets.base_dataset import BaseDataset
from datasets.cityscapes.helper import get_city_sample_paths, remap_19idxs
from datasets.transforms import get_transform, remap, mapbg, get_img_transfrom

from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

cfg = data_cfg['Cityscapes']
root = cfg['root']
base_size, crop_size = cfg['base_size'], cfg['crop_size']
num_classes = cfg['num_classes']
label_names, label_colors = cfg['label_colors']


def load_model():
    model = DeepLab('mobilenet', 16, num_classes, with_mask=True, with_pam=True)
    ckpt_path = 'runs/Cityscapes/diff_score_200_Jun30_133011/runs_040/checkpoint.pth.tar'
    model.load_pretrain(ckpt_path)
    return model.cuda().eval()


@torch.no_grad()
def demo_city(city='hanover', split='train'):
    img_paths, target_paths = get_city_sample_paths(city, split)

    dataset = BaseDataset(img_paths, target_paths)
    dataset.transform = get_transform('test', base_size=(800, 400))

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, pin_memory=False, num_workers=4)
    model = load_model()

    bg_idx = 0
    mapbg_fn = mapbg(bg_idx)
    remap_fn = remap(bg_idx)

    for idx, sample in enumerate(dataloader):
        img, target = sample['img'].cuda(), sample['target']
        img_name = os.path.basename(sample['img_path'][0])
        print(img_name)

        output, diff_map = model(img)

        pred = torch.argmax(output, dim=1)
        target, pred = to_numpy(target, toint=True), to_numpy(pred, toint=True)  # H,W
        target = mapbg_fn(remap_19idxs(target))  # BaseDataset 加载的对象，没有预处理
        target_error_mask = generate_target_error_mask(pred, target,
                                                       class_aware=True, num_classes=num_classes)
        img = recover_color_img(img)
        target, pred = remap_fn(target), remap_fn(pred)
        target_error_mask = remap_fn(target_error_mask)
        diff_map = to_numpy(diff_map)

        plt_img_target_diff_error(
            img, target, target_error_mask, diff_map, label_colors
        )


@torch.no_grad()
def demo_video():
    scene = 'stuttgart_02'
    img_dir = f'/nfs/xs/Datasets/Segment/Cityscapes/leftImg8bit/demoVideo/{scene}'
    diff_dir = f'result/{scene}/diff'
    pred_dir = f'result/{scene}/pred'
    mkdir(diff_dir)
    mkdir(pred_dir)

    img_names = sorted(os.listdir(img_dir))
    transform = get_img_transfrom(base_size=(800, 400))

    model = load_model()
    remap_fn = remap(0)

    for img_name in tqdm(img_names):
        img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
        img = transform(img).unsqueeze(0).cuda()

        output, diff_map = model(img)

        pred = torch.argmax(output, dim=1)
        pred = to_numpy(pred, toint=True)
        pred = remap_fn(pred)

        cv2.imwrite(f'{pred_dir}/{img_name}', color_code_target(pred, label_colors)[:, :, ::-1])

        diff_map = to_numpy(diff_map)

        plt.figure(figsize=(8, 4))
        plt.imshow(diff_map, cmap='jet')
        plt.axis('off')
        plt.savefig(f'{diff_dir}/{img_name}', bbox_inches='tight', pad_inches=0.)
        plt.close()

        # plt_img_target_error(img, pred, diff_map, label_colors)


def save_video(scene='stuttgart_02'):
    img_dir = f'/nfs/xs/Datasets/Segment/Cityscapes/leftImg8bit/demoVideo/{scene}'
    diff_dir = f'assests/demo/{scene}/diff'
    pred_dir = f'assests/demo/{scene}/pred'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_w, video_h = 400, 200

    def write_video(src_dir, out):
        print('read from', src_dir)
        vw = cv2.VideoWriter(out, fourcc, 20, (video_w, video_h))
        imgs = sorted(os.listdir(src_dir))
        while '@eaDir' in imgs:
            imgs.remove('@eaDir')
        for img in tqdm(imgs):
            frame = cv2.imread(f'{src_dir}/{img}')
            frame = cv2.resize(frame, dsize=(video_w, video_h))
            vw.write(frame)
        vw.release()

    write_video(img_dir, out=f'assests/demo/{scene}/{scene}.mp4')
    write_video(diff_dir, out=f'assests/demo/{scene}/{scene}_diff.mp4')
    write_video(pred_dir, out=f'assests/demo/{scene}/{scene}_pred.mp4')


if __name__ == '__main__':
    # demo_city()
    # demo_video()
    # save_video()
    for scene in ['stuttgart_00', 'stuttgart_01', 'stuttgart_02']:
        save_video(scene)
