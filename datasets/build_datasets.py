from datasets import data_cfg


def build_datasets(dataset, base_size, crop_size, init_percent=None):
    if dataset not in data_cfg:
        raise NotImplementedError('no such dataset')

    cfg = data_cfg[dataset]
    root = cfg['root']
    cls, active_cls = cfg['cls']

    if init_percent is None:  # 构建普通数据集
        trainset = cls(root, 'train', base_size, crop_size)  # model input size
    else:
        trainset = active_cls(root, 'train', base_size, crop_size, init_percent)

    valset = cls(cfg['root'], 'val', base_size, crop_size)
    testset = cls(cfg['root'], 'test', base_size, crop_size)

    return trainset, valset, testset


if __name__ == '__main__':
    from utils.vis import plt_img_target, recover_color_img

    dataset = 'Cityscapes'
    cfg = data_cfg[dataset]
    label_names, label_colors = cfg['label_colors']

    trainset, valset, testset = build_datasets(dataset,
                                               base_size=(1024, 512),
                                               crop_size=(512, 512), init_percent=10)
    print(len(trainset))  # 300, 10%
    print(len(valset))  # 300
    print(len(testset))  # 500

    for idx, sample in enumerate(trainset):
        img, target = sample['img'], sample['target']
        img = recover_color_img(img)
        target = trainset.remap_fn(target.numpy().astype('uint8'))

        plt_img_target(img, target, label_colors)

        if idx == 10:
            exit(0)
