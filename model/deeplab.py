import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .backbone import build_backbone
from .aspp import build_aspp
from .decoder import SegmentHead, MaskHead, MaskHead_branch
import os


class DeepLab(nn.Module):

    def __init__(self,
                 backbone='resnet50', output_stride=16, num_classes=11,
                 sync_bn=False, mc_dropout=False,
                 with_mask=False, with_pam=False, branch_early=False):
        super(DeepLab, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, mc_dropout)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)

        # low level features
        if backbone.startswith('resnet') or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.low_level_conv = nn.Sequential(nn.Conv2d(low_level_inplanes, 48, 1, bias=False),
                                            BatchNorm(48),
                                            nn.ReLU())
        # segment
        self.seg_head = SegmentHead(num_classes, BatchNorm)

        # error mask -> difficulty branch
        self.with_mask = with_mask
        self.branch_early = branch_early
        if with_mask:
            if branch_early:
                self.mask_head = MaskHead_branch(304, num_classes, BatchNorm, with_pam)
            else:
                self.mask_head = MaskHead(num_classes, with_pam)

        self.return_features = False
        self.return_attention = False

    def forward(self, inputs):
        backbone_feat, low_level_feat = self.backbone(inputs)  # 1/16, 1/4;
        x = self.aspp(backbone_feat)  # 1/16 -> aspp -> 1/16

        # low + high features
        low_level_feat = self.low_level_conv(low_level_feat)  # 256->48
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)  # 1/4
        second_to_last_features = torch.cat((x, low_level_feat), dim=1)  # 304=256+48

        # segment
        x = self.seg_head(second_to_last_features)

        if self.with_mask:
            if self.branch_early:
                mask, attention = self.mask_head(second_to_last_features)  # 1/4 features same to seg_head
            else:
                mask, attention = self.mask_head(x)  # segment output

            x = F.interpolate(x, size=inputs.size()[2:], mode='bilinear', align_corners=True)
            mask = F.interpolate(mask, size=inputs.size()[2:], mode='bilinear', align_corners=True)  # nearest can't use align_corners
            mask = torch.sigmoid(mask)
            if self.return_attention:
                return x, mask, attention
            else:
                return x, mask
        else:
            x = F.interpolate(x, size=inputs.size()[2:], mode='bilinear', align_corners=True)
            if self.return_features:
                return x, second_to_last_features  # for coreset
            else:
                return x

    def set_return_features(self, return_features):  # True or False
        self.return_features = return_features

    def set_return_attention(self, return_attention):  # True or False
        self.return_attention = return_attention

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.low_level_conv, self.seg_head]
        if self.with_mask:
            modules.append(self.mask_head)
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def load_pretrain(self, pretrained):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')['state_dict']
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}  # 不加载最后的 head 参数
            # for k, v in pretrained_dict.items():
            #     print('=> loading {} | {}'.format(k, v.size()))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            print('No such file {}'.format(pretrained))
