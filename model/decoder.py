import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import PAM
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import constants


class SegmentHead(nn.Module):

    def __init__(self, num_classes, BatchNorm):
        super(SegmentHead, self).__init__()
        self.segment_head = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          BatchNorm(256),  # 304=256+48
                                          nn.ReLU(),
                                          nn.Dropout(0.5),  # decoder dropout 1
                                          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          BatchNorm(256),
                                          nn.ReLU(),
                                          nn.Dropout(constants.MC_DROPOUT_RATE),  # MC dropout
                                          nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x):
        x = self.segment_head(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


att_map_size = {
    11: (60, 80),  # CamVid
    19: (86, 86)  # Cityscapes, 86*8=688
}


class MaskHead(nn.Module):
    def __init__(self, num_classes, with_pam=False):
        super(MaskHead, self).__init__()
        self.att_size = att_map_size[num_classes]
        self.pam = PAM() if with_pam else None
        self.mask_head = nn.Conv2d(num_classes, 1, kernel_size=1, stride=1, bias=True)
        self.with_pam = with_pam

    def forward(self, inputs):  # 1/4 feature
        if self.with_pam:
            x = F.interpolate(inputs, size=self.att_size, mode='bilinear', align_corners=True)
            feat, attention = self.pam(x)
            mask = self.mask_head(feat)
            return mask, attention
        else:
            mask = self.mask_head(inputs)
            return mask, None


class MaskHead_branch(nn.Module):
    def __init__(self, in_dim, num_classes, BatchNorm, with_pam=False):
        super(MaskHead_branch, self).__init__()
        self.att_size = att_map_size[num_classes]
        self.branch = nn.Sequential(nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(256),  # 304=256+48
                                    nn.ReLU(),
                                    nn.Dropout(0.5),  # decoder dropout 1
                                    nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(num_classes),  # nc features
                                    nn.ReLU())
        self.pam = PAM() if with_pam else None
        self.mask_head = nn.Conv2d(num_classes, 1, kernel_size=1, stride=1, bias=True)
        self.with_pam = with_pam

        self._init_weight()

    def forward(self, inputs):
        inputs = self.branch(inputs)  # independent features
        if self.with_pam:
            x = F.interpolate(inputs, size=self.att_size, mode='bilinear', align_corners=True)
            feat, attention = self.pam(x)
            mask = self.mask_head(feat)
            return mask, attention
        else:
            mask = self.mask_head(inputs)
            return mask, None

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
