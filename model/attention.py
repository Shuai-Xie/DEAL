"""
ref: Positional attention module in `Dual attention network for scene segmentation`.
"""

import torch
import torch.nn as nn


class PAM(nn.Module):
    """ Probability attention module """

    #
    def __init__(self):
        super(PAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的 attention 系数
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        B, C, H, W = x.size()
        x = x.softmax(dim=1)  # logits -> prob

        if self.gamma < -0.01:
            out = x
            attention = None
        else:
            proj_value = x.view(B, -1, W * H)  # D reshape (B,C,H*W)

            proj_query = x.view(B, -1, W * H).permute(0, 2, 1)  # B reshape & transpose (B,H*W,C)
            proj_key = x.view(B, -1, W * H)  # C reshape (B,C,H*W)
            energy = torch.bmm(proj_query, proj_key)  # batch matrix multiplication, (B, H*W, H*W)
            attention = self.softmax(energy)

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B,C,H*W)
            out = out.view(B, C, H, W)  # new attentioned features

            out = self.gamma * out + x

        return out, attention
