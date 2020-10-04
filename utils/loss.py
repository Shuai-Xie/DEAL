import torch
import torch.nn.functional as F
import constants


class CELoss:
    def __call__(self, output, target):
        ph, pw = output.size(2), output.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            output = F.interpolate(input=output, size=(h, w), mode='bilinear', align_corners=True)
        return F.cross_entropy(output, target, ignore_index=constants.BG_INDEX)


class MaskLoss:

    def __init__(self, mode):
        if mode == 'bce':
            self.loss_fn = self.BCE
        elif mode == 'wce':
            self.loss_fn = self.weightBCE
        else:
            raise NotImplementedError

    def __call__(self, soft_mask, target_error_mask):
        soft_mask = soft_mask.squeeze(1)  # B,1,H,W -> B,H,W
        return self.loss_fn(soft_mask, target_error_mask)

    @staticmethod
    def BCE(soft_mask, target_error_mask):
        loss = F.binary_cross_entropy(soft_mask, target_error_mask)
        return loss

    @staticmethod
    def weightBCE(soft_mask, target_error_mask):
        loss = F.binary_cross_entropy(soft_mask, target_error_mask, reduction='none')

        error_mask = target_error_mask.bool()
        right_mask = ~error_mask
        weights = cal_class_weights(right_mask, error_mask)

        loss = mask_mean(loss, right_mask) * weights[0] + mask_mean(loss, error_mask) * weights[1]
        return loss


def mask_mean(x, mask):
    return x[mask].mean() if mask.sum() > 0 else torch.tensor([0.]).cuda()


def cal_class_weights(right_mask, error_mask):
    pixel_num = torch.tensor([right_mask.sum(), error_mask.sum()]).float().cuda()
    class_weights = 1 / torch.sqrt(pixel_num + 1)
    class_weights = class_weights / class_weights.sum()
    return class_weights
