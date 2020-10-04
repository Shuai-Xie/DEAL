import numpy as np


# https://stats.stackexchange.com/questions/179835/how-to-build-a-confusion-matrix-for-a-multiclass-classifier

def calculate_miou(confusion_matrix):
    MIoU = np.divide(np.diag(confusion_matrix), (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix)))
    MIoU = np.nanmean(MIoU)
    return MIoU


class Evaluator(object):

    def __init__(self, num_class):
        np.seterr(divide='ignore', invalid='ignore')
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  # TP / ALL pixels
        return Acc

    def Acc_of_each_class(self):
        Accs = np.divide(np.diag(self.confusion_matrix),  # TP of each class
                         self.confusion_matrix.sum(axis=1))  # (TP+FP) of each class 列
        return Accs  # vector

    def Mean_Pixel_Accuracy(self):
        Acc = np.nanmean(self.Acc_of_each_class())  # mean of vector
        return Acc

    def IOU_of_each_class(self):
        inter = np.diag(self.confusion_matrix)  # TP
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - inter  # TP+FP+FN
        IoUs = np.divide(inter, union)  # IoU of each class
        return IoUs  # vector

    def Mean_Intersection_over_Union(self):
        MIoU = np.nanmean(self.IOU_of_each_class())  # mIoU
        return MIoU

    def Mean_Intersection_over_Union_20(self):  # 20类之后
        MIoU = 0
        if self.num_class > 20:
            subset_20 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 23, 27, 32, 33, 35, 38])
            confusion_matrix = self.confusion_matrix[subset_20[:, None], subset_20]  # 取出子矩阵
            MIoU = np.divide(np.diag(confusion_matrix), (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix)))
            MIoU = np.nanmean(MIoU)
        return MIoU

    def Mean_Dice(self):
        inter = np.diag(self.confusion_matrix)  # vector
        dices = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0)
        dice = np.divide(2 * inter, dices)
        dice = np.nanmean(dice)
        return dice

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.divide(np.diag(self.confusion_matrix), (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix)))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]  # i:gt, j:pre
        count = np.bincount(label, minlength=self.num_class ** 2)  # total classes on confusion_matrix
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image, return_miou=False):
        assert gt_image.shape == pre_image.shape  # np img, B,H,W
        confusion_matrix = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += confusion_matrix
        if return_miou:
            return calculate_miou(confusion_matrix)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def dump_matrix(self, path):
        np.save(path, self.confusion_matrix)
