import numpy as np

'''若因分母为0导致警告很烦，可以在分母上加1e-10'''

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
    def Overall_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        # Acc = np.nanmean(Acc)
        return Acc

    def Recall_Class(self):
        Rec = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # Rec = np.nanmean(Rec)
        return Rec

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Kappa(self):
        pe_rows = np.sum(self.confusion_matrix, axis=0)
        pe_cols = np.sum(self.confusion_matrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        po = np.trace(self.confusion_matrix) / float(sum_total)
        return (po - pe) / (1 - pe)

    def F1_Class(self):
        p, r = self.Pixel_Accuracy_Class(), self.Recall_Class()
        F1 = 2*(p*r) / (p+r)
        return F1

    def Mean_F1(self):
        p, r = self.Pixel_Accuracy_Class(), self.Recall_Class()
        F1 = 2*(p*r) / (p+r)
        mf1 = np.nanmean(F1)
        return mf1

    def _generate_matrix(self, gt_image, pre_img):
        pre_image = pre_img.copy()
        mask = (gt_image >= 0) & (gt_image < self.num_class)        # 用来掩膜掉一部分像元如忽略像元
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # print(gt_image.shape,pre_image.shape)
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




