import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
#sys.path.append("C:/Users/Administrator/Desktop/bilateralfilter/build/lib.linux-x86_64-3.6")
from bilateralfilter import bilateralfilter, bilateralfilter_batch


class DenseEnergyLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):     # （scale后的）原始图像,分割结果softmax,sigma_rgb,sigma_xy*scale_factor,ROIs,unlabel_region
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape    # BCHW
        Gate = ROIs.cuda()                                  # BHW

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)
        seg_max = torch.max(segmentations, dim=1)[0]        # torch.max返回两个结果，第一个结果为value,第二个为indice
        Gate = Gate - seg_max                               # 有标记像元的权重为1-max(P)
        Gate[unlabel_region] = 1                            # 无标记像元的权重较大（为1）
        Gate[Gate < 0] = 0                                  # 还有负值？？？
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)    # B1HW ==> BCHW

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())        # e.g. not interested for padded region ？？？？？？
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()           # 注意进行了flatten
        AS = np.zeros(segmentations.shape, dtype=np.float32)            # 注意shape是flatten后的
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)   # 猜测：通过原始img和预测score计算像元之间的能量函数，保存在AS中
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)                                      # 对应元素相乘（由于进行了flatten故相当于向量相乘）
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None
    

class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):  # 原始图像，分割网络softmax，cropping，reliable label
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False)   # BCHW (SOTFMAX)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor).squeeze(1)                # BWH (F.interpolate对WH进行resize时需要输入四维（对后两个维度内插）
        scaled_seg_label = F.interpolate(seg_label.float(),scale_factor=self.scale_factor,mode='nearest')               # B1HW
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)                                            # BHW
        return self.weight*DenseEnergyLossFunction.apply(               # xxx.apply
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )
