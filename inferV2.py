from glob import glob
import numpy as np
from tqdm import tqdm
from torch.backends import cudnn
import imageio
import PIL.Image
from utils import pyutils
import argparse
import cv2
import os.path
from core.networks import *
from utils.metrics import Evaluator
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import shutil
import math
from dataloaders import imutils, data_loader
import torch

'''文件夹中逐图像读取进行预测并使用overlapping strategy计算精度'''
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
cudnn.enabled = True


def _crf_with_alpha(pred_prob, ori_img):
    bgcam_score = pred_prob.cpu().data.numpy()
    crf_score = imutils.crf_inference_inf(ori_img, bgcam_score, labels=2)
    return crf_score

def infer(model_weights, args):
    transform = transforms.Compose([
        imutils.FixScaleResizeImage(args.crop_size),
        imutils.NormalizeImage(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 归一化和标准化 runways2
        imutils.ToTensorImage()
    ])

    model = DeepLabv3_Plus('resnet101', num_classes = 2, mode='fix', use_group_norm=True)
    # from modeling.unet import UNet
    # model = UNet(n_channels=4, n_classes=2, bilinear=True)

    model.load_state_dict(torch.load(model_weights))
    model.eval()  # 测试模式（将BN和DropOut固定住，不会取平均，而是用训练好的值）
    model.cuda()

    if args.save_log:
        print(vars(args))
        pyutils.Logger(args.out_dir + 'print.txt')  # 输出日志文件

    evaluator_seg = Evaluator(args.numclasses)
    evaluator_crf = Evaluator(args.numclasses)

    crop_imgs_list = glob(os.path.join(args.crop_img_root,'*.tif'))         # 所有小图的路径
    whole_imgs_list = glob(os.path.join(args.whole_img_root,'*.tif'))       # 所有大图的路径

    for whole_img in tqdm(whole_imgs_list, desc='whole', position=0):      # 循环读取每个大图（路径）
        torch.cuda.empty_cache()
        whole_img_name = Path(whole_img).stem                     # 获得大图文件名
        width, height = cv2.imread(whole_img).shape[0], cv2.imread(whole_img).shape[1]      # 大图的尺寸
        crop_imgs = [i for i in crop_imgs_list if Path(i).stem.startswith(whole_img_name)]    # 提取大图对应的所有小图
        final_result = np.zeros((2, height, width))
        final_crf = np.zeros((2, height, width))

        h_nums = math.ceil((height-args.size)/args.stride)+1      # height方向切割的图片数量
        w_nums = math.ceil((width-args.size)/args.stride)+1       # width方向切割的图片数量

        for img in tqdm(crop_imgs, desc='crop', colour='#00ff00', leave=False):
            ori_img = PIL.Image.open(img).convert("RGB")
            input = transform(ori_img).unsqueeze(0).cuda()   # 小图预处理
            with torch.no_grad():
                output = model(input, mode='seg')
            output = torch.squeeze(output,0)        # 小图预测结果
            img_name = Path(img).stem               # 小图名称
            number = int(img_name.split('_')[1])    # 小图序号
            num_h, num_w = (number - 1) // w_nums, (number - 1) % w_nums    # img所在的height号与width号

            '''result'''
            if num_w == w_nums-1:
                w_begin = width - args.size
            else:
                w_begin = num_w * args.stride
            if num_h == h_nums-1:
                h_begin = height - args.size
            else:
                h_begin = num_h * args.stride
            final_result[:, h_begin:h_begin + args.size, w_begin:w_begin + args.size] += output.cpu().numpy()  # 整合为大图

            '''crf'''
            if args.crf_post:
                if args.crop_size: ori_img = imutils.FixScaleResizeImage(args.crop_size)(ori_img)
                ori_img = np.array(ori_img)
                pred_prob = torch.nn.Softmax(dim=0)(output)
                crf_la = _crf_with_alpha(pred_prob, ori_img)
                final_crf[:, h_begin:h_begin + args.size, w_begin:w_begin + args.size] += crf_la

        result = np.argmax(final_result, 0)
        if args.crf_post:
            res_crf = np.argmax(final_crf, 0)

        if args.save_img:
            cv2.imwrite(os.path.join(args.out_dir, whole_img_name + '_res.png'), result.astype(np.uint8) * 255)
            if args.crf_post:
                cv2.imwrite(os.path.join(args.out_dir, whole_img_name + '_crf.png'), res_crf.astype(np.uint8) * 255)

        """整幅大图片精度评价"""
        pixel_label = cv2.imread(os.path.join(args.whole_mask_root,whole_img_name+'.png'),0)
        evaluator_seg.add_batch(pixel_label,result)
        if args.crf_post: evaluator_crf.add_batch(pixel_label,res_crf)

        """灰度图上色"""
        # color_map = np.zeros([height,width,3])
        # color_map = np.piecewise(output,[output==i for i in [0,1,2]],[color for color in [[0,0,0],[0,0,255],[255,0,0]]])

    if args.numclasses == 2:  # 二分类精度
        acc_dict_seg = {'seg_IoU': evaluator_seg.Intersection_over_Union()[1], 'seg_F1': evaluator_seg.F1_Class()[1],
                        'seg_pre': evaluator_seg.Pixel_Accuracy_Class()[1], 'seg_rec': evaluator_seg.Recall_Class()[1]}
        for key, value in acc_dict_seg.items():  # 遍历精度字典
            acc_dict_seg[key] = '%.4f' % (value)  # 保留小数点4后四位
            print(key, acc_dict_seg[key], sep=': ', end='\t')  # 输出精度字典
        print()  # 输出换行

        if args.crf_post:
            acc_dict_crf = {'crf_IoU': evaluator_crf.Intersection_over_Union()[1],
                            'crf_F1': evaluator_crf.F1_Class()[1],
                            'crf_pre': evaluator_crf.Pixel_Accuracy_Class()[1],
                            'crf_rec': evaluator_crf.Recall_Class()[1]}
            for key, value in acc_dict_crf.items():  # 遍历精度字典
                acc_dict_crf[key] = '%.4f' % (value)  # 保留小数点4后四位
                print(key, acc_dict_crf[key], sep=': ', end='\t')  # 输出精度字典
            print()  # 输出换行

    else:  # 多类别精度
        acc_dict_seg = {'seg_mIoU': evaluator_seg.Mean_Intersection_over_Union(), 'seg_mF1': evaluator_seg.Mean_F1(),
                        'seg_OA': evaluator_seg.Overall_Accuracy()}
        for key, value in acc_dict_seg.items():  # 遍历精度字典
            acc_dict_seg[key] = '%.4f' % (value)  # 保留小数点4后四位
            print(key, acc_dict_seg[key], sep=': ', end='\t')  # 输出精度字典
        print()  # 输出换行
        if args.crf_post:
            acc_dict_crf = {'crf_mIoU': evaluator_crf.Mean_Intersection_over_Union(),
                            'crf_mF1': evaluator_crf.Mean_F1(),
                            'crf_OA': evaluator_crf.Overall_Accuracy()}
            for key, value in acc_dict_crf.items():  # 遍历精度字典
                acc_dict_crf[key] = '%.4f' % (value)  # 保留小数点4后四位
                print(key, acc_dict_crf[key], sep=': ', end='\t')  # 输出精度字典
            print()  # 输出换行

    return acc_dict_seg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--crop_size", default=None, type=int)
    parser.add_argument("--numclasses", default=2, type=int)
    parser.add_argument("--crf_post", default=False, type=bool)
    parser.add_argument("--save_img", default=True, type=bool)
    parser.add_argument("--save_log", default=True, type=bool)
    parser.add_argument("--stride", default=128, type=int)
    parser.add_argument("--size", default=256, type=int)

    # infer runways2
    parser.add_argument("--weights",
                        default=r'D:\WSSS_code\PuzzleCAM-master\experiments\building_potsdam_FlipCAM_resnet50_RGB_scale=1.0_CRF_0.5_inference=2_seg_resnet101_DeepLabV3+\models\building_potsdam_FlipCAM_resnet50_RGB_scale=1.0_CRF_0.5_inference=2_seg_resnet101_DeepLabV3+.pth',
                        type=str)
    parser.add_argument("--out_dir",
                        default=r'D:\Experiments\Potsdam\seg\building\test\out/',
                        type=str)
    parser.add_argument("--whole_img_root",
                        default=r'D:\Experiments\Potsdam\4_Ortho_RGBIR/',
                        type=str)
    parser.add_argument("--crop_img_root",
                        default=r'D:\Experiments\Potsdam\data\test\JPEGImages/',
                        type=str)
    parser.add_argument("--whole_mask_root",
                        default=r'D:\Experiments\Potsdam\seg\building\test\label/',
                        type=str)
    args = parser.parse_args()
    
    infer(args.weights, args)  # 预测结果