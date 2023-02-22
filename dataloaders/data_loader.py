import PIL.Image
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import scipy.misc
import glob
# from osgeo import gdal
import imageio

#TODO：当前读取的label均为0~n的值，可以考虑改为彩色图.png，再encode为0~n.

# '''包括pixel-level label和自定义seg_label的数据获取，用于image-level弱监督。通过label.npy获取训练样本名称及路径'''
# class Train_Dataset(Dataset):
#     def __init__(self, data_dir, label_path, pixellabel_dir,seglabel_dir,size, transform_label=None,transform=None, transform2=None, ):
#         self.data_dir = data_dir
#         self.label_path = label_path
#         self.pixellabel_dir = pixellabel_dir
#         self.seglabel_dir = seglabel_dir
#         self.transform_label = transform_label
#         self.transform = transform
#         self.transform2 = transform2
#         self.h,self.w = size[0],size[1]
#
#         self.labels_dict = np.load(self.label_path,allow_pickle=True).item()                    # image-level label dict
#         self.images_name = [i for i in self.labels_dict]                                        # 图像名
#         self.images_dir = [self.data_dir+image_name+'.jpg' for image_name in self.labels_dict]  # 图像路径
#         self.labels = list(self.labels_dict.values())                                           # image-label label (即labels_dict的value)
#
#     def __len__(self):
#         return len(self.labels_dict)
#     def __getitem__(self, idx):
#         label = torch.tensor(self.labels[idx],dtype=float).view(-1)
#         name = self.images_name[idx]
#         img = PIL.Image.open(self.images_dir[idx]).convert("RGB")    # 保证图片格式为RGB模式
#         img = img.resize((self.h, self.w))                           # PIL库的图像缩放
#         if label == 0 :
#             pixel_label = PIL.Image.fromarray(np.zeros([self.h, self.w]))
#             seg_label = PIL.Image.fromarray(np.zeros([self.h, self.w]))
#         else:
#             pixel_label = PIL.Image.open(self.pixellabel_dir+name+'.png').resize((self.h, self.w))     # 逐项元label
#             if self.seglabel_dir == None:
#                 seg_label = PIL.Image.fromarray(np.zeros([self.h, self.w]))
#             else: seg_label = PIL.Image.open(self.seglabel_dir+name+'.png').resize((self.h, self.w))
#         if self.transform_label:                                       # 对img, pixel-label, pseudo-mask同时进行操作（随机水平翻转）
#             img,pixel_label,seg_label = self.transform_label(img,pixel_label,seg_label)
#             pixel_label = torch.from_numpy(np.asarray(pixel_label,dtype=float))
#             seg_label = torch.from_numpy(np.asarray(seg_label,dtype=float))
#         if self.transform:
#             img = self.transform(img)
#         if self.transform2:
#             ori_img = img
#             croppings = np.zeros_like(img)
#             img_dict = []
#             img_dict.append(img)            # 第一次transform后的img
#             img_dict.append(ori_img)        # 第一次transform后的img
#             img_dict.append(croppings)      # 全零矩阵
#             img, ori_img, croppings = self.transform2(img_dict)
#         img = torch.from_numpy(img)
#         return name, img, label, ori_img, croppings, pixel_label, seg_label
# 如果Train_Dataset基础上需要加上pixel-label,可以继承该类

class Cls_Dataset(Dataset):
    def __init__(self, data_dir, label_npy, transform=None):
        self.data_dir = data_dir
        self.label_npy = label_npy
        self.transform = transform

        self.labels_dict = np.load(self.label_npy,allow_pickle=True).item()                    # image-level label dict
        self.images_name = [i for i in self.labels_dict]                                        # 图像名
        self.images_dir = [os.path.join(self.data_dir,image_name+'.jpg') for image_name in self.labels_dict]  # 图像路径
        self.labels = list(self.labels_dict.values())                                           # label (即labels_dict的value)

    def __len__(self):
        return len(self.labels_dict)
    def __getitem__(self, idx):
        name = self.images_name[idx]
        label = torch.tensor(self.labels[idx],dtype=float).view(-1)
        # if self.labels[idx]:
        #     label = torch.tensor([0,1], dtype=float)
        # else: label = torch.tensor([0,0], dtype=float)
        img = PIL.Image.open(self.images_dir[idx]).convert("RGB")    # 保证图片格式为RGB模式
        if self.transform:
            img = self.transform(img)
        return img, label
        # return name, img, label

class Train_Dataset(Dataset):
    def __init__(self, data_dir, seglabel_dir, transform1=None, transform2=None):      # transform1：对img和pixellabel同时进行变换，transform2：生成用于CRFloss的图像并进行变换
        self.data_dir = data_dir
        self.transform1 = transform1
        self.transform2 = transform2

        self.seglabels_dir = glob.glob(os.path.join(seglabel_dir,'*.png'))   # 训练样本mask路径
        self.images_name = [os.path.splitext(os.path.basename(i))[0] for i in self.seglabels_dir]   # 训练样本名称
        self.images_dir = [os.path.join(self.data_dir,i+'.jpg') for i in self.images_name]                        # img路径

    def __len__(self):
        return len(self.images_dir)
    def __getitem__(self, idx):
        name = self.images_name[idx]
        img = PIL.Image.open(self.images_dir[idx]).convert("RGB")    # 保证图片格式为RGB模式
        seglabel = PIL.Image.open(self.seglabels_dir[idx])     # 逐像元label

        if self.transform1:                                       # 对img和pixellabel同时进行transform
            img_list = [img, seglabel]
            img, seglabel = self.transform1(img_list)

        if self.transform2:
            ori_img = img
            croppings = np.zeros_like(img)
            img_list = [img, ori_img, croppings, seglabel]      # 输入图像、原始图像、裁剪掩膜、分割标记
            img, ori_img, croppings, seglabel = self.transform2(img_list)
            return name, img, seglabel, ori_img, croppings,
        return name, img, seglabel


class Val_Dataset(Dataset):
    def __init__(self, data_dir, pixellabel_dir,transform=None):      # transform：对img和mask进行操作
        self.data_dir = data_dir
        self.pixellabel_dir = pixellabel_dir
        self.transform = transform

        self.pixellabels_dir = glob.glob(os.path.join(pixellabel_dir,'*.png'))                                  # ground-truth路径
        self.images_name = [os.path.splitext(os.path.basename(i))[0] for i in self.pixellabels_dir]               # 训练样本名称
        self.images_dir = [os.path.join(self.data_dir,i+'.jpg') for i in self.images_name]                                    # img路径

    def __len__(self):
        return len(self.images_dir)
    def __getitem__(self, idx):
        name = self.images_name[idx]
        img = PIL.Image.open(self.images_dir[idx]).convert("RGB")
        pixel_label = PIL.Image.open(self.pixellabels_dir[idx])     # 逐像元label
        img_list = [img, pixel_label]
        img, pixel_label = self.transform(img_list)

        return name, img, pixel_label

'''以逐像素标记进行全监督训练的数据读取，可以支持img为多波段图像，包括pixel-label的数据获取。通过pixel-label获取训练样本路径及名称'''
class Train_Dataset_MSS(Dataset):
    def __init__(self, data_dir, pixellabel_dir, transform_label=None,transform=None, transform2=None):      # transform_label：对img和pixellabel同时进行随机水平翻转，transform：对原图进行操作，transform2：将原图进行一些转化
        self.data_dir = data_dir
        self.transform_label = transform_label
        self.transform = transform
        self.transform2 = transform2

        self.pixellabels_dir = glob.glob(os.path.join(pixellabel_dir,'*.png'))   # 训练样本mask路径
        self.images_name = [os.path.splitext(os.path.basename(i))[0] for i in self.pixellabels_dir]   # 训练样本名称
        self.images_dir = [self.data_dir+i+'.jpg' for i in self.images_name]                        # img路径

    def __len__(self):
        return len(self.images_dir)
    def __getitem__(self, idx):
        name = self.images_name[idx]
        # img = imageio.imread(self.images_dir[idx])  # HWC
        img = PIL.Image.open(self.images_dir[idx]).convert("RGB")    # 保证图片格式为RGB模式
        pixel_label = PIL.Image.open(self.pixellabels_dir[idx])     # 逐像元label

        if self.transform_label:                                       # 对img和pixellabel同时进行随机水平翻转
            img,pixel_label = self.transform_label(img,pixel_label)
            pixel_label = torch.from_numpy(np.asarray(pixel_label,dtype=float))
        if self.transform:
            img = self.transform(img)
        if self.transform2:
            ori_img = img
            croppings = np.zeros_like(img)
            img_dict = []
            img_dict.append(img)            # 第一次transform后的img
            img_dict.append(ori_img)        # 第一次transform后的img
            img_dict.append(croppings)      # 全零矩阵
            img, ori_img, croppings = self.transform2(img_dict)
        img = torch.from_numpy(img)
        return name, img, pixel_label, ori_img, croppings,

'''以模拟标记进行全监督训练的数据读取，仅支持RGB图像，包括pixel-label和pseudo-mask。通过seglabels_dir（pseudo-mask）获取训练样本路径及名称'''
class Train_Dataset_fully(Dataset):
    def __init__(self, data_dir, pixellabel_dir,seglabel_dir, transform_label=None,transform=None, transform2=None):      # transform_label：对img和pixellabel同时进行随机水平翻转，transform：对原图进行操作，transform2：将原图进行一些转化
        self.data_dir = data_dir
        self.pixellabel_dir = pixellabel_dir
        self.transform_label = transform_label
        self.transform = transform
        self.transform2 = transform2

        self.seglabels_dir = glob.glob(seglabel_dir+'*.png')   # 训练样本mask路径
        self.images_name = [os.path.splitext(os.path.basename(i))[0] for i in self.seglabels_dir]   # 训练样本名称
        self.images_dir = [self.data_dir+i+'.jpg' for i in self.images_name]                        # img路径
        self.pixellabels_dir = [self.pixellabel_dir+i+'.png' for i in self.images_name]             # ground-truth路径

    def __len__(self):
        return len(self.images_dir)
    def __getitem__(self, idx):
        name = self.images_name[idx]
        img = PIL.Image.open(self.images_dir[idx]).convert("RGB")    # 保证图片格式为RGB模式 (HWC)
        pixel_label = PIL.Image.open(self.pixellabels_dir[idx])     # 逐像元label
        seg_label = PIL.Image.open(self.seglabels_dir[idx])         # pseudo-masks

        if self.transform_label:                                       # 对img和pixellabel同时进行随机水平翻转
            img,pixel_label,seg_label = self.transform_label(img,pixel_label,seg_label)
            pixel_label = torch.from_numpy(np.asarray(pixel_label,dtype=float))
            seg_label = torch.from_numpy(np.asarray(seg_label,dtype=float))
        if self.transform:
            img = self.transform(img)
        if self.transform2:
            ori_img = img
            croppings = np.zeros_like(img)
            img_dict = []
            img_dict.append(img)            # 第一次transform后的img
            img_dict.append(ori_img)        # 第一次transform后的img
            img_dict.append(croppings)      # 全零矩阵
            img, ori_img, croppings = self.transform2(img_dict)
        img = torch.from_numpy(img)
        return name, img, seg_label, ori_img, croppings, pixel_label

'''以模拟标记进行全监督训练的数据读取，支持img为多波段图像，包括pixel-label和pseudo-mask的数据获取。通过seglabels_dir（pseudo-mask）获取训练样本路径及名称'''
class Train_Dataset_fully_MSS(Dataset):
    def __init__(self, data_dir, pixellabel_dir, seglabel_dir, transform_label=None,transform=None, transform2=None):      # transform_label：对img和pixellabel同时进行随机水平翻转，transform：对原图进行操作，transform2：将原图进行一些转化
        self.data_dir = data_dir
        self.pixellabel_dir = pixellabel_dir
        self.transform_label = transform_label
        self.transform = transform
        self.transform2 = transform2

        self.seglabels_dir = glob.glob(os.path.join(seglabel_dir,'*.png'))   # 训练样本mask路径
        self.images_name = [os.path.splitext(os.path.basename(i))[0] for i in self.seglabels_dir]   # 训练样本名称
        self.images_dir = [self.data_dir+i+'.tif' for i in self.images_name]                        # img路径
        self.pixellabels_dir = [os.path.join(self.pixellabel_dir,i+'.png') for i in self.images_name]             # ground-truth路径

    def __len__(self):
        return len(self.images_dir)
    def __getitem__(self, idx):
        name = self.images_name[idx]
        img = imageio.imread(self.images_dir[idx])  # HWC

        # img = PIL.Image.open(self.images_dir[idx]).convert("RGB")    # 保证图片格式为RGB模式
        pixel_label = PIL.Image.open(self.pixellabels_dir[idx])     # 逐像元label
        seg_label = PIL.Image.open(self.seglabels_dir[idx])         # pseudo-masks

        if self.transform_label:                                       # 对img和pixellabel同时进行随机水平翻转
            img,pixel_label,seg_label = self.transform_label(img,pixel_label,seg_label)
            pixel_label = torch.from_numpy(np.asarray(pixel_label,dtype=float))
            seg_label = torch.from_numpy(np.asarray(seg_label,dtype=float))
        if self.transform:
            img = self.transform(img)
        if self.transform2:
            ori_img = img
            croppings = np.zeros_like(img)
            img_dict = []
            img_dict.append(img)            # 第一次transform后的img
            img_dict.append(ori_img)        # 第一次transform后的img
            img_dict.append(croppings)      # 全零矩阵
            img, ori_img, croppings = self.transform2(img_dict)
        img = torch.from_numpy(img)
        return name, img, seg_label, ori_img, croppings,pixel_label

'''验证loader1, 包含seglabel，通过seglabel获取样本路径及名称'''
class Train_Dataset_fully_MSS_val(Dataset):
    def __init__(self, data_dir, pixellabel_dir,seglabel_dir,transform=None):      # transform：对原图进行操作
        self.data_dir = data_dir
        self.pixellabel_dir = pixellabel_dir
        self.seglabel_dir = seglabel_dir
        self.transform = transform

        self.seglabels_dir = glob.glob(os.path.join(seglabel_dir,'*.png'))                                      # 训练样本mask路径
        self.images_name = [os.path.splitext(os.path.basename(i))[0] for i in self.seglabels_dir]               # 训练样本名称
        self.images_dir = [self.data_dir+i+'.tif' for i in self.images_name]                                    # img路径
        self.pixellabels_dir = [os.path.join(self.pixellabel_dir,i+'.png') for i in self.images_name]           # ground-truth路径


    def __len__(self):
        return len(self.images_dir)
    def __getitem__(self, idx):
        name = self.images_name[idx]
        img = imageio.imread(self.images_dir[idx])  # HWC
        pixel_label = PIL.Image.open(self.pixellabels_dir[idx])     # 逐像元label
        seg_label = PIL.Image.open(self.seglabels_dir[idx])         # pseudo-masks

        if self.transform:
            img = self.transform(img)
        pixel_label = torch.from_numpy(np.asarray(pixel_label))
        seg_label = torch.from_numpy(np.asarray(seg_label))

        return name, img, pixel_label, seg_label

'''验证loader2, 不包含seglabel，只有img和pixellabel，通过pixellabel获取样本路径及名称'''
class Train_Dataset_fully_val2(Dataset):
    def __init__(self, data_dir, pixellabel_dir,transform=None):      # transform：对原图进行操作
        self.data_dir = data_dir
        self.pixellabel_dir = pixellabel_dir
        self.transform = transform

        self.pixellabels_dir = glob.glob(os.path.join(pixellabel_dir,'*.png'))                                  # ground-truth路径
        self.images_name = [os.path.splitext(os.path.basename(i))[0] for i in self.pixellabels_dir]               # 训练样本名称
        self.images_dir = [self.data_dir+i+'.jpg' for i in self.images_name]                                    # img路径

    def __len__(self):
        return len(self.images_dir)
    def __getitem__(self, idx):
        name = self.images_name[idx]
        img = imageio.imread(self.images_dir[idx])  # HWC
        pixel_label = PIL.Image.open(self.pixellabels_dir[idx])     # 逐像元label

        if self.transform:
            img = self.transform(img)
        pixel_label = torch.from_numpy(np.asarray(pixel_label))

        return name, img, pixel_label

'''测试loader, 只有img'''
class Train_Dataset_fully_MSS_test(Dataset):
    def __init__(self, data_dir, transform=None):      # transform：对原图进行操作
        self.data_dir = data_dir
        self.transform = transform

        self.images_dir = glob.glob(os.path.join(self.data_dir,'*.tif'))                                  # img路径
        self.images_name = [os.path.splitext(os.path.basename(i))[0] for i in self.images_dir]               # 训练样本名称

    def __len__(self):
        return len(self.images_dir)
    def __getitem__(self, idx):
        name = self.images_name[idx]
        img = imageio.imread(self.images_dir[idx])  # HWC
        # img = imageio.imread(self.images_dir[idx])[:,:,:4]  # HWC
        if self.transform:
            img = self.transform(img)

        return name, img

