# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

from model_loss_semseg_gatedcrf import ModelLossSemsegGatedCRF
from DenseEnergyLoss import DenseEnergyLoss
from hrnet import HRNet
from torch.cuda.amp import autocast as autocast


parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default=r'D:\Experiments\Vaihingen\data\train/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='DeepLabv3+', type=str)
parser.add_argument('--backbone', default='resnet101', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--use_gn', default=True, type=str2bool)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=48, type=int)
parser.add_argument('--max_epoch', default=50, type=int)

parser.add_argument('--lr', default=0.007, type=float)
parser.add_argument('--wd', default=4e-5, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--min_image_size', default=256, type=int)
parser.add_argument('--max_image_size', default=256, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default=r'building_vaihingen_FlipCAM_alpha=0.25_seg_resnet101_DeepLabv3+', type=str)
# parser.add_argument('--tag', default=r'building_1024_BiSeNetV2', type=str)

# parser.add_argument('--label_name', default='resnet50@seed=0@aug=Affinity_ResNet50@ep=3@nesterov@train_aug@beta=10@exp_times=8@rw', type=str)

if __name__ == '__main__':
    ###################################################################################
    # Arguments.
    ###################################################################################
    args = parser.parse_args()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    log_dir = create_directory(f'./experiments/{args.tag}/logs/')
    data_dir = create_directory(f'./experiments/{args.tag}/data/')
    model_dir = create_directory(f'./experiments/{args.tag}/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')
    # pred_dir = './experiments/predictions/{}/'.format(args.label_name)
    pred_dir = r'D:\Experiments\Vaihingen\cam\building\train_FlipCAM_resnet50_RGB_alpha=0.25_binary=0.5_CRF_inference=1/'
    # pred_dir = r'D:\Experiments\Potsdam_1024\data\train\SegmentationClass_num/'
    # pred_dir = r'D:\Experiments\Potsdam\data\train\SegmentationClass/'
    log_path = log_dir + f'{args.tag}.txt'
    data_path = data_dir + f'{args.tag}.json'
    model_path = model_dir + f'{args.tag}.pth'
    
    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)
    
    log_func('[i] {}'.format(args.tag))
    log_func()

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    # imagenet_mean = [0.485, 0.456, 0.406, 0.456]
    # imagenet_std = [0.229, 0.224, 0.225, 0.224]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    train_transforms = [
        # RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip_For_Segmentation(),
        
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args.image_size),
    ]
    
    # if 'Seg' in args.architecture:
    #     if 'C' in args.architecture:
    #         train_transforms.append(Resize_For_Mask(args.image_size // 4))
    #     else:
    #         train_transforms.append(Resize_For_Mask(args.image_size // 8))

    train_transform = transforms.Compose(train_transforms + [Transpose_For_Segmentation()])
    
    test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        # Top_Left_Crop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])
    
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    
    train_dataset = VOC_Dataset_For_WSSS(args.data_dir, 'train', pred_dir, train_transform)
    valid_dataset = VOC_Dataset_For_Segmentation(args.data_dir, 'val', test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=True)
    
    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration
    
    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    
    ###################################################################################
    # Network
    ###################################################################################
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=meta_dic['classes'], mode=args.mode, use_group_norm=args.use_gn)
        param_groups = model.get_parameter_groups(None)
        params = [
            {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
            {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
            {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
        ]
    elif args.architecture == 'HRNet':
        model = HRNet(3,16,8) ##
        params_dict = dict(model.named_parameters())##
        params = [{'params': list(params_dict.values()), 'lr': 0.01}]##


    model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

        # for sync bn
        # patch_replication_callback(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn_for_backup = lambda: save_model(model, model_path.replace('.pth', f'_backup.pth'), parallel=the_number_of_gpu > 1)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.CrossEntropyLoss(ignore_index=255).cuda()

    # log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    # log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    # log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    # log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))
    
    optimizer = PolyOptimizer(params, lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)

    # 在训练最开始之前实例化一个GradScaler对象
    scaler = torch.cuda.amp.GradScaler()


    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : [],
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss'])

    best_valid_mIoU = -1
    best_valid_mIoU_fg = -1

    def evaluate(loader):
        model.eval()
        eval_timer.tik()

        meter = Calculator_For_mIoU('./data/VOC_2012.json') 

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()

                logits = model(images)
                predictions = torch.argmax(logits, dim=1)
                
                # for visualization
                # if step == 0:
                #     for b in range(4):
                #         image = get_numpy_from_tensor(images[b])
                #         pred_mask = get_numpy_from_tensor(predictions[b])
                #
                #         image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
                #         h, w, c = image.shape
                #
                #         pred_mask = decode_from_colormap(pred_mask, train_dataset.colors)
                #         pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                #
                #         image = cv2.addWeighted(image[:,:,:3], 0.5, pred_mask, 0.5, 0)[..., ::-1]
                #         image = image.astype(np.float32) / 255.
                #
                #         writer.add_image('Mask/{}'.format(b + 1), image, iteration, dataformats='HWC')
                #
                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index])

                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    meter.add(pred_mask, gt_mask)

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        
        print(' ')
        model.train()
        
        return meter.get(clear=True)
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    torch.autograd.set_detect_anomaly(True)

    for iteration in range(max_iteration):
        images, labels = train_iterator.get()
        images, labels = images.cuda(), labels.cuda()
        
        #################################################################################################
        # Inference
        #################################################################################################
        optimizer.zero_grad()

        # 前向过程(model + loss)开启 autocast
        with autocast():
            logits = model(images)

            ###############################################################################
            # The part is to calculate losses.

            ##without CRFLoss
            loss = class_loss_fn(logits, labels)

            ##with CRFLoss
            # ce_loss = class_loss_fn(logits, labels).float()
            # labels = labels.unsqueeze(1)
            # pred_softmax = torch.nn.Softmax(dim=1)
            # pred_probs = pred_softmax(logits)
            # croppings = np.zeros((256, 256, args.batch_size)) + 1
            # croppings = torch.from_numpy(croppings.astype(np.float32).transpose(2, 0, 1))
            #
            # DenseEnergyLosslayer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15.0,
            #                                        sigma_xy=100.0, scale_factor=0.5)
            # crf_loss = DenseEnergyLosslayer(images, pred_probs, croppings, labels)
            # crf_loss = crf_loss.cuda()

            # γ = 0.1
            # loss = ce_loss + γ * crf_loss

            #################################################################################################

        # loss.backward()
        # optimizer.step()

        # Scales loss. 为了梯度放大.
        scaler.scale(loss).backward()

        # scaler.step() 首先把梯度的值unscale回来.
        # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
        # 否则，忽略step调用，从而保证权重不更新（不被破坏）
        scaler.step(optimizer)

        # 准备着，看是否要增大scaler
        scaler.update()

        train_meter.add({
            'loss' : loss.item(), 
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        
        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            mIoU, mIoU_fg = evaluate(valid_loader)

            if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                best_valid_mIoU = mIoU

            if best_valid_mIoU_fg == -1 or best_valid_mIoU_fg < mIoU_fg:
                best_valid_mIoU_fg = mIoU_fg

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'mIoU' : mIoU,
                'best_valid_mIoU' : best_valid_mIoU,
                'mIoU_fg': mIoU_fg,
                'best_valid_mIoU_fg': best_valid_mIoU_fg,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                mIoU={mIoU:.2f}%, \
                best_valid_mIoU={best_valid_mIoU:.2f}%, \
                mIoU_fg={mIoU_fg:.2f}%, \
                best_valid_mIoU_fg={best_valid_mIoU_fg:.2f}%, \
                time={time:.0f}sec'.format(**data)
            )
            
            writer.add_scalar('Evaluation/mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_valid_mIoU', best_valid_mIoU, iteration)
            writer.add_scalar('Evaluation/mIoU_fg', mIoU_fg, iteration)
            writer.add_scalar('Evaluation/best_valid_mIoU_fg', best_valid_mIoU_fg, iteration)
    
    write_json(data_path, data_dic)
    writer.close()

    print(args.tag)