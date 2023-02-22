import numpy as np
import dataloaders.imutils as imutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
import os

class Expansion_and_CPSDropout(nn.Module):
    def __init__(self, dropout_rate, stride):
        super(Expansion_and_CPSDropout, self).__init__()
        assert 0<=dropout_rate<=1
        assert stride % 2 == 1
        self.dropout_rate = dropout_rate
        self.s = stride

    def forward(self, x):
        assert len(x.shape)==4
        s_2 = self.s//2
        padding = torch.nn.ZeroPad2d(s_2)
        x_padding = padding(x)
        b,c,h,w = x.shape
        # h,w = x.shape
        x_new = torch.empty([b,c,h*self.s,w*self.s],device='cuda')       # x_expand                                             如果不指定device='cuda'则会特别特别特别慢（尤其是反向传播时特别慢！）
        for i in range(h):
            for j in range(w):
                x_new[:,:,i*self.s:(i+1)*self.s,j*self.s:(j+1)*self.s] = x_padding[:,:,i:i+self.s,j:j+self.s]

        dropout_mask = torch.rand([x_new.shape[2],x_new.shape[3]],device='cuda')>self.dropout_rate
        for i in range(h):
            for j in range(w):
                dropout_mask[s_2+i*self.s,s_2+j*self.s]=1

        x_new = x_new * dropout_mask/(1-self.dropout_rate)    # dropout
        del(x_padding,dropout_mask)
        return x_new

class Expansion_and_CPSDropout1(nn.Module):
    def __init__(self, dropout_rate, stride):
        super(Expansion_and_CPSDropout1, self).__init__()
        assert 0<=dropout_rate<=1
        assert stride % 2 == 1
        self.dropout_rate = dropout_rate
        self.s = stride

    def forward(self, x):
        assert len(x.shape)==4
        s_2 = self.s//2
        padding = torch.nn.ZeroPad2d(s_2)
        x_padding = padding(x)              # padding
        b,c,h,w = x.shape

        # expand
        feature = x_padding.unfold(2,self.s,1).unfold(3,self.s,1).contiguous()
        feature = feature.transpose(3,4).contiguous()
        feature_size = feature.shape
        x_new = feature.view(feature_size[0],feature_size[1],feature_size[2]*self.s,feature_size[4]*self.s)

        # dropout
        p = np.pad([[1]],((s_2,s_2),(s_2,s_2)))
        p = torch.tensor(p,device='cuda')
        p = p.repeat(h,w).bool()
        dropout_mask = torch.rand([x_new.shape[2],x_new.shape[3]],device='cuda')>self.dropout_rate
        dropout_mask = dropout_mask | p

        x_new = x_new * dropout_mask    # dropout
        del(x_padding,dropout_mask)
        return x_new




def _crf_with_alpha(ori_img,cam_dict, alpha):                           # 此时的cam_dict不含bg，alpha为bg的decay rate
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)                 # cam score map，包含fg和bg
    crf_score = imutils.crf_inference(ori_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = np.zeros([21, bg_score.shape[1], bg_score.shape[2]])
    n_crf_al[0, :, :] = crf_score[0, :, :]          # 加入bg
    for i, key in enumerate(cam_dict.keys()):       # 加入fg，得到深度为21的crf优化后的cam
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al

def compute_seg_label_lzs(ori_img, cam_label, multi_cam):
    w,h = ori_img.shape[0],ori_img.shape[1]
    if cam_label == 0: return np.zeros([w,h]), 0, 0                 # 如果图片中无物体，直接返回全零矩阵作为pixel-label
    multi_cam = F.interpolate(multi_cam.unsqueeze(1), (w, h), mode='bilinear', align_corners=False).squeeze(1)
    multi_cam = multi_cam.cpu().data.numpy()

    # norm_cam = multi_cam / (np.max(multi_cam, keepdims=True) + 1e-5)    # cam归一化，crf输入必须有两类
    # bg = np.power(1 - norm_cam, 4)
    # final = np.stack((bg,norm_cam))
    # cam_crf = imutils.crf_inference(ori_img, final, labels=2)
    # cam_crf = np.argmax(cam_crf, 0).astype(np.int16)
    # cam_label = np.piecewise(multi_cam,[multi_cam>np.max(multi_cam)*0.6],[255])       # cam二值化

    # reliable_condition = np.logical_and(cam_crf == 1,multi_cam>np.max(multi_cam)*0.6)             # crf label & cam label
    # reliable_label = np.array(reliable_condition,dtype=np.int16)
    # reliable_label = cam_crf                                          # crf label directly as mask
    reliable_label = np.array(multi_cam>np.max(multi_cam)*0.3,dtype=np.int16)

    mask_val = reliable_label.copy()                                      # 用于计算精度的mask（0/1）
    cam_val = np.array(multi_cam>np.max(multi_cam)*0.15,dtype=np.int8)   # 用于计算精度的cam
    reliable_label[reliable_label==0]=255                               # 让交集外的区域都是忽略像元（用于计算交叉熵）

    bg_pixel = np.array(multi_cam<np.max(multi_cam)*0.2,dtype=np.int8)  # 不忽略背景，背景为0
    reliable_label[bg_pixel==1]=0
    mask_val[mask_val==0]=2                                             # 预测时可视化用,忽略值为2（*127后显示为白色）
    mask_val[bg_pixel==1]=0                                   # 预测时可视化用,背景值为0（*127后显示为黑色）

    return reliable_label, cam_val, mask_val

def quant_analysis(img, label):
    # print(img.shape,label.shape)
    assert img.shape == label.shape

    iou = np.sum(np.logical_and(img==1,label==1))/np.sum(np.logical_or(img==1,label==1))
    if np.sum(img==1)==0: precision = 0
    else: precision = np.sum(np.logical_and(img==1,label==1))/np.sum(img==1)
    recall = np.sum(np.logical_and(img==1,label==1))/np.sum(label==1)
    if iou>1:print('???')
    return iou,precision,recall

# '''
def compute_seg_label(ori_img, cam_label, norm_cam):
    cam_label = cam_label.astype(np.uint8)

    cam_dict = {}
    cam_np = np.zeros_like(norm_cam)
    for i in range(20):
        if cam_label[i] > 1e-5:             # 只对存在的类别进行操作
            cam_dict[i] = norm_cam[i]       # 字典形式的cam，key为类别号，没有的类别则不存在其key
            cam_np[i] = norm_cam[i]         # np形式的cam，没有的类别值为0

    bg_score = np.power(1 - np.max(cam_np, 0), 32)      # 32为decay rate 这么大？？？？？？？？
    bg_score = np.expand_dims(bg_score, axis=0)
    cam_all = np.concatenate((bg_score, cam_np))        # 完整的cam score map，包括fg和bg，不包括不存在的类别
    _, bg_w, bg_h = bg_score.shape

    cam_img = np.argmax(cam_all, 0)                     #

    crf_la = _crf_with_alpha(ori_img, cam_dict, 4)      # 注意：传入的cam dict不包含bg。得到crf优化后的cam(score)，np格式，深度为21
    crf_ha = _crf_with_alpha(ori_img, cam_dict, 32)
    crf_la_label = np.argmax(crf_la, 0)                 # low decay rate的类别cam图，即位置seed
    crf_ha_label = np.argmax(crf_ha, 0)                 # high decay rate的类别cam图
    crf_label = crf_la_label.copy()
    crf_label[crf_la_label == 0] = 255                  # 让此时判断为背景的像元也变为255即not_true，即除了fg之外的所有像元都为255即not_true,后面再加上背景像元（即high decay rate的背景）

    single_img_classes = np.unique(crf_la_label)
    cam_sure_region = np.zeros([bg_w, bg_h], dtype=bool)            # 对每个类提取score大于0.6*类内最大值的像元的位置（bg为0.8）,得到reliable CAM label，易知
    for class_i in single_img_classes:
        if class_i != 0:
            class_not_region = (cam_img != class_i)
            cam_class = cam_all[class_i, :, :]                      # 提取这一类别的cam score map
            cam_class[class_not_region] = 0                         # 将不是该类的像元置零
            cam_class_order = cam_class[cam_class > 0.1]
            cam_class_order = np.sort(cam_class_order)
            confidence_pos = int(cam_class_order.shape[0] * 0.6)
            confidence_value = cam_class_order[confidence_pos]
            class_sure_region = (cam_class > confidence_value)      # 按行计算取0.6最大值而不是整体取？？？？？？？？？？？？？？？？？？？？？？？？？？
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)     # 一个像元为sure_region的条件为：既满足cam_argmax后是该类，又满足该像元的score在该类中分数大于0.6
        else:
            class_not_region = (cam_img != class_i)
            cam_class = cam_all[class_i, :, :]
            cam_class[class_not_region] = 0
            class_sure_region = (cam_class > 0.8)
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)

    cam_not_sure_region = ~cam_sure_region

    crf_label[crf_ha_label == 0] = 0        # 高decay rate情况下（背景的score低）仍是背景的区域
    crf_label_np = np.concatenate([np.expand_dims(crf_ha[0, :, :], axis=0), crf_la[1:, :, :]])      # 将高decay rate的背景和低decay rate的前景合并得到更准确的crf label(score)
    crf_not_sure_region = np.max(crf_label_np, 0) < 0.8                                             # 将crf label score小于0.8的区域视为not_sure
    not_sure_region = np.logical_or(crf_not_sure_region, cam_not_sure_region)                       # 结合cam和crf的not true区域

    crf_label[not_sure_region] = 255        # 让not_true区域值为空白

    return crf_label
# '''

def compute_joint_loss_lzs(ori_img, pred, seg_label, croppings, critersion, DenseEnergyLosslayer):   # seg为分割网络输出，seg_label为refined label
    # seg_label = np.expand_dims(seg_label,axis=1)
    # seg_label = torch.from_numpy(seg_label)
    celoss = critersion(pred, seg_label.long())
    seg_label = seg_label.unsqueeze(1)
    pred_softmax = torch.nn.Softmax(dim=1)
    pred_probs = pred_softmax(pred)
    ori_img = torch.from_numpy(ori_img.astype(np.float32))
    croppings = torch.from_numpy(croppings.astype(np.float32).transpose(2,0,1))     # cwh
    dloss = DenseEnergyLosslayer(ori_img,pred_probs,croppings, seg_label)
    dloss = dloss.cuda()
    # seg_label_tensor = seg_label.long().cuda()

    # seg_label_copy = torch.squeeze(seg_label_tensor.clone())
    # fg_label = seg_label_copy.clone()
    # fg_celoss = critersion(pred, fg_label.long().cuda())
    # celoss = fg_celoss
    return celoss, dloss

def compute_joint_loss_lzs1(ori_img, pred, seg_label, croppings, critersion, DenseEnergyLosslayer):   # 没有dloss
    seg_label = np.expand_dims(seg_label,axis=1)
    seg_label = torch.from_numpy(seg_label)
    # pred_softmax = torch.nn.Softmax(dim=1)
    # pred_probs = pred_softmax(pred)
    # ori_img = torch.from_numpy(ori_img.astype(np.float32))
    # croppings = torch.from_numpy(croppings.astype(np.float32).transpose(2,0,1))     # cwh
    # dloss = DenseEnergyLosslayer(ori_img,pred_probs,croppings, seg_label)
    # dloss = dloss.cuda()
    seg_label_tensor = seg_label.long().cuda()

    seg_label_copy = torch.squeeze(seg_label_tensor.clone())
    # bg_label = seg_label_copy.clone()
    fg_label = seg_label_copy.clone()
    # bg_label[seg_label_copy != 0] = 255
    # fg_label[seg_label_copy == 0] = 255
    # bg_celoss = critersion(pred, bg_label.long().cuda())

    fg_celoss = critersion(pred, fg_label.long().cuda())

    # celoss = bg_celoss + fg_celoss
    celoss = fg_celoss
    # return celoss, dloss
    return celoss

# '''
def compute_joint_loss(ori_img, seg, seg_label, croppings, critersion, DenseEnergyLosslayer):   # seg为分割网络输出，seg_label为refined label
    seg_label = np.expand_dims(seg_label,axis=1)
    seg_label = torch.from_numpy(seg_label)

    w = seg_label.shape[2]
    h = seg_label.shape[3]
    pred = F.interpolate(seg,(w,h),mode="bilinear",align_corners=False)
    pred_softmax = torch.nn.Softmax(dim=1)
    pred_probs = pred_softmax(pred)
    ori_img = torch.from_numpy(ori_img.astype(np.float32))
    croppings = torch.from_numpy(croppings.astype(np.float32).transpose(2,0,1))     # cwh
    dloss = DenseEnergyLosslayer(ori_img,pred_probs,croppings, seg_label)
    dloss = dloss.cuda()

    seg_label_tensor = seg_label.long().cuda()

    seg_label_copy = torch.squeeze(seg_label_tensor.clone())
    bg_label = seg_label_copy.clone()
    fg_label = seg_label_copy.clone()
    bg_label[seg_label_copy != 0] = 255
    fg_label[seg_label_copy == 0] = 255
    bg_celoss = critersion(pred, bg_label.long().cuda())

    fg_celoss = critersion(pred, fg_label.long().cuda())

    celoss = bg_celoss + fg_celoss

    return celoss, dloss
# '''

def compute_cam_up(cam, label, w, h, b):
    cam_up = F.interpolate(cam, (w, h), mode='bilinear', align_corners=False)
    cam_up = cam_up * label.clone().view(b, 20, 1, 1)   # Unlike copy_(), clone() is recorded in the computation graph. Gradients propagating to the cloned tensor will propagate to the original tensor.
    cam_up = cam_up.cpu().data.numpy()                  # .data ？？？？？？？？？？？？？
    return cam_up


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def resize_label_batch(label, size):
    label_resized = np.zeros((size, size, 1, label.shape[3]))
    interp = torch.nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = torch.autograd.Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)
    label_resized[label_resized>21] = 255
    return label_resized


def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def scale_gt(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims, interpolation=cv2.INTER_NEAREST).astype(float)

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy',allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]


def RandomCrop(imgarr, cropsize):

    h, w, c = imgarr.shape

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space+1)
    else:
        cont_left = random.randrange(-w_space+1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space+1)
    else:
        cont_top = random.randrange(-h_space+1)
        img_top = 0

    img_container = np.zeros((cropsize, cropsize, imgarr.shape[-1]), np.float32)

    cropping =  np.zeros((cropsize, cropsize), np.bool)


    img_container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        imgarr[img_top:img_top+ch, img_left:img_left+cw]
    cropping[cont_top:cont_top + ch, cont_left:cont_left + cw] = 1

    return img_container, cropping

def get_data_from_chunk_v2(chunk, args):
    img_path = args.IMpath

    scale = np.random.uniform(0.7, 1.3)
    dim = args.crop_size
    images = np.zeros((dim, dim, 3, len(chunk)))
    ori_images = np.zeros((dim, dim, 3, len(chunk)),dtype=np.uint8)
    croppings = np.zeros((dim, dim, len(chunk)))
    labels = load_image_label_list_from_npy(chunk)
    labels = torch.from_numpy(np.array(labels))

    for i, piece in enumerate(chunk):
        flip_p = np.random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg'))
        img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB).astype(np.float)
        img_temp = scale_im(img_temp, scale)
        img_temp = flip(img_temp, flip_p)
        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225
        img_temp, cropping = RandomCrop(img_temp, dim)
        ori_temp = np.zeros_like(img_temp)
        ori_temp[:, :, 0] = (img_temp[:, :, 0] * 0.229 + 0.485) * 255.
        ori_temp[:, :, 1] = (img_temp[:, :, 1] * 0.224 + 0.456) * 255.
        ori_temp[:, :, 2] = (img_temp[:, :, 2] * 0.225 + 0.406) * 255.
        ori_images[:, :, :, i] = ori_temp.astype(np.uint8)
        croppings[:,:,i] = cropping.astype(np.float32)

        images[:, :, :, i] = img_temp

    images = images.transpose((3, 2, 0, 1))
    ori_images = ori_images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images).float()
    return images, ori_images, labels, croppings
