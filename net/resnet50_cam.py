import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
from net import resnet


class Net(nn.Module):

    def __init__(self, stride=16):
        super(Net, self).__init__()
        if stride == 16:
            self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic['resnet50'], strides=(2, 2, 2, 1))
            # self.stage1 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
            #                             self.model.layer1)
            self.stage1 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool)
        else:
            self.model = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
                                        self.model.layer1)
        # self.stage2 = nn.Sequential(self.model.layer2)
        # self.stage3 = nn.Sequential(self.model.layer3)
        # self.stage4 = nn.Sequential(self.model.layer4)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)

        self.classifier = nn.Conv2d(2048, 1, 1, bias=False)

        # self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        # self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        # x = self.stage1(x)
        # x = self.stage2(x)
        #
        # x = self.stage3(x)
        # x = self.stage4(x)
        #
        #
        # x = torchutils.gap2d(x, keepdims=True)
        # x = self.classifier(x)
        # x = x.view(-1, 2)

        N, C, H, W = x.size()

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.global_average_pooling_2d(x, keepdims=True)
        logits = self.classifier(x).view(-1, self.num_classes)
        return logits



        # return x

    def train(self, mode=True):
        for p in self.model.conv1.parameters():
            p.requires_grad = False
        for p in self.model.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self, stride=16):
        super(CAM, self).__init__(stride=stride)

    def forward(self, x, separate=False):
        # x = self.stage1(x)
        #
        # x = self.stage2(x)
        #
        # x = self.stage3(x)
        #
        # x = self.stage4(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = F.conv2d(x, self.classifier.weight)
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x
