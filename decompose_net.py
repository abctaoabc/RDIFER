import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
from torch.nn import Module, Parameter
from tqdm import tqdm


class Baseline(nn.Module):
    def __init__(self, args, NumOfLayer, num_classes=7, drop_rate=0):
        super(Baseline, self).__init__()
        self.drop_rate = drop_rate

        if NumOfLayer == 18:
            resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)
        weights = torch.load("/home/zhongtao/code/EAC/model/resnet18_msceleb.pth")["state_dict"]
        resnet.load_state_dict(weights)
        self.pre_conv = nn.Sequential(*list(resnet.children())[0:4]).cuda()
        self.layer1 = nn.Sequential(*list(resnet.children())[4:5]).cuda()
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6]).cuda()
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7]).cuda()
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8]).cuda()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512, num_classes)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.output_dim = self.fc.weight.size(1)

    def forward(self, x, phase=None):
        x = self.pre_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature = self.avgpool(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)

        if phase == 'test':
            return out

        return out

    def output_num(self):
        return self.output_dim

    def shuffleDomain(self, feature_exp_source, feature_exp_target, feature_dom_source, feature_dom_target):
        b = feature_exp_source.shape[0]

        ST = feature_exp_source + feature_dom_target
        TS = feature_exp_target + feature_dom_source

        return ST, TS


class My_Model_hire(nn.Module):
    def __init__(self, args, NumOfLayer, pretrained=True, num_classes=7, drop_rate=0):
        super(My_Model_hire, self).__init__()
        self.drop_rate = drop_rate

        if NumOfLayer == 18:
            resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)

        self.pre_conv = nn.Sequential(*list(resnet.children())[0:4]).cuda()
        self.layer1 = nn.Sequential(*list(resnet.children())[4:5]).cuda()
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6]).cuda()
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7]).cuda()
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8]).cuda()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

        self.compose1 = nn.Conv2d(64, 64, 1, 1, 0)
        self.fc1 = nn.Linear(64, 512)
        self.compose2 = nn.Conv2d(128, 128, 1, 1, 0)
        self.fc2 = nn.Linear(128, 512)
        self.compose3 = nn.Conv2d(256, 256, 1, 1, 0)
        self.fc3 = nn.Linear(256, 512)
        self.compose4 = nn.Conv2d(512, 512, 1, 1, 0)
        self.fc4 = nn.Linear(512, 512)

        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b = x.shape[0]

        x = self.pre_conv(x)
        x = self.layer1(x)
        temp1 = self.compose1(x)
        d1 = self.avgpool(temp1).view(b, -1)
        d1 = self.fc1(d1)
        x = x - temp1

        x = self.layer2(x)
        temp2 = self.compose2(x)
        d2 = self.avgpool(temp2).view(b, -1)
        d2 = self.fc2(d2)
        x = x - temp2

        x = self.layer3(x)
        temp3 = self.compose3(x)
        d3 = self.avgpool(temp3).view(b, -1)
        d3 = self.fc3(d3)
        x = x - temp3

        x = self.layer4(x)
        temp4 = self.compose4(x)
        d4 = self.avgpool(temp4).view(b, -1)
        d4 = self.fc4(d4)
        x = x - temp4

        x = self.avgpool(x)
        exp = x.view(x.size(0), -1)

        # 自适应
        domain = d1 + d2 + d3 + d4
        domain = domain * self.fc(domain)
        domain = self.relu(domain)

        return domain, exp


    def output_num(self):
        return self.output_dim

    def shuffleDomain(self, feature_exp_source, feature_exp_target, feature_dom_source, feature_dom_target):
        b = feature_exp_source.shape[0]

        ST = feature_exp_source + feature_exp_target
        TS = feature_dom_target + feature_dom_source

        return ST, TS

    def output_num(self):
        return self.output_dim

    def shuffleDomain(self, feature_exp_source, feature_exp_target, feature_dom_source, feature_dom_target):
        b = feature_exp_source.shape[0]

        ST = feature_exp_source + feature_exp_target
        TS = feature_dom_target + feature_dom_source

        return ST, TS

class Decomposer(nn.Module):
    def __init__(self, feature_dim):
        super(Decomposer, self).__init__()
        self.encoder1 = nn.Linear(feature_dim, 128)
        self.decoder1 = nn.Linear(128, feature_dim)
        # self.encoder2 = nn.Linear(feature_dim, 128)
        # self.decoder2 = nn.Linear(128, feature_dim)
        # self.encoder3 = nn.Linear(feature_dim, 128)
        # self.decoder3 = nn.Linear(128, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        domain1 = self.encoder1(x)
        domain1 = self.relu(domain1)
        domain1 = self.decoder1(domain1)

        return domain1


class Domain_Classifier(nn.Module):
    def __init__(self, feature_dim):
        super(Domain_Classifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        domain = self.classifier(x)
        domain = self.sigmoid(domain)

        return domain


class Expression_Classifier(nn.Module):
    def __init__(self, feature_dim, num_classers=7):
        super(Expression_Classifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classers)

    def forward(self, x):
        out = self.classifier(x)

        return out


class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes=7):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        out = self.classifier(x)

        return out







