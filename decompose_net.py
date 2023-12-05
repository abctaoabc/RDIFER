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

        return out, 0, 0, 0, 0, 0, 0, 0

    def output_num(self):
        return self.output_dim


class My_Model(nn.Module):
    def __init__(self, args, NumOfLayer, pretrained=True, num_classes=7, drop_rate=0):
        super(My_Model, self).__init__()
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

        self.decomposer = Decomposer(512)
        self.domain_classifier = Domain_Classifier(512)
        self.exp_classifier = Expression_Classifier(512, 7)
        self.classifier = Classifier(512)
        self.output_dim = self.classifier.classifier.weight.size(1)

    def forward(self, x, phase):
        b = x.shape[0]

        x = self.pre_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if phase == 'train':
            domain = self.decomposer(x)
            exp = x - domain

            exp_s, exp_aug = exp[0:b // 2, :], exp[b // 2:, :]
            domain_s, domain_aug = domain[0:b // 2, :], domain[b // 2:, :]
            exp_s_dom_aug, exp_aug_dom_s = self.shuffleDomain(exp_s, exp_aug, domain_s, domain_aug)

            out_exp = self.exp_classifier(exp)
            out_domain = self.domain_classifier(domain)
            out1 = self.classifier(torch.cat((exp_s_dom_aug, exp_aug_dom_s), dim=0))
            # out2 = self.classifier(x)

            # exp = self.avgpool(exp).view(exp.shape[0], -1)
            # domain = self.avgpool(domain).view(domain.shape[0], -1)
            # exp_s_dom_aug = self.avgpool(exp_aug_dom_s).view(exp_aug_dom_s.shape[0], -1)
            # exp_aug_dom_s = self.avgpool(exp_aug_dom_s).view(exp_aug_dom_s.shape[0], -1)

            return out_exp, out_domain, out1, 0, exp, domain, exp_s_dom_aug, exp_aug_dom_s
        else:
            out = self.classifier(x)

            # domain = self.decomposer(x)
            # exp = x - domain
            # out = self.exp_classifier(exp)

            return out

    def output_num(self):
        return self.output_dim

    def shuffleDomain(self, feature_exp_source, feature_exp_target, feature_dom_source, feature_dom_target):
        b = feature_exp_source.shape[0]

        ST = feature_exp_source + feature_dom_target
        TS = feature_exp_target + feature_dom_source

        # dis_source = F.pairwise_distance(S, TS, p=2)
        # dis_source = torch.sum(dis_source)
        # dis_target = F.pairwise_distance(T, ST, p=2)
        # dis_target = torch.sum(dis_target)
        # dis_st = F.pairwise_distance(S, T, p=2)
        # dis_st = torch.sum(dis_st)
        # dis_ts = F.pairwise_distance(ST, TS, p=2)
        # dis_ts = torch.sum(dis_ts)

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

        self.SA = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                        attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=0,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=4)

        self.attention_ds = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.attention_da = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.attention_es = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.attention_ea = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.attention2 = Attention(128, 28)
        self.attention3 = Attention(256, 14)
        self.attention4 = Attention(512, 7)
        # self.norm = nn.BatchNorm2d(512)
        self.domain_classifier = Domain_Classifier(512)
        self.exp_classifier = Expression_Classifier(512, 7)
        self.classifier = Classifier(512)
        self.output_dim = self.classifier.classifier.weight.size(1)

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

        # SA
        # d1, d2, d3, d4 = d1.unsqueeze(1), d2.unsqueeze(1), d3.unsqueeze(1), d4.unsqueeze(1)
        # d = torch.cat((d1, d2, d3, d4), dim=1)
        # d = self.SA(d)
        # domain = d[:, 0, :]

        # 自适应
        domain = d1 + d2 + d3 + d4
        domain = domain * self.fc(domain)
        domain = self.relu(domain)
        return domain, exp

        # d = torch.cat((d1.unsqueeze(1), d2.unsqueeze(1), d3.unsqueeze(1), d4.unsqueeze(1)), dim=1)
        # weight = self.fc(d1 + d2 + d3 + d4).unsqueeze(2)
        # d = d * weight
        # domain = d[:, 0, :] + d[:, 1, :] + d[:, 2, :] + d[:, 3, :]
        # domain = self.relu(domain)

        # if phase == 'train':
        #     exp_s, exp_aug = exp[0:b // 2, :], exp[b // 2:, :]
        #     domain_s, domain_aug = domain[0:b // 2, :], domain[b // 2:, :]
        #     exp_s, exp_aug, domain_s, domain_aug = exp_s * self.attention_es(exp_s), exp_aug * self.attention_ea(exp_aug), domain_s * self.attention_ds(domain_s), domain_aug * self.attention_da(domain_aug)
        #     exp_s_dom_aug, exp_aug_dom_s = self.shuffleDomain(exp_s, exp_aug, domain_s, domain_aug)
        #
        #     out_exp = self.exp_classifier(exp)
        #     out_domain = self.domain_classifier(domain)
        #     out1 = self.classifier(torch.cat((exp_s_dom_aug, exp_aug_dom_s), dim=0))
        #
        #     # exp = self.avgpool(exp).view(exp.shape[0], -1)
        #     # domain = self.avgpool(domain).view(domain.shape[0], -1)
        #     # exp_s_dom_aug = self.avgpool(exp_aug_dom_s).view(exp_aug_dom_s.shape[0], -1)
        #     # exp_aug_dom_s = self.avgpool(exp_aug_dom_s).view(exp_aug_dom_s.shape[0], -1)
        #
        #     return out_exp, out_domain, out1, 0, exp, domain, exp_s_dom_aug, exp_aug_dom_s
        # else:
        #     out = self.exp_classifier(exp)
        #
        #     # domain = self.decomposer(x)
        #     # exp = x - domain
        #     # out = self.exp_classifier(exp)

        #    return out

    def output_num(self):
        return self.output_dim

    def shuffleDomain(self, feature_exp_source, feature_exp_target, feature_dom_source, feature_dom_target):
        b = feature_exp_source.shape[0]

        ST = feature_exp_source + feature_exp_target
        TS = feature_dom_target + feature_dom_source

        # dis_source = F.pairwise_distance(S, TS, p=2)
        # dis_source = torch.sum(dis_source)
        # dis_target = F.pairwise_distance(T, ST, p=2)
        # dis_target = torch.sum(dis_target)
        # dis_st = F.pairwise_distance(S, T, p=2)
        # dis_st = torch.sum(dis_st)
        # dis_ts = F.pairwise_distance(ST, TS, p=2)
        # dis_ts = torch.sum(dis_ts)

        return ST, TS


class Attention(nn.Module):
    def __init__(self, in_channels, size):
        super(Attention, self).__init__()
        self.size = size
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv_h = nn.Conv2d(in_channels, 1, 1, 1, 0)
        self.conv_w = nn.Conv2d(in_channels, 1, 1, 1, 0)
        self.channel = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels)
        )
        self.weight = nn.Sequential(
            nn.Linear(size, size * 2),
            nn.ReLU(),
            nn.Linear(size * 2, size),
        )
        self.height = nn.Sequential(
            nn.Linear(size, size * 2),
            nn.ReLU(),
            nn.Linear(size * 2, size),
        )

    def forward(self, x):
        b = x.shape[0]

        c = self.avg(x).view(b, -1)
        c = self.channel(c)
        h = self.conv_h(x)
        h = h.squeeze(1)
        h = torch.sum(h, dim=2) / self.size
        h = self.height(h)
        w = self.conv_w(x)
        w = w.squeeze(1)
        w = torch.sum(w, dim=1) / self.size
        w = self.weight(w)

        c = c.unsqueeze(2).unsqueeze(3)
        h = h.unsqueeze(1).unsqueeze(3)
        w = w.unsqueeze(1).unsqueeze(2)

        return c * h * w


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
        # domain2 = self.encoder2(x)
        # domain2 = self.relu(domain2)
        # domain2 = self.decoder2(domain2)
        # domain3 = self.encoder3(x)
        # domain3 = self.relu(domain3)
        # domain3 = self.decoder3(domain3)
        #
        # weight = [torch.cosine_similarity(domain1, x).unsqueeze(1), torch.cosine_similarity(domain2, x).unsqueeze(1), torch.cosine_similarity(domain3, x).unsqueeze(1)]
        # weight = torch.cat(weight, dim=1)
        # weight = F.softmax(weight, dim=1)
        #
        # for i in range(3):
        #     if i == 0:
        #         domain1 = domain1 * weight[:, i].unsqueeze(1)
        #     elif i == 1:
        #         domain2 = domain2 * weight[:, i].unsqueeze(1)
        #     elif i == 2:
        #         domain3 = domain3 * weight[:, i].unsqueeze(1)
        #
        # domain = domain1 + domain2 + domain3

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


class My_Model_Trans(nn.Module):
    def __init__(self, args, NumOfLayer, pretrained=True, num_classes=7, drop_rate=0):
        super(My_Model_Trans, self).__init__()
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
        self.SA = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                        attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=0,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=49)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.decomposer = Decomposer()
        self.domain_classifier = Domain_Classifier()
        self.exp_classifier = Expression_Classifier(7)
        self.classifier = Classifier()
        self.output_dim = self.classifier.classifier.weight.size(1)

    def forward(self, x, phase):
        b = x.shape[0]

        x = self.pre_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        bs, c, _, _ = x.shape
        t = x.view(bs, c, -1)
        t = t.permute(0, 2, 1)
        t = self.SA(t)
        t = t.permute(0, 2, 1)
        t = t.view(bs, c, 7, 7)

        x = self.avgpool(torch.cat((x, t), dim=1))
        x = x.view(x.size(0), -1)

        if phase == 'train':
            domain = self.decomposer(x)
            exp = x - domain

            exp_s, exp_aug = exp[0:b // 2, :], exp[b // 2:, :]
            domain_s, domain_aug = domain[0:b // 2, :], domain[b // 2:, :]
            exp_s_dom_aug, exp_aug_dom_s = self.shuffleDomain(exp_s, exp_aug, domain_s, domain_aug)

            out_exp = self.exp_classifier(exp)
            out_domain = self.domain_classifier(domain)
            out1 = self.classifier(torch.cat((exp_s_dom_aug, exp_aug_dom_s), dim=0))
            out2 = self.classifier(x)

            # exp = self.avgpool(exp).view(exp.shape[0], -1)
            # domain = self.avgpool(domain).view(domain.shape[0], -1)
            # exp_s_dom_aug = self.avgpool(exp_aug_dom_s).view(exp_aug_dom_s.shape[0], -1)
            # exp_aug_dom_s = self.avgpool(exp_aug_dom_s).view(exp_aug_dom_s.shape[0], -1)

            return out_exp, out_domain, out1, out2, exp, domain, exp_s_dom_aug, exp_aug_dom_s
        else:
            out = self.classifier(x)

            # domain = self.decomposer(x)
            # exp = x - domain
            # out = self.exp_classifier(exp)

            return out

    def output_num(self):
        return self.output_dim

    def shuffleDomain(self, feature_exp_source, feature_exp_target, feature_dom_source, feature_dom_target):
        b = feature_exp_source.shape[0]

        ST = feature_exp_source + feature_dom_target
        TS = feature_exp_target + feature_dom_source

        # dis_source = F.pairwise_distance(S, TS, p=2)
        # dis_source = torch.sum(dis_source)
        # dis_target = F.pairwise_distance(T, ST, p=2)
        # dis_target = torch.sum(dis_target)
        # dis_st = F.pairwise_distance(S, T, p=2)
        # dis_st = torch.sum(dis_st)
        # dis_ts = F.pairwise_distance(ST, TS, p=2)
        # dis_ts = torch.sum(dis_ts)

        return ST, TS


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SAttention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 head_drop_ratio=0.,
                 num_patches=196):
        super(SAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.head_drop_ratio = head_drop_ratio
        # self.fc = nn.Sequential(
        #     nn.Linear(num_patches * num_patches, num_patches),
        #     nn.ReLU(),
        #     nn.Linear(num_patches, 1))
        self.fc = nn.Linear(num_patches * num_patches, 1)
        # self.bias = nn.Parameter(torch.randn(1, num_heads, num_patches, num_patches))
        # nn.init.kaiming_uniform_(self.bias, mode='fan_in')

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn = self.attn_drop(attn) + self.bias

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 head_drop_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 num_patches=196):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,
                               head_drop_ratio=head_drop_ratio, num_patches=num_patches)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
