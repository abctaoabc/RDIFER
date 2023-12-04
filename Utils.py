import argparse
import os
import errno
import os.path as osp
import sys

import cv2

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from AdversarialNetwork import AdversarialNetwork


class AverageMeter(object):
    '''Computes and stores the sum, count and average'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val
        self.count += count

        if self.count == 0:
            self.avg = 0
        else:
            self.avg = float(self.sum) / self.count


def str2bool(input):
    if isinstance(input, bool):
        return input
    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def Compute_Accuracy(args, pred, target, acc, prec, recall):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples.'''

    pred = pred.cpu().data.numpy()  # [64, 7]
    pred = np.argmax(pred, axis=1)
    target = target.cpu().data.numpy()

    pred = pred.astype(np.int32).reshape(pred.shape[0], )
    target = target.astype(np.int32).reshape(target.shape[0], )

    for i in range(7):
        TP = np.sum((pred == i) * (target == i))
        TN = np.sum((pred != i) * (target != i))

        # Compute Accuracy of All --> TP+TN / All
        acc[i].update(np.sum(pred == target), pred.shape[0])

        # Compute Precision of Positive --> TP/(TP+FP)
        prec[i].update(TP, np.sum(pred == i))

        # Compute Recall of Positive --> TP/(TP+FN)
        recall[i].update(TP, np.sum(target == i))


def get_wrong_image(args, pred, target):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples.'''

    pred = pred.cpu().data.numpy()  # [64, 7]
    pred = np.argmax(pred, axis=1)
    target = target.cpu().data.numpy()

    pred = pred.astype(np.int32).reshape(pred.shape[0], )
    target = target.astype(np.int32).reshape(target.shape[0], )
    for i in range(len(pred)):
        print(pred[i] == target[i])


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1
    return conf_matrix


def print_confusion_matrix(conf_matrix):
    """draw confusion matrix

    Args:
        data (dict): Contain config data
        path (path-like): The path to save picture
    """
    conf_matrix = conf_matrix / torch.sum(conf_matrix, dim=1, keepdim=True)
    # draw
    plt.figure()
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)  # 可以改变颜色
    labels = ["SU", 'FE', 'DI', 'HA', 'SA', 'AN', 'NE']  # 每种类别的标签
    indices = list(range(7))
    plt.xticks(indices, labels, rotation=45)
    plt.yticks(indices, labels)
    # plt.xlabel('pred')
    # plt.ylabel('true')
    # 显示数据
    for first_index in range(7):  # trues
        for second_index in range(7):  # preds
            if conf_matrix[second_index][first_index] < 0.55:
                plt.text(first_index, second_index, "{:.2f}".format(conf_matrix[second_index][first_index].item()),
                         verticalalignment='center', horizontalalignment='center')
            else:
                plt.text(first_index, second_index, "{:.2f}".format(conf_matrix[second_index][first_index].item()),
                         verticalalignment='center', horizontalalignment='center', color='white')

    plt.tight_layout()
    plt.savefig(fname='confusion_matrix.png', format="png")
    plt.close()


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def BulidModel(args):
    """Bulid Model."""

    if args.Backbone == 'ResNet18':
        if args.Network == 'Baseline':
            model = None

    if args.Resume_Model != 'None':
        print('Resume Model: {}'.format(args.Resume_Model))
        checkpoint = torch.load(args.Resume_Model, map_location='cpu')

        model.load_state_dict(checkpoint, strict=False)
    else:
        print('No Resume Model')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.cuda()

    return model


def BulidAdversarialNetwork(args, model_output_num, class_num=7):
    """Bulid Adversarial Network."""

    ad_net = AdversarialNetwork(model_output_num, 128)
    # ad_net = AdversarialNetwork(model_output_num * class_num, 512)

    ad_net.cuda()

    return ad_net


def DANN(features, ad_net):
    '''
    Paper Link : https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
    Github Link : https://github.com/thuml/CDAN
    '''

    ad_out = ad_net(features)
    batch_size = ad_out.size(0)
    dc_target = torch.from_numpy(np.array([[1]] * (batch_size // 2) + [[0]] * (batch_size // 2))).float()

    ad_out = ad_out.cuda()
    dc_target = dc_target.cuda()

    return nn.BCELoss()(ad_out, dc_target)


def Show_Accuracy(acc, prec, recall, class_num=7):
    """Compute average of accuaracy/precision/recall/f1"""

    # Compute F1 value
    f1 = [AverageMeter() for i in range(class_num)]
    for i in range(class_num):
        if prec[i].avg == 0 or recall[i].avg == 0:
            f1[i].avg = 0
            continue
        f1[i].avg = 2 * prec[i].avg * recall[i].avg / (prec[i].avg + recall[i].avg)

    # Compute average of accuaracy/precision/recall/f1
    acc_avg, prec_avg, recall_avg, f1_avg = 0, 0, 0, 0

    for i in range(class_num):
        acc_avg += acc[i].avg
        prec_avg += prec[i].avg
        recall_avg += recall[i].avg
        f1_avg += f1[i].avg

    acc_avg, prec_avg, recall_avg, f1_avg = acc_avg / class_num, prec_avg / class_num, recall_avg / class_num, f1_avg / class_num

    # Logs Accuracy Infomation
    Accuracy_Info = ''

    Accuracy_Info += '    Accuracy'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(acc[i].avg)
    Accuracy_Info += '\n'

    Accuracy_Info += '    Precision'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(prec[i].avg)
    Accuracy_Info += '\n'

    Accuracy_Info += '    Recall'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(recall[i].avg)
    Accuracy_Info += '\n'

    Accuracy_Info += '    F1'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(f1[i].avg)
    Accuracy_Info += '\n'

    return Accuracy_Info, acc_avg, prec_avg, recall_avg, f1_avg


def show_feature_map(args, image, label, conv_features, num=0, epoch=0, stage=0):
    '''可视化卷积层特征图输出
    img_src:源图像文件路径
    conv_feature:得到的卷积输出,[b, c, h, w]
    '''
    for i in range(conv_features.shape[0]):
        num += 1
        img = image[i, :, :, :]
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # minmax归一化处理
        img = np.uint8(255 * img)  # 像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
        # img = img.squeeze(0)
        heat = conv_features[i, :, :, :]
        # heat = conv_features.squeeze(0)  # 降维操作,尺寸变为(2048,7,7)
        heat_mean = torch.sum(heat, dim=0)  # 对各卷积层(2048)求平均值,尺寸变为(7,7)

        # print(heat_mean)
        # print(torch.sum(heat_mean))
        heatmap = heat_mean.detach().cpu().numpy()  # 转换为numpy数组
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # minmax归一化处理
        heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))  # 变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
        heatmap = np.uint8(255 * heatmap)  # 像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 颜色变换
        # heatmap = np.array(Image.fromarray(heatmap).convert('L'))
        superimg = heatmap * 0.4 + img  # 图像叠加，注意翻转通道，cv用的是bgr
        path = '/home/zhongtao/code/CrossDomainFER/my_method/CAM/test/{}_{}_{:>02d}_{:>05d}_{:>01d}_{}.png'.format(
            args.sourceDataset, args.targetDataset, epoch, num, stage, label[i].item())
        cv2.imwrite(path, superimg)  # 保存结果


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.2):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def VisualizationForTwoDomain(path, model, source_dataloader, target_dataloader):
    '''Feature Visualization in Source and Target Domain.'''

    model.eval()

    Feature_Source, Label_Source = [], []

    # Get Feature and Label in Source Domain
    for batch_i, (imgs, label, indexes) in enumerate(source_dataloader):
        imgs, label = imgs.cuda(), label.cuda()
        with torch.no_grad():
            output, _, _, _, _, _, feature = model(imgs)

        Feature_Source.append(feature.cpu().data.numpy())
        Label_Source.append(label.cpu().data.numpy())

    Feature_Source = np.vstack(Feature_Source)
    Label_Source = np.concatenate(Label_Source)

    Feature_Target, Label_Target = [], []
    iter_target_dataloader = iter(target_dataloader)

    # Get Feature and Label in Target Domain
    for batch_index, (input, label, _) in enumerate(iter_target_dataloader):
        input, label = input.cuda(), label.cuda()
        with torch.no_grad():
            output, _, _, _, _, _, feature = model(input)

        Feature_Target.append(feature.cpu().data.numpy())
        Label_Target.append(label.cpu().data.numpy())

    Feature_Target = np.vstack(Feature_Target)
    Label_Target = np.concatenate(Label_Target)

    Label_Target += 7

    Feature = np.vstack((Feature_Source, Feature_Target))
    Label = np.concatenate((Label_Source, Label_Target))

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    plot_num = 13000
    embedding = tsne.fit_transform(Feature[:plot_num, :])

    # Draw Visualization of Feature
    colors = {0: 'red', 1: 'blue', 2: 'olive', 3: 'green', 4: 'orange', 5: 'purple', 6: 'darkslategray', \
              7: 'red', 8: 'blue', 9: 'olive', 10: 'green', 11: 'orange', 12: 'purple', 13: 'darkslategray'}
    labels = {0: 'Surprised', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Neutral', \
              7: 'Surprised', 8: 'Fear', 9: 'Disgust', 10: 'Happy', 11: 'Sad', 12: 'Angry', 13: 'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    # ax = plt.subplot(111)

    for i in range(7):

        data_source_x, data_source_y = data_norm[Label[:plot_num] == i][:, 0], data_norm[Label[:plot_num] == i][:, 1]
        source_scatter = plt.scatter(data_source_x, data_source_y, color="none", edgecolor=colors[i], s=20,
                                     label=labels[i], marker="o", alpha=0.4, linewidth=0.5)

        data_target_x, data_target_y = data_norm[Label[:plot_num] == (i + 7)][:, 0], data_norm[
                                                                                         Label[:plot_num] == (i + 7)][:,
                                                                                     1]
        target_scatter = plt.scatter(data_target_x, data_target_y, color=colors[i], edgecolor="none", s=30,
                                     label=labels[i], marker="x", alpha=0.6, linewidth=0.2)

        if i == 0:
            source_legend = source_scatter
            target_legend = target_scatter

    # tmp = [0, 1]
    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in tmp ], loc='upper right', prop = {'size':8})

    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in range(7)], loc='upper right', prop = {'size':8})
    # plt.legend([source, target], ['Source Domain', 'Target Domain'], loc='upper left', prop = {'size':8})

    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(7)],
    #                 loc='upper left', prop={'size': 8}, bbox_to_anchor=(1.05, 0.85), borderaxespad=0)
    # plt.legend([source, target], ['Source Domain', 'Target Domain'], loc='upper left', prop={'size': 7},
    #            bbox_to_anchor=(1.05, 1.0), borderaxespad=0)
    # plt.gca().add_artist(l1)

    plt.savefig(fname=path, format="png", bbox_inches='tight')


def VisualizationForOneDomain(path, model, dataloader):
    '''Feature Visualization in Source and Target Domain.'''

    model.eval()

    Feature_Target, Label_Target = [], []
    iter_target_dataloader = iter(dataloader)

    # Get Feature and Label in Target Domain
    for batch_index, (input, label, _) in enumerate(iter_target_dataloader):
        input, label = input.cuda(), label.cuda()
        with torch.no_grad():
            output, _, _, _, _, _, feature = model(input)

        Feature_Target.append(feature.cpu().data.numpy())
        Label_Target.append(label.cpu().data.numpy())

    Feature_Target = np.vstack(Feature_Target)
    Label_Target = np.concatenate(Label_Target)

    Feature = Feature_Target
    Label = Label_Target

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3, )
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    colors = {0: 'red', 1: 'blue', 2: 'olive', 3: 'green', 4: 'orange', 5: 'purple', 6: 'darkslategray'}
    labels = {0: 'Surprised', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    # ax = plt.subplot(111)

    for i in range(7):

        data_target_x, data_target_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
        target_scatter = plt.scatter(data_target_x, data_target_y, color=colors[i], edgecolor="none", s=30,
                                     label=labels[i], marker="x", alpha=0.6, linewidth=0.2)

        if i == 0:
            target_legend = target_scatter

    # tmp = [0, 1]
    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in tmp ], loc='upper right', prop = {'size':8})

    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in range(7)], loc='upper right', prop = {'size':8})
    # plt.legend([source, target], ['Source Domain', 'Target Domain'], loc='upper left', prop = {'size':8})

    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(7)],
    #                 loc='upper left', prop={'size': 8}, bbox_to_anchor=(1.05, 0.85), borderaxespad=0)
    # plt.legend([source, target], ['Source Domain', 'Target Domain'], loc='upper left', prop={'size': 7},
    #            bbox_to_anchor=(1.05, 1.0), borderaxespad=0)
    # plt.gca().add_artist(l1)

    plt.savefig(fname=path, format="png", bbox_inches='tight')
