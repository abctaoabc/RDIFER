import time
from random import random

import functorch.dim
import torch.nn
import torch.optim as optim

import torch.utils.data as data
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib import tenumerate

from Datasets import RafDataSet, SFEWDataSet, JAFFEDataSet, FER2013DataSet, ExpWDataSet, AffectNetDataSet, \
    FER2013PlusDataSet
from Datasets import *
from Utils import *
from model import FERAE

parser = argparse.ArgumentParser(description='Expression Classification Training')
##
parser.add_argument('--Log_Name', type=str, default='train', help='Logs Name')
parser.add_argument('--OutputPath', default='/home/zhongtao/code/RDIFER/checkpoints', type=str,
                    help='Output Path')
parser.add_argument('--Backbone', type=str, default='ResNet18', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--Network', type=str, default='FERAE',
                    choices=['Baseline', 'FERAE'])
parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='None')
parser.add_argument('--pretrained', type=str,
                    default="/home/zhongtao/code/RDIFER/resume.pth",
                    help='Pretrained weights')
parser.add_argument('--device', default='cuda:0', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--faceScale', type=int, default=224, help='Scale of face (default: 112)')
parser.add_argument('--sourceDataset', type=str, default='RAFDB',
                    choices=['RAFDB', 'SFEW', 'FER2013', 'AffectNet', 'AffectNet', 'ExpW'])
parser.add_argument('--targetDataset', type=str, default='ExpW',
                    choices=['RAFDB', 'SFEW', 'FER2013', 'FER2013Plus', 'AffectNet', 'JAFFE', 'ExpW'])
parser.add_argument('--raf_path', type=str, default='/home/zhongtao/datasets/RAFDB',
                    help='Raf-DB dataset path.')
parser.add_argument('--jaffe-path', type=str, default='/home/zhongtao/datasets/jaffedbase',
                    help='JAFFE dataset path.')
parser.add_argument('--fer2013-path', type=str, default='/home/zhongtao/datasets/FER2013',
                    help='FER2013 dataset path.')
parser.add_argument('--fer2013plus-path', type=str, default='/home/zhongtao/datasets/FER2013+',
                    help='FER2013Plus dataset path.')
parser.add_argument('--expw-path', type=str, default='/home/zhongtao/datasets/ExpW',
                    help='ExpW dataset path.')
parser.add_argument('--sfew-path', type=str, default='/home/zhongtao/datasets/SFEW2.0',
                    help='SFEW dataset path.')
parser.add_argument('--affectnet-path', type=str, default='/home/zhongtao/datasets/AffectNet',
                    help='AffectNet dataset path.')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')

parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 60)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.01, help='SGD weight decay (default: 0.0001)')

parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model')
parser.add_argument('--isSave', type=str2bool, default=True, help='whether to save model')
parser.add_argument('--showFeature', type=str2bool, default=False, help='whether to show feature')

parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')
parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 1)')

num = 0


def Train(args, model, train_dataloader, optimizer, scheduler, epoch, writer):
    """Train."""

    model.train()
    torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in
                                                                                                 range(7)]
    loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_i, (imgs, label, indexes, domain_label) in tenumerate(train_dataloader):
        imgs, label = imgs.to(args.device), label.to(args.device)
        domain_target = domain_label.to(args.device)
        data_time.update(time.time() - end)

        # Forward propagation
        end = time.time()
        feat_domain, logit_exp = model(imgs)
        batch_time.update(time.time() - end)

        # Compute Loss
        global_cls_loss_ = torch.nn.CrossEntropyLoss()(logit_exp, label)
        domain_clc_loss_ = torch.nn.BCELoss()(feat_domain, domain_target.unsqueeze(1).float())
        # global_cls_loss_ = LabelSmoothLoss()(output, label)

        loss_ = global_cls_loss_ + domain_clc_loss_

        # Back Propagation
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

        output = F.softmax(logit_exp)
        # Compute accuracy, recall and loss
        Compute_Accuracy(args, output, label, acc, prec, recall)

        # Logs loss
        loss.update(float(loss_.cpu().data.item()))


        end = time.time()

    scheduler.step()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Accuracy', acc_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)
    writer.add_scalar('Loss', loss.avg, epoch)

    LoggerInfo = '''
    [Tain]: 
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {1}\n'''.format(epoch, scheduler.get_lr(), data_time=data_time, batch_time=batch_time)

    LoggerInfo += AccuracyInfo

    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f} Norm Loss {4:.4f} Center Loss {5:.4f} Total Loss {loss:.4f}'''.format(
        acc_avg, prec_avg, recall_avg, f1_avg, 0, 0, loss=loss.avg)

    print(LoggerInfo)


def Test(args, model, test_source_dataloader, test_target_dataloader, Best_Acc_Source, Best_Acc_Target, epoch, writer):
    """Test."""
    global num

    model.eval()
    torch.autograd.set_detect_anomaly(True)

    iter_source_dataloader = iter(test_source_dataloader)
    iter_target_dataloader = iter(test_target_dataloader)

    # Test on Source Domain
    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in
                                                                                                 range(7)]
    loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_index, (input, label, _, domain_label) in tenumerate(iter_source_dataloader):
        data_time.update(time.time() - end)

        imgs, label = input.to(args.device), label.to(args.device)
        domain_target = domain_label.to(args.device)

        with torch.no_grad():
            end = time.time()
            feat_domain, logit_exp = model(imgs)

            batch_time.update(time.time() - end)

        global_cls_loss_ = torch.nn.CrossEntropyLoss()(logit_exp, label)
        domain_clc_loss_ = torch.nn.BCELoss()(feat_domain, domain_target.unsqueeze(1).float())
        loss_ = global_cls_loss_ + domain_clc_loss_

        output = F.softmax(logit_exp)
        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, label, acc, prec, recall)

        # Logs loss
        loss.update(float(loss_.cpu().data.numpy()))

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Test_Recall_SourceDomain', recall_avg, epoch)
    writer.add_scalar('Test_Accuracy_SourceDomain', acc_avg, epoch)

    LoggerInfo = '''
    [Test (Source Domain)]: 
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)

    LoggerInfo += AccuracyInfo

    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)

    print(LoggerInfo)

    # Save Checkpoints
    if acc_avg > Best_Acc_Source and not args.isTest:
        Best_Acc_Source = acc_avg


    # Test on Target Domain
    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in
                                                                                                 range(7)]
    loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_index, (input, label, _) in tenumerate(iter_target_dataloader):
        data_time.update(time.time() - end)

        imgs, label = input.to(args.device), label.to(args.device)
        # domain_target = domain_label.to(args.device)

        with torch.no_grad():
            end = time.time()
            _, logit_exp = model(imgs)
            batch_time.update(time.time() - end)

        global_cls_loss_ = torch.nn.CrossEntropyLoss()(logit_exp, label)
        # domain_clc_loss_ = torch.nn.BCELoss()(feat_domain, domain_target.unsqueeze(1).float())
        loss_ = global_cls_loss_

        output = F.softmax(logit_exp)
        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, label, acc, prec, recall)

        # Logs loss
        loss.update(float(loss_.cpu().data.numpy()))

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Test_Recall_TargetDomain', recall_avg, epoch)
    writer.add_scalar('Test_Accuracy_TargetDomain', acc_avg, epoch)

    LoggerInfo = '''
    [Test (Target Domain)]:
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)

    LoggerInfo += AccuracyInfo

    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)

    print(LoggerInfo)
    num = 0

    if acc_avg > Best_Acc_Target and not args.isTest:
        Best_Acc_Target = acc_avg
        print('[Save] Best Accuracy: %.4f.' % Best_Acc_Target)

        if not args.isTest and args.isSave:
            # Save Checkpoints
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(args.OutputPath,
                                                                   '{}_{}_{}_test.pkl'.format(args.sourceDataset,
                                                                                          args.targetDataset,
                                                                                          args.Network, )))
            else:
                torch.save(model.state_dict(), os.path.join(args.OutputPath,
                                                            '{}_{}_{}_test.pkl'.format(args.sourceDataset,
                                                                                   args.targetDataset, args.Network)))

    return Best_Acc_Source, Best_Acc_Target


def main():
    """Main."""

    # Parse Argument
    args = parser.parse_args()
    if not args.isTest:
        sys.stdout = Logger(
            osp.join('./Logs/', '{}_{}_{}_test.txt'.format(args.sourceDataset, args.targetDataset, args.Network)))
    if args.seed:
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print('set seed:{}'.format(args.seed))

    # Experiment Information
    print('Logs Name: %s' % args.Log_Name)
    print('Output Path: %s' % args.OutputPath)
    print('Backbone: %s' % args.Backbone)
    print('Network: %s' % args.Network)
    print('CUDA_USE: %s' % args.device)

    print('================================================')

    print('Use {} * {} Image'.format(args.faceScale, args.faceScale))
    print('SourceDataset: %s' % args.sourceDataset)
    print('TargetDataset: %s' % args.targetDataset)
    print('Batch Size: %d' % args.batch_size)

    print('================================================')

    if args.isTest:
        print('Test Model.')
    else:
        print('Train Epoch: %d' % args.epochs)
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Weight Decay: %f' % args.weight_decay)
        print('Number of classes : %d' % args.class_num)

    print('================================================')

    # Bulid Dataloder
    print("Buliding Train and Test Dataloader...")
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    if args.sourceDataset == 'RAFDB':
        train_dataset = domain_RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)
    elif args.sourceDataset == 'FER2013':
        train_dataset = FER2013DataSet(args.fer2013_path, phase='train', transform=data_transforms, basic_aug=True)
    elif args.sourceDataset == 'FER2013Plus':
        train_dataset = FER2013PlusDataSet(args.fer2013plus_path, phase='train', transform=data_transforms,
                                           basic_aug=True)
    elif args.sourceDataset == 'SFEW':
        train_dataset = SFEWDataSet(args.sfew_path, phase='train', transform=data_transforms, basic_aug=True)
    elif args.sourceDataset == 'ExpW':
        train_dataset = ExpWDataSet(args.expw_path, phase='train', transform=data_transforms, basic_aug=True)
    elif args.sourceDataset == 'AffectNet':
        train_dataset = AffectNetDataSet(args.affectnet_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())
    print('The Source Train dataset distribute:', train_dataset.__distribute__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    if args.sourceDataset == 'RAFDB':
        val_dataset_source = domain_RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    elif args.sourceDataset == 'FER2013':
        val_dataset_source = FER2013DataSet(args.fer2013_path, phase='test', transform=data_transforms_val)
    elif args.sourceDataset == 'FER2013Plus':
        val_dataset_source = FER2013PlusDataSet(args.fer2013plus_path, phase='test', transform=data_transforms_val)
    elif args.sourceDataset == 'SFEW':
        val_dataset_source = SFEWDataSet(args.sfew_path, phase='test', transform=data_transforms_val)
    elif args.sourceDataset == 'ExpW':
        val_dataset_source = ExpWDataSet(args.expw_path, phase='test', transform=data_transforms_val)
    elif args.sourceDataset == 'AffectNet':
        val_dataset_source = AffectNetDataSet(args.affectnet_path, phase='test', transform=data_transforms_val)

    if args.targetDataset == 'RAFDB':
        val_dataset_target = domain_RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    elif args.targetDataset == 'JAFFE':
        val_dataset_target = JAFFEDataSet(args.jaffe_path, transform=data_transforms_val)
    elif args.targetDataset == 'FER2013':
        val_dataset_target = FER2013DataSet(args.fer2013_path, phase='test', transform=data_transforms_val)
    elif args.targetDataset == 'FER2013Plus':
        val_dataset_target = FER2013PlusDataSet(args.fer2013plus_path, phase='test', transform=data_transforms_val)
    elif args.targetDataset == 'SFEW':
        val_dataset_target = SFEWDataSet(args.sfew_path, phase='test', transform=data_transforms_val)
    elif args.targetDataset == 'ExpW':
        val_dataset_target = ExpWDataSet(args.expw_path, phase='test', transform=data_transforms_val)
    elif args.targetDataset == 'AffectNet':
        val_dataset_target = AffectNetDataSet(args.affectnet_path, phase='test', transform=data_transforms_val)

    print('Validation Source set size:', val_dataset_source.__len__())
    print('The Validation Source dataset distribute:', val_dataset_source.__distribute__())
    print('Validation Target set size:', val_dataset_target.__len__())
    print('The Validation Target dataset distribute:', val_dataset_target.__distribute__())

    val_loader_source = torch.utils.data.DataLoader(val_dataset_source,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.workers,
                                                    shuffle=False,
                                                    pin_memory=True)
    val_loader_target = torch.utils.data.DataLoader(val_dataset_target,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.workers,
                                                    shuffle=False,
                                                    pin_memory=True)
    print('Done!')

    print('================================================')

    # Bulid Model
    print('Buliding Model...')
    model = FERAE().to(args.device)
    resume_weight = torch.load(args.pretrained, map_location=args.device)
    model.mae_encoder.load_state_dict(resume_weight)
    print('Done!')

    print('================================================')

    # Set Optimizer
    print('Buliding Optimizer...')
    params = model.parameters()
    optimizer = optim.AdamW(params, betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print('Done!')

    print('================================================')

    # Save Best Checkpoint
    Best_Acc_Source, Best_Acc_Target = 0, 0

    # Running Experiment
    print("Run Experiment...")

    writer = SummaryWriter(os.path.join('/home/zhongtao/code/RDIFER/LogInfo',
                                        '{}_{}_{}'.format(args.sourceDataset, args.targetDataset, args.Network)))

    for epoch in range(1, args.epochs + 1):

        if not args.isTest:
            Train(args, model, train_loader, optimizer, scheduler, epoch, writer)

        Best_Acc_Source, Best_Acc_Target = Test(args, model, val_loader_source, val_loader_target, Best_Acc_Source,
                                                Best_Acc_Target, epoch, writer)

        if args.showFeature:
            VisualizationForTwoDomain(
                '/home/zhongtao/code/RDIFER/visualization/{}_{}_{:>02d}_test.pdf'.format(
                    args.sourceDataset, args.targetDataset, epoch), model, train_loader, val_loader_target)

        # torch.cuda.empty_cache()

    writer.close()
    print('Best Accuarcy on Source Domain:%.4f' % (Best_Acc_Source))
    print('Best Accuarcy on Target Domain:%.4f' % (Best_Acc_Target))


if __name__ == '__main__':
    main()
