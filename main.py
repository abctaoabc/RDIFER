# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 22:40
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : run_mae_vis.py
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from PIL import Image

from pathlib import Path

from timm.models import create_model
from torchvision import transforms as transforms

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import reconstruct_model_2
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('--img_path', type=str, help='input image path', default='/home/zhongtao/datasets/CelebAMask-HQ/')
    parser.add_argument('--save_path', type=str, help='save image path', default='./output')
    parser.add_argument('--model_path', type=str, help='checkpoint path of model',
                        default="/home/zhongtao/code/RDIFER/mae_pretrain_checkpoint.pth")
    parser.add_argument('--img_name', type=str, default='12223.jpg')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default="pretrain_mae_base_patch16_224_parsing_mask", type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model

def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with open(os.path.join(args.img_path+"CelebA-HQ-img", args.img_name), 'rb') as f:
        img = Image.open(f)
        img.convert('RGB')
        print("img path:", args.img_path)

    parsing_path = os.path.join(args.img_path+"CelebAMaskHQ-mask", args.img_name[:-4]+".png")
    parsing_face = Image.open(parsing_path).convert('P').resize((224,224))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    mean = IMAGENET_INCEPTION_MEAN if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    transform = transforms.Compose([
        transforms.Resize([args.input_size, args.input_size]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ])
    img= transform(img)

    with torch.no_grad():
        img = img.unsqueeze(0)
        img = img.to(device, non_blocking=True)
        mask_pos = model.parsing_mask(img, np.array(parsing_face).astype(np.int64))
        mask_pos = torch.from_numpy(mask_pos)[None, :]
        mask_pos = mask_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        outputs = model(img, mask_pos)

        # save original img
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
        ori_img = img * std + mean  # in [0, 1]
        img = ToPILImage()(ori_img[0, :])
        img.save(f"{args.save_path}/ori_img.jpg")

        img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[0])
        img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
                    img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        img_patch[mask_pos] = outputs

        # make mask
        mask = torch.ones_like(img_patch)
        mask[mask_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)

        # save reconstruction img
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
        rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(
            dim=-2, keepdim=True)
        rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14,
                            w=14)
        img = ToPILImage()(rec_img[0, :].clip(0, 0.996))
        img.save(f"{args.save_path}/rec_img.jpg")

        #save random mask img
        img_mask = rec_img * mask
        img = ToPILImage()(img_mask[0, :])
        img.save(f"{args.save_path}/mask_img.jpg")

if __name__ == '__main__':
    opts = get_args()
    main(opts)