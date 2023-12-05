import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from reconstruct_model import MaskedAutoencoderViT
from functools import partial
device = "cuda:0"
ckpt_pth = "/home/zhongtao/mae_finetuned_vit_base.pth"

model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

model_dict = model.state_dict()
ckpt_weight = torch.load(ckpt_pth, map_location = device)['model']
target_weights = {}
for k in model_dict.keys():
    if k in ckpt_weight:
        target_weights[k] = ckpt_weight[k].clone()
    else:
        print(f"these weights are not in our model: {k}")

model.load_state_dict(target_weights,strict=True)
torch.save(model.state_dict(),"./resume.pth")
