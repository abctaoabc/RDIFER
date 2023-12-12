import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from reconstruct_model import MaskedAutoencoderViT
from reconstruct_model_2 import *
from functools import partial
device = "cuda:0"
ckpt_pth = "/home/zhongtao/mae_pretrain_checkpoint.pth"

model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

model_dict = model.state_dict()
ckpt_weight = torch.load(ckpt_pth, map_location = device)['model']
target_weights = {}
for k in model_dict.keys():
    if k in ckpt_weight:
        target_weights[k] = ckpt_weight[k].clone()
    else:
        target_weights[k] = model_dict[k].clone()
        print(f"these weights are lack in pre_trained: {k}")

model.load_state_dict(target_weights,strict=True)
torch.save(model.state_dict(),"./new_reconstruct.pth")
print("Done!")