import torch.nn as nn

from UNet import UNet
from reconstruct_model import MaskedAutoencoderViT
from torchvision.models import resnet18
from decompose_net import My_Model_hire as decomposer_net
from functools import partial


class FERAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae_encoder = MaskedAutoencoderViT(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # self.cnn = UNet()
        self.decomposer = decomposer_net(args=None, NumOfLayer=18)

    def forward(self, x):
        x = self.mae_encoder(x)

        return x
