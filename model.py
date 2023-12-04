import torch.nn as nn

from UNet import UNet
from reconstruct_model import MaskedAutoencoderViT


class FERAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = MaskedAutoencoderViT()
        self.cnn = UNet()

    def forward(self, x):
        x = self.mae(x)
        x = self.cnn(x)

        return x
