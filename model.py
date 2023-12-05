import torch.nn as nn

from UNet import UNet
from reconstruct_model import MaskedAutoencoderViT
from torchvision.models import resnet18
from decompose_net import My_Model_hire as decomposer_net
from functools import partial


class FERAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mae_encoder = MaskedAutoencoderViT(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # self.to_next_proj = nn.Linear(kwargs["embed_dim"], kwargs["patch_size"]**2 * kwargs["in_chans"]) TODO: add after mae_encoder, equally project

        self.decomposer = decomposer_net(args=None, NumOfLayer=18)

        self.classify = nn.Linear(512, 7)

    def forward(self, x):
        # MAE encoder
        x = self.mae_encoder(x)
        x = x[:, 1:, :]  # remove cls token
        x = self.mae_encoder.unpatchify(x)
        feat_domain, feat_exp = self.decomposer(x)
        logit_exp = self.classify(feat_exp)
        #
        return feat_domain, logit_exp
