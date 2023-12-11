import copy
import random
import logging
import json
from typing import Sequence, Dict, Union
import cv2
import numpy
import torch
import numpy as np
import os
import PIL.Image as Image
import time
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pickle
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

logger = logging.getLogger(__name__)


class CelebAMask_dataset(data.Dataset):

    def __init__(
            self,
            file_list: str,
            out_size: int,
            trans
    ):
        super(CelebAMask_dataset, self).__init__()
        self.file_list = file_list

        # front face part
        files = []
        taget_face_list = os.listdir(file_list)
        for line in taget_face_list:
            path = line.strip()
            if path:
                files.append(os.path.join(self.file_list+"/CelebA-HQ-img", path))
        self.img_paths = files

        # parsing
        self.parsing_path = os.path.join(self.file_list, "CelebAMaskHQ-mask")

        self.out_size = out_size
        self.transfrom = trans

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.

        gt_path = self.img_paths[index]
        img_name = gt_path.split("/")[-1]
        parsing_path = os.path.join(self.parsing_path, img_name)
        face_img = Image.open(gt_path).convert("RGB")
        parsing_face = Image.open(parsing_path).convert('P')
        h, w, _ = np.array(face_img).shape

        # resize
        face_img = self.transfrom(face_img)

        # parsing part
        parsing_gt = np.array(parsing_face).astype(np.int64)

        return face_img, parsing_gt

    def __len__(self) -> int:
        return len(self.img_paths)
