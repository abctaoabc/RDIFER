import time
from random import random

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib import tenumerate

from Datasets import RafDataSet, SFEWDataSet, JAFFEDataSet, FER2013DataSet, ExpWDataSet, AffectNetDataSet, \
    FER2013PlusDataSet
from Utils import *
from model import FERAE

device = "cuda:0"


model = FERAE().to(device)
resume_weight = torch.load('./resume.pth',map_location=device)
model.mae_encoder.load_state_dict(resume_weight)
trans = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor()
])

img = Image.open("test.png")
img = trans(img).unsqueeze(0)
img = img.to(device)
model(img)