import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.decorder4 = Decoder(1024, 512)
        self.decorder3 = Decoder(512, 256)
        self.decorder2 = Decoder(256, 128)
        self.decorder1 = Decoder(128, 64)
        self.last = nn.Conv2d(64, 3, 1)

    def forward(self, input):
        # Encorder
        layer1 = self.layer1(input)
        layer2 = self.layer2(self.maxpool(layer1))
        layer3 = self.layer3(self.maxpool(layer2))
        layer4 = self.layer4(self.maxpool(layer3))
        layer5 = self.layer5(self.maxpool(layer4))

        # Decorder
        layer6 = self.decorder4(layer5, layer4)
        layer7 = self.decorder3(layer6, layer3)
        layer8 = self.decorder2(layer7, layer2)
        layer9 = self.decorder1(layer8, layer1)
        out = self.last(layer9)

        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # up-conv 2*2
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, high, low):
        x1 = self.up(high)
        offset = x1.size()[2] - low.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        # 计算应该填充多少（这里可以是负数）
        x2 = F.pad(low, padding)
        x1 = torch.cat((x1, x2), dim=1)  # 拼起来
        x1 = self.conv_relu(x1)  # 卷积走起
        return x1


