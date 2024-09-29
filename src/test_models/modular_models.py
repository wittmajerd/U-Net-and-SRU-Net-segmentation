import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from .test_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, down_conv, up_conv, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = down_conv(n_channels, 64)
        self.down1 = Down(64, 128, down_conv)
        self.down2 = Down(128, 256, down_conv)
        self.down3 = Down(256, 512, down_conv)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, down_conv)
        self.up1 = Up(1024, 512 // factor, up_conv, bilinear)
        self.up2 = Up(512, 256 // factor, up_conv, bilinear)
        self.up3 = Up(256, 128 // factor, up_conv, bilinear)
        self.up4 = Up(128, 64, up_conv, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes, down_conv, up_conv, bilinear=False):
        super(UNet4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = down_conv(n_channels, 64)
        self.down1 = Down(64, 128, down_conv)
        self.down2 = Down(128, 256, down_conv)
        self.down3 = Down(256, 512, down_conv)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, down_conv)
        self.up1 = Up(1024, 512 // factor, up_conv, bilinear)
        self.up2 = Up(512, 256 // factor, up_conv, bilinear)
        self.up3 = Up(256, 128 // factor, up_conv, bilinear)
        self.up4 = Up(128, 64 // factor, up_conv, bilinear)
        self.up5 = Up(64, 32 // factor, up_conv, bilinear)
        self.up6 = Up(32, 16, up_conv, bilinear)
        self.outc = OutConv(16, n_classes)

        self.up_s1=Upscaling(64, 32, up_conv)
        self.up_s2=Upscaling(32, 16, up_conv)

    def forward(self, xs):
        x1 = self.inc(xs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Upsample "negative" layers
        x0=self.up_s1(x1)
        x_1=self.up_s2(x0)
        
        x = self.up5(x, x0)
        x = self.up6(x, x_1)
        x = self.outc(x)
        return x

class UNet8(nn.Module):
    def __init__(self, n_channels, n_classes, down_conv, up_conv, bilinear=False):
        super(UNet8, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = down_conv(n_channels, 64)
        self.down1 = Down(64, 128, down_conv)
        self.down2 = Down(128, 256, down_conv)
        self.down3 = Down(256, 512, down_conv)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, down_conv)
        self.up1 = Up(1024, 512 // factor, up_conv, bilinear)
        self.up2 = Up(512, 256 // factor, up_conv, bilinear)
        self.up3 = Up(256, 128 // factor, up_conv, bilinear)
        self.up4 = Up(128, 64 // factor, up_conv, bilinear)
        self.up5 = Up(64, 32 // factor, up_conv, bilinear)
        self.up6 = Up(32, 16 // factor, up_conv, bilinear)
        self.up7 = Up(16, 8, up_conv, bilinear)
        self.outc = OutConv(8, n_classes)

        self.up_s1=Upscaling(64, 32)
        self.up_s2=Upscaling(32, 16)
        self.up_s3=Upscaling(16, 8)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x0=self.up_s1(x1)
        x_1=self.up_s2(x0)
        x_2=self.up_s3(x_1)

        x = self.up5(x, x0)
        x = self.up6(x, x_1)
        x = self.up7(x, x_2)
        x = self.outc(x)
        return x
