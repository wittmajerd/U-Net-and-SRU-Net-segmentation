import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from .test_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

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

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        

class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = SingleConv(n_channels, 64)
        self.down1 = DownSingle(64, 128)
        self.down2 = DownSingle(128, 256)
        self.down3 = DownSingle(256, 512)
        self.down4 = DownSingle(512, 512)
        self.up1 = UpSingle(1024, 256, bilinear)
        self.up2 = UpSingle(512, 128, bilinear)
        self.up3 = UpSingle(256, 64, bilinear)
        self.up4 = UpSingle(128, 32, bilinear)
        self.up5 = UpSingle(64, 16, bilinear)
        self.up6 = UpSingle(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

        self.up_s1=Upscaling(64,32)
        self.up_s2=Upscaling(32,16)

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
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet8, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = SingleConv(n_channels, 64)
        self.down1 = DownSingle(64, 128)
        self.down2 = DownSingle(128, 256)
        self.down3 = DownSingle(256, 512)
        self.down4 = DownSingle(512, 512)
        self.up1 = UpSingle(1024, 256, bilinear)
        self.up2 = UpSingle(512, 128, bilinear)
        self.up3 = UpSingle(256, 64, bilinear)
        self.up4 = UpSingle(128, 32, bilinear)
        self.up5 = UpSingle(64, 16, bilinear)
        self.up6 = UpSingle(32,8, bilinear)
        self.up7 = UpSingle(16,8, bilinear)
        self.outc = OutConv(8, n_classes)

        self.up_s1=Upscaling(64,32)
        self.up_s2=Upscaling(32,16)
        self.up_s3=Upscaling(16,8)

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
