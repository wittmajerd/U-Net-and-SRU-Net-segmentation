import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from .srunet_parts import *

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear)
        self.up2 = up(512, 128, bilinear)
        self.up3 = up(256, 64, bilinear)
        self.up4 = up(128, 32, bilinear)
        self.up5 = up(64, 16, bilinear)

        self.outc = outconv(16, n_classes)

        self.up_s1=up_s(64,32)
        #self.up_s=up_s(32,16)
        #self.up_s=up_s(16,8)

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

        x0 = self.up_s1(x1)
        x = self.up5(x, x0)

        # What is this??
        #xout2 = F.conv2d(x2.unsqueeze(1), self.weight1, padding=2)
        #xout3 = F.conv2d(x3.unsqueeze(1), self.weight1, padding=2)
        #x = self.beforeconv(x)
        #x = self.pixel_shuffle(x)

        x = self.outc(x)
        return x    #F.sigmoid(x)?

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear)
        self.up2 = up(512, 128, bilinear)
        self.up3 = up(256, 64, bilinear)
        self.up4 = up(128, 32, bilinear)
        self.up5 = up(64, 16, bilinear)
        self.up6 = up(32, 16, bilinear)
        self.outc = outconv(16, n_classes)  #64?

        self.up_s1=up_s(64,32)
        self.up_s2=up_s(32,16)
        #self.up_s=up_s(16,8)

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

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class UNet8(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet8, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.bilinear = bilinear

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)  #(128, 64)
        self.up5 = up(64, 16)
        self.up6 = up(32,8)
        self.up7 = up(16,8)
        self.outc = outconv(8, n_classes)   #64

        self.up_s1=up_s(64,32)
        self.up_s2=up_s(32,16)
        self.up_s3=up_s(16,8)

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

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class UNet16(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet16, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.bilinear = bilinear

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)  #(128, 64)
        self.up5 = up(64, 16)
        self.up6 = up(32,8)
        self.up7 = up(16,4)
        self.up8 = up(8,4)
        self.outc = outconv(4, n_classes)   #64

        self.up_s1=up_s(64,32)
        self.up_s2=up_s(32,16)
        self.up_s3=up_s(16,8)
        self.up_s4=up_s(8,4)

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
        x_3=self.up_s4(x_2)

        x = self.up5(x, x0)
        x = self.up6(x, x_1)
        x = self.up7(x, x_2)
        x = self.up8(x, x_3)
        x = self.outc(x)

        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)



def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    plt.pause(5)