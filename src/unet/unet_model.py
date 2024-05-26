""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 32, bilinear)
        self.up5 = Up(64, 16, bilinear)
        self.up6 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

        self.up_s1 = UpScale(64,32,bilinear)
        self.up_s2 = UpScale(32,16,bilinear)

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
        X_1 = self.up_s2(x0)

        x = self.up5(x, x0)
        x = self.up6(x, X_1)

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
        self.up5 = torch.utils.checkpoint(self.up5)
        self.up6 = torch.utils.checkpoint(self.up6)
        self.up_s1 = torch.utils.checkpoint(self.up_s1)
        self.up_s2 = torch.utils.checkpoint(self.up_s2)
        self.outc = torch.utils.checkpoint(self.outc)