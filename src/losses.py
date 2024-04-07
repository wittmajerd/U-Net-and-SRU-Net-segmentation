import torch

class DiceLoss(torch.nn.Module):
    def forward(self, pred, target, smooth = 1.):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

class IoULoss(torch.nn.Module):
    def forward(self, pred, target, smooth = 1.):
        intersection = (pred * target).sum()
        total = (pred + target).sum()
        union = total - intersection 
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou