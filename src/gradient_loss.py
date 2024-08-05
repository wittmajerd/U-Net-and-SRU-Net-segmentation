import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel1 = [[-1., -2., -1.],
                  [0., 0., 0.],
                  [1., 2., 1.]]
        kernel1 = torch.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        self.weight1 = nn.Parameter(data=kernel1, requires_grad=False)

        kernel2 = [[-1., 0., 1.],
                  [-2., 0., 2.],
                  [-1., 0., 1.]]
        kernel2 = torch.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)

    def forward(self, x, y):
        loss1 = 0
        loss2 = 0

        for i in range(x.shape[0]):  # loop over each image in the batch
            x1 = x[i, 0].float()
            x1 = F.conv2d(x1.unsqueeze(0).unsqueeze(0), self.weight1, padding=1)

            y1 = y[i, 0].float()
            y1 = F.conv2d(y1.unsqueeze(0).unsqueeze(0), self.weight1, padding=1)

            loss1 += torch.mean(torch.mean((x1 - y1)**2)) / 100.0

            x1 = F.conv2d(x1, self.weight2, padding=1)

            y1 = F.conv2d(y1, self.weight2, padding=1)

            loss2 += torch.mean(torch.mean((x1 - y1)**2)) / 10000.0

        loss1 /= x.shape[0]  # average the loss over the batch
        loss2 /= x.shape[0]  # average the loss over the batch

        loss = torch.sqrt(loss1*loss1 + loss2*loss2) * 2
        return loss