import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftLoULoss(nn.Module):
    def __init__(self):
        super(SoftLoULoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)

        return loss

class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, inputs, target, smooth=1e-5):
        intersection = (inputs * target).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 2.0 * (intersection + smooth) / (union + smooth)
        loss = 1.0 - dice
        return loss.sum()

class WeightedLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        
        return (wbce + wiou).mean()
