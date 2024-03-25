import torch
import torch.nn as nn

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