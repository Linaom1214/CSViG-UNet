import torch.nn as nn
import torch.nn.functional as F

from models.vig import *


def get_vig(num_class=1):
    return ViG(num_class)


if __name__ == '__main__':
    model = ViG(1)
    from torchsummary import summary
    summary(model, (3, 256, 256), device='cpu')
