import torch
from torch import nn
from timm.models.layers import DropPath
import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_
from models.head import _FCNHead


class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type

    K is the number of superpatches, therefore hops equals res // K.
    """

    def __init__(self, in_channels, out_channels, K=2, **kwargs):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )
        self.K = K
        self.shift_size = kwargs['shift_size']

    def forward(self, x):
        B, C, H, W = x.shape
        if self.shift_size is not None:
            shift_size = self.shift_size
        else:
            shift_size = self.K // 2
        x = torch.roll(x, shifts=(-(shift_size), -(shift_size)), dims=(2, 3))
        x_j = torch.zeros_like(x).to(x.device)
        for i in torch.arange(self.K, H, self.K):
            x_c = x - torch.roll(x, shifts=(-i, 0), dims=(2, 3))
            x_j = torch.max(x_j, x_c)
        for i in torch.arange(self.K, W, self.K):
            x_r = x - torch.roll(x, shifts=(0, -i), dims=(2, 3))
            x_j = torch.max(x_j, x_r)

        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, shift_size, drop_path=0.0, K=2, **kwargs):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels, in_channels * 2, K=self.K, shift_size=shift_size)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp

        return x

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features  # same as input
        hidden_features = hidden_features or in_features  # x4
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class ViGBlock(torch.nn.Module):
    def __init__(self, dim, depth, shift_size, dropout=0.):
        super().__init__()
        self.backbone = nn.ModuleList([])
        for i in range(depth):
            self.backbone += [nn.Sequential(
                Grapher(dim, drop_path=dropout, K=4, shift_size=shift_size[i] if shift_size != None else None),
                FFN(dim, dim * 4, drop_path=dropout),
            )]

    def forward(self, x):
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        return x

class VViGBlock(nn.Module):
    def __init__(self, dim, depth, channel, patch_size, dropout=0.0):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv2 = nn.Conv2d(channel, dim, kernel_size=1, stride=1, padding=0, bias=False)


        self.transformer = ViGBlock(dim, depth, dropout=dropout, shift_size=None)

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU())

    def forward(self, x):
        y = x.clone()
        x = self.conv2(x)
        x = self.transformer(x)
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

def autopad(kernel_size):
    return (kernel_size - 1) // 2


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, autopad(kernel_size), bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, autopad(dw_size), groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class Encoder(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2):
        super().__init__()
        self.ghost1 = GhostModule(inp, int(inp * 2), kernel_size)
        self.convdw = nn.Conv2d(in_channels=int(inp * 2),
                                out_channels=int(inp * 2),
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=autopad(kernel_size),
                                groups=int(inp * 2))
        self.bn = nn.BatchNorm2d(int(inp * 2))
        self.ghost2 = GhostModule(int(inp * 2), oup, kernel_size, stride=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size, stride,
                      autopad(kernel_size), groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        x = self.convdw(x)
        x = self.bn(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden, oup, kernel_size=3):
        super().__init__()
        self.ghost = GhostModule(hidden, oup, kernel_size)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)  # 1,256,256
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.ghost(x1)
        return x1

class ViG(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.layer1 = Encoder(3, 128)
        self.layer2 = Encoder(128, 128)
        self.layer3 = Encoder(128, 256)
        self.vit = VViGBlock(96, 6, 256, (2, 2),  0.1)
        self.decode2 = Decoder(256 + 128, 128)
        self.decode1 = Decoder(128 + 128, 128)
        self.head = _FCNHead(128, n_class)
        self.apply(self.__init_weights)
        self.vis = False

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)

    def forward(self, input):
        _, _, h, w = input.size()
        e1 = self.layer1(input) 
        e2 = self.layer2(e1)  
        e3 = self.layer3(e2)  
        f = self.vit(e3)
        d2 = self.decode2(f, e2) 
        d1 = self.decode1(d2, e1)  
        out = F.interpolate(d1, size=[h, w], mode='bilinear', align_corners=True)  # 1,256,256
        out = self.head(out)
        return out if not self.vis else (e1, e2, e3, f, d2, d1, out)


if __name__ == "__main__":
    model = ViG(1)
    from torchsummary import summary

    summary(model, (3, 256, 256), device='cpu')
