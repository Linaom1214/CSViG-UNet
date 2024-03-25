import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange
import math
from timm.models.layers import trunc_normal_
from models.head import _FCNHead
from mobile_sam import sam_model_registry


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding"""
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type

    K is the number of superpatches, therefore hops equals res // K.
    """

    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.roll(x, shifts=(-(self.K // 2), -(self.K // 2)), dims=(2, 3))
        x_j = torch.zeros_like(x).to(x.device)
        for i in torch.arange(self.K, H, self.K):
            x_c = x - torch.roll(x, shifts=(-i, 0), dims=(2, 3))
            x_j = torch.max(x_j, x_c)
        for i in torch.arange(self.K, W, self.K):
            x_r = x - torch.roll(x, shifts=(0, -i), dims=(2, 3))
            x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        x = torch.roll(x, shifts=(self.K // 2, self.K // 2), dims=(2, 3))
        return self.nn(x)

class MRConv4dTF(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type

    K is the number of superpatches, therefore hops equals res // K.
    """

    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4dTF, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape
        x_j = torch.zeros_like(x).to(x.device)
        
        q = rearrange(x, 'b c h w -> b (h w) c')
        k = q.clone()
        v = q.clone()
        att = torch.matmul(q, k.transpose(-2, -1))
        att = F.softmax(att, dim=-1)
        att = torch.matmul(att, v)
        att = rearrange(att, 'b (h w) c -> b c h w', h=H, w=W)

        x_r = x - att
        x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)

class MRConv4dSP(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type

    K is the number of superpatches, therefore hops equals res // K.
    """

    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )

        self.d  = in_channels ** -0.5

    def forward(self, x):
        x_j = torch.zeros_like(x).to(x.device)
        x_j[:, :, ::4, ::4] = 1
        select = x[:, :, ::4, ::4]
        B, C, H, W = select.shape
        q = rearrange(select, 'b c h w -> b (h w) c')
        k = q.clone()
        v = q.clone()
        att = torch.matmul(q, k.transpose(-2, -1))* self.d
        att = F.softmax(att, dim=-1)
        att = torch.matmul(att, v)
        att = rearrange(att, 'b (h w) c -> b c h w', h=H, w=W)
        att = F.interpolate(att, x_j.shape[2:], mode='nearest')
        att = att * x_j
        x = torch.cat([x, att], dim=1)
        return self.nn(x)


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, drop_path=0.0, K=2, mode='graph_vig'):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        
        if mode == 'mobile_vig':
            self.graph_conv = MRConv4d(in_channels, in_channels * 2, K=self.K)
        elif mode == 'graph_vig':
            self.graph_conv = MRConv4dSP(in_channels, in_channels * 2, K=self.K)

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, stride=1, padding=1),
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
            nn.ReLU(),
            nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        x = self.convdw(x)
        x = self.bn(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class Encoder3(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2, graph=True):
        super().__init__()
        if graph:
            self.conv = Grapher(inp, drop_path=0.0, mode='graph_vig')
        else:
            self.conv = nn.Conv2d(inp, inp, kernel_size, stride=1, padding=autopad(kernel_size), bias=False)
        self.bn = nn.BatchNorm2d(inp)
        self.convdw = nn.Conv2d(in_channels=inp,
                                out_channels=oup,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=autopad(kernel_size))
        
        self.dwbn = nn.BatchNorm2d(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.convdw(x)
        x = self.dwbn(x) #
        x = F.relu(x)    #
        return x

class Decoder3(nn.Module):
    def __init__(self, hidden, oup, kernel_size=3, graph=False):
        super().__init__()
        if graph:
            self.ghost = Grapher(hidden)
        else:
            self.ghost = nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False)
        self.conv = nn.Conv2d(hidden, oup, 1, 1, 0, bias=False)
        self.BN = nn.BatchNorm2d(oup)
        self.act = nn.ReLU()

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)  # 1,256,256
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.ghost(x1)
        x1 = self.conv(x1)
        x1 = self.BN(x1)
        x1 = self.act(x1)
        return x1

class ViG(nn.Module):
    def __init__(self, n_class, type="torch"):
        super().__init__()
        self.topk = 4
        self.layer1 = Encoder(3, 128)
        self.layer2 = Encoder3(128, 128, graph=True)
        self.layer3 = Encoder3(128, 256, graph=True)
        self.layer4 = Encoder3(256, 256, graph=True)
        self.decode3 = Decoder3(256 + 256, 256)
        self.decode2 = Decoder3(128 + 256, 128)
        self.decode1 = Decoder3(128 + 128, 128)
        self.head = _FCNHead(128, n_class) 

        model_type = "vit_t"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        mobile_sam = sam_model_registry[model_type](checkpoint="mobile_sam.pt")
        mobile_sam.to(device=device)
        
        self.model = mobile_sam.image_encoder

        for param in self.model.parameters():
            param.requires_grad = False

        self.type = type

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)
    
    def sam(self, input):
        h, w = input.shape[-2:]
        padh = self.model.img_size - h
        padw = self.model.img_size - w
        img = F.pad(input, (0, padw, 0, padh))

        if self.type == "torch":
            masks = self.model(img)
        return masks

    def forward(self, input):
        img = input.detach()  # B C  H W
        masks = self.sam(img)
        _, _, h, w = input.size()
        e1 = self.layer1(input) 
        e2 = self.layer2(e1)  
        e3 = self.layer3(e2)  
        e4 = self.layer4(e3) 

        mask1 = F.interpolate(masks, size=e4.shape[2:], mode='bilinear', align_corners=True)

        e4 = e4 + self.cross_attention(e4, mask1[:e4.shape[0], ...])

        d3 = self.decode3(e4, e3) 

        d2 = self.decode2(d3, e2)  

        d1 = self.decode1(d2, e1)  

        out = F.interpolate(d1, size=[h, w], mode='bilinear', align_corners=True) 
        out = self.head(out)
        return out

    def cross_attention(self, q, k): # k , q
        batch_size, channels, height, width = q.data.size()
        num = channels // k.shape[1]
        q_ = q.view(batch_size, num, k.shape[1], height, width)
        
        """
        交叉注意力机制
        """
        res = []
        for i in range(num):
            q = q_[:, i, ...]
            bs, c, h, w = q.shape
            q = F.adaptive_avg_pool2d(q, (1, 1)).view(bs, c, -1)  # bs, c
            k_ = F.adaptive_avg_pool2d(k, (1, 1)).view(bs, c, -1)  # bs, c
            qk = torch.matmul(q, k_.permute(0, 2, 1)) * c**(-0.5)  # bs, c, c1
            qk = torch.matmul(qk, k_)  # bs, c, h*w 
            qk = qk.view(bs, c, 1, 1)
            qk = F.interpolate(qk, size=[h, w], mode='bilinear', align_corners=True)  # 1,256,256
            res.append(qk)
        qk = torch.cat(res, dim=1)
        qk = F.gelu(qk)
        return qk
    
if __name__ == "__main__":
    model = ViG(1)
    from torchsummary import summary
    
    summary(model, (3, 256, 256), device='cpu')
