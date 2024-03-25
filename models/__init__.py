from models.network import *

def get_segmentation_model(name):
    if name == 'vig':
        net = get_vig()
    else:
        raise NotImplementedError

    return net


if __name__ == '__main__':
    net = get_segmentation_model('vig')
    from torchsummary import summary

    summary(net, (3, 256, 256), device='cpu')
