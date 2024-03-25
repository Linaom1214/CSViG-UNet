from models.vig import ViG

def get_vig(num_class=1):
    return ViG(num_class)


if __name__ == '__main__':
    model = get_vig(1)
    from torchsummary import summary

    summary(model, (3, 256, 256), device='cpu')
