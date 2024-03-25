import torch
from models import get_segmentation_model
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import cv2
import os
import time

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Inference of CSViG')

    #
    # Checkpoint parameters
    #
    parser.add_argument('--pkl-path', type=str, default=r'result/sirstaug/2023-06-02_11-19-13_transformer3/checkpoint/Iter-11720_mIoU-0.7159_fmeasure-0.8345.pkl',
                        help='checkpoint path') 
    parser.add_argument('--net-name', type=str, default='transformer3',
                        help='net name: fcn')
    #
    # Test image parameters
    #
    parser.add_argument('--image-path', type=str, default=r'data/IRSTD-1k/IRSTD1k_Img/XDU953.png', help='image path')
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')

    args = parser.parse_args()
    return args


def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img).unsqueeze(0)

    return preprocessed_img.to(device)


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load network
    print('...load checkpoint: %s' % args.pkl_path)
    net = get_segmentation_model(args.net_name)
    ckpt = torch.load(args.pkl_path, map_location=torch.device('cpu'))
    net.load_state_dict(ckpt)
    net.to(device)
    net.eval()
    net.vis = True
    # load image
    print('...loading test image: %s' % args.image_path)
    img = cv2.imread(args.image_path)
    img = np.float32(cv2.resize(img, (args.base_size, args.base_size))) / 255
    input = preprocess_image(img)

    out = net(input)
    for i, feat in enumerate(out):
        out = torch.max(feat, dim=1, keepdim=True)
        img = cv2.imread(args.image_path)
        img = cv2.resize(img, (args.base_size, args.base_size))
        result = overlay_mask(to_pil_image(input.squeeze(0).detach()), to_pil_image(out[0].squeeze(0).detach(), mode='F'), alpha=0.5)
        dir = os.path.splitext(os.path.basename(args.image_path))[0]
        os.makedirs(dir, exist_ok=True)
        result.save(f'{dir}/{i}.png')



