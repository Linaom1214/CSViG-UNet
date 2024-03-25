import torch
from models import get_segmentation_model
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import cv2
import os
import time

def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Inference of CSViG')

    #
    # Checkpoint parameters
    #
    parser.add_argument('--pkl-path', type=str, default=r'./results/mdfa_mIoU-0.4843_fmeasure-0.6525.pkl',
                        help='checkpoint path')
    parser.add_argument('--net-name', type=str, default='transformer',
                        help='net name: fcn')
    #
    # Test image parameters
    #
    parser.add_argument('--image-path', type=str, default=r'./data/1.bmp', help='image path')
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

    # load image
    print('...loading test image: %s' % args.image_path)
    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (args.base_size, args.base_size))) / 255
    input = preprocess_image(img)

    # warm up
    for _ in range(5):
        data = torch.ones((2, 3, args.base_size, args.base_size)).to(device)
        net(data)

    # inference in cpu
    print('...inference in progress')
    t1 = time.time()
    with torch.no_grad():
        output = net(input)
    t2 = time.time()
    print("FPS : ", round(1/(t2-t1), 3))
    output = output.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output = output > 0

    # show results
    plt.figure()
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(output, cmap='gray'), plt.title('Inference Result')
    plt.savefig(f'{os.path.splitext(os.path.basename(args.image_path))[0]}_infer.jpg')