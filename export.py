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
    parser = ArgumentParser(description='Inference of AGPCNet')

    #
    # Checkpoint parameters
    #
    parser.add_argument('--pkl-path', type=str, default=r'./results/mdfa_mIoU-0.4843_fmeasure-0.6525.pkl',
                        help='checkpoint path')
    parser.add_argument('--net-name', type=str, default='transformer',
                        help='net name: fcn')
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')
    args = parser.parse_args()
    return args



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

    dummy_input = torch.randn(1, 3, args.base_size, args.base_size).cuda()
    output_name = args.pkl_path.replace('pkl', 'onnx')
    torch.onnx._export(
        net,
        dummy_input,
        output_name,
        input_names=['images'],
        output_names=['bboxes'],
        opset_version=12,
    )
    import onnx
    from onnxsim import simplify
    # use onnxsimplify to reduce reduent model.
    onnx_model = onnx.load(output_name)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_name)