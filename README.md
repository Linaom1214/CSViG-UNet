# CS-ViG-UNet: Enhancing Infrared Small and Dim Target Detection with Vision Graph Convolution [ESWA]
- Our paper is published on [ESWA](https://www.sciencedirect.com/science/article/abs/pii/S095741742401251X).
- [Project](https://linaom1214.github.io/CSViG-UNet/)
- [Code](https://github.com/Linaom1214/CSViG-UNet/tree/code)
# Cite
```text
@article{LIN2024124385,
title = {CS-ViG-UNet: Infrared small and dim target detection based on cycle shift vision graph convolution network},
journal = {Expert Systems with Applications},
volume = {254},
pages = {124385},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.124385},
url = {https://www.sciencedirect.com/science/article/pii/S095741742401251X},
author = {Jian Lin and Shaoyi Li and Xi Yang and Saisai Niu and Binbin Yan and Zhongjie Meng},
keywords = {Infrared small and dim target, Vision graph network, Infrared target detection, U-shape architecture},
abstract = {Infrared small and dim target detection benefits from the exploration of correlations among targets, neighboring regions, and the background. However, existing methods that rely on convolutional neural networks and vision transformers cannot effectively capture long-range information correlations within images. To overcome this limitation, this paper proposes CS-ViG-UNet, a framework that introduces vision graph convolution for infrared small and dim target detection. Our framework employs a cyclic shift sparse graph attention mechanism to address the issue of reduced expressive power. Meanwhile, the CS-ViG module is designed to construct an effective graph structure using image patches, thereby capturing feature information relevant to target recognition. On the public datasets Sirst AUG and IRSTD-1K, our method obtained F1 scores of 0.8561 and 0.745, respectively, showing an improvement of 3.15 % and 4.1 % compared to the state-of-the-art methods. On the RTX3090 with TensorRT acceleration, CS-ViG-UNet can process approximately 357 images of size 256 × 256 pixels per second at FP16 precision. For detailed information, please visit our homepage: https://linaom1214.github.io/CSViG-UNet.}
}
```
