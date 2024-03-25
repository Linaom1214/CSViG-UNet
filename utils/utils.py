import cv2
import numpy as np
import torch
import random 
import torch.nn.functional as F

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

def Batch_Augmentation1 (img, mask): 
    
    Random_coefficient = random.randint(0, img.shape[0]-1) 
        
    img_s = []
    mask_s = []
        
    for i in range(img.shape[0]):

        img_s.append(img[i:i+1])
        mask_s.append(mask[i:i+1])

    data_aug = []
    label_aug = []
        
    for i in range(img.shape[0]):
            
        if Random_coefficient > img.shape[0]//2-1:
        
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),3))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),3))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),3))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),3))
                    
        else:
                
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),2))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),2))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),2))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),2))

    data = torch.cat(data_aug, dim=0)
    label = torch.cat(label_aug, dim=0)

    img = F.interpolate(data, size=[256, 256]) 
    mask = F.interpolate(label, size=[256, 256]) 
        
    # data = torch.cat((img,data),0)
    # label = torch.cat((mask,label),0)

    return img, mask
        
        
def Batch_Augmentation2 (img, mask, img_h=256, img_w=256): 
    
    Random_coefficient = random.randint(0, img.shape[0]-1) 
        
    img_s = []
    mask_s = []
        
    for i in range(img.shape[0]):

        img_s.append(img[i:i+1])
        mask_s.append(mask[i:i+1])

    data_aug = []
    label_aug = []
        
    for i in range(img.shape[0]):
            
        if Random_coefficient > img.shape[0]//2-1:
        
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),3))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),3))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),3))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),3))
                    
        else:
                
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),2))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),2))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),2))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),2))

    data = torch.cat(data_aug, dim=0)
    label = torch.cat(label_aug, dim=0)

    data = F.interpolate(data, size=[img_h, img_w]) 
    label = F.interpolate(label, size=[img_h, img_w]) 
        
    img = torch.cat((img,data),0)
    mask = torch.cat((mask,label),0)

    return img, mask