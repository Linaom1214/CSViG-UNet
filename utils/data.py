from glob import glob
import torch.utils.data as Data
import torchvision.transforms as transforms
import albumentations as A

from PIL import Image
import os
import os.path as osp
import numpy as np
import cv2

__all__ = ['SirstAugDataset', 'SirstDataset', 'MDFADataset', 'MergedDataset', 'IRSTDDataset']


class MergedDataset(Data.Dataset):
    def __init__(self, mdfa_base_dir='./data/MDFA', sirstaug_base_dir='./data/sirst_aug', mode='train', base_size=256):
        assert mode in ['train', 'test']

        self.sirstaug = SirstAugDataset(base_dir=sirstaug_base_dir, mode=mode, base_size=base_size)
        self.mdfa = MDFADataset(base_dir=mdfa_base_dir,
                                mode=mode, base_size=base_size)

    def __getitem__(self, i):
        if i < self.mdfa.__len__():
            return self.mdfa.__getitem__(i)
        else:
            inx = i - self.mdfa.__len__()
            return self.sirstaug.__getitem__(inx)

    def __len__(self):
        return self.sirstaug.__len__() + self.mdfa.__len__()


class MDFADataset(Data.Dataset):
    def __init__(self, base_dir='./data/MDFA', mode='train', base_size=256):
        assert mode in ['train', 'test']
        self.base_size = base_size
        self.mode = mode
        if mode == 'train':
            self.paths = glob(osp.join(base_dir, 'training') + '/*.*')
        elif mode == 'test':
            self.paths = glob(osp.join(base_dir, 'test_org') + '/*.*')
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        if self.mode == 'train':
            path = self.paths[i]
            if '_1' in path:
                img_path = path
                mask_path = path.replace('_1', '_2')
            elif '_2' in path:
                mask_path = path
                img_path = path.replace('_1', '_2')
        elif self.mode == 'test':
            img_path = self.paths[i]
            mask_path = img_path.replace('test_org', 'test_gt')
        else:
            raise NotImplementedError

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        img, mask = img.resize((self.base_size, self.base_size), Image.BILINEAR), mask.resize(
            (self.base_size, self.base_size), Image.NEAREST)

        TF = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std for preTrain model
            ]
        )

        img, mask = TF(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        if self.mode == 'train':
            return len(self.paths) // 2
        elif self.mode == 'test':
            return len(self.paths)
        else:
            raise NotImplementedError


class SirstAugDataset(Data.Dataset):
    def __init__(self, base_dir='./data/sirst_aug', mode='train', base_size=256):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.mode = mode
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

        self.base_size = base_size

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')

        img, mask = img.resize((self.base_size, self.base_size), Image.BILINEAR), mask.resize(
            (self.base_size, self.base_size), Image.NEAREST)

        TF = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std for preTrain model
            ]
        )

        img, mask = TF(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)


class IRSTDDataset(Data.Dataset):
    def __init__(self, base_dir='./data/IRSTD-1k', mode='train', base_size=256):
        assert mode in ['train', 'test']

        if mode == 'train':
            with open(osp.join(base_dir, 'trainval.txt'), 'r') as f:
                self.names = f.readlines()
        elif mode == 'test':
            with open(osp.join(base_dir, 'test.txt'), 'r') as f:
                self.names = f.readlines()
        else:
            raise NotImplementedError
        self.mode = mode
        self.base_dir = base_dir
        self.base_size = base_size
        self.names = [name.strip() + '.png' for name in self.names]

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.base_dir, 'IRSTD1k_Img', name)
        label_path = osp.join(self.base_dir, 'IRSTD1k_Label', name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')

        img, mask = img.resize((self.base_size, self.base_size), Image.BILINEAR), mask.resize(
            (self.base_size, self.base_size), Image.NEAREST)

        TF = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std for preTrain model
            ]
        )

        img, mask = TF(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)


class SirstDataset(Data.Dataset):
    def __init__(self, base_dir='./data/sirst', mode='train', base_size=256):
        super(SirstDataset, self).__init__()
        assert mode in ['train', 'test']
        self.dataset_dir = base_dir
        if mode == 'train':
            self.image_index = open(f'{self.dataset_dir}/idx_320/train.txt').readlines()
        else:
            self.image_index = open(f'{self.dataset_dir}/idx_320/test.txt').readlines()
        self.mode = mode
        self.base_size = base_size

    def __len__(self):
        return len(self.image_index)

    def __getitem__(self, index):
        image_index = self.image_index[index].strip('\n')
        image_path = os.path.join(self.dataset_dir, 'images', '%s.png' % image_index)
        label_path = os.path.join(self.dataset_dir, 'masks', '%s_pixels0.png' % image_index)
        img = Image.open(image_path)
        mask = Image.open(label_path)

        img, mask = img.resize((self.base_size, self.base_size), Image.BILINEAR), mask.resize(
            (self.base_size, self.base_size), Image.NEAREST)
        TF = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std for preTrain model
            ]
        )

        img, mask = TF(img), transforms.ToTensor()(mask)
        return img, mask
