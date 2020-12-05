import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)


        if self.training:  
            w, h = left_img.size

            # @ywt image size must be multiple of 32 or 16, according to github issues.
            # KITTI image size: [375, 1242], SCARED image size: [1016, 1231/1232]([height, width])
            # so I set the size to 512, 512, update: due to the GPU memory, I set the size to 256, 512
            # th, tw = 256, 512
            th, tw = 256, 512


            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img   = processed(left_img)
            right_img  = processed(right_img)

            return left_img, right_img, dataL
        else:
            w, h = left_img.size
            # @ywt image size must be multiple of 32 or 16, according to github issues.
            # KITTI image size: [375, 1242], SCARED image size: [1016, 1231/1232]([height, width]
            # so I set the crop size to
            # crop_size = w - 1232, h - 368, w, h
            crop_size = w - 1216, h - 1008, w, h

            left_img = left_img.crop(crop_size)
            right_img = right_img.crop(crop_size)
            w1, h1 = left_img.size

            dataL = dataL.crop(crop_size)
            dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
