import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import PIL.Image as pil_image
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

class TrainDataset(Dataset):
    def __init__(self, image_dir, is_train=1, scale=4):
        super(TrainDataset, self).__init__()
        lr_dir = os.path.join(image_dir, "lr/")
        hr_dir = os.path.join(image_dir, "hr/")
        self.lr_list = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
        self.hr_list = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])
        cur_len = len(self.hr_list)
        train_len = round(cur_len*0.9)

        if is_train == 1:
            self.lr_list = self.lr_list[:train_len]
            self.hr_list = self.hr_list[:train_len]
        else:
            self.lr_list = self.lr_list[train_len:]
            self.hr_list = self.hr_list[train_len:]

        self.scale = scale
        self.crop_size = 33
        self.is_train = is_train

    def __getitem__(self, idx):
        image = pil_image.open(self.hr_list[idx]).convert('RGB')

        image_width = (image.width // self.scale) * self.scale
        image_height = (image.height // self.scale) * self.scale
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        image = image.resize((image.width // self.scale, image.height // self.scale), resample=pil_image.BICUBIC)
        image = image.resize((image.width * self.scale, image.height * self.scale), resample=pil_image.BICUBIC)
        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        lr = ycbcr[..., 0]
        lr /= 255.
        lr = torch.from_numpy(lr)
        h, w = lr.size()

        hr = pil_image.open(self.hr_list[idx]).convert('RGB')
        hr = np.array(hr).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(hr)

        hr = ycbcr[..., 0]
        hr /= 255.
        hr = torch.from_numpy(hr)

        # random crop
        if self.is_train:
            rand_h = torch.randint(h - (self.crop_size), [1, 1])
            rand_w = torch.randint(w - (self.crop_size), [1, 1])
            lr = lr[rand_h:rand_h + self.crop_size, rand_w:rand_w + self.crop_size]
            hr = hr[rand_h:rand_h + self.crop_size, rand_w:rand_w + self.crop_size]

        lr = lr.unsqueeze(0)
        hr = hr.unsqueeze(0)
         # lr hr pair
        return lr, hr

    def __len__(self):
       return len(self.hr_list)


class TestDataset(Dataset):
    def __init__(self, image_dir, scale=4):
        super(TestDataset, self).__init__()
        lr_dir = os.path.join(image_dir, "lr/")
        hr_dir = os.path.join(image_dir, "hr/")
        self.lr_list = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
        self.hr_list = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])

        self.scale = scale
        self.crop_size = 33

    def __getitem__(self, idx):
        image = pil_image.open(self.hr_list[idx]).convert('RGB')

        image_width = (image.width // self.scale) * self.scale
        image_height = (image.height // self.scale) * self.scale
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        image = image.resize((image.width // self.scale, image.height // self.scale), resample=pil_image.BICUBIC)
        image = image.resize((image.width * self.scale, image.height * self.scale), resample=pil_image.BICUBIC)
        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        lr = ycbcr[..., 0]
        lr /= 255.
        lr = torch.from_numpy(lr)
        h, w = lr.size()

        hr = pil_image.open(self.hr_list[idx]).convert('RGB')
        hr = np.array(hr).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(hr)

        hr = ycbcr[..., 0]
        hr /= 255.
        hr = torch.from_numpy(hr)

        lr = lr.unsqueeze(0)
        hr = hr.unsqueeze(0)

        return lr, hr

    def __len__(self):
       return len(self.hr_list)
