# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision.io import read_image

class BasicFoodsThree(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = read_image(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            y_label = self.target_transform(y_label)
        return image, y_label