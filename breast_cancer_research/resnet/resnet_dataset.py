# ml
import numpy as np
import torch
from torch.utils.data import Dataset
# logging and typing
import logging
from typing import Optional, Mapping
# basic
import os
import pandas as pd
import cv2
from breast_cancer_research.utils.utils import hwc_2_chw
from torchvision import transforms

class ResnetDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, i: int) -> Mapping[str, torch.Tensor]:
        pass


class CifarDataset(Dataset):
    def __init__(self, dataset: dict, scale: int = 10, sample: Optional[int] = None):
        self.labels = dataset[b'labels']
        self.data = dataset[b'data']
        self.filenames = dataset[b'filenames']

        if sample is not None:
            assert sample <= len(self.labels), f"Sample size: {sample} is higher than size of dataset: {len(self.labels)}"
            self.labels = self.labels[0:sample]
            self.data = self.data[0:sample]
            self.filenames = self.filenames[0:sample]

        self.scale = scale
        self.transforms = self._get_transforms()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw_img = self.data[idx]
        img = np.rot90(np.reshape(raw_img, (32, 32, 3), order='F'), -1)
        img = cv2.resize(img, (int(32*self.scale), int(32*self.scale)))
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = self.transforms(img)
        label = self.labels[idx]

        return img, label

    def _get_transforms(self):
        data_transforms = transforms.Compose([
                transforms.ToPILImage(mode='RGB'),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return data_transforms