# ml
import numpy as np
import torch
from torch.utils.data import Dataset
# logging and typing
import logging
from typing import Optional, Mapping, Dict, Tuple, Any
# basic
import os
import pandas as pd
import cv2
from breast_cancer_research.utils.utils import hwc_2_chw
from torchvision import transforms
from breast_cancer_research.base.base_dataset import BaseDataset
from breast_cancer_research.unet.unet_model import UNet
from PIL import Image


class ResnetDataset(BaseDataset):
    def __init__(self, metadata: pd.DataFrame, root_img: str, root_mask: str, classes: Tuple[str], sample: Optional[int] = None,
                 unet_config: Optional[Dict] = None, multi_target: bool = False):
        self.metadata = metadata
        self.root_img = root_img
        self.root_mask = root_mask
        self.sample = sample
        self.ids = [*self.metadata.index.values]
        self.multi_target = multi_target
        self.classes = classes

        if sample is None:
            sample = len(self.ids)

        self.ids = self.ids[:sample]
        self.transforms = self._get_transforms()

        if unet_config is not None:
            self.unet = self._init_unet(**unet_config)
        else:
            self.unet = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self._load_from_metadata(i, 'image file path', self.root_img, self.metadata)
        img = self._uint16_2_uint8(img)

        if self.unet is None:
            mask_malignant = self._load_from_metadata(i, 'ROI malignant path', self.root_mask, self.metadata).astype(
                np.uint8)
            mask_benign = self._load_from_metadata(i, 'ROI benign path', self.root_mask, self.metadata).astype(np.uint8)

            assert img.size == mask_benign.size == mask_malignant.size, \
                f'Image and mask {i} should be the same size, but are different: img size -> {img.size}, ' \
                f'benign size -> {mask_benign.size}, malignant size -> {mask_malignant.size}'

            true_class_benign = self.metadata.iloc[i]['benign_findings']
            true_class_malignant = self.metadata.iloc[i]['malignant_findings']
            targets = [true_class_benign, true_class_malignant]

            input_img = np.stack([img, mask_benign, mask_malignant], axis=2)
        else:
            raise NotImplementedError

        if self.multi_target is False:
            targets = np.argmax(targets)
            targets = torch.tensor(targets).long()
        else:
            targets = torch.from_numpy(np.array(targets)).long()

        input_img = self.transforms(input_img)
        targets = torch.from_numpy(np.array(targets)).long()

        return input_img, targets

    def _uint16_2_uint8(self, image):
        return np.array(image.clip(10000, 50000) / (40000 / 255)).astype(np.uint8)

    def _get_transforms(self):
        data_transforms = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize((224, 224), Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return data_transforms

    def _init_unet(self, unet_params):
        unet = UNet(**unet_params)
        return unet
