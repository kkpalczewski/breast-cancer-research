# ml
import numpy as np
import torch
# logging and typing
import logging
from typing import Optional, Mapping
# basic
import pandas as pd
import cv2
# custom
from breast_cancer_research.base.base_dataset import BaseDataset


class UnetDataset(BaseDataset):
    ORIGINAL_RATIO = 1.7
    ORIGINAL_HEIGHT = 2601

    def __init__(self, metadata: pd.DataFrame,
                 root: str,
                 sample: Optional[int] = None,
                 scale: float = 0.4,
                 training_transforms_name: Optional[str] = None):
        self.metadata = metadata
        self.root = root
        self.scale = scale
        self.sample = sample

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [*self.metadata.index.values]
        if sample is None:
            sample = len(self.ids)
        self.ids = self.ids[:sample]

        # get new img shape
        self.new_shape = self._get_new_shape()
        # transforms
        if training_transforms_name is None:
            transforms_name = "UnetEval"
        else:
            transforms_name = training_transforms_name

        self.transforms = BaseDataset.get_transforms(transform_name=transforms_name,
                                                     new_size=self.new_shape)

        logging.info(f'Creating dataset with {len(self.ids)} examples.\n'
                     f'Image height: {self.new_shape[1]}, '
                     f'width: {self.new_shape[0]}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int) -> Mapping[str, torch.Tensor]:
        img = self._load_from_metadata(i, 'image file path', self.root, self.metadata)

        img = img.astype(np.float32)
        img = (img.clip(10000, 50000) - 10000) / 40000

        #img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        mask_malignant = self._load_from_metadata(i, 'ROI malignant path', self.root, self.metadata)
        mask_benign = self._load_from_metadata(i, 'ROI benign path', self.root, self.metadata)
        mask_background = self._get_background(mask_benign, mask_malignant)

        mask_malignant = np.where(mask_malignant > 1, 1, 0).astype(img.dtype)
        mask_benign = np.where(mask_benign > 1, 1, 0).astype(img.dtype)
        mask_background = np.where(mask_background > 1, 1, 0).astype(img.dtype)

        assert img.shape == mask_benign.shape == mask_malignant.shape == mask_background.shape, \
            f'Image and mask {i} should be the same size, but are different: img size -> {img.shape}, ' \
            f'benign size -> {mask_benign.shape}, malignant size -> {mask_malignant.shape}'

        all_masks = np.array([mask_benign, mask_malignant, mask_background])

        transformed_img, transformed_mask = BaseDataset.transform_images_masks(img=img, masks=all_masks, transform=self.transforms)

        training_records = {'image': transformed_img,
                            'mask': transformed_mask,
                            'classname': ["benign", "malignant", "background"]}

        return training_records

    @property
    def img_size(self):
        return self[0]['image'].numpy().shape

    def _get_new_shape(self):
        new_shape = (int(UnetDataset.ORIGINAL_HEIGHT * self.scale),
                     int(UnetDataset.ORIGINAL_HEIGHT / UnetDataset.ORIGINAL_RATIO * self.scale))
        assert new_shape[0] > 100 and new_shape[1] > 100, f"Desired shape of an image: {new_shape} is too small."
        return new_shape

    @classmethod
    def _get_background(cls, mask_benign: np.ndarray, mask_malignant: np.ndarray) -> np.ndarray:
        masks_sum = np.bitwise_or(mask_benign, mask_malignant)
        mask_background = np.bitwise_not(masks_sum)

        return mask_background
