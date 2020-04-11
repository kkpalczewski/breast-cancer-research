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
#custom
from breast_cancer_research.base.base_dataset import BaseDataset

class UnetDataset(BaseDataset):
    ORIGINAL_RATIO = 1.7
    ORIGINAL_HEIGHT = 2601

    def __init__(self, metadata: pd.DataFrame, root: str, sample: Optional[int] = None, scale: float = 0.05):
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

        logging.info(f'Creating dataset with {len(self.ids)} examples.\n'
                     f'Image height: {self.new_shape[1]}, '
                     f'width: {self.new_shape[0]}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int) -> Mapping[str, torch.Tensor]:
        img = self._load_from_metadata(i, 'image file path', self.root, self.metadata)
        mask_malignant = self._load_from_metadata(i, 'ROI malignant path', self.root, self.metadata)
        mask_benign = self._load_from_metadata(i, 'ROI benign path', self.root, self.metadata)
        mask_background = self._get_background(mask_benign, mask_malignant)

        img = self._prepare_single_image(i, img)
        mask_malignant = self._prepare_single_image(i, mask_malignant)
        mask_benign = self._prepare_single_image(i, mask_benign)
        mask_background = self._prepare_single_image(i, mask_background)

        mask = np.concatenate([mask_benign, mask_malignant, mask_background], axis=0)

        assert img.size == mask_benign.size == mask_malignant.size == mask_background.size, \
            f'Image and mask {i} should be the same size, but are different: img size -> {img.size}, ' \
            f'benign size -> {mask_benign.size}, malignant size -> {mask_malignant.size}'

        training_records = {'image': torch.from_numpy(img),
                            'mask': torch.from_numpy(mask)}

        return training_records

    @property
    def img_size(self):
        return self[0]['image'].numpy().shape

    def _prepare_single_image(self, i: int, image: np.ndarray) -> np.ndarray:
        """
        prepare single image for training
        """
        # TODO: Add thresholding for masks

        orientation = self.metadata.iloc[i]['left or right breast']
        processed_record = self.preprocess(image, self.new_shape, orientation)
        return processed_record

    def _get_new_shape(self):
        new_shape = (int(UnetDataset.ORIGINAL_HEIGHT / UnetDataset.ORIGINAL_RATIO * self.scale),
                     int(UnetDataset.ORIGINAL_HEIGHT * self.scale))
        assert new_shape[0] > 100 and new_shape[1] > 100, f"Desired shape of an image: {new_shape} is too small."
        return new_shape

    @classmethod
    def _get_background(cls, mask_benign: np.ndarray, mask_malignant: np.ndarray) -> np.ndarray:
        masks_sum = np.bitwise_or(mask_benign, mask_malignant)
        mask_background = np.bitwise_not(masks_sum)

        return mask_background
