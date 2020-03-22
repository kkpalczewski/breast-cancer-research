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


class UnetDataset(Dataset):
    ORIGINAL_RATIO = 1.7
    ORIGINAL_HEIGHT = 2601

    def __init__(self, metadata: pd.DataFrame, root: str, sample: Optional[int] = None, scale: float = 0.05):
        self.metadata = metadata
        self.root = root
        self.scale = scale
        self.sample = sample

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [*self.metadata.index.values]
        if sample == None:
            sample = len(self.ids)
        self.ids = self.ids[0:sample]

        # infer size
        self.img_size_ = self.img_size

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int) -> Mapping[str, torch.Tensor]:
        img = self._load_from_metadata(i, 'image file path')
        mask_malignant = self._load_from_metadata(i, 'ROI malignant path')
        mask_benign = self._load_from_metadata(i, 'ROI benign path')
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

    @classmethod
    def preprocess(cls, img: np.ndarray, scale: float, orientation: str):

        shape_size = len(img.shape)

        if shape_size == 2:
            h, w = img.shape
        else:
            raise AttributeError("Not implemented shape size: {}".format(shape_size))

        # check if initial format is the same
        if h != UnetDataset.ORIGINAL_HEIGHT or w != int(UnetDataset.ORIGINAL_HEIGHT * UnetDataset.ORIGINAL_RATIO):
            img = preprocess_single_img(img, orientation=orientation)

        new_h, new_w = int(scale * h), int(scale * w)
        assert new_h > 0 and new_w > 0, 'Scale is too small'

        img_resized = cv2.resize(img, (new_w, new_h))

        img_resized = cv2.normalize(img_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img_nd = np.expand_dims(img_resized, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans


    def _load_from_metadata(self, i: int, label: str) -> np.ndarray:

        img_file = self.metadata.iloc[i][label]
        img_path = os.path.join(self.root, img_file)
        img = cv2.imread(img_path, -cv2.IMREAD_ANYDEPTH)

        return img

    def _prepare_single_image(self, i: int, image: np.ndarray) -> np.ndarray:
        """
        prepare single image for training
        """
        # TODO: Add thresholding for masks
        # TODO: WYciągnij img_name przed tą funkcję

        orientation = self.metadata.iloc[i]['left or right breast']
        processed_record = self.preprocess(image, self.scale, orientation)
        return processed_record

    @classmethod
    def _get_background(cls, mask_benign: np.ndarray, mask_malignant: np.ndarray) -> np.ndarray:
        masks_sum = np.bitwise_or(mask_benign, mask_malignant)
        mask_background = np.bitwise_not(masks_sum)

        return mask_background

def preprocess_single_img(img, *, ratio: float = 1.7, height: int = 2601, orientation: str = 'RIGHT'):
    processed_img = cv2.resize(img, (int(height / ratio), height))
    if orientation == 'LEFT':
        processed_img = cv2.flip(processed_img, 1)

    return processed_img
