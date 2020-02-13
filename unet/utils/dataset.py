# ml
import numpy as np
import torch
from torch.utils.data import Dataset
# logging and typing
import logging
from typing import Optional, Mapping
# basic
from PIL import Image
import os
import pandas as pd

class BasicDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, root: str, sample: Optional[int] = None, scale: float = 0.05):
        self.metadata = metadata
        self.metadata = add_index(self.metadata)
        self.root = root
        self.scale = scale
        self.sample = sample

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [*self.metadata.index.values]
        if sample == None:
            sample = len(self.ids)
        self.ids = self.ids[0:sample]

        #infer size
        self.img_size_ = self.img_size

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @property
    def img_size(self):
        return self[0]['image'].numpy().shape

    @classmethod
    def preprocess(cls, pil_img, scale: float):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i: int) -> Mapping[str, torch.Tensor]:
        img = self._prepare_single_record(i, 'image file path')
        mask_malignant = self._prepare_single_record(i, 'ROI malignant path')
        mask_benign = self._prepare_single_record(i, 'ROI benign path')
        mask = np.concatenate([mask_benign, mask_malignant], axis=0)

        assert img.size == mask_benign.size == mask_malignant.size, \
            f'Image and mask {i} should be the same size, but are different: img size -> {img.size}, ' \
            f'benign size -> {mask_benign.size}, malignant size -> {mask_malignant.size}'

        training_records = {'image': torch.from_numpy(img),
                            'mask': torch.from_numpy(mask)}

        return training_records

    def _prepare_single_record(self, i: int, label: str) -> np.ndarray:
        """
        prepare single image for training
        """
        #TODO: Add thresholding for masks
        record_file = self.metadata.iloc[i][label]
        record_path = os.path.join(self.root, record_file)
        record = Image.open(record_path)
        processed_record = self.preprocess(record, self.scale)
        return processed_record

def add_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add index if not settled
    """
    if df.index.empty:
        index_columns = ['P_00004', 'MLO', 'LEFT']
        df = df.set_index(index_columns)
        logging.info("Added index on index_columns: {}".format(index_columns))
    return df