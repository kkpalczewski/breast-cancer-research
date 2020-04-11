from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import pandas as pd

class BaseDataset(Dataset, ABC):
    @staticmethod
    def _load_from_metadata(i: int, label: str, root: str, metadata: pd.DataFrame) -> np.ndarray:

        img_file = metadata.iloc[i][label]
        img_path = os.path.join(root, img_file)
        img = cv2.imread(img_path, -cv2.IMREAD_ANYDEPTH)

        return img

    @staticmethod
    def preprocess(img: np.ndarray, new_shape, orientation: str = None):

        shape_size = len(img.shape)

        if shape_size == 2:
            h, w = img.shape
        else:
            raise AttributeError("Not implemented shape size: {}".format(shape_size))

        # check if initial format is the same
        if h != new_shape[0] or w != new_shape[1]:
            img = BaseDataset.preprocess_single_img(img, new_shape=new_shape, orientation=orientation)

        # TODO: change this
        img_resized = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img_nd = np.expand_dims(img_resized, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans

    @staticmethod
    def preprocess_single_img(img, *, new_shape, orientation: str = 'RIGHT'):
        processed_img = cv2.resize(img, new_shape)
        if orientation is not None and orientation == 'LEFT':
            processed_img = cv2.flip(processed_img, 1)

        return processed_img
