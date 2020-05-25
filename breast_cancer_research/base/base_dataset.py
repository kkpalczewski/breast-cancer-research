from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import os
import cv2
import pandas as pd
from typing import Tuple, Union, List, Optional
import torch
import albumentations as A
from breast_cancer_research.utils.utils import chw_2_hwc, hwc_2_chw
import numpy as np


class BaseDataset(Dataset, ABC):
    @staticmethod
    def get_transforms(transform_name: str,
                       new_size: Tuple = (1040, 612),
                       center_crop_factor: float = 0.9):
        if transform_name == "UnetEval":
            full_transforms = A.Compose([
                A.Resize(height=new_size[0], width=new_size[1])])
        elif transform_name == "UnetFlipCropResize":
            full_transforms = A.Compose([
                A.HorizontalFlip(),

                A.RandomBrightnessContrast(brightness_limit=0.1,
                                           contrast_limit=0.1,
                                           brightness_by_max=False,
                                           p=0.25),
                A.Resize(height=new_size[0], width=new_size[1]),
                A.CenterCrop(height=int(new_size[0] * center_crop_factor), width=int(new_size[1] * center_crop_factor),
                             p=0.25),
                A.PadIfNeeded(min_height=new_size[0], min_width=new_size[1], border_mode=cv2.BORDER_CONSTANT,
                              value=np.random.randint(2), p=0.25),
                A.Resize(height=new_size[0], width=new_size[1])])
        elif transform_name == "ResnetFlipCropResize":
            full_transforms = A.Compose([
                A.HorizontalFlip(),
                A.Resize(new_size[0], new_size[1]),
                A.CenterCrop(height=int(new_size[0] * center_crop_factor), width=int(new_size[1] * center_crop_factor),
                             p=0.25),
                A.PadIfNeeded(min_height=new_size[0], min_width=new_size[1], border_mode=cv2.BORDER_CONSTANT,
                              value=np.random.randint(2), p=0.25),
                A.Resize(new_size[0], new_size[1]),
                # A.Normalize(mean=[0.485, 0.456, 0.406],
                #             std=[0.229, 0.224, 0.225])
            ])
        elif transform_name == "ResnetEval":
            full_transforms = A.Compose([
                A.Resize(new_size[0], new_size[1]),
                # A.Normalize(mean=[0.485, 0.456, 0.406],
                #             std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Transform {transform_name} not implemented")

        return full_transforms

    @staticmethod
    def transform_images_masks(img: np.array, transform: A.Compose, masks: Optional[Union[np.array, List]] = None):

        if len(img.shape) == 2:
            img = np.reshape(img, list(img.shape) + [1])
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = chw_2_hwc(img)

        if masks is not None:
            masks = chw_2_hwc(masks)
            augmented = transform(image=img, mask=masks)
            transformed_masks = torch.from_numpy(hwc_2_chw(augmented["mask"]))
        else:
            augmented = transform(image=img)
            transformed_masks = None

        transformed_img = torch.from_numpy(hwc_2_chw(augmented["image"]))

        return transformed_img, transformed_masks

    @staticmethod
    def _load_from_metadata(i: int, label: str, root: str, metadata: pd.DataFrame) -> np.ndarray:

        img_file = metadata.iloc[i][label]
        img_path = os.path.join(root, img_file)
        img = cv2.imread(img_path, -cv2.IMREAD_ANYDEPTH)

        return img

    @staticmethod
    def preprocess_img(img: np.ndarray, new_shape):
        img = img.clip(10000, 50000)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img = BaseDataset.reshape_resize(img, new_shape)

        return img

    @staticmethod
    def preprocess_mask(mask: np.ndarray, new_shape):
        mask = BaseDataset.reshape_resize(mask, new_shape)
        return mask

    @staticmethod
    def reshape_resize(img: np.array, new_shape):
        shape_size = len(img.shape)
        if shape_size == 2:
            h, w = img.shape
        else:
            raise AttributeError("Not implemented shape size: {}".format(shape_size))

        # check if initial format is the same
        if h != new_shape[0] or w != new_shape[1]:
            img = BaseDataset.preprocess_single_img(img, new_shape=new_shape)

        img_nd = np.expand_dims(img, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans

    @staticmethod
    def preprocess_single_img(img, *, new_shape, ):
        processed_img = cv2.resize(img, new_shape)

        return processed_img
