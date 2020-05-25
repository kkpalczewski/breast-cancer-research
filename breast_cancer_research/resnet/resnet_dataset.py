# ml
import numpy as np
import torch
# logging and typing
import logging
from typing import Optional, Dict, Tuple
# basic
import pandas as pd
from torchvision import transforms
from breast_cancer_research.base.base_dataset import BaseDataset
from breast_cancer_research.unet.unet_facade import BreastCancerSegmentator
from breast_cancer_research.base.base_model import BaseModel
from PIL import Image


class ResnetDataset(BaseDataset):
    def __init__(self,
                 metadata: pd.DataFrame,
                 root_img: str,
                 classes: Tuple[str],
                 input_masks: bool = True,
                 root_mask: Optional[str] = None,
                 sample: Optional[int] = None,
                 unet_config: Optional[Dict] = None,
                 unet_out_layer_name: Optional[str] = None,
                 unet_new_shape: Tuple[int] = (1040, 612),
                 unet_transforms_name: Optional[str] = None,
                 multi_target: bool = False,
                 training_transforms_name: Optional[str] = None):

        assert root_mask is not None or unet_config is not None, f"Mask generation not specified. root_mask or unet_config has to be specified. root_mask is None and unet_config is also None."

        self.metadata = metadata
        self.root_img = root_img
        self.root_mask = root_mask
        self.sample = sample
        self.ids = [*self.metadata.index.values]
        self.multi_target = multi_target
        self.classes = classes
        self.input_masks = input_masks

        if sample is None:
            sample = len(self.ids)

        self.ids = self.ids[:sample]
        if training_transforms_name is None:
            transforms_name = "ResnetEval"
        else:
            transforms_name = training_transforms_name

        self.transforms = BaseDataset.get_transforms(transforms_name)

        if unet_config is not None and input_masks is True:
            self.unet = self._init_unet(unet_config)

            if unet_transforms_name is None:
                unet_transforms_name = "UnetEval"

            self.unet_transforms = BaseDataset.get_transforms(unet_transforms_name)
            self.unet_new_shape = unet_new_shape
            self.unet_out_layer = BaseModel.get_out_layer(unet_out_layer_name)
            logging.info("Initialized UNet in resnet dataset ...")
        else:
            self.unet = None
            self.unet_transforms = None
            self.unet_new_shape = None
            self.unet_out_layer = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self._load_from_metadata(i, 'image file path', self.root_img, self.metadata)
        if self.input_masks is True:
            if self.unet is None:
                img = img.astype(np.float32)
                img = (img.clip(10000, 50000) - 10000) / 40000
                mask_malignant = (self._load_from_metadata(i, 'ROI malignant path', self.root_mask, self.metadata)/255).astype(
                    img.dtype)
                mask_benign = (self._load_from_metadata(i, 'ROI benign path', self.root_mask, self.metadata)/255).astype(img.dtype)
                assert img.size == mask_benign.size == mask_malignant.size, \
                    f'Image and mask {i} should be the same size, but are different: img size -> {img.size}, ' \
                    f'benign size -> {mask_benign.size}, malignant size -> {mask_malignant.size}'

                input_img = np.stack([img, mask_benign, mask_malignant], axis=2)
            else:
                img = img.astype(np.float32)
                img = (img.clip(10000, 50000) - 10000) / 40000
                img, _ = BaseDataset.transform_images_masks(img=img, transform=self.unet_transforms)
                img = img.view([1] + list(img.shape))
                with torch.no_grad():
                    imgs_tensor = img.to(device=self.unet.device, dtype=torch.float32)
                    pred_masks = self.unet.model(imgs_tensor)
                    pred_masks = self.unet_out_layer(pred_masks)

                input_img = torch.cat([imgs_tensor, pred_masks[:, 0:2]], dim=1).cpu()[0].numpy()
        else:
            img = img.astype(np.float32)
            img = (img.clip(10000, 50000) - 10000) / 40000
            dummy_mask_malignant = np.zeros(img.shape).astype(img.dtype)
            dummy_mask_benign = np.zeros(img.shape).astype(img.dtype)
            input_img = np.stack([img, dummy_mask_benign, dummy_mask_malignant], axis=2)

        true_class_benign = self.metadata.iloc[i]['benign_findings']
        true_class_malignant = self.metadata.iloc[i]['malignant_findings']
        targets = [true_class_benign, true_class_malignant]

        input_img, _ = BaseDataset.transform_images_masks(img=input_img,
                                                          transform=self.transforms)
        targets = torch.from_numpy(np.array(targets)).long()

        return input_img, targets

    def _uint16_2_uint8(self, image):
        return np.array(image.clip(10000, 50000) / (40000 / 255)).astype(np.uint8)

    def _init_unet(self, unet_params):
        unet = BreastCancerSegmentator(**unet_params)
        return unet
