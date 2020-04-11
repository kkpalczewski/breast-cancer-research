from breast_cancer_research.base.base_summary_logger import SummaryLogger
from typing import Optional, Union
import logging
from torch.utils.data import DataLoader
import torch
import cv2
from breast_cancer_research.utils.utils import hwc_2_chw
import numpy as np

class UnetSummaryWriter(SummaryLogger):
    def __init__(self, log_dir: Optional[str] = None, comment: Optional[str] = None):
        super().__init__(log_dir=log_dir, comment=comment)

    def evaluate(self, val_score_dict, model_step, mode: str):
        if mode == "val":
            metric_mode = "Validation"
        elif mode == "train":
            metric_mode = "Training"
        else:
            raise ValueError(f"Metric mode {mode} not implemented")

        logging.info(f'{metric_mode} metrics: {val_score_dict}')

        for k, v in val_score_dict.items():
            self.add_scalar(f'{metric_mode}/{k}', v, model_step)

    def predict(self, prediction_dict, model_step, sample_batch: Optional[int] = 10):
        for idx in range(len(prediction_dict["all_images"])):
            self.add_images(f'{idx}/original_images', prediction_dict["all_images"][idx])
            self.add_images(f'{idx}/benign/ground_truth', prediction_dict["all_truth_masks"][idx][0])
            self.add_images(f'{idx}/benign/predictions', prediction_dict["all_pred_masks"][idx][0],
                                   model_step)
            self.add_images(f'{idx}/malignant/ground_truth', prediction_dict["all_truth_masks"][idx][1])
            self.add_images(f'{idx}/malignant/predictions', prediction_dict["all_pred_masks"][idx][1],
                                   model_step)
            self.add_images(f'{idx}/background/ground_truth', prediction_dict["all_truth_masks"][idx][2])
            self.add_images(f'{idx}/background/predictions', prediction_dict["all_pred_masks"][idx][2],
                                   model_step)

            if sample_batch and sample_batch == idx:
                break
