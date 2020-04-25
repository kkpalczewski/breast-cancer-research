from breast_cancer_research.base.base_summary_logger import SummaryLogger
from typing import Optional
import logging
import torch
from breast_cancer_research.utils.visualize import overlay_mask

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
        all_steps = 0
        for idx1 in range(len(prediction_dict["all_images"])):
            classname = prediction_dict["classnames"][idx1]
            for idx2 in range(prediction_dict["all_images"][idx1].shape[0]):
                img_tensor = prediction_dict["all_images"][idx1][idx2]
                pred_masks = prediction_dict["all_pred_masks"][idx1][idx2]
                ground_truth_masks = prediction_dict["all_truth_masks"][idx1][idx2]

                benign_img = overlay_mask(tensor_img=img_tensor,
                                          tensor_masks=torch.stack([pred_masks[0], ground_truth_masks[0]]),
                                          classnames=[classname[0][0], classname[0][0] + "_ground_truth"])
                self.add_image(f'{all_steps}/benign', benign_img, model_step, dataformats='HWC')

                malignant_img = overlay_mask(tensor_img=img_tensor,
                                          tensor_masks=torch.stack([pred_masks[1], ground_truth_masks[1]]),
                                          classnames=[classname[1][0], classname[1][0] + "_ground_truth"])
                self.add_image(f'{all_steps}/malignant', malignant_img, model_step, dataformats='HWC')

                if sample_batch and sample_batch == all_steps:
                    break

                all_steps += 1
