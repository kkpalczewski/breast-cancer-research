from breast_cancer_research.base.base_summary_logger import SummaryLogger
from typing import Optional
import numpy as np
import torch
import logging


class ResnetSummaryWriter(SummaryLogger):
    def __init__(self, log_dir: Optional[str] = None, comment: Optional[str] = None):
        super().__init__(log_dir=log_dir, comment=comment)

    def evaluate(self, *, val_score_dict, classes, support, model_step, mode: str, excluded_classes=None):
        if mode == "val":
            metric_mode = "Validation"
        elif mode == "train":
            metric_mode = "Training"
        else:
            raise ValueError(f"Metric mode {mode} not implemented")

        if excluded_classes is not None:
            raise NotImplementedError

        for metric_name, metric_value in val_score_dict.items():
            # add wighted averages total
            weighted_avg = np.dot(metric_value, support) / np.sum(support)
            self.add_scalar(f'{metric_mode}_{metric_name}/weighted_average', weighted_avg, model_step)
            # add specifics for every class
            for idx, (single_class, s) in enumerate(zip(classes, support)):
                if s != 0:
                    self.add_scalar(f'{metric_mode}_{metric_name}/{single_class}', metric_value[idx], model_step)
                else:
                    logging.warning(
                        f"Support for class {single_class} equals 0. Setting metric_name {metric_name} to 0.")
                    self.add_scalar(f'{metric_mode}_{metric_name}/{single_class}', 0, model_step)
        logging.info(f"Saved metrics: {val_score_dict}")

    def predict(self, *, prediction_dict, model_step):
        for tp_or_fn, single_class_dict in prediction_dict.items():
            for class_name, img in single_class_dict.items():
                batch_tensor = torch.stack(img)
                batch_tensor = (batch_tensor*255).to(dtype=torch.uint8)

                self.add_images(f'{tp_or_fn}/{class_name}', batch_tensor, model_step)

    def train(self):
        pass
