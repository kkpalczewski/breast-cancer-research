from breast_cancer_research.base.base_summary_logger import SummaryLogger
from typing import Optional
import numpy as np
import torch

class ResnetSummaryWriter(SummaryLogger):
    def __init__(self, log_dir: Optional[str] = None, comment: Optional[str] = None):
        super().__init__(log_dir=log_dir, comment=comment)

    def evaluate(self, *, val_score_dict, classes, support, model_step, excluded_classes=None):
        if excluded_classes is not None:
            raise NotImplementedError

        for metric_name, metric_value in val_score_dict.items():
            #add wighted averages total
            weighted_avg = np.dot(metric_value, support)/np.sum(support)
            self.add_scalar(f'Eval_metric_{metric_name}/weighted_average', weighted_avg, model_step)
            #add specifics for every class
            for idx, single_class in enumerate(classes):
                self.add_scalar(f'Eval_metric_{metric_name}/{single_class}', metric_value[idx], model_step)

    def predict(self, *, prediction_dict, model_step):
        for tp_or_fn, single_class_dict in prediction_dict.items():
            for class_name, img in single_class_dict.items():
                batch_tensor = torch.stack(img)
                self.add_images(f'{tp_or_fn}/{class_name}', batch_tensor, model_step)

    def train(self):
        pass

    def totals(self, hparam_dict, metric_dict):
        pass

    def graph(self, dataloader_val, model, model_step):
        pass