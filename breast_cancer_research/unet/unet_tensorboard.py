from breast_cancer_research.base.base_summary_logger import SummaryLogger
from typing import Optional
import logging
from torch.utils.data import DataLoader
import torch

class UnetSummaryWriter(SummaryLogger):
    def __init__(self, log_dir: Optional[str] = None, comment: Optional[str] = None):
        super().__init__(log_dir=log_dir, comment=comment)

    def evaluate(self, val_score_dict, model_step):
        logging.info('Validation metrics: {}'.format(val_score_dict))
        for k, v in val_score_dict.items():
            self.add_scalar(f'Val/{k}', v, model_step)

    def predict(self, prediction_dict, model_step, sample_batch: Optional[int] = 5):
        for idx in range(len(prediction_dict["all_images"])):
            self.add_images(f'{idx}/original_images', prediction_dict["all_images"][idx])
            self.add_images(f'{idx}/benign/ground_truh', prediction_dict["all_truth_masks"][idx][0])
            self.add_images(f'{idx}/benign/predictions', prediction_dict["all_pred_masks"][idx][0],
                                   model_step)
            self.add_images(f'{idx}/malignant/ground_truh', prediction_dict["all_truth_masks"][idx][1])
            self.add_images(f'{idx}/malignant/predictions', prediction_dict["all_pred_masks"][idx][1],
                                   model_step)
            self.add_images(f'{idx}/background/ground_truh', prediction_dict["all_truth_masks"][idx][2])
            self.add_images(f'{idx}/background/predictions', prediction_dict["all_pred_masks"][idx][2],
                                   model_step)

            if sample_batch and sample_batch == idx:
                break

    def totals(self, hparam_dict, metric_dict):
        hparam_dict = self._totals_hparams(hparam_dict)
        metric_dict = self._totals_metric(metric_dict)

        self.add_hparams(hparam_dict, metric_dict)

    def graph(self, model, dataloader_val: DataLoader, device):
        # only for graph add
        for batch in dataloader_val:
            imgs = batch['image']
            imgs = imgs.to(device=device, dtype=torch.float32)
            break

        self.add_graph(model, imgs)

    def _totals_hparams(self, hparam_dict):
        hparam_dict = hparam_dict.copy()
        hparam_keys = [*hparam_dict.keys()]

        # parse optimizer
        if 'optimizer' in hparam_keys:
            for opt_key, opt_val in hparam_dict['optimizer'].param_groups[0].items():
                if opt_key in ['lr', 'betas', 'eps', 'weight_decay', 'initial_lr']:
                    param_name = "opt/" + opt_key
                    if opt_key == 'betas':
                        opt_val = str(opt_val)
                    hparam_dict[param_name] = opt_val
            hparam_dict['opt/class_name'] = type(hparam_dict['optimizer']).__name__
            del hparam_dict['optimizer']

        # parse lr scheduler
        if 'lr_scheduler' in hparam_keys:
            hparam_dict['lr_sched/gamma'] = hparam_dict['lr_scheduler'].gamma
            hparam_dict['lr_sched/step_size'] = hparam_dict['lr_scheduler'].step_size
            hparam_dict['lr_sched/class_name'] = type(hparam_dict['lr_scheduler']).__name__
            del hparam_dict['lr_scheduler']

        # parse criterion
        if 'criterion' in hparam_keys:
            if hasattr(hparam_dict['criterion'], 'weights'):
                hparam_dict['crit/weights'] = str(hparam_dict['criterion'].weights.cpu().numpy())
            hparam_dict['crit/class_name'] = type(hparam_dict['criterion']).__name__
            del hparam_dict['criterion']

        return hparam_dict

    def _totals_metric(self, metric_dict):
        metric_dict = metric_dict.copy()
        metric_keys = [*metric_dict.keys()]
        if 'last_eval_score' in metric_keys:
            for k, v in metric_dict['last_eval_score'].items():
                eval_name = 'eval/' + k
                metric_dict[eval_name] = v
            del metric_dict['last_eval_score']

        return metric_dict

    def hparams(self, hparams, model_step):
        for k, v in hparams.items():
            self.add_scalar(f'Hparams/{k}', v, model_step)