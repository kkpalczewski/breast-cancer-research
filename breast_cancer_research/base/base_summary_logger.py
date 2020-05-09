from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from typing import Dict, Optional
from torch import float32
from torch.utils.data import DataLoader


class SummaryLogger(SummaryWriter, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    def loss(self, epoch_metric_dict: Dict[str, float], model_step: int):
        for k, v in epoch_metric_dict.items():
            self.add_scalar(f'Epoch_train_metric/{k}', v, model_step)

    def hparams(self, hparams, model_step):
        for k, v in hparams.items():
            self.add_scalar(f'Hparams/{k}', v, model_step)

    def totals(self, hparam_dict, metric_dict, train_metadata: Optional[dict] = None):
        if train_metadata is None:
            train_metadata = {}
        hparam_dict = self._totals_hparams(hparam_dict)
        metric_dict = self._totals_metric(metric_dict)
        hparam_dict.update(train_metadata)
        self.add_hparams(hparam_dict, metric_dict)

    def graph(self, model, dataloader_val: DataLoader, device):
        # only for graph add
        for batch in dataloader_val:
            imgs = batch['image']
            imgs = imgs.to(device=device, dtype=float32)
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
            if hasattr(hparam_dict['criterion'], 'smooth'):
                hparam_dict['crit/smooth'] = str(hparam_dict['criterion'].smooth.cpu().numpy())
            if hasattr(hparam_dict['criterion'], 'beta'):
                hparam_dict['crit/beta'] = str(hparam_dict['criterion'].beta)
            hparam_dict['crit/class_name'] = type(hparam_dict['criterion']).__name__
            del hparam_dict['criterion']

        return hparam_dict

    def _totals_metric(self, metric_dict):
        metric_dict = metric_dict.copy()
        metric_keys = [*metric_dict.keys()]
        if 'last_eval_score' in metric_keys:
            if metric_dict['last_eval_score'] is not None:
                for k, v in metric_dict['last_eval_score'].items():
                    eval_name = 'eval/' + k
                    metric_dict[eval_name] = v
            del metric_dict['last_eval_score']

        return metric_dict
