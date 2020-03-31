from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from typing import Dict

class SummaryLogger(SummaryWriter, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def totals(self, *args, **kwargs):
        pass

    def loss(self, epoch_metric_dict: Dict[str, float], model_step: int):
        for k, v in epoch_metric_dict.items():
            self.add_scalar(f'Epoch_train_metric/{k}', v, model_step)