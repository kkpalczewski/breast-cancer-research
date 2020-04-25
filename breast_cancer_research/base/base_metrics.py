from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from typing import List

class BaseMetrics(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @classmethod
    def _get_activation(cls, activation_name):
        if activation_name == "softmax":
            activation = nn.Softmax(dim=1)
        else:
            raise NotImplementedError(f"Activation function {activation_name} not implemented")
        return activation

    @staticmethod
    def metric_dice_similarity(target, pred, smooth) -> torch.Tensor:
        target = target.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        target_pred_product = torch.sum(target * pred)
        target_sum = torch.sum(target)
        pred_sum = torch.sum(pred)
        single_dice_similarity = (2 * target_pred_product + smooth) / (target_sum + pred_sum + smooth)

        return single_dice_similarity

    @staticmethod
    def metric_target_coverage(target, pred, smooth) -> torch.Tensor:
        target = target.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        target_pred_product = torch.sum(target * pred)
        target_sum = torch.sum(target)

        single_target_coverage = (target_pred_product + smooth)/ (target_sum + smooth)

        return single_target_coverage
