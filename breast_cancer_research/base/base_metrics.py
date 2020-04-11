from abc import ABC, abstractmethod
import torch.nn as nn


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
