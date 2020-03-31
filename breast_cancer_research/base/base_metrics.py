from abc import ABC, abstractmethod


class BaseMetrics(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass