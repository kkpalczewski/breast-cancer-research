from breast_cancer_research.base.base_metrics import BaseMetrics
import torch.nn as nn
from typing import List, Dict, Tuple
from torch import Tensor
from sklearn.metrics import precision_recall_fscore_support

class CrossEntropyMetrics(nn.Module, BaseMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_criterion = nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward_criterion(input, target)

    def evaluate(self, preds_labels_dict: Dict[str, List[int]]) -> Tuple[Dict[str, List[float]], List[int]]:
        preds_labels_keys = [*preds_labels_dict.keys()]
        assert 'preds' in preds_labels_keys, f"\"preds\" not among in metric keys: {preds_labels_keys}"
        assert 'labels' in preds_labels_keys, f"\"labels\" not among in metric keys: {preds_labels_keys}"

        labels = preds_labels_dict['labels']
        preds = preds_labels_dict['preds']

        precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true=labels, y_pred=preds)

        metric_dict = dict(
            precision=precision,
            recall=recall,
            fbeta_score=fbeta_score,
        )

        return metric_dict, support
