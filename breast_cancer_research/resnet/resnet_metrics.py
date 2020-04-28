from breast_cancer_research.base.base_metrics import BaseMetrics
import torch.nn as nn
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, f1_score
import torch
from collections import defaultdict
import numpy as np

class CrossEntropyMetrics(nn.Module, BaseMetrics):
    def __init__(self,
                 activation_name: str = "softmax",
                 eval_threshold: float = 0.5):
        super().__init__()
        self.activation = self.get_activation(activation_name)
        self.forward_criterion = nn.CrossEntropyLoss()
        self.eval_threshold = eval_threshold

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward_criterion(input, target)

    def evaluate(self, preds_labels_dict: Dict[str, List[int]]) -> Tuple[Dict[str, List[float]], List[int]]:
        preds_labels_keys = [*preds_labels_dict.keys()]
        assert 'preds' in preds_labels_keys, f"\"preds\" not among in metric keys: {preds_labels_keys}"
        assert 'labels' in preds_labels_keys, f"\"labels\" not among in metric keys: {preds_labels_keys}"

        labels = np.array(preds_labels_dict['labels'])

        preds = preds_labels_dict['preds']
        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds)
        preds = np.array(self.activation(preds).tolist())

        metric_dict = defaultdict(list)

        preds = np.array(preds).reshape((preds.shape[1], preds.shape[0]), order="F")
        labels = np.array(labels).reshape((labels.shape[1], labels.shape[0]), order="F")

        support = []
        for pred, label in zip(preds, labels):
            metric_dict["auc"].append(roc_auc_score(y_true=label, y_score=pred))
            thresholded_pred = np.where(pred > self.eval_threshold, 1, 0)
            tn, fp, fn, tp = confusion_matrix(label, thresholded_pred).ravel()
            metric_dict["tn"].append(tn)
            metric_dict["fp"].append(fp)
            metric_dict["fn"].append(fn)
            metric_dict["tp"].append(tp)
            metric_dict["f1"].append(f1_score(y_true=label, y_pred=thresholded_pred))
            support.append(sum(label))

        return metric_dict, support
