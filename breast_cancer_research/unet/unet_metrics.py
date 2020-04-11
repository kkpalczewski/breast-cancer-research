import torch
import torch.nn as nn
from typing import List, Union
from breast_cancer_research.base.base_metrics import BaseMetrics

class BinaryDiceLoss(nn.Module, BaseMetrics):
    def __init__(self,
                 activation_name: str = "softmax",
                 weights: List[float] = (1, 1, 1),
                 device: str = "cuda",
                 reduction: str = "mean",
                 eval_threshold: float = 0.5):

        super().__init__()

        self.device = device
        self.eval_threshold = eval_threshold
        self.activation = self._get_activation(activation_name)

        # get weights
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=device, dtype=torch.float)
        weights = torch.nn.functional.normalize(weights, dim=0)
        self.weights = weights

        # get reduction
        self.reduction = reduction

    def forward(self, preds, targets):
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self.device)

        assert targets.shape == preds.shape, f"Target's shape {targets.shape} and pred's shape {preds.shape} are not equal"
        assert len(targets[0]) == len(self.weights), f"Weights' len {len(self.weights)} and" \
                                                     f" target's len {len(targets[0])} are not equal"
        assert targets.device == preds.device == self.weights.device, f"Devices for tensor " \
                                                                      f"targets: {targets.device}, " \
                                                                      f"preds: {preds.device}, " \
                                                                      f"weights: {self.weights.device} are not the same "

        dice_loss_with_weights = torch.zeros(1, dtype=torch.float, device=preds.device)

        # activation func
        preds = self.activation(preds)

        #TODO: check if this is a good way of taking mean from batch
        for target_batch, pred_batch, weight in zip(targets, preds, self.weights):
            for target, pred in zip(target_batch, pred_batch):
                dice_similarity = self.metric_dice_similarity(target, pred)
                dice_loss_with_weights += (torch.ones(1, device=self.device) - dice_similarity) * weight

        if self.reduction == "mean":
            reduced_dice_loss = dice_loss_with_weights/len(targets[0])/len(targets)
        elif self.reduction == "sum":
            reduced_dice_loss = dice_loss_with_weights
        else:
            raise NotImplementedError(f"Reduction function {self.reduction} not implemented")

        return reduced_dice_loss

    def evaluate(self, preds, targets, *,
                 eval_funcs: List[str] = ("dice_similarity", "target_coverage")):
        assert targets.shape == preds.shape, f"Target's shape {targets.shape} and " \
                                             f"pred's shape {preds.shape} are not equal"

        # activation func
        preds = self.activation(preds)

        eval_list = []
        for pred, target in zip(preds, targets):
            benign_eval_dict = self._eval_single_threshold(pred[0], target[0],
                                                           threshold=self.eval_threshold,
                                                           eval_funcs=eval_funcs)
            malignant_eval_dict = self._eval_single_threshold(pred[1], target[1],
                                                              threshold=self.eval_threshold,
                                                              eval_funcs=eval_funcs)
            eval_list.append(self._combine_single_metrics(benign_eval_dict, malignant_eval_dict, eval_funcs))

        return eval_list

    def _combine_single_metrics(self, benign_eval_dict, malignant_eval_dict,
                                eval_funcs: List[str] = ("dice_similarity", "target_coverage")):
        combined_metrics = dict()
        for eval_fun in eval_funcs:
            combined_metrics[f'benign_{eval_fun}'] = benign_eval_dict[eval_fun].data
            combined_metrics[f'malignant_{eval_fun}'] = malignant_eval_dict[eval_fun].data
            combined_metrics[f'combined_{eval_fun}'] = (
                        (malignant_eval_dict[eval_fun] + benign_eval_dict[eval_fun]) / 2).data

        return combined_metrics

    def _eval_single_threshold(self, target, pred, *, threshold: float,
                               eval_funcs: List[str] = ("dice_similarity", "target_coverage")):
        pred = (pred > threshold).float()
        eval_dict = dict()
        for eval_fun in eval_funcs:
            eval_dict[eval_fun] = self._eval_metrics(target, pred, eval_fun)

        return eval_dict

    def _eval_metrics(self, target, pred, eval_func: str) -> torch.Tensor:
        eval_metrics_dict = dict(
            dice_similarity=self.metric_dice_similarity,
            target_coverage=self.metric_target_coverage
        )

        assert eval_func in [*eval_metrics_dict.keys()], f"Metric {eval_func} not implemented"

        return eval_metrics_dict[eval_func](target, pred).cpu().float()

    def _eval_totals(self, metric, total: str):
        eval_totals_dict = dict(
            mean=torch.mean
        )

        assert total in [*eval_totals_dict.keys()], f"Total {total} not implemented"

        return eval_totals_dict[total](torch.stack(metric)).data

    @staticmethod
    def metric_dice_similarity(target, pred) -> torch.Tensor:
        target = target.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        target_pred_product = torch.sum(target * pred)
        target_sum = torch.sum(target)
        pred_sum = torch.sum(pred)

        single_dice_similarity = (2 * target_pred_product) / (target_sum + pred_sum)

        return single_dice_similarity

    @staticmethod
    def metric_target_coverage(target, pred) -> torch.Tensor:
        target = target.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        target_pred_product = torch.sum(target * pred)
        target_sum = torch.sum(target)

        single_target_coverage = target_pred_product / target_sum

        return single_target_coverage
