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
                 eval_threshold: Union[float, List[float]] = 0.5):

        super().__init__()

        self.activation_name = activation_name
        self.device = device
        self.eval_threshold = eval_threshold

        if activation_name == "softmax":
            self.activation = nn.Softmax(dim=0)
        else:
            raise NotImplementedError(f"Activation function {activation_name} not implemented")

        # get weights
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=device, dtype=torch.float)
        weights = torch.nn.functional.normalize(weights, dim=0)
        self.weights = weights

        # get reduction
        self.reduction = reduction

    def forward(self, preds, targets):

        preds, targets = self._prepare_pred_target(targets, preds)

        assert len(targets) == len(
            self.weights), f"Weights' len {len(self.weights)} anr target's len {len(targets)} are not equal"

        assert targets.device == preds.device == self.weights.device, f"Devices for tensor " \
                                                                      f"targets: {targets.device}, " \
                                                                      f"preds: {preds.device}, " \
                                                                      f"weights: {self.weights.device} are not the same "

        dice_loss_with_weights = torch.zeros(1, dtype=torch.float, device=preds.device)

        # activation func
        preds = self.activation(preds)

        for target, pred, weight in zip(targets, preds, self.weights):
            dice_similarity = self.metric_dice_similarity(target, pred)
            dice_loss_with_weights += (torch.ones(1, device=self.device) - dice_similarity) * weight

        if self.reduction == "mean":
            reduced_dice_loss = dice_loss_with_weights/len(self.weights)
        elif self.reduction == "sum":
            reduced_dice_loss = dice_loss_with_weights
        else:
            raise NotImplementedError(f"Reduction function {self.reduction} not implemented")

        return reduced_dice_loss

    def evaluate(self, preds, targets, *, eval_funcs: List[str] = ("dice_similarity", "target_coverage")):
        preds, targets = self._prepare_pred_target(targets, preds)
        # activation func
        preds = self.activation(preds)

        if isinstance(self.eval_threshold, float):
            benign_eval_dict = self._eval_single_threshold(preds[0], targets[0],
                                                           threshold=self.eval_threshold,
                                                           eval_funcs=eval_funcs)
            malignant_eval_dict = self._eval_single_threshold(preds[1], targets[1],
                                                              threshold=self.eval_threshold,
                                                              eval_funcs=eval_funcs)
            eval_dict = self._combine_single_metrics(benign_eval_dict, malignant_eval_dict, eval_funcs)

        elif isinstance(self.eval_threshold, list) or isinstance(self.eval_threshold, tuple):
            malignant_all_thresh = []
            benign_all_thresh = []
            for single_threshold in self.eval_threshold:
                benign_all_thresh.append(self._eval_single_threshold(preds[0], targets[0],
                                                                     threshold=single_threshold,
                                                                     eval_funcs=eval_funcs))
                malignant_all_thresh.append(self._eval_single_threshold(preds[1], targets[1],
                                                                        threshold=single_threshold,
                                                                        eval_funcs=eval_funcs))
            eval_dict = self._combine_multiple_metrics(benign_all_thresh, malignant_all_thresh, eval_funcs)
        else:
            raise TypeError(f'Eval threshold type {type(self.eval_threshold)} not permitted')

        return eval_dict

    def _combine_multiple_metrics(self, benign_all_thresh, malignant_all_thresh,
                                  eval_funcs: List[str] = ("dice_similarity", "target_coverage"),
                                  totals: List[str] = ("mean",)):
        combined_metrics = dict()
        for eval_fun in eval_funcs:
            for total in totals:
                benign_metrics = [x[eval_fun] for x in benign_all_thresh]
                malignant_metrics = [x[eval_fun] for x in malignant_all_thresh]
                combined_metrics[f'benign_{eval_fun}_{total}'] = self._eval_totals(benign_metrics, total)
                combined_metrics[f'malignant_{eval_fun}_{total}'] = self._eval_totals(malignant_metrics, total)
                combined_metrics[f'combined_{eval_fun}_{total}'] = (self._eval_totals(benign_metrics, total) +
                                                                    self._eval_totals(malignant_metrics, total)) / 2
        return combined_metrics

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

    def _prepare_pred_target(self, targets: torch.Tensor, preds: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self.device)

        assert targets.shape == preds.shape, f"Target's shape {targets.shape} and pred's shape {preds.shape} are not equal"
        assert len(targets) == 1, f"Check targets shape"

        targets = targets[0]
        preds = preds[0]

        return preds, targets

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
