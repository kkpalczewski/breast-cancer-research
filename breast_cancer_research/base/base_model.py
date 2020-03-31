from abc import ABC, abstractmethod

from typing import Dict, Optional
import torch.nn as nn
from breast_cancer_research.unet.unet_metrics import BinaryDiceLoss
from breast_cancer_research.resnet.resnet_metrics import CrossEntropyMetrics
import torch
import torch.optim as optim
from itertools import product

class BaseModel(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @classmethod
    def _get_scheduler(cls, optimizer, scheduler_params: Optional[Dict] = None):

        scheduler_name, scheduler_hparams = BaseModel._get_names_hparams(scheduler_params)

        if scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_hparams)
        else:
            raise ValueError(f"Scheduler {scheduler_name} not implemented")

        return scheduler

    @classmethod
    def _get_criterion(cls, criterion_params: Optional[Dict] = None):
        criterion_name, criterion_hparams = BaseModel._get_names_hparams(criterion_params)

        if criterion_name == "BCE":
            criterion = nn.BCEWithLogitsLoss(**criterion_hparams)
            out_layer_name = "sigmoid"
        elif criterion_name == "CrossEntropyLoss":
            criterion = CrossEntropyMetrics(**criterion_hparams)
            out_layer_name = "identity"
        elif criterion_name == "dice":
            criterion = BinaryDiceLoss(**criterion_hparams)
            out_layer_name = "softmax"
        else:
            raise ValueError(f"Loss {criterion_name} not implemented")

        return criterion, out_layer_name

    @classmethod
    def _get_optimizer(cls, params, optimizer_params: Optional[Dict] = None):
        optimizer_name, optimizer_hparams = BaseModel._get_names_hparams(optimizer_params)

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(params, **optimizer_hparams)
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(params, **optimizer_hparams)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(params, **optimizer_hparams)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not implemented")

        return optimizer

    @classmethod
    def _get_names_hparams(cls, params):
        params = params.copy()
        name = params["name"]
        params.pop("name")
        hparams = params
        return name, hparams

    @staticmethod
    def _cv_update_params(train_default_hparams, cross_val_params):
        def _rec_cv_update_params(train_params, cross_params):
            for k, v in cross_params.items():
                if not isinstance(v, dict):
                    train_params[k] = v
                else:
                    train_params[k] = _rec_cv_update_params(train_params[k], cross_params[k])
            return train_params

        adjusted_hparams = train_default_hparams.copy()

        _rec_cv_update_params(adjusted_hparams, cross_val_params)

        return adjusted_hparams

    @staticmethod
    def _cv_params_combination(params_dict: Dict):
        def _rec_cv_params_combination(master_dict, key, val):
            if len(key) == 1:
                master_dict[key[0]] = val
                return master_dict
            else:
                if key[0] not in [*master_dict.keys()]:
                    master_dict[key[0]] = dict()
                master_dict[key[0]] = _rec_cv_params_combination(master_dict[key[0]], key[1:], val)

        keys, vals = BaseModel._cv_key_val_list(params_dict)
        vals_combination = product(*vals)

        cv_params = list()
        for vals in vals_combination:
            cv_param = dict()
            for key, val in zip(keys, vals):
                _rec_cv_params_combination(cv_param, key, val)
            cv_params.append(cv_param)

        return cv_params

    @staticmethod
    def _cv_key_val_list(mapping):
        def _rec_cv_key_val_list(mapping, parent_key_path, all_keys_list, all_vals_list):
            for k, v in mapping.items():
                if not isinstance(v, dict):
                    key_path = parent_key_path + [k]
                    all_keys_list.append(key_path)
                    all_vals_list.append(v)
                else:
                    _rec_cv_key_val_list(mapping[k], parent_key_path + [k], all_keys_list, all_vals_list)

            return all_keys_list, all_vals_list

        mapping = mapping.copy()
        parent_key_path = []
        all_keys_list = []
        all_vals_list = []

        all_keys, all_vals = _rec_cv_key_val_list(mapping, parent_key_path, all_keys_list, all_vals_list)

        return all_keys, all_vals

    @staticmethod
    def _cv_check_config_alignement(train_default_hparams, cross_val_config):
        train_keys, _ = BaseModel._cv_key_val_list(train_default_hparams)
        cross_val_keys, _ = BaseModel._cv_key_val_list(cross_val_config)
        for k in cross_val_keys:
            assert k in train_keys, f"Cross val key {k} not found in training config. Train and cross val keys not aligned."