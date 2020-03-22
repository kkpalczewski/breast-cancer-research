# basic
import argparse
import json
from itertools import product
from typing import Dict
from collections import defaultdict
# ML
import torch
import pandas as pd
# custom
from breast_cancer_research.unet.unet_facade import BreastCancerSegmentator
from breast_cancer_research.unet.unet_model import UNet
from breast_cancer_research.unet.unet_dataset import UnetDataset


def main():
    parser = argparse.ArgumentParser(description='Check model predictions')
    parser.add_argument('--train_config_path', type=str, help='path to train config')
    parser.add_argument('--cross_val_config_path', type=str, default=None, help='path to cross-val config')
    parser.add_argument('--train_metadata_path', type=str, help='path to train metadata')
    parser.add_argument('--val_metadata_path', type=str, help='path to val metadata')
    args = parser.parse_args()

    # read train config
    with open(args.train_config_path) as f:
        config = json.load(f)
    # read
    with open(args.cross_val_config_path) as f:
        cross_val_config = json.load(f)

    model_params = config["model_params"]
    model = UNet
    device = config["device"]

    assert device in ["cuda", "cpu"], "Cuda device not implemented, change \"device\" in config: {}".format(
        args.config_path)
    if device == "cuda":
        assert torch.cuda.is_available(), "Cuda not available, change \"device\" in config: {}".format(args.config_path)

    # get metadata
    metadata_train = pd.read_csv(args.train_metadata_path)
    metadata_val = pd.read_csv(args.val_metadata_path)
    # get dataset
    dataset_train = UnetDataset(metadata_train,
                                root=config["root"],
                                scale=config["scale"])
    dataset_val = UnetDataset(metadata_val,
                              root=config["root"],
                              scale=config["scale"])

    # TODO: check out Skorch for smarter cross-validation

    train_default_hparams=dict(
        epochs=config["epochs"],
        criterion_params=config["criterion_params"],
        scheduler_params=config["scheduler_params"],
        optimizer_params=config["optimizer_params"]
    )

    _check_test_cross_val_alignement(train_default_hparams, cross_val_config)

    combinations = recursive_params_combination(cross_val_config)

    for cross_val_params in combinations:
        torch.cuda.empty_cache()
        unet = BreastCancerSegmentator(model=model,
                                       model_params=model_params,
                                       device=device)

        train_cross_val_hparams = recursive_update_params(train_default_hparams, cross_val_params)

        print(train_cross_val_hparams)

        unet.train(dataset_train=dataset_train,
                   dataset_val=dataset_val,
                   save_cp=config["save_cp"],
                   **train_cross_val_hparams)

def recursive_update_params(train_default_hparams, cross_val_params):
    def _recursive_update_params(train_params, cross_params):
        for k, v in cross_params.items():
            if not isinstance(v, dict):
                train_params[k] = v
            else:
                train_params[k] = _recursive_update_params(train_params[k], cross_params[k])
        return train_params

    adjusted_hparams = train_default_hparams.copy()

    _recursive_update_params(adjusted_hparams, cross_val_params)

    return adjusted_hparams

def recursive_params_combination(params_dict: Dict):
    def _recursive_params_combination(master_dict, key, val):
        if len(key) == 1:
            master_dict[key[0]] = val
            return master_dict
        else:
            if key[0] not in [*master_dict.keys()]:
                master_dict[key[0]] = dict()
            master_dict[key[0]] = _recursive_params_combination(master_dict[key[0]], key[1:], val)


    keys, vals = recursive_key_val_list(params_dict)
    vals_combination = product(*vals)

    cv_params = list()
    for vals in vals_combination:
        cv_param = dict()
        for key, val in zip(keys, vals):
            _recursive_params_combination(cv_param, key, val)
        cv_params.append(cv_param)

    return cv_params

def recursive_key_val_list(mapping):
    def _rec_key_list(mapping, parent_key_path, all_keys_list, all_vals_list):
        for k, v in mapping.items():
            if not isinstance(v, dict):
                key_path = parent_key_path + [k]
                all_keys_list.append(key_path)
                all_vals_list.append(v)
            else:
                _rec_key_list(mapping[k], parent_key_path + [k], all_keys_list, all_vals_list)

        return all_keys_list, all_vals_list

    mapping = mapping.copy()
    parent_key_path = []
    all_keys_list = []
    all_vals_list = []

    all_keys, all_vals = _rec_key_list(mapping, parent_key_path, all_keys_list, all_vals_list)

    return all_keys, all_vals

def _check_test_cross_val_alignement(train_default_hparams, cross_val_config):
    train_keys, _ = recursive_key_val_list(train_default_hparams)
    cross_val_keys, _ = recursive_key_val_list(cross_val_config)
    for k in cross_val_keys:
        assert k in train_keys, f"Cross val key {k} not found in training config. Train and cross val keys not aligned."

if __name__ == "__main__":
    main()
