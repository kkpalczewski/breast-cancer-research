# basic
import argparse
import json
from itertools import product
# ML
import torch
import pandas as pd
# custom
from breast_cancer_research.unet.unet_facade import BreastCancerSegmentator
from breast_cancer_research.unet.unet_model import UNet
from breast_cancer_research.unet.unet_dataset import UnetDataset


def main():
    parser = argparse.ArgumentParser(description='Check model predictions')
    parser.add_argument('--config_path', type=str, help='path to model config')
    parser.add_argument('--train_metadata_path', type=str, help='path to train metadata')
    parser.add_argument('--val_metadata_path', type=str, help='path to val metadata')
    args = parser.parse_args()

    # read config
    with open(args.config_path) as f:
        config = json.load(f)

    model_params = config["model_params"]
    model = UNet
    device = config["device"]
    cross_val_params = config["cross_validation_params"]

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

    combinations = get_params_combination(cross_val_params)

    for training_params in combinations:
        torch.cuda.empty_cache()
        unet = BreastCancerSegmentator(model=model,
                                       model_params=model_params,
                                       device=device)
        unet.train(dataset_train=dataset_train,
                   dataset_val=dataset_val,
                   save_cp=config["save_cp"],
                   criterion_name=config["criterion"]["name"],
                   criterion_params=config["criterion"]["params"],
                   scheduler_params=config["scheduler_params"],
                   optimizer_name=config["optimizer"]["name"],
                   optimizer_params=config["optimizer"]["params"],
                   **training_params)


def get_params_combination(params_dict):
    keys = [*params_dict.keys()]
    params = [params_dict[key] for key in keys]
    combinations = product(*params)

    cv_params = []

    for combination in combinations:
        new_param_dict = dict(zip(keys, combination))
        cv_params.append(new_param_dict)

    return cv_params


if __name__ == "__main__":
    main()
