# base
import argparse
import json
# ml
import pandas as pd
import torch
# custom
from breast_cancer_research.unet.unet_facade import BreastCancerSegmentator
from breast_cancer_research.unet.unet_model import UNet
from breast_cancer_research.unet.unet_dataset import UnetDataset


def main():
    parser = argparse.ArgumentParser(description='Check model predictions')
    parser.add_argument('--config_path', type=str, help='path to model config')
    parser.add_argument('--pretrained_dict_path', type=str, help='path to model pretrained dict')
    parser.add_argument('--metadata_path', type=str, help='path to metadata')
    args = parser.parse_args()

    # read config
    with open(args.config_path) as f:
        config = json.load(f)

    model_params = config["model_params"]
    model = UNet
    root = config["root"]
    device = config["device"]
    scale = config["scale"]

    assert device in ["cuda", "cpu"], "Cuda device not implemented, change \"device\" in config: {}".format(
        args.config_path)
    if device == "cuda":
        assert torch.cuda.is_available(), "Cuda not available, change \"device\" in config: {}".format(args.config_path)

    metadata_train = pd.read_csv(args.metadata_path)
    dataset_train = UnetDataset(metadata_train, root, scale=scale)

    pretrained_model_path = args.pretrained_dict_path
    unet_model = BreastCancerSegmentator(model=model, model_params=model_params, device=device,
                                         pretrained_model_path=pretrained_model_path)

    #predict and save to tensorbaord
    unet_model.predict(dataset_test=dataset_train, tensorboard_verbose=True)
    unet_model.writer.close() #TODO: make it more elegant


if __name__ == "__main__":
    main()
