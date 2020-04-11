# basic
import argparse
import json
import logging
# ML
from torch.utils.data import DataLoader
import pandas as pd
# custom
from breast_cancer_research.unet.unet_facade import BreastCancerSegmentator
from breast_cancer_research.unet.unet_dataset import UnetDataset

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Check model predictions')
    parser.add_argument('--unet_config_path', type=str, help='path to unet config')
    parser.add_argument('--train_metadata_path', type=str, help='path to train metadata')
    parser.add_argument('--val_metadata_path', type=str, default=None, help='path to val metadata')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # read train config
    with open(args.unet_config_path) as f:
        unet_config = json.load(f)

    train_config = unet_config['training_params']
    model_config = unet_config['inference_params']
    dataset_config = unet_config['dataset_params']
    cross_val_config = unet_config['cross_val_params']

    # get metadata
    metadata_train = pd.read_csv(args.train_metadata_path)

    if args.val_metadata_path is not None:
        metadata_val = pd.read_csv(args.val_metadata_path)
    else:
        metadata_val = None

    # get dataset
    dataset_train = UnetDataset(metadata_train,
                                root=dataset_config["root"],
                                scale=dataset_config["scale"])
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=dataset_config["batch_size"],
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True)
    if metadata_val is not None:
        dataset_val = UnetDataset(metadata_val,
                                  root=dataset_config["root"],
                                  scale=dataset_config["scale"])
        dataloader_val = DataLoader(dataset_val,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=8,
                                    pin_memory=True)
    else:
        dataloader_val = None

    #get model
    unet = BreastCancerSegmentator(**model_config)

    # perform cross-validation
    unet.cv(dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            train_config=train_config,
            cross_val_config=cross_val_config)


if __name__ == "__main__":
    main()
