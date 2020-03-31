# basic
import argparse
import json

from typing import Dict
from collections import defaultdict
import time
import logging
# ML
import torch
import pandas as pd
# custom
from breast_cancer_research.unet.unet_facade import BreastCancerSegmentator
from breast_cancer_research.unet.unet_model import UNet

import logging
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Check model predictions')
    parser.add_argument('--train_config_path', type=str, help='path to train config')
    parser.add_argument('--cross_val_config_path', type=str, default=None, help='path to cross-val config')
    parser.add_argument('--train_metadata_path', type=str, help='path to train metadata')
    parser.add_argument('--val_metadata_path', type=str, help='path to val metadata')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # read train config
    with open(args.train_config_path) as f:
        train_config = json.load(f)
    # read
    with open(args.cross_val_config_path) as f:
        cross_val_config = json.load(f)

    # get metadata
    metadata_train = pd.read_csv(args.train_metadata_path)
    metadata_val = pd.read_csv(args.val_metadata_path)


    unet = BreastCancerSegmentator(model_class=UNet,
                                   model_params=train_config["model_params"],
                                   device=train_config["device"])
    # perform cross-validation
    unet.cv(metadata_train=metadata_train,
            metadata_val=metadata_val,
            train_config=train_config,
            cross_val_config=cross_val_config)


if __name__ == "__main__":
    main()
