# basic
import argparse
import json
import logging
# ML
from torch.utils.data import DataLoader
import pandas as pd
# custom
from breast_cancer_research.resnet.resnet_facade import BreastCancerClassifier
from breast_cancer_research.resnet.resnet_dataset import ResnetDataset
import torchvision
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Check model predictions')
    parser.add_argument('--resnet_config_path', type=str, help='path to resnet config')
    parser.add_argument('--train_metadata_path', type=str, help='path to train metadata')
    parser.add_argument('--val_metadata_path', type=str, default=None, help='path to val metadata')

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # read train config
    with open(args.resnet_config_path) as f:
        resnet_config = json.load(f)

    train_config = resnet_config['training_params']
    model_config = resnet_config['inference_params']
    dataset_config = resnet_config['dataset_params']
    cross_val_config = resnet_config['cross_val_params']

    # get metadata
    metadata_train = pd.read_csv(args.train_metadata_path)

    if args.val_metadata_path is not None:
        metadata_val = pd.read_csv(args.val_metadata_path)
    else:
        metadata_val = None

    dataset_train = ResnetDataset(metadata=metadata_train,
                                  root_img=dataset_config["root_img"],
                                  root_mask=dataset_config["root_mask"],
                                  sample=dataset_config["sample"],
                                  unet_config=dataset_config["unet_config"],
                                  classes=dataset_config["classes"])
    dataloader_train = DataLoader(dataset_train, batch_size=dataset_config['batch_size'],
                                  shuffle=True, num_workers=8)

    if metadata_val is not None:
        dataset_val = ResnetDataset(metadata=metadata_val,
                                    root_img=dataset_config["root_img"],
                                    root_mask=dataset_config["root_mask"],
                                    sample=dataset_config["sample"],
                                    unet_config=dataset_config["unet_config"],
                                    classes=dataset_config["classes"])
        dataloader_val = DataLoader(dataset_val, dataset_config['batch_size'],
                                    shuffle=False, num_workers=8)
    else:
        dataloader_val = None

    # get model
    resnet = BreastCancerClassifier(model_params=model_config["model_params"],
                                    device=model_config["device"])

    # perform cross-validation
    resnet.cv(dataloader_train=dataloader_train,
              dataloader_val=dataloader_val,
              train_config=train_config,
              cross_val_config=cross_val_config)


if __name__ == "__main__":
    main()
