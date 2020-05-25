# base
import argparse
import json
import logging
# ml
import pandas as pd
from torch.utils.data import DataLoader
# custom
from breast_cancer_research.unet.unet_facade import BreastCancerSegmentator
from breast_cancer_research.unet.unet_dataset import UnetDataset

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Evaluate UNet')
    parser.add_argument('--unet_config_path', type=str, help='path to model config')
    parser.add_argument('--metadata_path', type=str, help='path to metadata')
    parser.add_argument('--predict_images', action='store_true', help='if set, sample predictions would '
                                                                      'be added to tensorboard')
    args = parser.parse_args()

    # read config
    with open(args.unet_config_path) as f:
        unet_config = json.load(f)

    train_config = unet_config['training_params']
    model_config = unet_config['inference_params']
    dataset_config = unet_config['dataset_params']

    metadata_train = pd.read_csv(args.metadata_path)
    dataset_eval = UnetDataset(metadata_train,
                               root=dataset_config["root"],
                               scale=dataset_config["scale"])
    dataloader_eval = DataLoader(dataset_eval,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

    unet = BreastCancerSegmentator(**model_config)

    # get out layer
    criterion, out_layer_name = unet.get_criterion(train_config["criterion_params"])

    # predict and save to tensorbaord
    unet.evaluate(eval_loader=dataloader_eval,
                  criterion=criterion)

    if args.predict_images is True:
        unet.predict(prediction_loader=dataloader_eval,
                     sample_batch=20,
                     out_layer_name=out_layer_name)


if __name__ == "__main__":
    main()
