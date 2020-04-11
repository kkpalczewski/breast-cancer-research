# base
import argparse
import json
# ml
import pandas as pd
from torch.utils.data import DataLoader
# custom
from breast_cancer_research.unet.unet_facade import BreastCancerSegmentator
from breast_cancer_research.unet.unet_dataset import UnetDataset


def main():
    parser = argparse.ArgumentParser(description='Check model predictions')
    parser.add_argument('--config_path', type=str, help='path to model config')
    parser.add_argument('--pretrained_dict_path', type=str, help='path to model pretrained dict')
    parser.add_argument('--metadata_path', type=str, help='path to metadata')
    parser.add_argument('--batch_sample', type=int, default=10, help='batch sample to make a prediction')
    args = parser.parse_args()

    # read config
    with open(args.config_path) as f:
        config = json.load(f)

    root = config["root"]
    scale = config["scale"]

    metadata_train = pd.read_csv(args.metadata_path)
    dataset_eval = UnetDataset(metadata_train,
                               root=root,
                               scale=scale)
    dataloader_eval = DataLoader(dataset_eval,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=True)

    pretrained_model_path = args.pretrained_dict_path

    unet = BreastCancerSegmentator(model_params=config["model_params"],
                                   device=config["device"],
                                   pretrained_model_path=pretrained_model_path)

    # get out layer
    _, out_layer_name = unet._get_criterion(config["criterion_params"])

    # predict and save to tensorbaord
    unet.predict(prediction_loader=dataloader_eval,
                 tensorboard_verbose=True,
                 out_layer_name=out_layer_name,
                 batch_size=1,
                 sample_batch=args.batch_sample)


if __name__ == "__main__":
    main()
