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
from breast_cancer_research.base.base_model import BaseModel

logging.basicConfig(level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description='Check model predictions')
    parser.add_argument('--resnet_config_path', type=str, help='path to resnet config')
    parser.add_argument('--val_metadata_path', type=str, help='path to val metadata')
    parser.add_argument('--unet_config_path', type=str, default=None, help='path to unet config (has to include path '
                                                                           'to pretrained unet)')
    parser.add_argument('--predict_images', action='store_true', help='if set, sample predictions would '
                                                                      'be added to tensorboard')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # read train config
    with open(args.resnet_config_path) as f:
        resnet_config = json.load(f)

    model_config = resnet_config['inference_params']
    dataset_config = resnet_config['dataset_params']

    # read unet config path if exist
    if args.unet_config_path is not None:
        with open(args.unet_config_path) as f:
            unet_config = json.load(f)
        unet_model_config = unet_config['inference_params']
        _, unet_out_layer_name = BaseModel.get_criterion(unet_config['training_params']['criterion_params'])
    else:
        unet_model_config = None
        unet_out_layer_name = None

    # get metadata
    metadata_val = pd.read_csv(args.val_metadata_path)

    dataset_val = ResnetDataset(metadata=metadata_val,
                                root_img=dataset_config["root_img"],
                                root_mask=dataset_config["root_mask"],
                                classes=dataset_config["classes"],
                                sample=dataset_config["sample"],
                                input_masks=dataset_config["input_masks"],
                                unet_config=unet_model_config,
                                unet_out_layer_name=unet_out_layer_name)
    dataloader_val = DataLoader(dataset=dataset_val,
                                batch_size=dataset_config["batch_size"],
                                shuffle=False,
                                num_workers=0)

    # get model
    resnet = BreastCancerClassifier(model_params=model_config["model_params"],
                                    device=model_config["device"])

    # additional metadata for tensorboard
    assert "pretrained_model_path" in [*unet_model_config.keys()], "Pretrained dict not specified"

    metadata_val['dataset/pretrained_model_path'] = unet_model_config["pretrained_model_path"]

    logging.info("Begin evaluation...")

    criterion, out_layer_name = BaseModel.get_criterion(resnet_config['training_params']['criterion_params'])

    resnet.evaluate(dataloader_val=dataloader_val,
                    criterion=criterion)

    if args.predict_images is True:
        resnet.predict(dataloader_val=dataloader_val,
                       num_images=20)


if __name__ == "__main__":
    main()
