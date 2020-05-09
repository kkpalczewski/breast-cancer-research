import argparse
import json
import logging
import os
import pandas as pd
from breast_cancer_research.utils.data import (instance_2_semantic_metadata, dcm_2_png, train_val_test_split)
logging.basicConfig(level=logging.DEBUG)

def main():
    parser = argparse.ArgumentParser(description='Preprocess CBIS-DDSM data')
    parser.add_argument('--data_config_path', type=str, help='path to data generation config')
    args = parser.parse_args()

    with open(args.data_config_path) as f:
        data_config = json.load(f)

    logging.info(f"Starting merging metadata ...")
    new_metadata = instance_2_semantic_metadata(data_config['downloaded_metadata_dir'])
    logging.info(f"Merged successful!")

    if "transform_dicom_to_png" in [*data_config.keys()] and data_config["transform_dicom_to_png"] is True:
        logging.info(f"Converting from .dcm to .png")
        new_metadata = dcm_2_png(new_metadata, data_config["downloaded_images_root"], data_config["images_out_dir"],
                                 sample=data_config["sample"])
        logging.info(f"Converting from .dcm to .png successful!")

    logging.info(f"Starting splitting metadata ...")
    if "default_split" in [*data_config.keys()] and data_config["default_split"] is not None:
        default_split = pd.read_csv(data_config["default_split"], index_col=0)
    else:
        default_split = None

    train_split, val_split, test_split = train_val_test_split(new_metadata,
                                                              default_split=default_split,
                                                              train_size=data_config["train_val_test_split"][0],
                                                              val_size=data_config["train_val_test_split"][1],
                                                              test_size=data_config["train_val_test_split"][2])
    logging.info(f"Split successful!")

    if "metadata_out_dir" not in [*data_config.keys()] or data_config["metadata_out_dir"] is None:
        metadata_out_dir = data_config['downloaded_metadata_dir']
    else:
        metadata_out_dir = data_config["metadata_out_dir"]

    train_metadata_path = os.path.join(metadata_out_dir, "train_metadata.csv")
    train_split.to_csv(train_metadata_path)
    logging.info(f"Train metadata saved in: {train_metadata_path}!")

    if val_split is not None:
        val_metadata_path = os.path.join(metadata_out_dir, "val_metadata.csv")
        val_split.to_csv(val_metadata_path)
        logging.info(f"Validation metadata saved in: {val_metadata_path}!")

    test_metadata_path = os.path.join(metadata_out_dir, "test_metadata.csv")
    test_split.to_csv(test_metadata_path)
    logging.info(f"Test metadata saved in: {test_metadata_path}!")


if __name__ == "__main__":
    main()
