import pandas as pd
import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path
import pydicom as dicom
import imageio
import numpy as np
from sklearn.model_selection import train_test_split

def train_val_test_split(metadata, default_split: Optional[pd.DataFrame] = None,
                         train_size: Optional[float]=0.5, val_size: Optional[float] =0.25, test_size: Optional[float]=0.25,
                         random_state=42, stratify_cols: List[str] = ('image_view', 'benign_findings', 'malignant_findings', 'subtlety',
                                                      'breast_density')):
    """
    Script which split metadata to train, validation and test split. It preserve equal (when possible) ratio of data from classes [default]:
    image view, left or right breast, subtlety, benign findings, malignant findings, breast density
    """
    assert default_split is not None or train_size is not None, f"default_split or train_size have to be specified"

    if default_split is not None:
        merged_metadata_png = pd.merge(metadata, default_split, on=["patient_id", "image_view", "left_or_right_breast"])
        train_split = merged_metadata_png[merged_metadata_png["train_val_test_split"] == "train"].drop(columns=["train_val_test_split"])
        val_split = merged_metadata_png[merged_metadata_png["train_val_test_split"] == "val"].drop(columns=["train_val_test_split"])
        test_split = merged_metadata_png[merged_metadata_png["train_val_test_split"] == "test"].drop(columns=["train_val_test_split"])
    else:
        stratify_cols = list(stratify_cols)
        if val_size is None and test_size is None:
            val_split = None
            train_split, test_split = train_test_split(metadata, train=train_size, stratify=metadata[stratify_cols],
                                                       shuffle=True, random_state=random_state)
        elif test_size is not None and val_size is None:
            #assert train_size + test_size == 1, f"Ratio of train: {train_size} and test set: {test_size} doesn't add up to 1"
            val_split = None
            train_split, test_split = train_test_split(metadata, train_size=train_size, test_size=test_size, stratify=metadata[stratify_cols],
                                                       shuffle=True, random_state=random_state)
        elif test_size is not None and val_size is not None:
            #assert train_size+val_size+test_size == 1, "Ratio of train, validation and test set doesn't add up to 1"
            train_split, val_test_split = train_test_split(metadata, test_size=test_size+val_size, stratify=metadata[stratify_cols],
                                                           shuffle=True, random_state=random_state)
            val_split, test_split = train_test_split(val_test_split, test_size=test_size/(test_size+val_size), stratify=val_test_split[stratify_cols],
                                                     shuffle=True, random_state=random_state)
        else:
            raise ValueError(f"Val split: {val_size} specified, but train split was not sepcified")

    return train_split, val_split, test_split

def dcm_2_png(source_metadata: pd.DataFrame, in_dcm_dir: str, out_png_dir: str, sample: Optional[int] = None):
    new_metadata = pd.DataFrame()

    i = 0
    for idx1, single_scan_df in source_metadata.groupby(["patient_id", "image_view", "left_or_right_breast"]):
        image_path = os.path.join(in_dcm_dir, single_scan_df.iloc[0]['image file path'])
        if not os.path.isfile(image_path):
            logging.debug(f"Image in path {image_path} doesn't exist. Skip file ...")
            continue

        single_scan_metadata = single_scan_df.iloc[0]

        # transform img
        try:
            image_png = dicom.dcmread(image_path).pixel_array
        except ValueError as e:
            logging.warning(f"Image {image_path} corrupted: {e}")
            continue

        scan_shape = image_png.shape
        single_scan_metadata["image file path"] = Path(single_scan_metadata["image file path"]).parent / "scan.png"
        image_png_path = os.path.join(out_png_dir, single_scan_metadata["image file path"])
        if not os.path.isfile(image_png_path):
            Path(image_png_path).parent.mkdir(parents=True, exist_ok=True)
            imageio.imsave(image_png_path, image_png)
            logging.debug(f"Saved img in path: {image_png_path}")

        # get benign masks
        benign_df = single_scan_df[single_scan_df.pathology == 0]
        possible_benign_masks = benign_df['cropped_image_file_path'].tolist() + benign_df['ROI_mask_file_path'].tolist()
        benign_mask = np.zeros(scan_shape).astype(np.int8)
        for possible_mask_path in possible_benign_masks:
            possible_mask_full_path = os.path.join(in_dcm_dir, possible_mask_path)
            if not os.path.isfile(possible_mask_full_path):
                logging.debug(f"Mask: {possible_mask_full_path} does not exist")
                continue
            try:
                potential_mask = dicom.dcmread(possible_mask_full_path).pixel_array
            except ValueError as e:
                logging.warning(f"Mask {possible_mask_full_path} corrupted: {e}")
                continue
            if potential_mask.shape != scan_shape:
                continue
            else:
                benign_mask = np.bitwise_or(benign_mask, potential_mask)
        if benign_mask.max() != 0:
            single_scan_metadata['benign_findings'] = True
        else:
            single_scan_metadata['benign_findings'] = False
        single_scan_metadata["ROI benign path"] = Path(
            single_scan_metadata["image file path"]).parent / "benign_mask.png"
        imageio.imsave(os.path.join(out_png_dir, single_scan_metadata["ROI benign path"]), benign_mask)

        # get malignant masks
        malignant_df = single_scan_df[single_scan_df.pathology == 1]
        possible_malignant_masks = malignant_df['cropped_image_file_path'].tolist() + malignant_df[
            'ROI_mask_file_path'].tolist()
        malignant_mask = np.zeros(scan_shape).astype(np.int8)
        for possible_mask_path in possible_malignant_masks:
            possible_mask_full_path = os.path.join(in_dcm_dir, possible_mask_path)
            if not os.path.isfile(possible_mask_full_path):
                logging.debug(f"Mask: {possible_mask_full_path} does not exist")
                continue
            try:
                potential_mask = dicom.dcmread(possible_mask_full_path).pixel_array
            except ValueError as e:
                logging.warning(f"Mask {possible_mask_full_path} corrupted: {e}")
                continue

            if potential_mask.shape != scan_shape:
                continue
            else:
                malignant_mask = np.bitwise_or(malignant_mask, potential_mask)
        if benign_mask.max() != 0:
            single_scan_metadata['malignant_findings'] = True
        else:
            single_scan_metadata['malignant_findings'] = False
        single_scan_metadata["ROI malignant path"] = Path(
            single_scan_metadata["image file path"]).parent / "malignant_mask.png"
        imageio.imsave(os.path.join(out_png_dir, single_scan_metadata["ROI malignant path"]), malignant_mask)

        # drop not unused columns
        single_scan_metadata.drop(labels=['cropped_image_file_path', 'ROI_mask_file_path'], inplace=True)

        logging.debug(f"Added to metadata: {image_png_path}")

        new_metadata = new_metadata.append(single_scan_metadata)

        i += 1
        logging.debug(f"{i} sample")
        if sample is not None and sample <= i:
            break

    return new_metadata


def instance_2_semantic_metadata(source_metadata_dir: str,
                                 columns: List[str] = ('patient_id', 'image_view', 'left_or_right_breast',
                                                       'pathology', 'subtlety', 'image file path',
                                                       'ROI_mask_file_path', 'cropped_image_file_path', 'breast_density')) -> \
        Tuple[pd.DataFrame, str]:
    """
    Prepare metadata for new model and NYU model
    :param source_metadata_dir: source metadata csv dir
    :param new_metadata_path: path to new metadata folder
    :param columns: columns in new metadata
    :return: metadata and saving path
    """
    columns = list(columns)
    # merge all metadata into one file
    all_metadata = _merge_metadata(source_metadata_dir, columns)

    # Change pathology names to int
    all_metadata['pathology'].replace(['BENIGN', 'BENIGN_WITHOUT_CALLBACK', 'MALIGNANT'], [0, 0, 1], inplace=True)
    tmp_metadata = all_metadata.copy()
    all_metadata = all_metadata.groupby(['patient_id', 'image_view', 'left_or_right_breast'],
                                        as_index=False)[['subtlety', 'pathology']].max()

    tmp_columns = columns
    for col in ['subtlety', 'pathology']:
        tmp_columns.remove(col)

    all_metadata = all_metadata.merge(tmp_metadata[tmp_columns], how='left',
                                      on=['patient_id', 'image_view', 'left_or_right_breast'])
    all_metadata.drop_duplicates(subset=['patient_id', 'image_view', 'left_or_right_breast'], inplace=True)

    all_metadata.sort_values(by='patient_id', inplace=True)

    # delete new lines signs
    all_metadata[['image file path', 'ROI_mask_file_path', 'cropped_image_file_path']] = \
        all_metadata[['image file path', 'ROI_mask_file_path', 'cropped_image_file_path']].replace('\n', '', regex=True)

    return all_metadata

def _merge_metadata(source_metadata_dir: str, columns: List[str]) -> pd.DataFrame:
    all_metadata = pd.DataFrame(columns=columns)

    source_metadata_files = os.listdir(source_metadata_dir)

    for metadata_path in source_metadata_files:
        if not metadata_path.endswith(".csv"):
            logging.debug(f"File {metadata_path} has different extension than .csv. Ommiting this one ...")
            continue

        metadata_full_path = os.path.join(source_metadata_dir, metadata_path)

        csv_metadata = pd.read_csv(metadata_full_path, index_col=None)

        # column name breast density is not the same in all metadata
        if "breast density" in csv_metadata.columns:
            csv_metadata.rename(columns={"breast density": "breast_density"}, inplace=True)
        # rename other columns to remove spaces
        csv_metadata.rename(columns={"image view": "image_view",
                                     "left or right breast": "left_or_right_breast",
                                     "ROI mask file path": "ROI_mask_file_path",
                                     "cropped image file path": "cropped_image_file_path"}, inplace=True)

        # merge all metadata
        all_metadata = pd.merge(all_metadata[columns], csv_metadata[columns],
                                on=columns, how='outer')

    all_metadata.drop_duplicates(inplace=True)

    return all_metadata
