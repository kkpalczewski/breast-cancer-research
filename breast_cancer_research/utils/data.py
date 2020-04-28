import pandas as pd
import os
import logging
from typing import List, Tuple
from pathlib import Path
import pydicom as dicom
import imageio
import numpy as np
from sklearn.model_selection import train_test_split

def train_val_test_split(path_to_metadata, destination_folder, train_size=0.5, val_size=0.25, test_size=0.25, random_state=42, verbose=False):
    """
    Script which split metadata to train, validation and test split. It preserve equal (when possible) ratio of data from classes:
    - image view
    - left or right breast
    - subtlety
    - pathology
    - breast density

    """
    #TODO: rewrite script so that it would accept any number of columns to make redistributed split

    if train_size+val_size+test_size != 1:
        raise Exception("Ratio of train, validation and test set doesn't add up to 1")

    metadata = pd.read_csv(path_to_metadata)
    image_view_col = metadata["image_view"].unique()
    left_or_right_breast_col = metadata["left_or_right_breast"].unique()
    subtlety_col = metadata["subtlety"].unique()
    pathology_col = metadata["pathology"].unique()
    breast_density_col = metadata["breast_density"].unique()

    train_array = []
    val_array = []
    test_array = []

    #iterate through every needed column and make split in every of those
    for image_view in image_view_col:
        for left_or_right_breast in left_or_right_breast_col:
            for subtlety in subtlety_col:
                for pathology in pathology_col:
                    for breast_density in breast_density_col:
                        tmp_metadata = metadata[((metadata["image_view"]==image_view) &
                                                (metadata["left_or_right breast"]==left_or_right_breast) &
                                                (metadata["subtlety"]==subtlety) &
                                                (metadata["pathology"]==pathology) &
                                                (metadata["breast_density"]==breast_density))]["image_file_path"]
                        if tmp_metadata.shape[0] > 1:
                            tmp_train, tmp_test, _, _ = train_test_split(tmp_metadata, tmp_metadata, test_size=test_size, random_state=random_state)
                            if tmp_train.shape[0] > 1:
                                tmp_train, tmp_val, _, _ = train_test_split(tmp_train, tmp_train, test_size=val_size/(1-test_size),
                                                                            random_state=random_state)
                                train_array.extend(list(tmp_train))
                                val_array.extend(list(tmp_val))
                                test_array.extend(list(tmp_test))
                            else:
                                train_array.extend(list(tmp_train))
                                test_array.extend(list(tmp_test))
                        elif tmp_metadata.shape[0] == 1:
                            tmp_train=tmp_metadata
                            train_array.extend(list(tmp_train))

    #save splitted data
    train_df = metadata[metadata["image file path"].isin(train_array)]
    val_df = metadata[metadata["image file path"].isin(val_array)]
    test_df = metadata[metadata["image file path"].isin(test_array)]

    print(f"STATISTICS:\n"
          "train size: {len(train_array)/all_records_size}, "
          "validation size: {len(val_array)/all_records_size}, "
          "test size: {len(test_array)/all_records_size)}")

    return train_df, val_df, test_df

def dcm_2_png(source_metadata: pd.DataFrame, in_dcm_dir: str, out_png_dir: str):
    new_metadata = pd.DataFrame()

    for idx1, single_scan_df in source_metadata.groupby(["patient_id", "image_view", "left_or_right_breast"]):
        image_path = os.path.join(in_dcm_dir, single_scan_df.iloc[0]['image_file_path'])
        if not os.path.isfile(image_path):
            logging.info(f"Image in path {image_path} doesn't exist. Skip file ...")
            continue

        single_scan_metadata = single_scan_df.iloc[0]

        # transform img
        image_png = dicom.dcmread(image_path).pixel_array
        scan_shape = image_png.shape
        single_scan_metadata["image_file_path"] = Path(single_scan_metadata["image_file_path"]).parent / "scan.png"
        image_png_path = os.path.join(out_png_dir, single_scan_metadata["image_file_path"])
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
            potential_mask = dicom.dcmread(possible_mask_full_path).pixel_array
            if potential_mask.shape != scan_shape:
                continue
            else:
                benign_mask = np.bitwise_or(benign_mask, potential_mask)
        if benign_mask.max() != 0:
            single_scan_metadata['benign_findings'] = True
        else:
            single_scan_metadata['benign_findings'] = False
        single_scan_metadata["roi_benign_path"] = Path(
            single_scan_metadata["image_file_path"]).parent / "benign_mask.png"
        imageio.imsave(os.path.join(out_png_dir, single_scan_metadata["roi_benign_path"]), benign_mask)

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
            potential_mask = dicom.dcmread(possible_mask_full_path).pixel_array
            if potential_mask.shape != scan_shape:
                continue
            else:
                malignant_mask = np.bitwise_or(malignant_mask, potential_mask)
        if benign_mask.max() != 0:
            single_scan_metadata['malignant_findings'] = True
        else:
            single_scan_metadata['malignant_findings'] = False
        single_scan_metadata["roi_malignant_path"] = Path(
            single_scan_metadata["image_file_path"]).parent / "malignant_mask.png"
        imageio.imsave(os.path.join(out_png_dir, single_scan_metadata["roi_malignant_path"]), malignant_mask)

        # drop not unused columns
        single_scan_metadata.drop(labels=['cropped_image_file_path', 'ROI_mask_file_path'], inplace=True)

        new_metadata = new_metadata.append(single_scan_metadata)

    return new_metadata


def instance_2_semantic_metadata(source_metadata_dir: str,
                                 columns: List[str] = ('patient_id', 'image_view', 'left_or_right_breast',
                                                       'pathology', 'subtlety', 'image_file_path',
                                                       'ROI_mask_file_path', 'cropped_image_file_path')) -> \
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
                                        as_index=False)['subtlety', 'pathology'].max()

    tmp_columns = columns
    for col in ['subtlety', 'pathology']:
        tmp_columns.remove(col)

    all_metadata = all_metadata.merge(tmp_metadata[tmp_columns], how='left',
                                      on=['patient_id', 'image_view', 'left_or_right_breast'])
    all_metadata.drop_duplicates(subset=['patient_id', 'image_view', 'left_or_right_breast'], inplace=True)

    all_metadata.sort_values(by='patient_id', inplace=True)

    # delet new lines signs
    all_metadata[['image_file_path', 'ROI_mask_file_path', 'cropped_image_file_path']] = \
        all_metadata[['image_file_path', 'ROI_mask_file_path', 'cropped_image_file_path']].replace('\n', '', regex=True)

    return all_metadata


def _create_png_img(image_path, metadata, idx, metadata_column_name, out_png_dir, desired_size=None):
    image_png = dicom.dcmread(image_path).pixel_array

    metadata.loc[idx, metadata_column_name] = str(metadata.loc[idx][metadata_column_name][:-3]) + "png"
    image_png_path = os.path.join(out_png_dir, metadata.loc[idx][metadata_column_name])
    if not os.path.isfile(image_png_path):
        Path(image_png_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imsave(image_png_path, image_png)
        logging.debug(f"Saved img in path: {image_png_path}")

    return metadata


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
                                     "image file path": "image_file_path",
                                     "ROI mask file path": "ROI_mask_file_path",
                                     "cropped image file path": "cropped_image_file_path"}, inplace=True)

        # merge all metadata
        all_metadata = pd.merge(all_metadata[columns], csv_metadata[columns],
                                on=columns, how='outer')

    all_metadata.drop_duplicates(inplace=True)

    return all_metadata
