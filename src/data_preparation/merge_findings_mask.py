import pandas as pd
import os
import imageio
import numpy as np
import PIL
import argparse

def merge_masks(metadata_path, old_metadata, data_folder, new_metadata_path):
    """
    Script which concatenate makss from all findings for specific case.
    This script also checks size of image and masks assigned to this image, if they are not the same, image is deleted
    from metadata
    :param metadata_path: path to metadata in .csv
    :param old_metadata: path to metadata for single findings - it is required to match masks with images
    :param data_folder: folder to stored data
    :param new_metadata_path: name of new metadata to save
    :return: None
    """
    metadata = pd.read_csv(metadata_path)
    new_metadata = metadata.copy()
    old_metadata = pd.read_csv(old_metadata)

    all_directories = os.listdir(data_folder)

    if "ROI benign path" in metadata.columns:
        print("ROI bening already exists")
    else:
        new_metadata["ROI benign path"] = None
    if "ROI malignent path" in metadata.columns:
        print("ROI malignant already exists")
    else:
        new_metadata["ROI malignant path"] = None

    for idx, record in metadata.iterrows():
        #this variable indicates if size of image and its mask are the same, if not image is deleted from metadata
        deleted=False
        image_file_path = os.path.join(data_folder, record["image file path"])
        img_size = imageio.imread(image_file_path).shape
        ROI_benign = np.zeros(img_size).astype(int)
        ROI_malignant = np.zeros(img_size).astype(int)

        # findings for an image is searched based on primary key, which consist of: "patient_id"+"image view"+"left or right breast"
        findings_directory = [x for x in all_directories if (record["patient_id"] in x) and (record["image view"] in x) and (record["left or right breast"] in x)]

        for finding in findings_directory:
            if not "mask.png" in os.listdir(os.path.join(data_folder, finding)):
                continue
            roi_mask_path = os.path.join(finding, "mask.png")
            roi_record = old_metadata[old_metadata["ROI mask file path"]==roi_mask_path]
            roi_mask = np.array(imageio.imread(os.path.join(data_folder, roi_mask_path))).astype(int)
            roi_record = roi_record.iloc[0]

            if img_size != roi_mask.shape:
                new_metadata.drop(new_metadata.index[[idx]], inplace=True)
                deleted = True
                print("Deleted: {}".format(record["image file path"]))
                break

            if roi_record["pathology"] == "MALIGNANT":
                ROI_malignant = np.bitwise_or(ROI_malignant, roi_mask)
            else:
                ROI_benign = np.bitwise_or(ROI_benign, roi_mask)

        if deleted:
            continue

        roi_benign_path = os.path.join(data_folder, record["image file path"].split("/")[0], "full_mask_benign.png")
        roi_malignant_path = os.path.join(data_folder, record["image file path"].split("/")[0], "full_mask_malignant.png")

        ROI_malignant = ROI_malignant.astype(np.uint8)
        ROI_benign = ROI_benign.astype(np.uint8)

        ROI_malignant = PIL.Image.fromarray(ROI_malignant, mode="L")
        ROI_benign = PIL.Image.fromarray(ROI_benign, mode="L")

        try:
            ROI_malignant.save(roi_malignant_path, "PNG")
            ROI_benign.save(roi_benign_path, "PNG")
        except Exception as e:
            print("Exception while saving {}, {}: {}".format(roi_malignant_path, roi_benign_path, e))

        new_metadata.at[idx, "ROI malignant path"] = os.path.join(record["image file path"].split("/")[0], "full_mask_malignant.png")
        new_metadata.at[idx, "ROI benign path"] = os.path.join(record["image file path"].split("/")[0], "full_mask_benign.png")

        if idx % 100 == 0:
            print("Iteration: {}".format(idx))
            new_metadata.to_csv(new_metadata_path)

    new_metadata.to_csv(new_metadata_path)

def main():
    """
    Merge masks from findings
    :return: None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('metadata_path', help='path to metadata (.csv file)')
    parser.add_argument('old_metadata', help='path to metadata for single findings - it is required to match masks with images')
    parser.add_argument('data_folder', help='folder to stored data')
    parser.add_argument('new_metadata_path', help='name of new metadata to save')

    args = parser.parse_args()

    merge_masks(args.metadata_path, args.old_metadata, args.data_folder, args.new_metadata_path)

if __name__=="__main__":
    main()

    # old_metadata_path = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/full_dataset.csv"
    # data_folder = "/media/krzysztof/ADATA_HD700/Breast_cancer_PNG/CBIS-DDSM"
    #
    # test_metadata_path = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels_for_whole_cases/test_set.csv"
    # val_metadata_path = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels_for_whole_cases/validation_set.csv"
    # train_metadata_path = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels_for_whole_cases/train_set.csv"
    # metadata_path = [train_metadata_path, val_metadata_path, test_metadata_path]
    #
    # test_new_metadata_path = "/breast-cancer-research/data/labels_with_masks/test_set_with_masks.csv"
    # val_new_metadata_path = "/breast-cancer-research/data/labels_with_masks/validation_set_with_masks.csv"
    # train_new_metadata_path = "/breast-cancer-research/data/labels_with_masks/train_set_with_masks.csv"
    # new_metadata_path = [train_new_metadata_path, val_new_metadata_path, test_new_metadata_path]
    #
    # for old_meta, new_meta in zip(metadata_path, new_metadata_path):
    #     merge_masks(old_meta, old_metadata_path, data_folder, new_meta)
