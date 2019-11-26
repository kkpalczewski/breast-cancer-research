import os
import shutil
import imageio
import pandas as pd
import argparse

def simplify_folders(data_folder):
    """
    Delete unnecessary nested folders and simplify filenames
    :param data_folder: folder with .png images
    :return: None
    """
    folder_list = os.listdir(data_folder)
    i = 0
    for f in folder_list:
        folder_path = os.path.join(data_folder, f)
        file_path_list = []
        dir_list = []

        for root, dirs, files in os.walk(folder_path):
            #list all files, to move them to simplify folder tree
            for file in files:
                file_path_list.append(os.path.join(root, file))
            # list all directories, to delete them after moving .png files
            for dir in dirs:
                dir_list.append(os.path.join(root, dir))

        # this case apply to folder with full mammogram
        if len(file_path_list) == 1 and "full" in file_path_list[0]:
            dest_path = os.path.join(folder_path, "full_image.png")
            shutil.move(file_path_list[0], dest_path)

        # this case apply to specific findings (ROI + cropped image)
        elif len(file_path_list) == 2 and ("ROI" in file_path_list[0] or "cropped" in file_path_list[0]):
            img_shape_0 = imageio.imread(file_path_list[0]).shape
            img_shape_1 = imageio.imread(file_path_list[1]).shape

            # size of cropped image has to be smaller than ROI image
            if img_shape_0[0] > img_shape_1[0]:
                source_path_mask = file_path_list[0]
                source_path_cropped = file_path_list[1]
            elif img_shape_0[0] < img_shape_1[0]:
                source_path_mask = file_path_list[1]
                source_path_cropped = file_path_list[0]
            else:
                raise Exception("Size of cropped image and ROI is the same. Chceck files: {} and {}".format(file_path_list[0], file_path_list[1]))
            dest_path_mask = os.path.join(folder_path, "mask.png")
            dest_path_cropped = os.path.join(folder_path, "cropped_image.png")
            shutil.move(source_path_mask, dest_path_mask)
            shutil.move(source_path_cropped, dest_path_cropped)
        else:
            raise Exception("Unknown content in folder: {}". format(folder_path))

        #delete unnecessary folder tree
        for dir in dir_list:
            shutil.rmtree(dir, ignore_errors=True)

        i += 1
        if i % 100:
            print("Moved {} files".format(i))

def adjust_metadata(data_folder, source_metadata, destination_metadata):
    """
    Adjust metadata for simplified folder tree
    :param data_folder: folder with .png images
    :param source_metadata: source csv file
    :param destination_metadata: new csv file
    :return: None
    """
    metadata =pd.read_csv(source_metadata, index_col=0)

    for idx, record in metadata.iterrows():

        full_img_path = os.path.join(data_folder, record["image file path"].split("/")[0], "full_image.png")
        roi_path = os.path.join(data_folder, record["ROI mask file path"].split("/")[0], "mask.png")
        cropped_path = os.path.join(data_folder, record["cropped image file path"].split("/")[0], "cropped_image.png")

        #check image file path
        if os.path.isfile(full_img_path):
            metadata.at[idx, "image file path"] = full_img_path
        else:
            raise Exception("File {} doesn't exist".format(full_img_path))

        #check roi path
        if os.path.isfile(roi_path):
            metadata.at[idx, "ROI mask file path"] = roi_path
        else:
            raise Exception("ERROR: file {} doesn't exist".format(roi_path))

        #check cropped path
        if os.path.isfile(cropped_path):
            metadata.at[idx, "cropped image file path"] = cropped_path
        else:
            raise Exception("ERROR: file {} doesn't exist".format(cropped_path))

        if idx % 100 == 0:
            print("Changed {} metadata records".format(idx))

    try:
        metadata.to_csv(destination_metadata)
    except Exception as e:
        raise Exception("Unable to save {}, dute to error: {}".format(destination_metadata, e))

def main():
    """
    Simplify directories and adjust metadata from cmd
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', help='source_folder')
    parser.add_argument('destination_folder', help='destination_folder')
    parser.add_argument('source_metadata_folder', help='source metadata folder (default:None)', const=None)
    parser.add_argument('destination_metadata_folder', help='destination metadata folder (default:None)', const=None)

    args = parser.parse_args()

    simplify_folders(args.data_folders)

    if args.source_metadata_folder != None and args.destination_metadata_folder != None:
        source_metadata = [
            os.path.join(args.source_metadata_folder, "calc_case_description_test_set.csv"),
            os.path.join(args.source_metadata_folder, "calc_case_description_train_set.csv"),
            os.path.join(args.source_metadata_folder, "mass_case_description_test_set.csv"),
            os.path.join(args.source_metadata_folder, "mass_case_description_train_set.csv"),
        ]
        destination_metadata = [
            os.path.join(args.destination_metadata_folder, "calc_case_description_test_set.csv"),
            os.path.join(args.destination_metadata_folder, "calc_case_description_train_set.csv"),
            os.path.join(args.destination_metadata_folder, "mass_case_description_test_set.csv"),
            os.path.join(args.destination_metadata_folder, "mass_case_description_train_set.csv"),
        ]

        for s, d in zip(source_metadata, destination_metadata):
            adjust_metadata(args.data_folder, s, d)


if __name__=="__main__":
    main()