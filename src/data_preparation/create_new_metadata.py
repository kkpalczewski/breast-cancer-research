import pandas as pd
import os
import shutil
import imageio

def change_spaces_to_underline_in_folder(source_folder):
    i = 0
    for root, dirs, files in os.walk(source_folder):
        # create folder
        for file in dirs:
            source_folder_name = os.path.join(root,file)
            if " " in source_folder_name:
                destination_folder_name = source_folder_name.replace(" ", "_")
                os.rename(source_folder_name, destination_folder_name)
                i += 1

        if i % 100 == 0:
            print("Iteration: {}".format(i))

def simplify_folders(data_folder):

    folder_list = os.listdir(data_folder)
    i = 0
    for f in folder_list:
        folder_path = os.path.join(data_folder, f)
        file_path_list = []
        dir_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path_list.append(os.path.join(root, file))
            for dir in dirs:
                dir_list.append(os.path.join(root, dir))

        if  len(file_path_list) == 1 and "full" in file_path_list[0]:
            dest_path = os.path.join(folder_path, "full_image.png")
            shutil.move(file_path_list[0], dest_path)
        elif len(file_path_list) == 2 and ("ROI" in file_path_list[0] or "cropped" in file_path_list[0]):
            img_shape_0 = imageio.imread(file_path_list[0]).shape
            img_shape_1 = imageio.imread(file_path_list[1]).shape

            if img_shape_0[0] > img_shape_1[0]:
                source_path_mask = file_path_list[0]
                source_path_cropped = file_path_list[1]
            elif img_shape_0[0] == img_shape_1[0]:
                return file_path_list
            else:
                source_path_mask = file_path_list[1]
                source_path_cropped = file_path_list[0]
            dest_path_mask = os.path.join(folder_path, "mask.png")
            dest_path_cropped = os.path.join(folder_path, "cropped_image.png")
            shutil.move(source_path_mask, dest_path_mask)
            shutil.move(source_path_cropped, dest_path_cropped)
        elif len(file_path_list) == 2:
            continue
        else:
            return f

        for dir in dir_list:
            shutil.rmtree(dir, ignore_errors=True)
        i += 1
        print("Iteration: {}".format(i))

def adjust_metadata(data_folder, source_metadata, destination_metadata):
    metadata =pd.read_csv(source_metadata, index_col=0)

    for idx, record in metadata.iterrows():

        full_img_path = os.path.join(data_folder, record["image file path"].split("/")[0], "full_image.png")
        roi_path = os.path.join(data_folder, record["ROI mask file path"].split("/")[0], "mask.png")
        cropped_path = os.path.join(data_folder, record["cropped image file path"].split("/")[0], "cropped_image.png")

        #check image file path
        if os.path.isfile(full_img_path):
            metadata.at[idx, "image file path"] = full_img_path
        else:
            print("ERROR: file {} doesn't exist".format(full_img_path))
        #check roi path
        if os.path.isfile(roi_path):
            metadata.at[idx, "ROI mask file path"] = roi_path
        else:
            print("ERROR: file {} doesn't exist".format(roi_path))
        #check cropped path
        if os.path.isfile(cropped_path):
            metadata.at[idx, "cropped image file path"] = cropped_path
        else:
            print("ERROR: file {} doesn't exist".format(cropped_path))

        if idx % 100 == 0:
            print("Iteration {}".format(idx))

    try:
        metadata.to_csv(destination_metadata)
    except Exception as e:
        print("[FAILURE] Unable to save {}, dute to error: {}".format(destination_metadata, e))


if __name__=="__main__":
    source_metadata_calc_test = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/calc_case_description_test_set_PNG.csv"
    destination_metadata_calc_test = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/calc_case_description_test_set_processed.csv"
    source_metadata_calc_train = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/calc_case_description_train_set_PNG.csv"
    destination_metadata_calc_train = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/calc_case_description_train_set_processed.csv"
    source_metadata_mass_test = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/mass_case_description_test_set_PNG.csv"
    destination_metadata_mass_test = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/mass_case_description_test_set_processed.csv"
    source_metadata_mass_train = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/mass_case_description_train_set_PNG.csv"
    destination_metadata_mass_train = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/mass_case_description_train_set_processed.csv"

    source_metadata = [source_metadata_calc_train, source_metadata_calc_test, source_metadata_mass_train, source_metadata_mass_test]
    destination_metadata = [destination_metadata_calc_train, destination_metadata_calc_test, destination_metadata_mass_train, destination_metadata_mass_test]

    for source, destination in zip(source_metadata, destination_metadata):
        adjust_metadata("/media/krzysztof/ADATA_HD700/Breast_cancer_PNG/CBIS-DDSM", source, destination)