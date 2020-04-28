import subprocess
import shutil
import time
import argparse
import pandas as pd
import os
from breast_cancer_research.utils.utils import mkdir

def convert_dicom_to_png(source_file, dest_file):
    """
    Convert DICOM to PNG images, using shell script.
    DICOM file is converted to PNG, placed in the same folder as DICOM and then copied to destination folder
    :param source_file: source .dcm file
    :param dest_file: destination of .png file
    :return: place of current .png file
    """
    subprocess.call(["mogrify", "-format", "png", "\"{}\"".format(source_file)])
    source_file = source_file.split(".dcm")[0]+".png"
    try:
        shutil.move(source_file, dest_file)
        print("Moved from: {}, to: {}".format(source_file, dest_file))
        return dest_file
    except:
        print("FAILED TO MOVE from: {}, to: {}".format(source_file, dest_file))
        time.sleep(10)
        return source_file

def change_dicom_to_png(source_folder, dest_folder):
    """
    Copy .dcm files to .png with preserved file tree
    :param source_folder: source folder with .dcm images
    :param dest_folder: dest folder with .png images
    :return: None
    """
    i = 0
    for root, dirs, files in os.walk(source_folder):
        #create folder
        for file in files:
            if file.endswith('.dcm'):
                dest_file = os.path.join(dest_folder, root.split(source_folder + "/")[-1], file).split(".dcm")[0] + ".png"
                source_file = os.path.join(root, file)
                convert_dicom_to_png(source_file, dest_file)
        i += 1
        for dir in dirs:
            new_path = os.path.join(dest_folder, root.split(source_folder + "/")[-1], dir)
            mkdir(new_path)

        if i % 100 == 0:
            print("Iteration: {}".format(i))

def change_extension_in_csv(source_csv, destination_csv):
    """
    Change extension, from .dcm to .png in metadata
    :param source_csv: csv file with metadata with .dcm extension
    :param destination_csv: new csv file with metadata with p.ng extension
    :return: None
    """
    source_csv_file = pd.read_csv(source_csv)

    fields_to_change = ['image file path', 'cropped image file path', 'ROI mask file path']

    try:
        first_record = source_csv_file.iloc[0]
        for field in fields_to_change:
            print("Field to change: {0:30} , sample value: {1}".format(field, first_record[field]))
    except KeyError as err:
        print("Change column name, which should be converted. ERROR: {}".format(err))
        return err

    for id, row in source_csv_file.iterrows():
        for field in fields_to_change:
            source_csv_file.at[id, field] = source_csv_file.iloc[id][field].split(".dcm")[0] + ".png"

    try:
        source_csv_file.to_csv(destination_csv)
        print("[SUCCESS] New metadata saved in path: {}".format(destination_csv))
    except BaseException as err:
        print("[FAILURE] of saving {}, due to exception: {}".format(destination_csv, err))


def main():
    """
    Create PNG directories from cmd
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('source_folder', help='source_folder')
    parser.add_argument('destination_folder', help='destination_folder')
    parser.add_argument('source_metadata_folder', help='source metadata folder (default:None)', const=None)
    parser.add_argument('destination_metadata_folder', help='destination metadata folder (default:None)', const=None)

    args = parser.parse_args()

    change_dicom_to_png(args.source_folder, args.destination_folder)

    if args.source_metadata_folder != None and args.destination_metadata_folder != None:
        source_metadata = [
            os.path.join(args.source_metadata_folder, "calc_case_description_test_set.csv"),
            os.path.join(args.source_metadata_folder, "calc_case_description_train_set.csv"),
            os.path.join(args.source_metadata_folder, "mass_case_description_test_set.csv"),
            os.path.join(args.source_metadata_folder, "mass_case_description_train_set.csv"),
        ]
        destination_metadata = [
            os.path.join(args.destination_metadata_folder, "calc_case_description_test_set_PNG.csv"),
            os.path.join(args.destination_metadata_folder, "calc_case_description_train_set_PNG.csv"),
            os.path.join(args.destination_metadata_folder, "mass_case_description_test_set_PNG.csv"),
            os.path.join(args.destination_metadata_folder, "mass_case_description_train_set_PNG.csv"),
        ]

        for s, d in zip(source_metadata, destination_metadata):
            change_extension_in_csv(s, d)


if __name__=="__main__":
    main()