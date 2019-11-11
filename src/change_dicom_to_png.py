import os
import subprocess
import shutil
import time
import argparse

def create_dir(dirName):
    """
    Script for creating new directory
    :param dirName: Name of new directory to create
    :return: Name of created directory or None, if directory already exists
    """
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
        return dirName
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
        return None

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
    Script for copied .dcm files to .png with preserved file tree
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
            create_dir(new_path)

        if i % 100 == 0:
            print("Iteration: {}".format(i))

def main():
    """
    Create PNG directories from cmd
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('source_folder', help='source_folder')
    parser.add_argument('destination_folder', help='destination_folder')
    args = parser.parse_args()
    change_dicom_to_png(args.source_folder, args.destination_folder)

if __name__=="__main__":
    main()