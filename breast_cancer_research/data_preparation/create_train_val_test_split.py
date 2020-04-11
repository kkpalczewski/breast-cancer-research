import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def train_val_test_split(path_to_metadata, destination_folder, train_size, val_size, test_size, random_state=42, verbose=False):
    """
    Script which split metadata to train, validation and test split. It preserve equal (when possible) ratio of data from classes:
    - image view
    - left or right breast
    - subtlety
    - pathology
    - breast density

    :param path_to_metadata: path to .csv metadata file
    :param destination_folder: path to destination folder where splitted metadata would be stored
    :param train_size: size of train set
    :param val_size: size of validation set
    :param test_size: size of test set
    :param random_state: radnom state
    :param verbose: verbosity of output
    :return: None
    """
    #TODO: rewrite script so that it would accept any number of columns to make redistributed split

    if train_size+val_size+test_size != 1:
        raise Exception("Ratio of train, validation and test set doesn't add up to 1")

    metadata = pd.read_csv(path_to_metadata)
    image_view_col = metadata["image view"].unique()
    left_or_right_breast_col = metadata["left or right breast"].unique()
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
                        tmp_metadata = metadata[((metadata["image view"]==image_view) &
                                                (metadata["left or right breast"]==left_or_right_breast) &
                                                (metadata["subtlety"]==subtlety) &
                                                (metadata["pathology"]==pathology) &
                                                (metadata["breast_density"]==breast_density))]["image file path"]
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
                        elif tmp_metadata.shape[0]==1:
                            tmp_train=tmp_metadata
                            train_array.extend(list(tmp_train))
                    if verbose:
                        print("Splitted data with parameters:\n"
                              "image_view: {}\n"
                              "left_or_right_breast: {}\n"
                              "subtlety: {}\n"
                              "pathology: {}\n"
                              "breast_density: {}\n".format(image_view, left_or_right_breast, subtlety, pathology, breast_density))

    #save splitted data
    metadata[metadata["image file path"].isin(train_array)].to_csv(os.path.join(destination_folder, "train_set.csv"))
    metadata[metadata["image file path"].isin(val_array)].to_csv(os.path.join(destination_folder, "validation_set.csv"))
    metadata[metadata["image file path"].isin(test_array)].to_csv(os.path.join(destination_folder, "test_set.csv"))

    all_records_size = len(train_array) + len(val_array) + len(test_array)

    print("STATISTICS:\n"
          "train size: {:.2f}%\n"
          "validation size: {:.2f}%\n"
          "test size: {:.2f}%\n".format(
        len(train_array)/all_records_size,
        len(val_array)/all_records_size,
        len(test_array)/all_records_size))

def main():
    """
    Create train, val, test split

    Returns: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_metadata', help='path to .csv file with meatdata')
    parser.add_argument('destination_folder', help='folder in which would be written splitted metadata')
    parser.add_argument('train_size', help='size of training set', const=0.6)
    parser.add_argument('val_size', help='size of validation set', const=0.2)
    parser.add_argument('test_size', help='size of test set', const=0.2)
    parser.add_argument('random_state', help='random state', const=42)
    parser.add_argument('verbose', help='verbosity', const=False)

    args = parser.parse_args()

    train_val_test_split(args.path_to_metadata, args.destination_folder, args.train_size, args.val_size, args.test_size,
                         args.random_state, args.verbose)

if __name__=="__main__":
    main()