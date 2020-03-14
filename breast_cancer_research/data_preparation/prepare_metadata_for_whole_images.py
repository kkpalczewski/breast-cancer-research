import pandas as pd
import os
import argparse

def prepare_metadata_for_whole_images(source_metadata, new_metadata_path, columns):
    """
    Prepare metadata for new model and NYU model
    :param source_metadata: source metadata csv
    :param new_metadata_path: path to new metadata folder
    :param columns: columns in new metadata
    :return: None
    """
    all_metadata = pd.DataFrame(columns=columns)

    for single_metadata in source_metadata:
        csv_metadata = pd.read_csv(single_metadata, index_col=0)

        #column name breast density is not the same in all metadata
        if "breast density" in csv_metadata.columns:
            csv_metadata.rename(columns={"breast density": "breast_density"}, inplace=True)

        #merge all metadata
        all_metadata = pd.merge(all_metadata[columns], csv_metadata[columns],
                                on=columns, how='outer')

    all_metadata.drop_duplicates(inplace=True)

    #Change pathology names to int
    all_metadata['pathology'].replace(['BENIGN', 'BENIGN_WITHOUT_CALLBACK', 'MALIGNANT'],[0, 0, 1], inplace=True)
    tmp_metadata = all_metadata.copy()
    all_metadata = all_metadata.groupby(['patient_id', 'image view', 'left or right breast'], as_index=False)['subtlety', 'pathology'].max()

    tmp_columns = columns
    for col in ['subtlety', 'pathology']:
        tmp_columns.remove(col)

    all_metadata = all_metadata.merge(tmp_metadata[tmp_columns], how='left', on=['patient_id', 'image view', 'left or right breast'])
    all_metadata.drop_duplicates(subset=['patient_id', 'image view', 'left or right breast'],inplace=True)

    all_metadata.sort_values(by='patient_id', inplace=True)

    #TODO: check if all files are in metadata

    #all_metadata = all_metadata[columns]

    new_metadata_file = os.path.join(new_metadata_path, "metadata_for_whole_images.csv")

    try:
        all_metadata.to_csv(new_metadata_file)
    except Exception as e:
        raise Exception("New metadata connot be saved in: {}, due to error: {}",format(new_metadata_file, e))


def main():
    """
    Create new metadata for developed and NYU model
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('source_metadata_folder', help='source metadata folder (default:None)')
    parser.add_argument('destination_metadata_folder', help='destination metadata folder (default:None)')
    parser.add_argument('columns', help='columns choosed to remain in the metadata', const=['patient_id', 'breast_density', 'image view', 'left or right breast',
            'pathology', 'image file path', 'subtlety'])

    args = parser.parse_args()

    required_columns = ['patient_id', 'breast_density', 'image view', 'left or right breast',
            'pathology', 'image file path', 'subtlety']
    for column in required_columns:
        if not column in args.columns:
            raise Exception("Required column {} is not among specified columns".format(column))

    source_metadata = [
        os.path.join(args.source_metadata_folder, "calc_case_description_test_set.csv"),
        os.path.join(args.source_metadata_folder, "calc_case_description_train_set.csv"),
        os.path.join(args.source_metadata_folder, "mass_case_description_test_set.csv"),
        os.path.join(args.source_metadata_folder, "mass_case_description_train_set.csv"),
    ]

    prepare_metadata_for_whole_images(source_metadata, args.destination_metadata_folder, args.columns)


if __name__=="__main__":
    main()