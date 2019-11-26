import pandas as pd
import os


def make_complete_metadata(calc_mass_path_list, data_folder, new_metadata_path):
    all_metadata = pd.DataFrame(columns=['patient_id', 'breast_density', 'image view', 'left or right breast',
            'pathology', 'image file path', 'subtlety'])

    for single_metadata in calc_mass_path_list:
        csv_metadata = pd.read_csv(single_metadata, index_col=0)

        if "breast density" in csv_metadata.columns:
            csv_metadata.rename(columns={"breast density": "breast_density"}, inplace=True)

        all_metadata = pd.merge(all_metadata[['patient_id', 'breast_density', 'image view', 'left or right breast',
                'pathology', 'image file path', 'subtlety']], csv_metadata[['patient_id', 'breast_density', 'image view', 'left or right breast',
               'pathology', 'image file path', 'subtlety']], on=['patient_id', 'breast_density', 'image view','left or right breast', 'pathology',
               'image file path', 'subtlety'], how='outer')

    all_metadata.drop_duplicates(inplace=True)

    #Change pathology names to int
    all_metadata['pathology'].replace(['BENIGN', 'BENIGN_WITHOUT_CALLBACK', 'MALIGNANT'],[0, 0, 1], inplace=True)

    all_metadata = all_metadata.groupby(['patient_id', 'breast_density', 'image view', 'left or right breast',
            'image file path'], as_index=False)['pathology', 'subtlety'].max()

    all_metadata.sort_values(by='patient_id', inplace=True)

    i = 0

    data_folder_list = sorted(os.listdir(data_folder))

    for folder in data_folder_list:
        try:
            last = eval(folder[-1])
        except NameError:
            last = None

        if not isinstance(last, int):
            full_img_path = os.path.join(data_folder, folder, "full_image.png")
            if full_img_path not in all_metadata['image file path'].values:
                new_record = pd.DataFrame({"patient_id": ["P_"+full_img_path.split("/")[-2].split("_")[2]],
                                           "breast density": [-1],
                                           "image view": [full_img_path.split("/")[-2].split("_")[-1]],
                                           "left or right breast": [full_img_path.split("/")[-2].split("_")[-2]],
                                           "image file path": [full_img_path],
                                           "pathology": [0]})
                all_metadata = all_metadata.append(new_record, ignore_index=True)
                i+=1
                print("Added: {}".format(full_img_path))

    columns = ['patient_id',  'image view', 'left or right breast', 'breast_density', 'subtlety',
                'pathology', 'image file path']

    all_metadata = all_metadata[columns]

    all_metadata.to_csv(new_metadata_path)

def make_split(x):
    separator = "/"
    return separator.join([x.split("/")[-2], x.split("/")[-1]])

def change_master_file_path(metadata_path, new_metadata_path):
    metadata = pd.read_csv(metadata_path, index_col=0)
    metadata["image file path"] = metadata.apply(lambda x: make_split(x["image file path"]), axis=1)
    metadata.to_csv(new_metadata_path, index=False)


if __name__=="__main__":
    os.chdir("/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research")
    calc_test_path = "./data/labels/calc_case_description_test_set_processed.csv"
    calc_train_path = "./data/labels/calc_case_description_train_set_processed.csv"
    mass_test_path = "./data/labels/mass_case_description_test_set_processed.csv"
    mass_train_path = "./data/labels/mass_case_description_train_set_processed.csv"
    data_folder = "/media/krzysztof/ADATA_HD700/Breast_cancer_PNG/CBIS-DDSM/"
    new_metadata_path = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/full_metadata.csv"

    #calc_mass_paht_list = [calc_test_path, calc_train_path, mass_test_path, mass_train_path]
    #make_complete_metadata(calc_mass_paht_list, data_folder, new_metadat_path)

    processed_metadata = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/full_metadata_without_master_folder.csv"

    change_master_file_path(new_metadata_path, processed_metadata)