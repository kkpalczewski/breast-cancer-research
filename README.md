# Computer-aid diagnosis (CADx) and detection (CADe) for breast cancer in mammography with combined UNet and ResNet

## Introduction
This project aims to develop an approach which could improve current CADx, CADe performance in breast cancer assesment using mammogrpahy studies.

## Setting up environment
1. Create new virtaulenv:
```
$ virtualenv -p python3 breast_cancer_research_venv
```
2. Source environment:
```
$ source breast_cancer_rsearch_venv/bin/activate
```
3. Install required packages:
```
$ pip install -r requirements.txt
```
4. Setup internal sources (*for debugging advised in development mode*):
```
$ pip install -e .
```

## Data

### CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
CBIS-DDSM is a database of 2,620 scanned film mammography studies. 
It contains benign, and malignant cases with verified pathology information. 
The scale of the database along with ground truth validation makes the DDSM a useful tool in the development 
and testing of decision support systems.

### Data preparation
1. Install NBIA Data Retriever and search CBIS-DDSM in Data Portal. 
Instruction for this task is available in [Cancer Imaging Archive website](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images).
2. Download [CBIS-DDSM data](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) using NBIA Data Retriever.
It is important to download data into "Classic Directory Name", since CBIS-DDSM metadata are prepared to work in this folder tree type.
3. Change data generation config `/config/data_generation_config.json` so it fits your directories.
Config description:
-  `downloaded_images_root`: dir to downloaded images (by default ending `CBIS-DDSM`)
- `downloaded_metadata_dir`: dir to downloaded metadata (`.csv` files)
- `transform_dicom_to_png`: boolean flag which indicates if images would be 
- `images_out_dir`: dir where images in png format should be stored
- `metadata_out_dir`: dir where new metadata would be stored (if `None` they would be stored in `downloaded_metadata_dir`)
- `train_val_test_split`: list of floats, which should add up to 1,
- `default_split`: prepare train, val, test split for enabling comparing results between researchers
4. Run data convertion and metadata preparation scirpt (*it could take several hours (i.e. all files have to be converted to .png and single findings would be merged*):
```
$ python scripts/instance_2_semantic_data.py --data_config_path configs/data_generation_config.csv
```
Arguments:
- `--data_config_path`: config for data convertion and metadata preparation

### Scripts
#### Unet training:
```
python scripts/train_unet.py --unet_config_path configs/unet_config_train.json 
--train_metadata_path 
```
Arguments:
- `--unet_config_path`: unet configuration file
- `--train_metadata_path`: path to train metadata
- `--val_metadata_path`: path to validation metadata
- `--cross_validation`: boolean which indicateas whethear cross-validation should be used.
If set, params which would be used for cross validation are taken from `--train_metadata_path`

#### ResNet training:

## Acknowledgments
This work was significantly influences by recent [New York University research](https://arxiv.org/pdf/1903.08297.pdf).
UNet implementation is based on [this repository](https://github.com/milesial/Pytorch-UNet).
