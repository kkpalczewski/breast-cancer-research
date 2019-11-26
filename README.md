# Computer-aid diagnosis (CADx) and detection (CADe) algorithm for breast cancer in mammography

### Introduction

### Prerequisites

### License

## How to run code

## Data

### CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
This dataset is available for download [here](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM). CBIS-DDSM is a database of 2,620 scanned film mammography studies. It contains normal, benign, and malignant cases with verified pathology information. The scale of the database along with ground truth validation makes the DDSM a useful tool in the development and testing of decision support systems.

#### Initial pre-processing
1. To download CBIS-DDSM data one has to install NBIA Data Retriever and search CBIS-DDSM in Data Portal. Instruction for this task is available in [Cancer Imaging Archive website](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)

2. Downloaded data are stored in DICOM format. To transform from DICOM to PNG format one should use:
```
python /src/data_preparation/change_dicom_to_png.py <SOURCE_DICOM_DATA_FOLDER> <DESTINATION_PNG_DATA_FOLDER> <SOURCE_DICOM_METADATA> <DESTINATION_PNG_METADATA>
```
3. Metadata points to non-existing directories. There are folders which are unnessecerly nested despite they contains only one directory. Additionally directories have spaces in names, which makes it incovenient to work with and names of files are not meaningful (00000.png and 00000.png). All those issues could be fixed by running:
```
python /src/data_preparation/simplify_folder_paths_and_filenames.py <DATA_FOLDER> <SOURCE_METADATA> <DESTINATION_METADATA>
```
