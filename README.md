# Computer-aid diagnosis (CADx) and detection (CADe) algorithm for breast cancer in mammography

### Introudction

### Prerequisites

### License

## How to run code

## Data

### CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
This dataset is available for download [here](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM). CBIS-DDSM is a database of 2,620 scanned film mammography studies. It contains normal, benign, and malignant cases with verified pathology information. The scale of the database along with ground truth validation makes the DDSM a useful tool in the development and testing of decision support systems.

#### Initial pre-processing
Downloaded data are stored in DICOM format. To transform from DICOM to PNG format one should use:
```
python change_dicom_to_png.py <SOURCE_FOLDER> <DESTINATION_FOLDER>
```