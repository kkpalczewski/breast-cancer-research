import pandas as pd
import torch

from breast_cancer_research.unet.model_facade import BreastCancerSegmentator
from breast_cancer_research.unet.unet_model import UNet
from breast_cancer_research.utils.dataset import UnetDataset


def main():
    model_params = dict(
        n_channels=1,
        n_classes=2
    )
    model = UNet

    root = "/home/kpalczew/CBIS_DDSM_2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scale = 0.05

    # get dataset
    metadata_path_train = "/home/kpalczew/breast-cancer-research/data/labels_with_masks/train_set_with_masks.csv"
    metadata_path_val = "/home/kpalczew/breast-cancer-research/data/labels_with_masks/train_set_with_masks.csv"

    metadata_train = pd.read_csv(metadata_path_train)
    # TODO: fix it :)
    metadata_train = metadata_train[metadata_train["ROI malignant path"].str.split("/", expand=True)[0].astype(
        str) == "Mass-Training_P_01194_LEFT_MLO"]
    dataset_train = UnetDataset(metadata_train, root, scale=scale)

    pretrained_model_path = "/home/kpalczew/breast-cancer-research/scripts/checkpoints/CP_epoch100.pth"
    #pretrained_model_path = None

    UnetModel = BreastCancerSegmentator(model=model, model_params=model_params, device=device,
                                        pretrained_model_path=pretrained_model_path)
    all_pred_masks = UnetModel.predict(dataset_test=dataset_train)

    print("end")

if __name__ == "__main__":
    main()