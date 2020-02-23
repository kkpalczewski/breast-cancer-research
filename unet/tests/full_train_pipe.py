# logging & typing
import logging
# basic
import os
import sys
import pandas as pd
# ml
import torch
# custom
#TODO: check it
from torch.backends import cudnn
from torch.utils.data import Dataset


from unet.unet_model import UNet
from pipeline.train import train_net
from utils.dataset import BasicDataset


def train(net: UNet, device: torch.device, dataset_train: Dataset, dataset_val: Dataset, epochs: str = 5):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    net.to(device=device)
    # faster convolutions, but more memory
    cudnn.benchmark = True

    train_params = dict(
        net=net,
        device=device,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        epochs=epochs
    )
    try:
        train_net(**train_params)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    root = "/home/kpalczew/CBIS_DDSM_2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metadata_path_train = "/home/kpalczew/Pytorch-UNet/data/labels/train_set_with_masks.csv"
    metadata_path_val = "/home/kpalczew/Pytorch-UNet/data/labels/validation_set_with_masks.csv"

    n = 10
    #metadata_train = pd.read_csv(metadata_path_train).sample(n=10)
    metadata_train = pd.read_csv(metadata_path_train)
    metadata_train = metadata_train[metadata_train["ROI malignant path"].str.split("/", expand=True)[0].astype(str)=="Mass-Training_P_01194_LEFT_MLO"]
    print(metadata_train)

    metadata_val = pd.read_csv(metadata_path_val).sample(n=5)

    scale = 0.05
    epochs = 100
    dataset_train = BasicDataset(metadata_train, root, scale=scale)
    dataset_val = BasicDataset(metadata_train, root, scale=scale)

    torch.cuda.empty_cache()

    net = UNet(n_channels=1, n_classes=2)

    train_params = dict(net=net,
                        device=device,
                        dataset_train=dataset_train,
                        dataset_val=dataset_val,
                        epochs=epochs)
    train(**train_params)
