import argparse
import pandas as pd
import os
import cv2
from breast_cancer_research.unet.unet_dataset import preprocess_single_img
from breast_cancer_research.utils.utils import mkdir
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Preprocess data (flip and resize)")
    parser.add_argument('--root_dir', help='source folder of images')
    parser.add_argument('--metadata_path', help='path to metadata in .cscd ny       v format')
    parser.add_argument('--out_dir', help='path where to store processed images')
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_path)

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        orientation = row['left or right breast']

        _preprocess_from_path(args.root_dir, row['image file path'], orientation, args.out_dir)
        _preprocess_from_path(args.root_dir, row['ROI malignant path'], orientation, args.out_dir)
        _preprocess_from_path(args.root_dir, row['ROI benign path'], orientation, args.out_dir)


def _preprocess_from_path(root, relative_path, orientation, out_dir):
    img_path = os.path.join(root, relative_path)

    img = cv2.imread(img_path, -cv2.IMREAD_ANYDEPTH)
    preprocessed_img = preprocess_single_img(img, orientation=orientation)

    img_out_path = os.path.join(out_dir, relative_path)

    mkdir(os.path.dirname(img_out_path))
    cv2.imwrite(img_out_path, preprocessed_img)


if __name__ == "__main__":
    main()
