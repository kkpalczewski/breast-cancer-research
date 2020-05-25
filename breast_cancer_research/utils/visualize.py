import numpy as np
from breast_cancer_research.utils.utils import chw_2_hwc
import cv2
import logging
from typing import Tuple

def string_to_rgb(string: str) -> Tuple[int, int, int]:
    if string == "benign":
        color = (153, 204, 0) #bright green
    elif string == "benign_ground_truth":
        color = (255, 102, 0) #light orange
    elif string == "malignant":
        color = (0, 255, 0) # green
    elif string == "malignant_ground_truth":
        color = (255, 0, 0) # red
    else:
        color = [int(abs(hash(string * idx))) % 256 for idx in range(1, 4)]
        logging.debug(f"String: {string} not specified in color mappings. Random color applied.")

    return color


def tensor_2_rgb(image, color=(255, 255, 255)):
    color = np.array(color)
    if not isinstance(image, np.ndarray):
        image = image.numpy()
    uint8_image = image * 255
    hwc_img = chw_2_hwc(uint8_image)
    rgb_list_img = [np.dot(channel, color).astype(np.uint8) for channel, color in zip((hwc_img,) * 3, color / 255)]
    rgb_img = np.concatenate(rgb_list_img, axis=2)
    return rgb_img


def overlay_mask(tensor_img, tensor_masks, classnames, alpha: float = 0.5):
    tensor_img = tensor_img.clone()
    rgb_img = tensor_2_rgb(tensor_img)
    for mask, classname in zip(tensor_masks, classnames):
        tensor_mask = mask.reshape(tuple(np.append([1], np.array(mask.shape))))
        mask_color = string_to_rgb(classname)
        rgb_mask = tensor_2_rgb(tensor_mask, color=mask_color)
        tmp_masked_img = cv2.bitwise_or(rgb_mask, cv2.bitwise_and(rgb_img, rgb_img, mask=np.bitwise_not(
            rgb_mask[:, :, 0].astype(bool)).astype(np.uint8)))
        rgb_img = cv2.addWeighted(rgb_img, alpha, tmp_masked_img, 1 - alpha, 0)
    return rgb_img
