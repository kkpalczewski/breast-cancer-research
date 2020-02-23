import torch
from tqdm import tqdm

from utils.dice_loss import dice_coeff
from collections import defaultdict
import numpy as np


def eval_net(net, loader, device, n_val, threshold: float = 0.5):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = defaultdict(float)

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)

            mask_pred = net(imgs)

            true_masks = true_masks.to(device=device, dtype=mask_pred.dtype)

            for true_mask, pred in zip(true_masks, mask_pred):
                # TODO: avoid cpu()
                pred = torch.where(pred < threshold, torch.ones(1, device=device), torch.zeros(1, device=device))
                tot["benign"] += dice_coeff(pred[0], true_mask[0].squeeze(dim=1)).item()
                tot["malignant"] += dice_coeff(pred[1], true_mask[1].squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])

    tot["benign"] = tot["benign"] / n_val
    tot["malignant"] = tot["malignant"] / n_val

    return tot
