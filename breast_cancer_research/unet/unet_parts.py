""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [int(diffX / 2), int(diffX - diffX / 2),
                        int(diffY / 2), int(diffY - diffY / 2)])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BinaryDiceLoss(nn.Module):
    def __init__(self, activation_name: str = "softmax", weights: List[float] = (1, 1, 1), device: str = "cuda",
                 reduction: str = "mean"):
        super(BinaryDiceLoss, self).__init__()

        self.activation_name = activation_name
        self.device = device

        if activation_name == "softmax":
            self.activation = nn.Softmax(dim=0)
        else:
            raise NotImplementedError(f"Activation function {activation_name} not implemented")

        # get weights
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=device, dtype=torch.float)
        weights = torch.nn.functional.normalize(weights, dim=0)
        self.weights = weights

        # get reduction
        self.reduction = reduction

    def forward(self, preds, targets):

        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self.device)

        assert targets.shape == preds.shape, f"Target's shape {targets.shape} and pred's shape {preds.shape} are not equal"
        assert targets.device == preds.device == self.weights.device, f"Devices for tensor " \
                                                                      f"targets: {targets.device}, " \
                                                                      f"preds: {preds.device}, " \
                                                                      f"weights: {self.weights.device} are not the same "
        assert len(targets) == 1, f"Check targets shape"

        targets = targets[0]
        preds = preds[0]

        assert len(targets) == len(
            self.weights), f"Weights' len {len(self.weights)} anr target's len {len(targets)} are not equal"

        dice_loss_with_weights = torch.zeros(1, dtype=torch.float, device=preds.device)

        # activation func
        preds = self.activation(preds)

        for target, pred, weight in zip(targets, preds, self.weights):
            dice_loss = self._single_dice_loss(target, pred)
            dice_loss_with_weights += (torch.ones(1, device=self.device) - dice_loss) * weight

        if self.reduction == "mean":
            reduced_dice_loss = torch.sum(dice_loss_with_weights)
        elif self.reduction == "sum":
            reduced_dice_loss = torch.sum(dice_loss_with_weights)
        else:
            raise NotImplementedError(f"Reduction function {self.reduction} not implemented")

        return reduced_dice_loss

    def _single_dice_loss(self, target, pred):
        target = target.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        target_pred_product = torch.sum(target * pred)
        target_sum = torch.sum(target)
        pred_sum = torch.sum(pred)

        single_dice_loss = (2 * target_pred_product) / (target_sum + pred_sum)

        return single_dice_loss
