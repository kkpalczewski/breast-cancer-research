# basic
from abc import ABC, abstractmethod
from typing import Mapping, ClassVar, Optional, Dict, Union, Any
import logging
from tqdm import tqdm
from collections import defaultdict
import numpy as np
# ML
# TODO: check it
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn
# custom
from breast_cancer_research.utils.utils import counter_global_step, mkdir
from breast_cancer_research.utils.metrics import dice_coeff


class BaseModel(ABC):
    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass


class BreastCancerSegmentator(BaseModel):
    def __init__(self, model: ClassVar, model_params: Mapping, device: torch.device = "cuda",
                 dir_checkpoint: str = "checkpoints/", cudnn_benchmark: bool = True,
                 pretrained_model_path: Optional[str] = None):
        self.model = model(**model_params)
        # load pretrained dict
        if pretrained_model_path is not None:
            self._load_pretrained_model(pretrained_model_path)

        self.device = device
        self.dir_checkpoint = dir_checkpoint
        self.writer = None
        # faster convolutions, but more memory
        cudnn.benchmark = cudnn_benchmark

        self.model.to(device=self.device)

        self._global_step = self.global_step
        self.init_global_step = self.global_step
        self._model_step = self.model_step

    @property
    def global_step(self):
        return self._train_single_batch.counter

    @property
    def model_step(self):
        return self.global_step - self.init_global_step

    def train(self, *, dataset_train: Dataset, dataset_val: Optional[Dataset] = None, epochs: int = 5,
              batch_size: int = 1, lr: float = 0.1, save_cp: bool = True):
        # TODO: check num_workers
        # TODO: check pin_memory
        # FIXME: add custom method for Dataset to get same size images

        if dataset_val is None:
            size_val = None
            dataloader_val = None
        else:
            size_val = len(dataset_val)
            dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8,
                                        pin_memory=True)

        size_train = len(dataset_train)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8,
                                      pin_memory=True)

        # check scales
        if dataset_val is not None:
            assert dataset_train.scale == dataset_val.scale, f'Different scales for train and val dataset: {dataset_train.scale} ' \
                                                             f'and {dataset_val.scale}. It would cause val metrics irrelevant.'

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {size_train}
            Validation size: {size_val}
            Checkpoints:     {save_cp}
            Images scaling:  {dataset_train.scale}
        ''')

        optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=1e-8)
        # TODO: check it out
        # criterion init
        # img_size = dataset_train.img_size
        # pos_weight = torch.ones(img_size)
        # check if this pos_weight is correct
        criterion = nn.BCEWithLogitsLoss()

        self.writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')

        for epoch in range(epochs):
            self._train_one_epoch(dataloader_train=dataloader_train,
                                  dataloader_val=dataloader_val,
                                  epoch=epoch,
                                  epochs=epochs,
                                  criterion=criterion,
                                  optimizer=optimizer)

        if save_cp:
            mkdir(self.dir_checkpoint)  # create checkpoint dir
            torch.save(self.model.state_dict(),
                       self.dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        for batch in dataloader_val:
            imgs = batch['image']
            imgs = imgs.to(device=self.device, dtype=torch.float32)
            break

        self.writer.add_graph(self.model, imgs)
        self.writer.close()

    def predict(self, *, dataset_test: Union[Dataset, DataLoader], scale_factor: float = 1, out_threshold: float = 0.5,
                batch_size: int = 1, out_layer_name: str = "sigmoid", tensorboard_verbose: bool = True) -> Dict[
        str, Any]:
        """
        Predict mask
        """

        if out_layer_name == "sigmoid":
            out_layer = torch.nn.Sigmoid()
        else:
            raise AttributeError("Not implemented out layer parameter: {}".format(out_layer_name))

        # TODO: change dataset_test for imgs?
        all_images = []
        all_pred_masks_benign = []
        all_pred_masks_malignant = []
        all_truth_masks_benign = []
        all_truth_masks_malignant = []

        with torch.no_grad():
            if isinstance(dataset_test, Dataset) or isinstance(dataset_test, DataLoader):
                if not isinstance(dataset_test, DataLoader):
                    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True)
                else:
                    dataloader_test = dataset_test
                for batch in dataloader_test:
                    imgs = batch['image']
                    true_masks = batch['mask']
                    imgs_tensor = imgs.to(device=self.device, dtype=torch.float32)
                    mask_pred = self.model(imgs_tensor)
                    all_images.append(imgs)

                    gt_shape = true_masks.shape
                    all_truth_masks_benign.append(true_masks[0][0].reshape(1, 1, gt_shape[2], gt_shape[3]))
                    all_truth_masks_malignant.append(true_masks[0][1].reshape(1, 1, gt_shape[2], gt_shape[3]))

                    pred_shape = mask_pred.shape
                    all_pred_masks_benign.append(out_layer(mask_pred[0][0]).reshape(1, 1, pred_shape[2], pred_shape[3]))
                    all_pred_masks_malignant.append(
                        out_layer(mask_pred[0][1]).reshape(1, 1, pred_shape[2], pred_shape[3]))

        prediction_dict = dict(
            all_images=all_images,
            all_pred_masks_benign=all_pred_masks_benign,
            all_pred_masks_malignant=all_pred_masks_malignant,
            all_truth_masks_benign=all_truth_masks_benign,
            all_truth_masks_malignant=all_truth_masks_malignant
        )

        if tensorboard_verbose is True:
            self._check_tensorboard_writer()
            self._tensorboard_predict(prediction_dict)

        return prediction_dict

    def evaluate(self, loader: DataLoader, threshold: float = 0.5, tensorboard_verbose: bool = True):
        """Evaluation without the densecrf with the dice coefficient"""
        n_val = len(loader.dataset)

        self.model.eval()
        tot = defaultdict(float)

        with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
            for batch in loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=self.device, dtype=torch.float32)

                mask_pred = self.model(imgs)

                true_masks = true_masks.to(device=self.device, dtype=mask_pred.dtype)

                for true_mask, pred in zip(true_masks, mask_pred):
                    # TODO: avoid cpu()
                    pred = torch.where(pred < threshold, torch.ones(1, device=self.device),
                                       torch.zeros(1, device=self.device))
                    tot["benign"] += dice_coeff(pred[0], true_mask[0].squeeze(dim=1)).item()
                    tot["malignant"] += dice_coeff(pred[1], true_mask[1].squeeze(dim=1)).item()
                pbar.update(imgs.shape[0])

        tot["benign"] = tot["benign"] / n_val
        tot["malignant"] = tot["malignant"] / n_val

        if tensorboard_verbose is True:
            self._tensorboard_evaluate(tot)

        return tot

    def _train_one_epoch(self, *, dataloader_train: DataLoader, dataloader_val: Optional[DataLoader], epoch: int,
                         epochs: int, criterion, optimizer):
        # TODO: check it
        size_train = len(dataloader_train.dataset)

        self.model.train()

        epoch_loss = 0
        with tqdm(total=size_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in dataloader_train:
                imgs = batch['image']
                masks_ground_truth = batch['mask']

                assert imgs.shape[1] == self.model.n_channels, \
                    f'Network has been defined with {self.model.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=self.device, dtype=torch.float32)
                masks_ground_truth = masks_ground_truth.to(device=self.device, dtype=torch.long)

                loss = self._train_single_batch(imgs, masks_ground_truth, optimizer, criterion)

                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])

                if dataloader_val and epoch % 24 == 0:
                    self._check_tensorboard_writer()
                    self.evaluate(loader=dataloader_val, tensorboard_verbose=True)
                    self.predict(dataset_test=dataloader_val, tensorboard_verbose=True)

    @counter_global_step
    def _train_single_batch(self, imgs, masks_ground_truth, optimizer, criterion):
        masks_pred = self.model(imgs)
        masks_ground_truth = masks_ground_truth.type_as(masks_pred)  # in other case BCELoss gets error

        loss = criterion(masks_pred, masks_ground_truth)

        self.writer.add_scalar('Loss/train', loss.item(), self.model_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def _load_pretrained_model(self, pretrained_model_path):
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(pretrained_model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def _tensorboard_evaluate(self, val_score):
        logging.info('Validation Dice Coeff: {}'.format(val_score))
        self.writer.add_scalar('Dice/test/benign', val_score['benign'], self.model_step)
        self.writer.add_scalar('Dice/test/malignant', val_score['malignant'], self.model_step)

    def _tensorboard_predict(self, prediction_dict):
        self.writer.add_images('Images/original_images', prediction_dict["all_images"][0], self.model_step)
        self.writer.add_images('Images/benign/ground_truh', prediction_dict["all_truth_masks_benign"][0],
                               self.model_step)
        self.writer.add_images('Images/benign/predictions', prediction_dict["all_pred_masks_benign"][0],
                               self.model_step)
        self.writer.add_images('Images/malignant/ground_truh', prediction_dict["all_truth_masks_malignant"][0],
                               self.model_step)
        self.writer.add_images('Images/malignant/predictions', prediction_dict["all_pred_masks_malignant"][0],
                               self.model_step)

    def _check_tensorboard_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter()
