# basic
from typing import Mapping, ClassVar, Optional, Dict, Union, Any
import logging
from tqdm import tqdm
from collections import defaultdict
import os
# ML
import numpy as np
# TODO: check it
from torch.backends import cudnn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
# custom
from breast_cancer_research.utils.utils import elapsed_time
from breast_cancer_research.utils.utils import counter_global_step, mkdir, get_converted_timestamp
import breast_cancer_research.base.base_model
from breast_cancer_research.unet.unet_tensorboard import UnetSummaryWriter
from time import time
# basic
from typing import Dict
from collections import defaultdict
import time
import logging
# ML
import torch
import pandas as pd
# custom
from breast_cancer_research.unet.unet_model import UNet
from breast_cancer_research.unet.unet_dataset import UnetDataset


class BreastCancerSegmentator(breast_cancer_research.base.base_model.BaseModel):
    def __init__(self, model_class: ClassVar,
                 model_params: Mapping,
                 device: torch.device = "cuda",
                 cp_dir: str = "checkpoints/",
                 run_dir: str = "runs/",
                 summary_mode: str = "tensorboard",
                 cudnn_benchmark: bool = True,
                 pretrained_model_path: Optional[str] = None):
        # device
        self.device = device

        # init model
        self._init_model(model_class=model_class,
                         model_params=model_params,
                         pretrained_model_path=pretrained_model_path)

        # writer
        self.summary_mode = summary_mode
        if summary_mode == "tensorboard":
            self.cp_dir = cp_dir
            self.run_dir = run_dir
        self.writer = self._initialize_writer()

        # faster convolutions, but more memory
        cudnn.benchmark = cudnn_benchmark

        # global step vars
        self._global_step = self.global_step
        self.init_global_step = self.global_step
        self._model_step = self.model_step

    @property
    def global_step(self):
        return self._train_one_epoch.counter

    @property
    def model_step(self):
        return self.global_step - self.init_global_step

    @elapsed_time
    def train(self,
              *,
              dataset_train: Dataset,
              dataset_val: Optional[Dataset] = None,
              epochs: int = 5,
              batch_size: int = 1,
              lr: float = 0.1,
              weight_decay: float = 1e-8,
              save_cp: bool = True,
              scheduler_params: Optional[dict] = None,
              criterion_params: Optional[dict] = None,
              optimizer_params: Optional[dict] = None):
        # TODO: check num_workers
        # TODO: check pin_memory
        # FIXME: add custom method for Dataset to get same size images

        # get mutable defaults
        if scheduler_params is None:
            scheduler_params = {"name": "StepLR"}
        if criterion_params is None:
            criterion_params = {"name": "BCE"}
        if optimizer_params is None:
            optimizer_params = {"name": "Adam"}

        if dataset_val is None:
            dataloader_val = None
        else:
            dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=12,
                                        pin_memory=True)

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=12,
                                      pin_memory=True)

        # check scales
        if dataset_val is not None:
            assert dataset_train.scale == dataset_val.scale, f'Different scales for train and val dataset: {dataset_train.scale} ' \
                                                             f'and {dataset_val.scale}. It would cause val metrics irrelevant.'

        # get scheduler, optimizer, criterion
        optimizer = BreastCancerSegmentator._get_optimizer(self.model.parameters(), optimizer_params)
        lr_scheduler = BreastCancerSegmentator._get_scheduler(optimizer, scheduler_params)
        criterion, out_layer_name = BreastCancerSegmentator._get_criterion(criterion_params)

        for epoch in tqdm(range(epochs), desc="Training epochs", total=epochs):
            epoch_loss = self._train_one_epoch(dataloader_train=dataloader_train,
                                               criterion=criterion,
                                               optimizer=optimizer,
                                               lr_scheduler=lr_scheduler)

            epoch_metrics = dict(epoch_loss=epoch_loss)
            self.writer.loss(epoch_metrics, self.model_step)
            self.writer.hparams(dict(lr=optimizer.param_groups[0]['lr']), self.model_step)

            if dataloader_val and epoch % 25 == 0:
                self.evaluate(loader=dataloader_val, criterion=criterion)
                self.predict(dataset_test=dataloader_val,
                             out_layer_name=out_layer_name)

        if save_cp:
            self._save_checkpoint()

        # self.writer.graph(self.model, dataloader_val, self.device)  # get model graph

        last_eval_score = self.evaluate(loader=dataloader_val, criterion=criterion)
        metric_dict = dict(last_epoch_loss=epoch_loss,
                           last_eval_score=last_eval_score)
        hparam_dict = dict(epochs=epochs,
                           batch_size=batch_size,
                           scale=dataloader_train.dataset.scale,
                           optimizer=optimizer,
                           lr_scheduler=lr_scheduler,
                           criterion=criterion)
        self.writer.totals(hparam_dict, metric_dict)

        self.writer.close()

    def cv(self, *, metadata_train, metadata_val, train_config, cross_val_config):
        # get dataset
        dataset_train = UnetDataset(metadata_train,
                                    root=train_config["root"],
                                    scale=train_config["scale"])
        dataset_val = UnetDataset(metadata_val,
                                  root=train_config["root"],
                                  scale=train_config["scale"])

        # TODO: check out Skorch for smarter cross-validation

        train_default_hparams = dict(
            epochs=train_config["epochs"],
            criterion_params=train_config["criterion_params"],
            scheduler_params=train_config["scheduler_params"],
            optimizer_params=train_config["optimizer_params"]
        )

        BreastCancerSegmentator._cv_check_config_alignement(train_default_hparams, cross_val_config)

        combinations = BreastCancerSegmentator._cv_params_combination(cross_val_config)

        for cross_val_params in combinations:
            train_cross_val_hparams = BreastCancerSegmentator._cv_update_params(train_default_hparams, cross_val_params)

            print(train_cross_val_hparams)

            self.train(dataset_train=dataset_train,
                       dataset_val=dataset_val,
                       save_cp=train_config["save_cp"],
                       **train_cross_val_hparams)

    def predict(self,
                *,
                dataset_test: Union[Dataset, DataLoader],
                scale_factor: float = 1,
                out_threshold: float = 0.5,
                batch_size: int = 1,
                out_layer_name: str = "sigmoid",
                tensorboard_verbose: bool = True,
                sample_batch: Optional[int] = 5) -> Dict[
        str, Any]:
        """
        Predict mask
        """

        init_training_state = self.model.training
        self.model.eval()

        if out_layer_name == "sigmoid":
            out_layer = torch.nn.Sigmoid()
        elif out_layer_name == "softmax":
            out_layer = nn.Softmax(dim=1)
        else:
            raise AttributeError("Not implemented out layer parameter: {}".format(out_layer_name))

        # TODO: change dataset_test for imgs?
        prediction_dict = defaultdict(list)

        with torch.no_grad():
            if isinstance(dataset_test, Dataset) or isinstance(dataset_test, DataLoader):
                if not isinstance(dataset_test, DataLoader):
                    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True)
                else:
                    dataloader_test = dataset_test

            for idx, batch in enumerate(dataloader_test):
                imgs = batch['image']
                true_masks = batch['mask']
                imgs_tensor = imgs.to(device=self.device, dtype=torch.float32)
                pred_masks = self.model(imgs_tensor)
                pred_masks = out_layer(pred_masks)
                prediction_dict['all_images'].append(imgs)

                # [mask_benign, mask_malignent, mask_background]
                prediction_dict['all_truth_masks'].append([mask.reshape(1, 1, *mask.shape) for mask in true_masks[0]])
                prediction_dict['all_pred_masks'].append([mask.reshape(1, 1, *mask.shape) for mask in pred_masks[0]])

                # if there is a sample break
                if sample_batch and sample_batch == idx:
                    break

        if tensorboard_verbose is True:
            self.writer.predict(prediction_dict, self.model_step)

        # go back to training state, if model was in training state
        if init_training_state is True:
            self.model.train()

        return prediction_dict

    def evaluate(self,
                 loader: DataLoader,
                 criterion,
                 threshold: float = 0.5,
                 tensorboard_verbose: bool = True):
        init_training_state = self.model.training
        self.model.eval()

        multiple_metrics = defaultdict(list)

        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=self.device, dtype=torch.float32)
            mask_pred = self.model(imgs)
            true_masks = true_masks.to(device=self.device, dtype=mask_pred.dtype)

            single_metric = criterion.evaluate(mask_pred, true_masks)
            for k, v in single_metric.items():
                multiple_metrics[k].append(v)

        tot = {k: np.mean(v) for k, v in multiple_metrics.items()}

        if tensorboard_verbose is True:
            self.writer.evaluate(tot, self.model_step)

        # go back to training state, if model was in training state
        if init_training_state is True:
            self.model.train()

        return tot

    @counter_global_step
    def _train_one_epoch(self, *, dataloader_train: DataLoader, criterion, optimizer, lr_scheduler):
        self.model.train()

        epoch_loss = 0
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

            # update learning rate
            lr_scheduler.step()

        return epoch_loss

    def _train_single_batch(self, imgs, masks_ground_truth, optimizer, criterion):
        masks_pred = self.model(imgs)
        masks_ground_truth = masks_ground_truth.type_as(masks_pred)  # in other case BCELoss gets error

        loss = criterion(masks_pred, masks_ground_truth)

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

    def _get_tensorboard_dir(self):
        converted_timestamp = get_converted_timestamp()

        run_dir = os.path.join(self.run_dir, converted_timestamp)
        cp_dir = os.path.join(self.cp_dir, converted_timestamp)

        return run_dir, cp_dir

    def _initialize_writer(self):
        if self.summary_mode == "tensorboard":
            current_runs_dir, current_cp_dir = self._get_tensorboard_dir()
            writer = UnetSummaryWriter(log_dir=current_runs_dir)
        elif self.summary_mode == "neptune":
            raise NotImplementedError
        else:
            raise ValueError(f"Summary mode {self.summary_mode} not implemented")

        return writer

    def _save_checkpoint(self):
        mkdir(self.cp_dir)
        model_dir = os.path.join(self.cp_dir, f"CP_epochs_{self.model_step + 1}")
        torch.save(self.model.state_dict(),
                   model_dir)
        logging.info(f'Checkpoint ({self.model_step + 1} saved !')

    def _init_model(self, model_params, pretrained_model_path, model_class=UNet):
        # model
        self.model = model_class(**model_params)
        # load pretrained dict
        if pretrained_model_path is not None:
            self._load_pretrained_model(pretrained_model_path)
        self.model.to(device=self.device)
