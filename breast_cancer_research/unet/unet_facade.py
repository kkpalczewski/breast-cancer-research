# basic
from abc import ABC, abstractmethod
from typing import Mapping, ClassVar, Optional, Dict, Union, Any
import logging
from tqdm import tqdm
from collections import defaultdict
import os
# ML
# TODO: check it
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn
# custom
from breast_cancer_research.utils.utils import counter_global_step, mkdir, get_converted_timestamp
from breast_cancer_research.utils.metrics import dice_coeff
from breast_cancer_research.unet.unet_parts import BinaryDiceLoss


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
                 cp_dir: str = "checkpoints/", run_dir: str = "runs/", cudnn_benchmark: bool = True,
                 pretrained_model_path: Optional[str] = None):
        self.model = model(**model_params)
        # load pretrained dict
        if pretrained_model_path is not None:
            self._load_pretrained_model(pretrained_model_path)

        self.device = device

        # get tensorboard dirs
        self.cp_dir = cp_dir
        self.run_dir = run_dir

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

    def train(self, *, dataset_train: Dataset, dataset_val: Optional[Dataset] = None,
              epochs: int = 5,
              batch_size: int = 1,
              lr: float = 0.1,
              weight_decay: float = 1e-8,
              save_cp: bool = True,
              scheduler_name: str = "StepLR",
              scheduler_params: Optional[dict] = None,
              criterion_name: str = "BCE",
              criterion_params: Optional[dict] = None,
              optimizer_name: str = "Adam",
              optimizer_params: Optional[dict] = None):
        # TODO: check num_workers
        # TODO: check pin_memory
        # FIXME: add custom method for Dataset to get same size images

        #get mutable defaults
        if scheduler_params is None: scheduler_params = {}
        if criterion_params is None: criterion_params = {}
        if optimizer_params is None: optimizer_params = {}

        if dataset_val is None:
            dataloader_val = None
        else:
            dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8,
                                        pin_memory=True)

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8,
                                      pin_memory=True)

        # check scales
        if dataset_val is not None:
            assert dataset_train.scale == dataset_val.scale, f'Different scales for train and val dataset: {dataset_train.scale} ' \
                                                             f'and {dataset_val.scale}. It would cause val metrics irrelevant.'

        #get scheduler, optimizer, criterion
        optimizer = BreastCancerSegmentator._get_optimizer(self.model.parameters(), optimizer_name, optimizer_params)
        lr_scheduler = BreastCancerSegmentator._get_scheduler(optimizer, scheduler_name, **scheduler_params)
        criterion = BreastCancerSegmentator._get_criterion(criterion_name, criterion_params)

        #tensorboard dirs
        current_runs_dir, current_cp_dir = self._get_tensorboard_dir(dict(lr=lr, batch_size=batch_size, epochs=epochs))

        self.writer = SummaryWriter(log_dir=current_runs_dir, comment='LR_{lr}_BS_{batch_size}_E_{epochs}')

        for epoch in tqdm(range(epochs), desc="Training epochs", total=epochs):
            epoch_loss = self._train_one_epoch(dataloader_train=dataloader_train,
                                               criterion=criterion,
                                               optimizer=optimizer,
                                               lr_scheduler=lr_scheduler)

            self._tensorboard_loss(epoch_loss)
            self._tensorboard_hparams(dict(lr=lr))

            if dataloader_val and epoch % 24 == 0:
                self._check_tensorboard_writer()
                self.evaluate(loader=dataloader_val, tensorboard_verbose=True)
                self.predict(dataset_test=dataloader_val, tensorboard_verbose=True)

        if save_cp:
            mkdir(current_cp_dir)
            model_dir = os.path.join(current_cp_dir, f"CP_epochs_{epoch + 1}")
            torch.save(self.model.state_dict(),
                       model_dir)
            logging.info(f'Checkpoint {epoch + 1} saved !')

        self._tensorboard_graph(dataloader_val) # get model graph
        self.writer.close()

    def predict(self, *, dataset_test: Union[Dataset, DataLoader], scale_factor: float = 1, out_threshold: float = 0.5,
                batch_size: int = 1, out_layer_name: str = "sigmoid", tensorboard_verbose: bool = True,
                sample_batch: Optional[int] = 5) -> Dict[
        str, Any]:
        """
        Predict mask
        """

        init_training_state = self.model.training
        self.model.eval()

        if out_layer_name == "sigmoid":
            out_layer = torch.nn.Sigmoid()
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
                prediction_dict['all_images'].append(imgs)

                # [mask_benign, mask_malignent, mask_background]
                prediction_dict['all_truth_masks'].append([mask.reshape(1, 1, *mask.shape) for mask in true_masks[0]])
                prediction_dict['all_pred_masks'].append([mask.reshape(1, 1, *mask.shape) for mask in pred_masks[0]])

                # if there is a sample break
                if sample_batch and sample_batch == idx:
                    break

        if tensorboard_verbose is True:
            self._check_tensorboard_writer()
            self._tensorboard_predict(prediction_dict)

        # go back to training state, if model was in training state
        if init_training_state is True:
            self.model.train()

        return prediction_dict

    def evaluate(self, loader: DataLoader, threshold: float = 0.5, tensorboard_verbose: bool = True):
        """Evaluation without the densecrf with the dice coefficient"""
        n_val = len(loader.dataset)

        init_training_state = self.model.training
        self.model.eval()
        tot = defaultdict(float)

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

        tot["benign"] = tot["benign"] / n_val
        tot["malignant"] = tot["malignant"] / n_val

        if tensorboard_verbose is True:
            self._tensorboard_evaluate(tot)

        # go back to training state, if model was in training state
        if init_training_state is True:
            self.model.train()

        return tot

    def _train_one_epoch(self, *, dataloader_train: DataLoader, criterion, optimizer, lr_scheduler):
        # TODO: check it

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

    @counter_global_step
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

    def _tensorboard_evaluate(self, val_score):
        logging.info('Validation Dice Coeff: {}'.format(val_score))
        self.writer.add_scalar('Dice/test/benign', val_score['benign'], self.model_step)
        self.writer.add_scalar('Dice/test/malignant', val_score['malignant'], self.model_step)

    def _tensorboard_predict(self, prediction_dict, sample_batch: Optional[int] = 5):
        for idx in range(len(prediction_dict["all_images"])):
            self.writer.add_images(f'{idx}/original_images', prediction_dict["all_images"][idx])
            self.writer.add_images(f'{idx}/benign/ground_truh', prediction_dict["all_truth_masks"][idx][0])
            self.writer.add_images(f'{idx}/benign/predictions', prediction_dict["all_pred_masks"][idx][0],
                                   self.model_step)
            self.writer.add_images(f'{idx}/malignant/ground_truh', prediction_dict["all_truth_masks"][idx][1])
            self.writer.add_images(f'{idx}/malignant/predictions', prediction_dict["all_pred_masks"][idx][1],
                                   self.model_step)
            self.writer.add_images(f'{idx}/background/ground_truh', prediction_dict["all_truth_masks"][idx][2])
            self.writer.add_images(f'{idx}/background/predictions', prediction_dict["all_pred_masks"][idx][2],
                                   self.model_step)

            if sample_batch and sample_batch == idx:
                break

    def _tensorboard_loss(self, loss):
        self.writer.add_scalar('Loss/train', loss, self.model_step)

    def _tensorboard_hparams(self, hparams):
        for k, v in hparams.items():
            self.writer.add_scalar(f'Hparams/{k}', v, self.model_step)

    def _tensorboard_graph(self, dataloader_val: DataLoader):
        # only for graph add
        for batch in dataloader_val:
            imgs = batch['image']
            imgs = imgs.to(device=self.device, dtype=torch.float32)
            break

        self.writer.add_graph(self.model, imgs)

    def _check_tensorboard_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter()

    def _get_tensorboard_dir(self, params):
        converted_timestamp = get_converted_timestamp()

        params_str = ""
        for key in params:
            params_str += "_" + key + "_" + str(params[key])

        run_dir = os.path.join(self.run_dir, converted_timestamp + params_str)
        cp_dir = os.path.join(self.cp_dir, converted_timestamp + params_str)

        return run_dir, cp_dir

    @classmethod
    def _get_criterion(cls, criterion_name: str = "BCE", criterion_params: Optional[Dict] = None):
        if criterion_params is None:
            criterion_params = {}

        if criterion_name == "BCE":
            criterion = nn.BCEWithLogitsLoss(**criterion_params)
        elif criterion_name == "dice":
            criterion = BinaryDiceLoss(**criterion_params)
        else:
            raise NotImplementedError(f"Loss {criterion_name} not implemented")

        return criterion

    @classmethod
    def _get_optimizer(cls, params, optimizer_name: str = "Adam", optimizer_params: Optional[Dict] = None):
        if optimizer_params is None:
            optimizer_params = {}

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(params, **optimizer_params)
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(params, **optimizer_params)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")

        return optimizer

    @classmethod
    def _get_scheduler(cls, optimizer, scheduler_name: str = "StepLR", scheduler_params: Optional[Dict] = None):
        if scheduler_params is None:
            scheduler_params = {}

        if scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        else:
            raise NotImplementedError(f"Optimizer {scheduler_name} not implemented")

        return scheduler