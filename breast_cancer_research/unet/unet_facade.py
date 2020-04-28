# basic
from typing import Mapping, Optional, Any
from tqdm import tqdm, trange
import os
# ML
import numpy as np
from torch.backends import cudnn
from torch.utils.data import DataLoader
# custom
from breast_cancer_research.utils.utils import elapsed_time
from breast_cancer_research.utils.utils import counter_global_step, mkdir, get_converted_timestamp, prevent_oom_error
from breast_cancer_research.unet.unet_tensorboard import UnetSummaryWriter
# basic
from typing import Dict, Tuple
from collections import defaultdict
import logging
# ML
import torch
# custom
from breast_cancer_research.unet.unet_model import UNet
from breast_cancer_research.base.base_model import BaseModel


class BreastCancerSegmentator(BaseModel):
    def __init__(self,
                 model_params: Mapping,
                 device: torch.device = "cuda",
                 cp_dir: str = "checkpoints_unet/",
                 run_dir: str = "runs_unet/",
                 summary_mode: str = "tensorboard",
                 cudnn_benchmark: bool = True,
                 pretrained_model_path: Optional[str] = None,
                 preventing_oom_mode: bool = True):
        # device
        self.device = device
        # init model
        self._init_model(model_params=model_params,
                         pretrained_model_path=pretrained_model_path)
        # writer
        self.summary_mode = summary_mode
        if summary_mode == "tensorboard":
            self.cp_dir = cp_dir
            self.run_dir = run_dir
            self.writer, self.run_dir, self.cp_dir = self._initialize_writer()
        elif summary_mode == "none":
            pass
        else:
            raise NotImplementedError

        # faster convolutions, but more memory
        cudnn.benchmark = cudnn_benchmark

        # global step vars
        self._global_step = self.global_step
        self.init_global_step = self.global_step
        self._model_step = self.model_step

        # preventing oom error -> "cuda runtime error(2): out of memory"
        # get additional context managet "try:" in parts of code which generate high gpu usage
        # this mode is made due to gpu limitations: https://pytorch.org/docs/stable/notes/faq.html for reference
        self.preventing_oom_mode = preventing_oom_mode

    @property
    def global_step(self):
        return self._train_one_epoch.counter

    @property
    def model_step(self):
        return self.global_step - self.init_global_step

    @elapsed_time
    def train(self,
              *,
              dataloader_train: DataLoader,
              dataloader_val: Optional[DataLoader] = None,
              epochs: int = 5,
              lr: float = 0.1,
              weight_decay: float = 1e-8,
              save_cp: bool = True,
              evaluation_interval: int = 10,
              evaluate_train: bool = False,
              train_metadata: Optional[dict] = None,
              scheduler_params: Optional[dict] = None,
              criterion_params: Optional[dict] = None,
              optimizer_params: Optional[dict] = None):

        # get mutable defaults
        if scheduler_params is None:
            scheduler_params = {"name": "StepLR"}
        if criterion_params is None:
            criterion_params = {"name": "BCE"}
        if optimizer_params is None:
            optimizer_params = {"name": "Adam"}

        # check scales
        if dataloader_val is not None:
            assert dataloader_train.dataset.scale == dataloader_val.dataset.scale, \
                f'Different scales for train and val dataset: {dataloader_train.dataset.scale} ' \
                f'and {dataloader_val.dataset.scale}. It would cause val metrics irrelevant.'

        # get scheduler, optimizer, criterion
        optimizer = BaseModel.get_optimizer(self.model.parameters(), optimizer_params)
        lr_scheduler = BaseModel.get_scheduler(optimizer, scheduler_params)
        criterion, out_layer_name = BaseModel.get_criterion(criterion_params)

        #init training params
        hparam_dict = dict(epochs=epochs,
                           batch_size=dataloader_train.batch_size,
                           scale=dataloader_train.dataset.scale,
                           optimizer=optimizer,
                           lr_scheduler=lr_scheduler,
                           criterion=criterion)
        self.writer.totals(hparam_dict, {}, train_metadata)

        for epoch in trange(epochs, desc="Training epochs", total=epochs, position=0, leave=True):
            epoch_loss = self._train_one_epoch(dataloader_train=dataloader_train,
                                               criterion=criterion,
                                               optimizer=optimizer)

            # update learning rate
            lr_scheduler.step()

            epoch_metrics = dict(epoch_loss=epoch_loss)
            self.writer.loss(epoch_metrics, self.model_step)
            self.writer.hparams(dict(lr=optimizer.param_groups[0]['lr']), self.model_step)

            if epoch % evaluation_interval == 0 and epoch != 0:
                if dataloader_val is not None:
                    totals = self.evaluate(eval_loader=dataloader_val, criterion=criterion, tensorboard_metric_mode="val")
                    logging.info(f"Validation totals for epoch {epoch}: {totals}")
                    self.predict(prediction_loader=dataloader_val,
                                 out_layer_name=out_layer_name,
                                 )

                if evaluate_train is True:
                    totals = self.evaluate(eval_loader=dataloader_train, criterion=criterion, tensorboard_metric_mode="train")
                    logging.info(f"Training totals for epoch {epoch}: {totals}")
                if save_cp:
                    self._save_checkpoint()

        if dataloader_val is not None:
            last_eval_score = self.evaluate(eval_loader=dataloader_val, criterion=criterion,
                                            tensorboard_metric_mode="val")
            logging.info(f"Validation totals for epoch {epoch}: {last_eval_score}")
        else:
            last_eval_score = None

        if evaluate_train is True:
            totals = self.evaluate(eval_loader=dataloader_train, criterion=criterion, tensorboard_metric_mode="train")
            logging.info(f"Training totals for epoch {epoch}: {totals}")
        if save_cp:
            self._save_checkpoint()
        # self.writer.graph(self.model, dataloader_val, self.device)  # get model graph

        metric_dict = dict(last_epoch_loss=epoch_loss,
                           last_eval_score=last_eval_score)
        hparam_dict = dict(epochs=epochs,
                           batch_size=dataloader_train.batch_size,
                           scale=dataloader_train.dataset.scale,
                           optimizer=optimizer,
                           lr_scheduler=lr_scheduler,
                           criterion=criterion)
        self.writer.totals(hparam_dict, metric_dict)

        self.writer.close()

    def cv(self,
           *,
           dataloader_train: DataLoader,
           dataloader_val: DataLoader,
           train_config: Dict,
           cross_val_config: Dict):
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

            torch.cuda.empty_cache()

            self.train(dataloader_train=dataloader_train,
                       dataloader_val=dataloader_val,
                       save_cp=train_config["save_cp"],
                       evaluation_interval=train_config["evaluation_interval"],
                       evaluate_train=train_config["evaluate_train"],
                       **train_cross_val_hparams)

    @prevent_oom_error
    def predict(self,
                *,
                prediction_loader: DataLoader,
                scale_factor: float = 1,
                out_threshold: float = 0.5,
                batch_size: int = 1,
                out_layer_name: str = "sigmoid",
                tensorboard_verbose: bool = True,
                sample_batch: Optional[int] = 10) -> Dict[str, Any]:
        """
        Predict mask
        """

        out_layer = BaseModel.get_out_layer(out_layer_name)

        prediction_dict = defaultdict(list)

        with torch.no_grad():
            for idx, batch in enumerate(prediction_loader):
                imgs = batch['image']
                true_masks = batch['mask']
                classnames = batch['classname']
                imgs_tensor = imgs.to(device=self.device, dtype=torch.float32)
                pred_masks = self.model(imgs_tensor)
                pred_masks = out_layer(pred_masks)
                pred_masks = torch.where(pred_masks.cpu() > out_threshold, torch.tensor(1), torch.tensor(0)).to(dtype=torch.float32)

                prediction_dict['all_images'].append(imgs.cpu())
                # [mask_benign, mask_malignent, mask_background]
                prediction_dict['all_truth_masks'].append(true_masks)
                prediction_dict['all_pred_masks'].append(pred_masks)
                prediction_dict['classnames'].append(classnames)

                # if there is a sample break
                if sample_batch and sample_batch == idx:
                    break

        if tensorboard_verbose is True:
            self.writer.predict(prediction_dict, model_step=self.model_step)

        return prediction_dict

    @prevent_oom_error
    def evaluate(self,
                 eval_loader: DataLoader,
                 criterion: Any,
                 threshold: float = 0.5,
                 tensorboard_verbose: bool = True,
                 tensorboard_metric_mode: str = "val") -> Dict:

        multiple_metrics = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(eval_loader, total=len(eval_loader), position=0, leave=True):
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=self.device, dtype=torch.float32)
                mask_pred = self.model(imgs)
                mask_pred = mask_pred.detach().cpu()
                batch_metrics = criterion.evaluate(mask_pred, true_masks)

                for single_metric in batch_metrics:
                    for k, v in single_metric.items():
                        multiple_metrics[k].append(v)

        tot = {k+"_mean": np.mean(v) for k, v in multiple_metrics.items()}

        if tensorboard_verbose is True:
            self.writer.evaluate(tot, self.model_step, mode=tensorboard_metric_mode)
            logging.info(tot)

        return tot

    @counter_global_step
    def _train_one_epoch(self, *, dataloader_train: DataLoader, criterion, optimizer):
        self.model.train()
        epoch_loss = 0

        #for batch in tqdm(dataloader_train, total=len(dataloader_train), position=1):
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

            if loss is not None:
                epoch_loss += loss

        return epoch_loss

    @prevent_oom_error
    def _train_single_batch(self, imgs: torch.Tensor, masks_ground_truth: torch.Tensor, optimizer, criterion) -> float:
        masks_pred = self.model(imgs)
        masks_ground_truth = masks_ground_truth.type_as(masks_pred)  # in other case BCELoss gets error

        loss = criterion(masks_pred, masks_ground_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # detach to save gpu memory
        loss = loss.detach().cpu().item()

        return loss

    def _load_pretrained_model(self, pretrained_model_path: str):
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(pretrained_model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def _get_tensorboard_dir(self) -> Tuple[str, str]:
        converted_timestamp = get_converted_timestamp()

        run_dir = os.path.join(self.run_dir, converted_timestamp)
        cp_dir = os.path.join(self.cp_dir, converted_timestamp)

        return run_dir, cp_dir

    def _initialize_writer(self) -> Tuple[Any, str, str]:
        if self.summary_mode == "tensorboard":
            current_runs_dir, current_cp_dir = self._get_tensorboard_dir()
            writer = UnetSummaryWriter(log_dir=current_runs_dir)
        elif self.summary_mode == "neptune":
            raise NotImplementedError
        else:
            raise ValueError(f"Summary mode {self.summary_mode} not implemented")

        return writer, current_runs_dir, current_cp_dir

    def _save_checkpoint(self):
        mkdir(self.cp_dir)
        model_dir = os.path.join(self.cp_dir, f"CP_epochs_{self.model_step + 1}_{get_converted_timestamp()}.pth")
        torch.save(self.model.state_dict(),
                   model_dir)
        logging.info(f'Checkpoint {self.model_step + 1} saved !')

    def _init_model(self, model_params: Dict, pretrained_model_path: str):
        # model
        self.model = UNet(**model_params)
        # load pretrained dict
        if pretrained_model_path is not None:
            self._load_pretrained_model(pretrained_model_path)
        self.model.to(device=self.device)
