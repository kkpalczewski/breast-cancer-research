from __future__ import division
from __future__ import print_function

from typing import Optional, Dict, Any
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import trange, tqdm
import numpy as np
from collections import defaultdict
from torchvision import transforms
import logging

from breast_cancer_research.base.base_model import BaseModel
from breast_cancer_research.resnet.resnet_tensorboard import ResnetSummaryWriter
from breast_cancer_research.utils.utils import elapsed_time, get_converted_timestamp, counter_global_step, mkdir


class BreastCancerClassifier(BaseModel):
    def __init__(self,
                 model_params: Dict[str, Any],
                 device: torch.device = "cuda",
                 summary_mode: str = "tensorboard",
                 cp_dir: str = "checkpoints_resnet/",
                 run_dir: str = "runs_resnet/"):
        # device
        self.device = device

        # writer
        self.summary_mode = summary_mode
        if summary_mode == "tensorboard":
            self.cp_dir = cp_dir
            self.run_dir = run_dir
        self.writer, self.run_dir, self.cp_dir = self._initialize_writer()

        # global step vars
        self._global_step = self.global_step
        self.init_global_step = self.global_step
        self._model_step = self.model_step

        # model
        self.model, self.input_size, self.feature_extract = self._initialize_model(**model_params)

        self.model.to(self.device)

        # transforms
        self.transform = self._get_transform()

    @property
    def global_step(self):
        return self._train_one_epoch.counter

    @property
    def model_step(self):
        return self.global_step - self.init_global_step

    @elapsed_time
    def train(self,
              dataloader_train: DataLoader,
              dataloader_val: DataLoader,
              *,
              epochs: int = 25,
              is_inception: bool = False,
              batch_sample: Optional[int] = None,
              save_cp: bool = True,
              scheduler_params: Optional[dict] = None,
              criterion_params: Optional[dict] = None,
              optimizer_params: Optional[dict] = None,
              evaluation_interval: int = 10,
              evaluate_train: bool = True,
              get_graph: bool = False,
              eval_mask_threshold: float = 0.5,
              train_metadata: Optional[dict] = None,
              multi_target: bool = False):

        # get mutable defaults
        if scheduler_params is None or scheduler_params == {}:
            scheduler_params = {"name": "StepLR", "step_size": 7, "gamma": 0.1}
        if criterion_params is None or criterion_params == {}:
            criterion_params = {"name": "CrossEntropyLoss"}
        if optimizer_params is None or optimizer_params == {}:
            optimizer_params = {"name": "SGD", "lr": 0.001, "momentum": 0.9}

        # Observe that all parameters are being optimized
        params_to_update = self._get_params_to_update()

        criterion, out_layer_name = BaseModel.get_criterion(criterion_params)
        optimizer = BaseModel.get_optimizer(params_to_update, optimizer_params)
        lr_scheduler = BaseModel.get_scheduler(optimizer, scheduler_params)

        hparam_dict = dict(epochs=epochs,
                           batch_size=dataloader_train.batch_size,
                           optimizer=optimizer,
                           lr_scheduler=lr_scheduler,
                           criterion=criterion)
        self.writer.totals(hparam_dict, {}, train_metadata)

        self.model.train()
        for epoch in trange(epochs):
            epoch_loss, epoch_acc = self._train_one_epoch(dataloader_train=dataloader_train,
                                                          optimizer=optimizer,
                                                          criterion=criterion,
                                                          is_inception=is_inception,
                                                          lr_scheduler=lr_scheduler,
                                                          batch_sample=batch_sample,
                                                          epoch=epoch,
                                                          multi_target=multi_target)

            epoch_metrics = dict(epoch_loss=epoch_loss, epoch_acc=epoch_acc)
            self.writer.loss(epoch_metrics, self.model_step)
            self.writer.hparams(dict(lr=optimizer.param_groups[0]['lr']), self.model_step)

            if epoch % evaluation_interval == 0 and epoch != 0:
                if dataloader_val is not None:
                    totals_val = self.evaluate(dataloader_val, criterion, tensorboard_metric_mode="val")
                    logging.info(f"Validation totals for epoch {epoch}: {totals_val}")
                    self.predict(dataloader_val, multi_target=multi_target)
                if evaluate_train is True:
                    totals_train = self.evaluate(dataloader_train, criterion, tensorboard_metric_mode="train")
                    logging.info(f"Training totals for epoch {epoch}: {totals_train}")
                if save_cp:
                    self._save_checkpoint()
        # last eval
        if dataloader_val is not None:
            last_eval_score_val = self.evaluate(dataloader_val, criterion, tensorboard_metric_mode="val")
            logging.info(f"Validation totals for epoch {epoch}: {last_eval_score_val}")
            self.predict(dataloader_val, multi_target=multi_target)
        else:
            last_eval_score = None
        if evaluate_train is True:
            last_eval_score_train = self.evaluate(dataloader_train, criterion, tensorboard_metric_mode="train")
            logging.info(f"Training totals for epoch {epoch}: {last_eval_score_train}")
        if save_cp:
            self._save_checkpoint()

        if get_graph is True:
            self.writer.graph(self.model, dataloader_val, self.device)  # get model graph

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
            batch_sample=train_config["batch_sample"],
            criterion_params=train_config["criterion_params"],
            scheduler_params=train_config["scheduler_params"],
            optimizer_params=train_config["optimizer_params"]
        )

        BreastCancerClassifier._cv_check_config_alignement(train_default_hparams, cross_val_config)

        combinations = BreastCancerClassifier._cv_params_combination(cross_val_config)

        for cross_val_params in combinations:
            train_cross_val_hparams = BreastCancerClassifier._cv_update_params(train_default_hparams, cross_val_params)

            torch.cuda.empty_cache()

            self.train(dataloader_train=dataloader_train,
                       dataloader_val=dataloader_val,
                       save_cp=train_config["save_cp"],
                       evaluation_interval=train_config["evaluation_interval"],
                       evaluate_train=train_config["evaluate_train"],
                       **train_cross_val_hparams)

    def evaluate(self,
                 dataloader_val: DataLoader,
                 criterion: Any,
                 tensorboard_metric_mode: str = "val") -> Dict:

        metric_dict = defaultdict(list)

        with torch.no_grad():
            for idx, (inputs, labels) in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):
                inputs = inputs.to(self.device)
                preds = self.model(inputs)

                preds = preds.detach().to("cpu").tolist()
                labels = labels.to("cpu").tolist()

                metric_dict['preds'].extend(preds)
                metric_dict['labels'].extend(labels)

        tot, support = criterion.evaluate(metric_dict)

        classes = dataloader_val.dataset.classes

        self.writer.evaluate(val_score_dict=tot,
                             classes=classes,
                             support=support,
                             model_step=self.model_step,
                             mode=tensorboard_metric_mode)

        return tot

    def predict(self,
                dataloader_val: DataLoader,
                num_images: int = 5,
                multi_target: bool = False):

        prediction_dict = defaultdict(lambda: defaultdict(list))
        classes = dataloader_val.dataset.classes

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(dataloader_val):
                inputs_tensor = inputs.to(self.device)
                preds = self.model(inputs_tensor)

                preds = preds.detach().to("cpu").tolist()
                preds = np.argmax(preds, axis=1)
                labels = labels.to("cpu").tolist()

                for pred, label, input in zip(preds, labels, inputs):
                    if multi_target is False:
                        label = np.argmax(label)
                    else:
                        raise NotImplementedError("Multi target for predict method not implemented")
                    pred_class = classes[pred]
                    label_class = classes[label]
                    match = (pred_class == label_class)

                    input = self._resize_tensor(input)
                    if match and len(prediction_dict['true_positive'][pred_class]) <= num_images:
                        prediction_dict['true_positive'][pred_class].append(input)
                    elif not match and len(prediction_dict['false_positive'][pred_class]) <= num_images:
                        prediction_dict['false_positive'][pred_class].append(input)

        self.writer.predict(prediction_dict=prediction_dict,
                            model_step=self.model_step)

    def set_parameter_requires_grad(self, feature_extracting: bool, model_ft: Any):
        if feature_extracting:
            for param in model_ft.parameters():
                param.requires_grad = False

    @counter_global_step
    def _train_one_epoch(self, dataloader_train: DataLoader, optimizer: Any, criterion: Any, is_inception: bool,
                         lr_scheduler: Any, batch_sample: int, epoch: int, multi_target: bool):
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for idx, (inputs, labels) in enumerate(dataloader_train):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            loss, preds = self._train_single_batch(inputs, labels, optimizer, is_inception, criterion, multi_target)

            # statistics
            running_loss += loss * dataloader_train.batch_size
            running_corrects += np.sum(preds == labels.to("cpu").numpy())

            lr_scheduler.step(epoch=epoch)

            if batch_sample and batch_sample == idx:
                break

        epoch_loss = running_loss / idx * dataloader_train.batch_size
        epoch_acc = running_corrects / idx * dataloader_train.batch_size

        return epoch_loss, epoch_acc

    def _train_single_batch(self, inputs, labels, optimizer, is_inception, criterion, multi_target) -> (float, np.ndarray):
        # zero the parameter gradients
        optimizer.zero_grad()

        if multi_target is False:
            _, labels = torch.max(labels, axis=1)
        else:
            raise NotImplementedError("Multi target for training not implemented")
        # Get model outputs and calculate loss
        # Special case for inception because in training it has an auxiliary output. In train
        #   mode we calculate the loss by summing the final output and the auxiliary output
        #   but in testing we only consider the final output.
        if is_inception:
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            outputs, aux_outputs = self.model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
        else:
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        loss = loss.item()
        preds = preds.to("cpu").numpy()

        return loss, preds

    def _initialize_model(self, model_name, use_pretrained, num_classes, feature_extract, pretrained_model_path):
        if model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract, model_ft)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract, model_ft)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract, model_ft)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract, model_ft)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract, model_ft)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(feature_extract, model_ft)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        else:
            raise ValueError(f"Invalid model name {model_name}")

        if pretrained_model_path is not None:
            self._load_pretrained_model(model_ft, pretrained_model_path)

        return model_ft, input_size, feature_extract

    def _initialize_writer(self):
        if self.summary_mode == "tensorboard":
            current_runs_dir, current_cp_dir = self._get_tensorboard_dir()
            writer = ResnetSummaryWriter(log_dir=current_runs_dir)
        elif self.summary_mode == "neptune":
            raise NotImplementedError
        else:
            raise ValueError(f"Summary mode {self.summary_mode} not implemented")

        return writer, current_runs_dir, current_cp_dir

    def _get_transform(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
            ]),
        }

        return data_transforms

    def _get_params_to_update(self):
        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        self.params_to_update = self.model.parameters()
        # print("Params to learn:")
        if self.feature_extract:
            self.params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    self.params_to_update.append(param)
                    # print("\t", name)
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    # print("\t", name)
                    pass

        return self.params_to_update

    def _load_pretrained_model(self, model, pretrained_model_path):
        # TODO: check this fun

        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrained_model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def _save_checkpoint(self):
        # TODO: check this fun
        mkdir(self.cp_dir)
        model_dir = os.path.join(self.cp_dir, f"CP_epochs_{self.model_step + 1}_{get_converted_timestamp()}.pth")
        torch.save(self.model.state_dict(),
                   model_dir)
        logging.info(f'Checkpoint {self.model_step + 1} saved !')

    def _get_tensorboard_dir(self):
        converted_timestamp = get_converted_timestamp()

        run_dir = os.path.join(self.run_dir, converted_timestamp)
        cp_dir = os.path.join(self.cp_dir, converted_timestamp)

        return run_dir, cp_dir

    def _resize_tensor(self, img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor()
        ])

        return transform(img)
