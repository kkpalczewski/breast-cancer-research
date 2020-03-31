from __future__ import division
from __future__ import print_function

from typing import Optional
import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from torchvision import transforms

from breast_cancer_research.base.base_model import BaseModel
from breast_cancer_research.resnet.resnet_tensorboard import ResnetSummaryWriter
from breast_cancer_research.utils.utils import elapsed_time, get_converted_timestamp
from breast_cancer_research.utils.utils import counter_global_step



class BreastCancerClassifier(BaseModel):
    TENSORBOARD_SUMMARY_DIR = "runs_resnet/"

    def __init__(self,
                 model_name: str = "resnet",
                 num_classes: int = 2,
                 feature_extract: bool = False,
                 use_pretrained: bool = True,
                 device: torch.device = "cuda",
                 summary_mode: str = "tensorboard",
                 pretrained_model_path: Optional[str] = None,
                 cp_dir: str = "checkpoints/",
                 run_dir: str = "runs/"):
        # device
        self.device = device

        # writer
        self.summary_mode = summary_mode
        if summary_mode == "tensorboard":
            self.cp_dir = cp_dir
            self.run_dir = run_dir
        self.writer = self._initialize_writer()

        # global step vars
        self._global_step = self.global_step
        self.init_global_step = self.global_step
        self._model_step = self.model_step

        # model
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
        self.num_classes = num_classes
        self.model_name = model_name
        self.model, self.input_size = self._initialize_model()
        if pretrained_model_path is not None:
            self._load_pretrained_model(pretrained_model_path)
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
              dataloader_train,
              dataloader_val,
              *,
              num_epochs=25,
              is_inception=False,
              batch_sample: Optional[int] = None,
              save_cp: bool = True,
              scheduler_params: Optional[dict] = None,
              criterion_params: Optional[dict] = None,
              optimizer_params: Optional[dict] = None,
              evaluation_interval: int = 1):

        #get mutable defaults
        if scheduler_params is None:
            scheduler_params = {"name": "StepLR", "step_size": 7, "gamma": 0.1}
        if criterion_params is None:
            criterion_params = {"name": "CrossEntropyLoss"}
        if optimizer_params is None:
            optimizer_params = {"name": "SGD", "lr": 0.001, "momentum": 0.9}

        # Observe that all parameters are being optimized
        params_to_update = self._get_params_to_update()

        criterion = BreastCancerClassifier._get_criterion(criterion_params)
        optimizer = BreastCancerClassifier._get_optimizer(params_to_update, optimizer_params)
        lr_scheduler = BreastCancerClassifier._get_scheduler(optimizer, scheduler_params)

        self.model.train()
        for epoch in tqdm(range(num_epochs), total=num_epochs):
            epoch_loss, epoch_acc = self._train_one_epoch(dataloader_train=dataloader_train,
                                                          optimizer=optimizer,
                                                          criterion=criterion,
                                                          is_inception=is_inception,
                                                          lr_scheduler=lr_scheduler,
                                                          batch_sample=batch_sample,
                                                          epoch=epoch)

            epoch_metrics = dict(epoch_loss=epoch_loss, epoch_acc=epoch_acc)
            self.writer.loss(epoch_metrics, self.model_step)

            if dataloader_val and epoch % evaluation_interval == 0 and epoch != 0:
                self.evaluate(dataloader_val, criterion)
                self.predict(dataloader_val)
        #last eval
        self.evaluate(dataloader_val, criterion)
        self.predict(dataloader_val)


    def evaluate(self, dataloader_val, criterion, batch_sample=50):
        init_training_state = self.model.training
        self.model.eval()

        metric_dict = defaultdict(list)

        for idx, (inputs, labels) in enumerate(dataloader_val):
            inputs = inputs.to(self.device)
            preds = self.model(inputs)

            preds = preds.detach().to("cpu").tolist()
            preds = np.argmax(preds, axis=1)
            labels = labels.to("cpu").tolist()

            metric_dict['preds'].extend(preds)
            metric_dict['labels'].extend(labels)

            #TODO: remove after testing
            if batch_sample and batch_sample == idx:
                break

        tot, support = criterion.evaluate(metric_dict)

        classes = dataloader_val.dataset.classes

        self.writer.evaluate(val_score_dict=tot,
                             classes=classes,
                             support=support,
                             model_step=self.model_step)

        # go back to training state, if model was in training state
        if init_training_state is True:
            self.model.train()

        return tot

    def predict(self, dataloader_val, num_images=5, batch_samples=100, resize_factor=4):
        init_training_state = self.model.training
        self.model.eval()

        prediction_dict = defaultdict(lambda: defaultdict(list))
        classes = dataloader_val.dataset.classes

        for idx, (inputs, labels) in enumerate(dataloader_val):
            inputs_tensor = inputs.to(self.device)
            preds = self.model(inputs_tensor)

            preds = preds.detach().to("cpu").tolist()
            preds = np.argmax(preds, axis=1)
            labels = labels.to("cpu").tolist()

            for pred, label, input in zip(preds, labels, inputs):
                pred_class = classes[pred]
                label_class = classes[label]
                match = (pred_class == label_class)

                if match and len(prediction_dict['true_positive'][pred_class]) <= num_images:
                    input = self._resize_tensor(input)
                    prediction_dict['true_positive'][pred_class].append(input)
                elif not match and len(prediction_dict['false_positive'][pred_class]) <= num_images:
                    input = self._resize_tensor(input)
                    prediction_dict['false_positive'][pred_class].append(input)

            if idx == batch_samples:
                break

        self.writer.predict(prediction_dict=prediction_dict,
                            model_step=self.model_step)

        # go back to training state, if model was in training state
        if init_training_state is True:
            self.model.train()

    def set_parameter_requires_grad(self, feature_extracting, model_ft):
        if feature_extracting:
            for param in model_ft.parameters():
                param.requires_grad = False

    @counter_global_step
    def _train_one_epoch(self, dataloader_train, optimizer, criterion, is_inception, lr_scheduler, batch_sample,
                         epoch):
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for idx, (inputs, labels) in enumerate(dataloader_train):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            loss, preds = self._train_single_batch(inputs, labels, optimizer, is_inception, criterion)

            # statistics
            running_loss += loss * dataloader_train.batch_size
            running_corrects += np.sum(preds == labels.to("cpu").numpy())

            lr_scheduler.step(epoch=epoch)

            if batch_sample and batch_sample == idx:
                break

        epoch_loss = running_loss / idx*dataloader_train.batch_size
        epoch_acc = running_corrects / idx*dataloader_train.batch_size

        return epoch_loss, epoch_acc


    def _train_single_batch(self, inputs, labels, optimizer, is_inception, criterion) -> (float, np.ndarray):
        # zero the parameter gradients
        optimizer.zero_grad()

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

    def _initialize_model(self):

        if self.model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(self.feature_extract, model_ft)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(self.feature_extract, model_ft)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(self.feature_extract, model_ft)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(self.feature_extract, model_ft)
            model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = self.num_classes
            input_size = 224

        elif self.model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(self.feature_extract, model_ft)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(self.feature_extract, model_ft)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 299

        else:
            raise ValueError(f"Invalid model name {self.model_name}")

        return model_ft, input_size

    def _initialize_writer(self):
        if self.summary_mode == "tensorboard":
            current_runs_dir, current_cp_dir = self._get_tensorboard_dir()
            writer = ResnetSummaryWriter(log_dir=current_runs_dir)
        elif self.summary_mode == "neptune":
            raise NotImplementedError
        else:
            raise ValueError(f"Summary mode {self.summary_mode} not implemented")

        return writer

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

    def _load_pretrained_model(self, pretrained_model_path):
        raise NotImplementedError

    def _get_tensorboard_dir(self):
        converted_timestamp = get_converted_timestamp()

        run_dir = os.path.join(self.run_dir, converted_timestamp)
        cp_dir = os.path.join(self.cp_dir, converted_timestamp)

        return run_dir, cp_dir

    def _resize_tensor(self, img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(128),
            transforms.ToTensor()
        ])

        return transform(img)