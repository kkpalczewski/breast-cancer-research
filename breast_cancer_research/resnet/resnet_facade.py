from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from tqdm import tqdm

from breast_cancer_research.unet.unet_facade import BaseModel
from breast_cancer_research.utils.utils import elapsed_time


class BreastCancerClassifier(BaseModel):
    def __init__(self, model_name: str = "resnet", num_classes: int = 2,
                 feature_extract: bool = False, use_pretrained: bool = True,
                 device: torch.device = "cuda"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
        self.device = device
        self.model, self.input_size = self.initialize_model()
        self.transform = self._get_transform()

    @elapsed_time
    def train(self, dataloader_train, dataloader_test, *,
              lr=0.001, momentum=0.9, num_epochs=25, is_inception=False):

        val_acc_history = []

        # Observe that all parameters are being optimized
        params_to_update = self._get_params_to_update()

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        self.model.train()

        for epoch in tqdm(range(num_epochs), total=num_epochs):

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader_train:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                #inputs = self.transform['train'](inputs)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(self.feature_extract):
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

                    # backward + optimize
                    loss.requires_grad = True

                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                exp_lr_scheduler.step(epoch=epoch)

            epoch_loss = running_loss / len(dataloader_train.dataset)
            epoch_acc = running_corrects.double() / len(dataloader_train.dataset)

            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            if epoch % 10 == 0:
                #TODO: add evaluation
                pass

        print(val_acc_history)

    def evaluate(self):
        pass

    def predict(self):
        pass

    def set_parameter_requires_grad(self, feature_extracting, model_ft):
        if feature_extracting:
            for param in model_ft.parameters():
                param.requires_grad = False

    def initialize_model(self):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

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
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

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
        print("Params to learn:")
        if self.feature_extract:
            self.params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    self.params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        return self.params_to_update


def main():
    # Initialize the model for this run
    model_ft, input_size = BreastCancerClassifier()

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))


if __name__ == "__main__":
    main()
