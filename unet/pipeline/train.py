import logging

import torch.nn as nn
import torch
from torch import optim
from tqdm import tqdm

from utils.eval import eval_net

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from unet.unet_model import UNet

from utils.utils import mkdir

dir_img = '../data/imgs/'
dir_mask = '../data/masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net: UNet, device: torch.device, dataset_train: Dataset,
              dataset_val: Dataset, epochs: int = 5, batch_size: int = 1,
              lr: float = 0.1, save_cp: bool = True):
    # TODO: check num_workers
    # TODO: check pin_memory
    # FIXME: add custom method for Dataset to get same size images
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    size_train = len(dataset_train)
    size_val = len(dataset_val)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    assert dataset_train.scale == dataset_val.scale, f'Different scales for train and val dataset: {dataset_train.scale} ' \
                                                     f'and {dataset_val.scale}. It would cause val metrics irrelevant.'

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {size_train}
        Validation size: {size_val}
        Checkpoints:     {save_cp}
        Device:          {device}
        Images scaling:  {dataset_train.scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    # TODO: check it out
    # criterion init
    img_size = dataset_train.img_size
    # pos_weight = torch.ones(img_size)
    # check if this pos_weight is correct
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        # TODO: check it
        net.train()

        epoch_loss = 0
        with tqdm(total=size_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in dataloader_train:
                imgs = batch['image']
                masks_ground_truth = batch['mask']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                masks_ground_truth = masks_ground_truth.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                masks_ground_truth = masks_ground_truth.type_as(masks_pred)  # in other case BCELoss gets error

                loss = criterion(masks_pred, masks_ground_truth)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if size_train>10 and global_step % (size_train // (10 * batch_size)) == 0:
                    val_score = eval_net(net, dataloader_val, device, size_val)
                    logging.info('Validation Dice Coeff: {}'.format(val_score))

                    writer.add_scalar('Dice/test/benign', val_score['benign'], global_step)
                    writer.add_scalar('Dice/test/malignant', val_score['malignant'], global_step)
                    writer.add_images('Images/ground_truth', imgs, global_step)

        # if save_cp:
        #     mkdir(dir_checkpoint)  # create checkpoint dir
        #     torch.save(net.state_dict(),
        #                dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        #     logging.info(f'Checkpoint {epoch + 1} saved !')

    if save_cp:
        mkdir(dir_checkpoint)  # create checkpoint dir
        torch.save(net.state_dict(),
                   dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == '__main__':
    pass
