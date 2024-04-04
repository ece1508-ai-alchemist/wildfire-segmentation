import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

from data.load_data import POST_FIRE_DIR
from src.data_loader.dataloader import get_loader
from src.data_loader.dataset import WildfireDataset
from src.model.unet import UNet
from src.train.train import load_checkpoint


def adjust_image(image):
    rgb_image = image[:, :, [2, 1, 0]]
    # Normalize the values to range between 0 and 1
    # ABSOLUTELY BAD HACK
    if np.min(rgb_image) != 0:
        rgb_image = (rgb_image + -1 * np.min(rgb_image)) / (-2 * np.min(rgb_image))
    # print(np.min(rgb_image))

    rgb_image = rgb_image.astype(np.float32)
    # print(rgb_image)
    if np.max(rgb_image) != 0:
        rgb_image /= np.max(rgb_image)

    # Apply some adjustments to enhance the image visibility
    gamma = 1.05
    rgb_image = np.clip(rgb_image ** (1 / gamma), 0, 1)  # Apply gamma correction

    return rgb_image


# This works with CPU only!
def print_sample(batch, sample_id):
    image = batch["image"]
    mask = batch["mask"]
    flip_img = image[sample_id].permute(1, 2, 0)
    pre_fire = np.array(flip_img)

    flip_mask = mask[sample_id].permute(1, 2, 0)

    pre_fire_rgb = adjust_image(pre_fire)

    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(pre_fire_rgb)

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(flip_mask)


def print_sample_and_pred(batch, pred, sample_id):
    image = batch["image"]
    mask = batch["mask"]

    flip_img = image[sample_id].permute(1, 2, 0)
    pre_fire = np.array(flip_img)

    flip_mask = mask[sample_id].permute(1, 2, 0)

    flip_pred = pred[sample_id].permute(1, 2, 0)

    pre_fire_rgb = adjust_image(pre_fire)

    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(pre_fire_rgb)
    ax1.set_title("Satellite Image")

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(flip_mask)
    ax1.set_title("True Mask")

    ax1 = fig.add_subplot(1, 3, 3)
    ax1.imshow(flip_pred)
    ax1.set_title("Predicted Mask")


if __name__ == "__main__":
    print("Adjust the loop below to print the sample you want")

    SAVE_PATH = "model_weights"
    LOAD_MODEL = 1  # set to 1 if there are model weights ready to be loaded
    CHECKPOINT_NAME = "pre_trained.pt"
    THRESHOLD = 0.5
    NUM_SAMPLES = 8  # Number of samples to print

    device = "cpu"

    print(device)

    dt = WildfireDataset(POST_FIRE_DIR, transforms=None)

    val_loader = get_loader(dt.train, is_train=False, loader_args=dict(batch_size=1, shuffle=True))

    if LOAD_MODEL:
        model = UNet(n_channels=12, n_classes=1, bilinear=False)
        model.to(device=device)

        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        scaler = torch.cuda.amp.GradScaler()

        CUR_EPOCH = 0
        # Load checkpoint
        CUR_EPOCH = load_checkpoint(os.path.join(SAVE_PATH, CHECKPOINT_NAME), model, scaler, optimizer)

        model.eval()

    num_samples_done = 0
    for batch in val_loader:
        images, true_masks = batch["image"], batch["mask"]

        images = images.to(device)
        true_masks = true_masks.to(device)

        if LOAD_MODEL:
            with torch.cuda.amp.autocast():
                masks_pred = model(images)
            # Calculate metrics
            preds_binary = (torch.sigmoid(masks_pred) > THRESHOLD).float()
            print_sample_and_pred(batch, preds_binary, 0)
        else:
            print_sample(batch, 0)
        num_samples_done += 1
        if num_samples_done == NUM_SAMPLES:
            break
