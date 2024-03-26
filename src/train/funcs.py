from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader.dataloader import get_loader
from src.data_loader.dataset import WildfireDataset
from src.train.config import ConfigSchema, DataSource


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    criterion: torch.nn.BCEWithLogitsLoss,
    config: ConfigSchema,
):
    device = get_device()
    print(f"💻 Device: {device}")

    model.to(device)

    print("📊 Initializing dataset")
    dataset = WildfireDataset(use_local=config.dataset.source == DataSource.local.value)

    # Create directory to save model weights
    weights_cache_folder = Path(__file__).parent / config.train.weights_cache_path

    if not weights_cache_folder.exists():
        weights_cache_folder.mkdir(exist_ok=True)

    # Initialize metrics dataframe
    metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "train_f1", "val_f1", "train_iou", "val_iou"])

    train_loader = get_loader(
        dataset.train, is_train=True, loader_args=dict(batch_size=config.train.batch_size, shuffle=True)
    )
    val_loader = get_loader(
        dataset.val, is_train=False, loader_args=dict(batch_size=config.train.batch_size, shuffle=False)
    )

    model.to(device=device)

    print(f"Total Epoch: {config.train.epoch_num}")
    for epoch in range(config.train.epoch_num):
        model.train()
        train_loss, train_f1, train_iou = 0, 0, 0
        num_samples = 0
        train_loop = tqdm(train_loader, leave=True)

        for batch in train_loop:
            images, true_masks = batch["image"], batch["mask"]

            images = images.to(device)

            true_masks = true_masks.to(device=device, dtype=torch.long)

            # Autocast for mixed precision
            with torch.autocast(device_type=device if device == "cuda" else "cpu"):
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate and update metrics
            batch_f1, batch_iou = get_scores(masks_pred, true_masks, config.validation.threshold)

            train_loss += loss.item() * images.size(0)

            train_f1 += batch_f1
            train_iou += batch_iou
            num_samples += images.size(0)

            train_loop.set_description(f"Epoch [{epoch + 1}/{config.train.epoch_num}]")
            train_loop.set_postfix(loss=train_loss / num_samples)

        # Validation step
        val_loss, val_f1, val_iou, num_val_samples = validate(model, val_loader, criterion, config)

        # Calculating average metrics
        avg_train_loss = train_loss / num_samples
        avg_val_loss = val_loss / num_val_samples
        avg_train_f1 = train_f1 / len(train_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_val_iou = val_iou / len(val_loader)

        # Saving metrics to dataframe
        new_row = pd.DataFrame(
            [
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_f1": avg_train_f1,
                    "val_f1": avg_val_f1,
                    "train_iou": avg_train_iou,
                    "val_iou": avg_val_iou,
                },
            ]
        )
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        # Save model weights
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), weights_cache_folder / f"unet_epoch_{epoch + 1} / model.pt")
        # Saving metrics to CSV file
        metrics_df.to_csv(
            weights_cache_folder / f"unet_epoch_{epoch + 1}" / config.train.training_metrics_cache_file, index=False
        )

    print("Training completed and metrics saved.")


def get_device() -> str:
    if torch.backends.cuda.is_built():
        device = "cuda"
    elif torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    return device


def get_scores(predictions: torch.Tensor, true_masks: torch.Tensor, threshold: float) -> (float, float):
    preds_binary = (torch.sigmoid(predictions) > threshold).float()

    # Convert true masks to float
    true_masks_binary = true_masks.float()

    # Move tensors to CPU and then convert to NumPy for scikit-learn functions
    # Flatten the tensors and move to CPU
    preds_binary_np = preds_binary.view(-1).detach().cpu().numpy()
    true_masks_binary_np = true_masks_binary.view(-1).detach().cpu().numpy()

    # Ensure no NaNs or Infs
    np.nan_to_num(preds_binary_np, copy=False)
    np.nan_to_num(true_masks_binary_np, copy=False)

    # Calculate and update metrics
    batch_f1 = f1_score(true_masks_binary_np, preds_binary_np, average="macro")
    batch_iou = jaccard_score(true_masks_binary_np, preds_binary_np, average="macro")

    return batch_f1, batch_iou


def validate(model: torch.nn.Module, validation_loader: DataLoader, criterion: torch.nn.BCELoss, config: ConfigSchema):
    model.eval()
    val_loss, val_f1, val_iou, num_val_samples = 0, 0, 0, 0
    val_loop = tqdm(validation_loader, leave=True, desc="Validation")

    device = get_device()
    with torch.no_grad():
        for batch in val_loop:
            images, true_masks = batch["image"], batch["mask"]

            images = images.to(device)
            true_masks = true_masks.to(device)
            with torch.autocast(device_type=device if device == "cuda" else "cpu"):
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks.float())

            val_loss += loss.item() * images.size(0)

            # Calculate and update metrics
            batch_f1, batch_iou = get_scores(masks_pred, true_masks, config.validation.threshold)

            val_f1 += batch_f1
            val_iou += batch_iou
            num_val_samples += images.size(0)
            val_loop.set_postfix(loss=val_loss / num_val_samples)

    return val_loss, val_f1, val_iou, num_val_samples
