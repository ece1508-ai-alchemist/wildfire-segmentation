import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score

from src.data_loader.dataset import WildfireDataset
from src.model.unet import UNet
from data.load_data import POST_FIRE_DIR
from utils.split_dataset import split_dataset

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
print(device)
dt = WildfireDataset(POST_FIRE_DIR, transforms=None)

# Set the number of epochs
NUM_EPOCHS = 25  
# Directory to save model weights
SAVE_PATH = 'model_weights'
# File to save metrics
CSV_FILE = 'training_metrics.csv'  

# Create directory to save model weights
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


# Initialize metrics dataframe
metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_f1', 'val_f1', 'train_iou', 'val_iou'])

# # 80% data for training
# n_val_test = int(len(dt) * 0.2)
# n_train = len(dt) - n_val_test
# train_set, val_test_set = random_split(dt, [n_train, n_val_test], generator=torch.Generator().manual_seed(0))

# # 10% data for validating and testing
# n_val = int(n_val_test / 2)
# n_test = n_val_test - n_val
# val_set, test_set = random_split(val_test_set, [n_val, n_test], generator=torch.Generator().manual_seed(0))
train_set, val_set, test_set = split_dataset(dt, train_frac=0.8, seed=0)

loader_args = dict(batch_size=8, num_workers=4)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

dt = WildfireDataset(POST_FIRE_DIR, data_transforms["train"])

model = UNet(n_channels=12, n_classes=1, bilinear=False)
model.to(device=device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()


 # Define a threshold for F1 score
THRESHOLD = 0.5 
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, train_f1, train_iou = 0, 0, 0
    num_samples = 0
    train_loop = tqdm(train_loader, leave=True)
    for batch in train_loop:
        images, true_masks = batch["image"], batch["mask"]
        # reduce img size
        images = torch.nn.functional.interpolate(images, size=(256, 256))
        true_masks = torch.nn.functional.interpolate(true_masks.float(), size=(256, 256))

        images = images.to(device)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        # Autocast for mixed precision
        with torch.cuda.amp.autocast():
            masks_pred = model(images)
            loss = criterion(masks_pred, true_masks.float())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Apply sigmoid to convert to probabilities and threshold to convert to binary mask
        preds_binary = (torch.sigmoid(masks_pred) > THRESHOLD).float()
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
        batch_f1 = f1_score(true_masks_binary_np, preds_binary_np, average='macro')
        batch_iou = jaccard_score(true_masks_binary_np, preds_binary_np, average='macro')
        train_loss += loss.item() * images.size(0)
        train_f1 += batch_f1
        train_iou += batch_iou
        num_samples += images.size(0)

        train_loop.set_description(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        train_loop.set_postfix(loss=train_loss / num_samples)

    # Validation step
    model.eval()
    val_loss, val_f1, val_iou, num_val_samples = 0, 0, 0, 0
    val_loop = tqdm(val_loader, leave=True, desc='Validation')
    with torch.no_grad():
        for batch in val_loop:
            images, true_masks = batch["image"], batch["mask"]
            images = F.interpolate(images, size=(256, 256))
            true_masks = F.interpolate(true_masks.float(), size=(256, 256))

            images = images.to(device)
            true_masks = true_masks.to(device)
            with torch.cuda.amp.autocast():
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)
            val_loss += loss.item() * images.size(0)

            # Calculate metrics
            preds_binary = (torch.sigmoid(masks_pred) > THRESHOLD).float()
            true_masks_binary = true_masks.float()

            # Move tensors to CPU and then convert to NumPy for scikit-learn functions
            # Flatten the tensors and move to CPU
            preds_binary_np = preds_binary.view(-1).detach().cpu().numpy()
            true_masks_binary_np = true_masks_binary.view(-1).detach().cpu().numpy()

            # Ensure no NaNs or Infs
            np.nan_to_num(preds_binary_np, copy=False)
            np.nan_to_num(true_masks_binary_np, copy=False)

            batch_f1 = f1_score(true_masks_binary_np, preds_binary_np, average='macro')
            batch_iou = jaccard_score(true_masks_binary_np, preds_binary_np, average='macro')

            val_f1 += batch_f1
            val_iou += batch_iou
            num_val_samples += images.size(0)
            val_loop.set_postfix(loss=val_loss / num_val_samples)

    # Calculating average metrics
    avg_train_loss = train_loss / num_samples
    avg_val_loss = val_loss / num_val_samples
    avg_train_f1 = train_f1 / len(train_loader)
    avg_val_f1 = val_f1 / len(val_loader)
    avg_train_iou = train_iou / len(train_loader)
    avg_val_iou = val_iou / len(val_loader)

     # Saving metrics to dataframe
    metrics_df = metrics_df.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 
                                    'val_loss': avg_val_loss, 'train_f1': avg_train_f1, 
                                    'val_f1': avg_val_f1, 'train_iou': avg_train_iou, 
                                    'val_iou': avg_val_iou}, ignore_index=True)

    # Save model weights
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'unet_epoch_{epoch + 1}.pt'))

# Saving metrics to CSV file
metrics_df.to_csv(CSV_FILE, index=False)

print("Training completed and metrics saved.")   