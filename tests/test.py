import torch
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score

from src.data_loader.dataset import WildfireDataset
from src.model.unet import UNet
from data.load_data import POST_FIRE_DIR
from utils.split_dataset import split_dataset

# Initialize metrics dataframe
metrics_df = pd.DataFrame(columns=['test_loss', 'test_f1', 'test_iou'])
# File to save metrics
CSV_FILE = 'test_metrics.csv'  

test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
# Load and split the dataset
dt = WildfireDataset(POST_FIRE_DIR, test_transforms)
_, _, test_set = split_dataset(dt, train_frac=0.8, seed=0)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

# Load the model
model = UNet(n_channels=12, n_classes=1, bilinear=False)
# Replace X with the epoch number 
model_path = 'model_weights/unet_epoch_X.pt'  
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
criterion = torch.nn.BCEWithLogitsLoss()

# Evaluate the model
test_loss, test_f1, test_iou = 0, 0, 0
num_test_samples = 0
 # Define a threshold for F1 score
THRESHOLD = 0.5 

with torch.no_grad():
    for batch in tqdm(test_loader):
        images, true_masks = batch["image"], batch["mask"]
        images = F.interpolate(images, size=(256, 256))
        true_masks = F.interpolate(true_masks.float(), size=(256, 256))
        with torch.cuda.amp.autocast():
            masks_pred = model(images)
            loss = criterion(masks_pred, true_masks)
        test_loss += loss.item() * images.size(0)
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

        test_f1 += batch_f1
        test_iou += batch_iou
        num_test_samples += images.size(0)
    

avg_test_loss = test_loss / num_test_samples
avg_test_f1 = test_f1 / len(test_loader)
avg_test_iou = test_iou / len(test_loader)
# Saving metrics to dataframe
metrics_df = metrics_df.append({'test_loss': avg_test_loss, 
                                'test_f1': avg_test_f1, 
                                'test_iou': avg_test_iou
                                },ignore_index=True)

# Saving metrics to CSV file
metrics_df.to_csv(CSV_FILE, index=False)
print(f"Test Loss: {avg_test_loss}, Test F1: {avg_test_f1}, Test IoU: {avg_test_iou}")