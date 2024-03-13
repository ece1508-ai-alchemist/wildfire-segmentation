import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data.load_data import POST_FIRE_DIR
from src.data_loader.dataset import WildfireDataset
from src.model.unet import UNet

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

n_val = int(len(dt) * 0.1)
n_train = len(dt) - n_val
train_set, val_set = random_split(dt, [n_train, n_val], generator=torch.Generator().manual_seed(0))

loader_args = dict(batch_size=8)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

model = UNet(n_channels=12, n_classes=1, bilinear=False)


for batch in train_loader:
    images, true_masks = batch["image"], batch["mask"]
    masks_pred = model(images)
    print("1")
