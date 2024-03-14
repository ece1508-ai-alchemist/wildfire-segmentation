import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from data.load_data import POST_FIRE_DIR
from src.data_loader.dataset import WildfireDataset
from src.model.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dt = WildfireDataset(POST_FIRE_DIR, transforms=None)

# 80% data for training
n_val_test = int(len(dt) * 0.2)
n_train = len(dt) - n_val_test
train_set, val_test_set = random_split(dt, [n_train, n_val_test], generator=torch.Generator().manual_seed(0))

# 10% data for validating and testing
n_val = int(n_val_test / 2)
n_test = n_val_test - n_val
val_set, test_set = random_split(val_test_set, [n_val, n_test], generator=torch.Generator().manual_seed(0))

loader_args = dict(batch_size=8)
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

loop = tqdm(n_train)

for i, batch in enumerate(tqdm(train_loader)):
    images, true_masks = batch["image"], batch["mask"]

    # reduce img size
    images = torch.nn.functional.interpolate(images, size=(256, 256))
    true_masks = torch.nn.functional.interpolate(true_masks.float(), size=(256, 256))

    images = images.to(device)
    true_masks = true_masks.to(device=device, dtype=torch.long)

    with torch.cuda.amp.autocast():
        predictions = model(images)
        masks_pred = model(images)
        loss = criterion(masks_pred, true_masks.float())
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    loop.set_postfix(loss=loss.item())


m1 = masks_pred.cpu()[0]
m1 = torch.sigmoid(m1) > 0.6
m1 = m1.long().squeeze().numpy()
plt.imshow(m1)
plt.imshow(true_masks.cpu()[0].squeeze().numpy())
