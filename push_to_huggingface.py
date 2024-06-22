from src.model.unet import UNet
import torch
from torch import optim
from src.train.train import load_checkpoint

configs = {
    "n_channels":12,
    "n_classes":1,
    "bilinear":False
}

model = UNet(**configs)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scaler = torch.cuda.amp.GradScaler()
load_checkpoint("model_weights/pre_trained.chkpt", model, scaler, optimizer)
model.save_pretrained("claudezss/wildfire-segmentation")
model.push_to_hub("claudezss/wildfire-segmentation")

# model = UNet.from_pretrained("claudezss/wildfire-segmentation")
