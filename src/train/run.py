from pathlib import Path

import torch
import yaml

from src.model.unet import UNet
from src.train.config import ConfigSchema
from src.train.funcs import train

ConfigFile = Path(__file__).parent / "config.yaml"

Config = ConfigSchema(**yaml.safe_load(open(ConfigFile, "r")))


def execute():
    model = UNet(n_channels=12, n_classes=1, bilinear=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.train.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    train(model, optimizer, scaler, criterion, Config)


if __name__ == "__main__":
    execute()
