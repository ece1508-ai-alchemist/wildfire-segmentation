from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import augmentation
from augmentation import (
    DoubleToTensor, DoubleCompose,
    DoubleHorizontalFlip, DoubleVerticalFlip, DoubleElasticTransform,
    GaussianNoise
)

# Helper function to set up data transforms and get the data loader
def get_loader(data_set, is_train, mean, std, loader_args):

    if is_train:
      image_mask_transform = DoubleCompose([
          DoubleToTensor(),
          #DoubleElasticTransform(alpha=250, sigma=10),
          DoubleHorizontalFlip(),
          DoubleVerticalFlip()
      ])
      image_transform = transforms.Compose([
          #transforms.ColorJitter(brightness=0.4),
          #transforms.Normalize(mean, std),
          GaussianNoise()
      ])
    else:
      image_mask_transform = DoubleCompose([
          DoubleToTensor()
      ])
      image_transform = transforms.Compose([
          #transforms.Normalize(mean, std)
      ])

    data_set.__set_transforms__(image_mask_transform, 
                                  image_transform)

    loader = DataLoader(
        data_set,
        **loader_args
    )
    return loader

# Helper function to calculate means
def get_mean_std(trainLoader):
  
    imgs = None
    for batch in trainLoader:
        images = batch['image']
        if imgs is None:
            imgs = images
        else:
            imgs = torch.cat((imgs, images), dim=0)
    mean = imgs.mean(dim=(0, 2, 3))
    std = imgs.std(dim=(0, 2, 3))

    print(mean, std)
    return mean, std


  