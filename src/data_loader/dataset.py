from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from src.data_loader.augmentation import (
    DoubleToTensor
)


class WildfireDataset(Dataset):
    def __init__(self, path_to_data: Union[str, Path], transforms):
        self.path_to_data = Path(path_to_data)
        self.transforms = transforms
        self._datasets = load_from_disk(str(self.path_to_data.absolute()))

        self.keys: List[Tuple[str, int]] = [
            (k, i) for k in self._datasets.data.keys() for i in range(len(self._datasets[k]))
        ]
        self.idxs = range(len(self.keys))
        self.mask_key = "mask"
        self.pre_fire_key = "pre_fire"
        self.post_fire_key = "post_fire"

        # Initialize
        # by default, just convert the NP images to Tensor
        self.image_mask_transform = DoubleToTensor()
        self.image_transform = None

    def __getitem__(self, idx):
        key, img_index = self.keys[idx]
        data = self._datasets[key][img_index]
        img = np.array(data[self.post_fire_key])/10000
        mask = np.array(data[self.mask_key])

        if self.image_mask_transform:
            img, mask = self.image_mask_transform(
                img, mask
            )
        if self.image_transform:
            img = self.image_transform(img)

        return {
            "image": img,
            "mask": mask,
        }
    def __set_transforms__(self, image_mask_transforms, image_transforms):
        self.image_transform = image_transforms
        self.image_mask_transform = image_mask_transforms


    def __len__(self):
        return len(self.idxs)
