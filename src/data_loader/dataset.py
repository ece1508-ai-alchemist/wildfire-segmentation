from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


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

    def __getitem__(self, idx):
        key, img_index = self.keys[idx]
        data = self._datasets[key][img_index]
        img = np.array(data[self.post_fire_key])
        mask = np.array(data[self.mask_key])

        if self.image_mask_transform:
            img, mask = self.image_mask_transform(
                img, mask
            )
        if self.image_transform:
            img = self.image_transform(img)

        return {
            "image": torch.tensor(img, dtype=torch.float32).permute((2, 0, 1)),
            "mask": torch.tensor(mask).permute((2, 0, 1)),
        }
    def __set_transforms__(self, image_mask_transforms, image_transforms):
        self.image_transform = image_transforms
        self.image_mask_transform = image_mask_transforms


    def __len__(self):
        return len(self.idxs)
