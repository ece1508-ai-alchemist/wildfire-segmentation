from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

from data.load_data import POST_FIRE_DIR
from src.data_loader.augmentation import DoubleToTensor


class BaseDataset(Dataset):
    mask_key = "mask"
    pre_fire_key = "pre_fire"
    post_fire_key = "post_fire"

    def __init__(self, datasets, keys):
        self._datasets = datasets
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __set_transforms__(self, image_mask_transforms, image_transforms):
        self.image_transform = image_transforms
        self.image_mask_transform = image_mask_transforms

    def __getitem__(self, idx):
        key, img_index = self.keys[idx]
        data = self._datasets[key][img_index]
        img = np.array(data[self.post_fire_key]) / 10000
        mask = np.array(data[self.mask_key])

        if self.image_mask_transform:
            img, mask = self.image_mask_transform(img, mask)
        if self.image_transform:
            img = self.image_transform(img)

        return {
            "image": img,
            "mask": mask,
        }


class WildfireDataset:
    def __init__(self, path_to_data: Union[str, Path] | None = POST_FIRE_DIR, transforms=None, use_local: bool = True):
        self.transforms = transforms

        if use_local:
            print("Load data from local")
            self.path_to_data = Path(path_to_data)
            self._datasets = load_from_disk(str(self.path_to_data.absolute()))
        else:
            print("Load data from network")
            self._datasets = load_dataset("DarthReca/california_burned_areas", name="post-fire", trust_remote_code=True)

        self.keys: List[Tuple[str, int]] = [
            (k, i) for k in self._datasets.data.keys() for i in range(len(self._datasets[k]))
        ]

        self.idxs = range(len(self.keys))

        n_val_test = int(len(self.keys) * 0.2)
        n_train = len(self.keys) - n_val_test
        self.train_idxs = self.idxs[:n_train]
        val_and_test_idxs = self.idxs[n_train:]

        mid_index = n_val_test // 2

        self.val_idxs = val_and_test_idxs[:mid_index]
        self.test_idxs = val_and_test_idxs[mid_index:]

        # Initialize
        # by default, just convert the NP images to Tensor
        self.image_mask_transform = DoubleToTensor()
        self.image_transform = None

        self.train = BaseDataset(self._datasets, [self.keys[i] for i in self.train_idxs])
        self.val = BaseDataset(self._datasets, [self.keys[i] for i in self.val_idxs])
        self.test = BaseDataset(self._datasets, [self.keys[i] for i in self.test_idxs])
