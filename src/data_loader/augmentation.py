"""Model Trainer

author: Masahiro Hayashi

This script defines custom image transformations that simultaneously transform
both images and segmentation masks.
"""
# Most of the code from From https://github.com/hayashimasa/UNet-PyTorch

import torchvision.transforms.functional as TF
# from torchvision.transforms import Compose
from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class GaussianNoise:
    """Apply Gaussian noise to tensor."""

    def __init__(self, mean=0., std=1., p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        noise = 0
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class DoubleToTensor:
    """Apply horizontal flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        if weight is None:
            return torch.tensor(image, dtype=torch.float32).permute((2, 0, 1)), torch.tensor(mask).permute((2, 0, 1))
        weight = weight.view(1, *weight.shape)
        return TF.to_tensor(image), TF.to_tensor(mask), weight

    def __repr__(self):
        return self.__class__.__name__ + '()'

class DoubleHorizontalFlip:
    """Apply horizontal flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        p = random.random()
        if p < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if weight is None:
            return image, mask
        elif p > self.p:
            weight = TF.hflip(weight)
        return image, mask, weight

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class DoubleVerticalFlip:
    """Apply vertical flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        p = random.random()
        if p < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if weight is None:
            return image, mask
        elif p > self.p:
            weight = TF.hflip(weight)
        return image, mask, weight

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'
    
class CustomColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.transform = transforms.ColorJitter(brightness=self.brightness,contrast=self.contrast,saturation=self.saturation,hue=self.hue)

    def __call__(self, img):
        # Separate the 12 channels into 4 groups of 3 channels each
        channels = torch.chunk(img, 4, dim=0)

        augmented_img = None
        # Apply ColorJitter to each group of 3 channels
        for channel_group in channels:
            # Combine the 3 channels into one image
            #print(channel_group.shape)

            # Apply ColorJitter to the image
            augmented_channel_group_img = self.transform(channel_group)
            
            # Put it back into original img    
            if augmented_img is None: 
                augmented_img = augmented_channel_group_img
            else:
                augmented_img = torch.cat((augmented_img, augmented_channel_group_img), dim=0)      

        return augmented_img

class DoubleElasticTransform:
    """Based on implimentation on
    https://gist.github.com/erniejunior/601cdf56d2b424757de5"""

    def __init__(self, alpha=250, sigma=10, p=0.5, seed=None, randinit=True):
        if not seed:
            seed = random.randint(1, 100)
        self.random_state = np.random.RandomState(seed)
        self.alpha = alpha
        self.sigma = sigma
        self.p = p 
        self.randinit = randinit


    def __call__(self, image, mask, weight=None):
        if random.random() < self.p:
            if self.randinit:
                seed = random.randint(1, 100)
                self.random_state = np.random.RandomState(seed)
                self.alpha = random.uniform(100, 500)
                self.sigma = random.uniform(10, 30)
                #self.alpha = 1000
                #self.sigma = 40
                #print(self.alpha)
                #print(self.sigma)

            dim = image.shape
            dx = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            dy = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )

            #print(dx, dy)

            #image = image.view(*dim[1:]).numpy()
            #mask = mask.view(*dim[1:]).numpy()
            #x, y = np.meshgrid(np.arange(dim[1]), np.arange(dim[2]))
            #indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            #image = map_coordinates(image, indices, order=1)
            #mask = map_coordinates(mask, indices, order=1)

            x, y = np.meshgrid(np.arange(dim[1]), np.arange(dim[2]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            for i in range(len(image)):
                # Apply transformations
                image[i] = torch.Tensor(map_coordinates(image[i], indices, order=1).reshape(*dim[1:]))
            for i in range(len(mask)):
                mask[i] = torch.Tensor(map_coordinates(mask[i], indices, order=1).reshape(*dim[1:]))

            #image, mask = image.reshape(dim), mask.reshape(dim)
            #image, mask = torch.Tensor(image), torch.Tensor(mask)
            if weight is None:
                return image, mask
            weight = weight.view(*dim[1:]).numpy()
            weight = map_coordinates(weight, indices, order=1)
            weight = weight.reshape(dim)
            weight = torch.Tensor(weight)

        return (image, mask) if weight is None else (image, mask, weight)


class DoubleCompose(transforms.Compose):

    def __call__(self, image, mask, weight=None):
        if weight is None:
            for t in self.transforms:
                image, mask = t(image, mask)
            return image, mask
        for t in self.transforms:
            image, mask, weight = t(image, mask, weight)
        return image, mask, weight

###############################################################################
# For testing
###############################################################################
if __name__ == '__main__':
    print("placeholder")
    