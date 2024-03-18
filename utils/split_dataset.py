import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score

def split_dataset(dataset, train_frac=0.8, seed=0):
    n_total = len(dataset)
    n_train = int(n_total * train_frac)
    n_val_test = n_total - n_train
    
    # Ensure that the sum of n_val and n_test equals n_val_test
    n_val = n_val_test // 2
    n_test = n_val_test - n_val

    train_set, val_test_set = random_split(dataset, [n_train, n_val_test], generator=torch.Generator().manual_seed(seed))
    val_set, test_set = random_split(val_test_set, [n_val, n_test], generator=torch.Generator().manual_seed(seed))

    return train_set, val_set, test_set
