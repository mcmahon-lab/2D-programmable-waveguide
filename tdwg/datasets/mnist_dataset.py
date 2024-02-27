"""
1. Code for loading up MNIST dataset
2. Simple functions for preprocessing MNIST digits
"""


import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

import torch.nn.functional as F
import torch.nn as nn
import tdwg.lib.ftutils_torch as ftutils_torch

def filter_dataset_by_labels(dataset, labels):
    mask = torch.isin(dataset.targets, torch.tensor(labels))
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]
    return dataset

def create_MNIST_datasets(label_list = [0, 1, 2]):
    data_dir = "../ml_datasets"
    torch.manual_seed(0)
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),])
    # transform = transforms.Compose(
    #     [transforms.ToTensor()])
    
    MNIST(data_dir, train=True, download=True)
    MNIST(data_dir, train=False, download=True)
    
    mnist_full = MNIST(data_dir, train=True, transform=transform)
    mnist_test = MNIST(data_dir, train=False, transform=transform)
    
    mnist_full = filter_dataset_by_labels(mnist_full, label_list)
    mnist_test = filter_dataset_by_labels(mnist_test, label_list)
    
    train_len = int(np.floor(len(mnist_full)*0.95))
    val_len = len(mnist_full)-train_len
    
    mnist_train, mnist_val = random_split(mnist_full, [train_len, val_len])
    return mnist_train, mnist_val, mnist_test


def ft_downsampling(x):
    x = x.to(torch.complex128)
    x = x[:, 0, :, :]
    x_ft = ftutils_torch.fft2_centered(x)
    x_ft_crop = x_ft[:, 10:17, 10:17]
    x = x_ft_crop.reshape(x_ft_crop.shape[0], -1)
    x = x/600
    return x

def avgpool_downsampling(x):
    x = F.avg_pool2d(x, kernel_size=4, stride=4)
    x = x[:, 0, :, :].flatten(start_dim = 1)
    # x = 3.0*x #hardcoding in data_rescale_factor that Martin found
    return x