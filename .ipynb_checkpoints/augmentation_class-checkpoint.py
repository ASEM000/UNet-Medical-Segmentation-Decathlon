import torch 
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

import matplotlib.pyplot as plt
import os
from typing import List,Tuple,Callable,Dict
from utility_functions import *
from dataset_dataloader import *
import math


class PreprocessAug:
    """
    Performs random cropping, flipping, and returns as feedable to NN
    """
    def __init__(self, device, batch_size=1, crop_size=256, full_size=320):
        self.device = device
        self.batch_size = batch_size

        # Size to crop if using augmentation at traning stage
        self.crop_size = crop_size
        # Size of the full image
        self.full_size = full_size
        
    def transform(self, input: torch.Tensor, label: torch.Tensor):
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            input, output_size=(self.crop_size, self.crop_size))
        input = TF.crop(input, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            input = TF.hflip(input)
            label = TF.hflip(label)

        # Random vertical flipping
        if random.random() > 0.5:
            input = TF.vflip(input)
            label = TF.vflip(label)
        return input, label

    def __call__(self, input: torch.Tensor, label: torch.Tensor, training=True):
        with torch.no_grad():
            if training:
                input, label = self.transform(input, label)
                input = input.view(self.batch_size, 1, self.crop_size, self.crop_size)
                label = label.view(self.batch_size, 1, self.crop_size, self.crop_size)
            else:
                input = input.view(self.batch_size, 1, self.full_size, self.full_size)
                label = label.view(self.batch_size, 1, self.full_size, self.full_size)
            input = input.to(self.device)
            label = label.to(self.device)
        return input, label
