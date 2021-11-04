
import torch 
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

import matplotlib.pyplot as plt
import os

from typing import List,Tuple,Callable,Dict

from utility_functions import *
from dataset_dataloader import *
from variable_unet import *
from augmentation_class import *


from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib as mpl

from collections import defaultdict
from time import time


# Define forward pass used for both training/validation
def forward_pass( input : torch.Tensor, 
                  label : torch.Tensor, 
                  model : torch.nn.Module,
                  dice_gam):
    
    # forward pass
    output = model(input).float()

    # calculate loss
    step_bceloss = BCE_LOSS(output, label)
    step_diceloss = DICE_LOSS(output,label)
    step_loss = step_bceloss
    step_metric = step_diceloss

    return output, step_loss, step_metric



def normalize_im(tensor: torch.Tensor)->torch.Tensor:
    
    large = torch.max(tensor).cpu().data
    small = torch.min(tensor).cpu().data
    diff = large - small
    
    normalized_tensor = (tensor.clamp(min=small, max=large) - small) * (torch.tensor(1) / diff)
    
    return normalized_tensor

def postprocess_im(tensor: torch.Tensor ,th: float = 0.5):
    """
    1. Applies sigmoid layer to U-Net output
    2. Thresholds to certain value. 
    i.e.) Values lower than th = 0 / Values higher than th = 1
    """
    
    tensor_th = tensor.clone()
    tensor_th = sigmoid(tensor_th)
    tensor_th[tensor_th <= th] = 0
    tensor_th[tensor_th > th] = 1
    
    return tensor_th

# def visualize_image( input: torch.Tensor, 
#                      output: torch.Tensor,
#                      label: torch.Tensor, 
#                      th=0.5, 
#                      training=True, 
#                      save=False,
#                      save_fname=None):
    
#     input = normalize_im(input.squeeze())
#     output_th = postprocess_im(output, th).squeeze()
    
#     output = normalize_im(output.squeeze())
#     label = label.squeeze()
    
#     iol = torch.cat((input, output, output_th, label), dim=1)

#     plt.figure(figsize=(10,40))
    
#     plt.imshow(iol.cpu().detach().numpy())
    
#     if not training:
#         plt.title('validation')
        
#     plt.axis('off')
#     plt.show()
    
#     if save:
#         plt.savefig(save_fname)
        
#     return output_th



def dice_coeff(input, label):
    smooth = 1.
    iflat = input.contiguous().view(-1)
    lflat = label.contiguous().view(-1)
    intersection = (iflat * lflat).sum()
    
    return ((2. * intersection + smooth) / (iflat.sum() + lflat.sum() + smooth))


class dice_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, label):
        dice_val = dice_coeff(input, label)
        return 1 - dice_val


# logger
def log_epoch(epoch, epoch_loss, epoch_metrics, writer, elapsed_secs, training=True):
    
    mode = 'Training' if training else 'Validation'
    
    string = \
    f'''Epoch {epoch:03d} {mode}.\tloss: {epoch_loss[i-1]:.4e}.\tTime: {elapsed_secs // 60} min {elapsed_secs % 60} sec\tDice={epoch_metrics:.4e}'''
    
    print(string)

# checkpointer
def save(model, epoch, ckpt_path):
    
    save_dict = {'model_state_dict': model.state_dict()}
    save_path = ckpt_path / f'ckpt_{epoch:03d}.tar'

    torch.save(save_dict, str(save_path))
    print(f'Saved Checkpoint to {str(save_path)}')

# metric calculator
def get_step_metric(output, label):
    
    dice = dice_coeff(output, label)   
    
    return dice
