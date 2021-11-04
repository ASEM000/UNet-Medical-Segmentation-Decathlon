
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class dense_conv_unit(nn.Module):
    def __init__(self,in_channels:int,out_channels:int)->None:
        super().__init__()
        self.conv = nn.Sequential(
                                nn.Conv2d(in_channels=in_channels,out_channels = 4*out_channels ,kernel_size=1,padding=0) , 
                                nn.BatchNorm2d(4*out_channels) , 
                                nn.ReLU(),

                                nn.Conv2d(in_channels=4*out_channels,out_channels = out_channels ,kernel_size=3,padding=1) , 
                                nn.BatchNorm2d( out_channels) , 
                                nn.ReLU()
                                )
    
    def forward(self,tensor:torch.Tensor)->torch.Tensor :
        return self.conv(tensor)
    
class densenet_conv_block_class(nn.Module):
    
    def __init__(self,in_channels:int,out_channels:int, reps:int=4)->None:
        super().__init__()

        self.layers=nn.ModuleDict()
        self.reps = reps

        channels_shape = torch.arange(in_channels,out_channels*(reps+1),out_channels)

        for i in range(reps):
            self.layers[f'c{i}'] =dense_conv_unit(in_channels = channels_shape[i],out_channels=out_channels) 
        
        self.layers[f'f0_0'] =nn.Conv2d(in_channels=channels_shape[-1],out_channels= out_channels,kernel_size=1,padding = 0)

    def forward(self,tensor:torch.Tensor)->torch.Tensor :

        for i in range(self.reps):
            x = self.layers[f'c{i}'](tensor)
            tensor = torch.cat([x,tensor],dim=1)

        tensor = self.layers['f0_0'](tensor)

        return tensor
    
    
class unet_conv_block_class(nn.Module):
    def __init__(self,in_channels:int,out_channels:int)->None:
        super().__init__()

        '''
        Modified convolution block in UNet : https://arxiv.org/abs/1505.04597

        (conv2d->batchnorm->relu) x 2

        '''

        self.block = nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels,out_channels = out_channels,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=out_channels,out_channels = out_channels,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True)
                                   )

    def forward(self,tensor:torch.Tensor)->torch.Tensor:

        return self.block(tensor)

class upscale_class(nn.Module):
    ''' 
    -Use upsample instead of Conv2DTranspose to avoid checkboard artifacts [https://distill.pub/2016/deconv-checkerboard/]
    '''
    def __init__(self,in_channels:int,out_channels:int)->None:
        super().__init__()

        self.c0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0)

    def forward(self,tensor:torch.Tensor)->torch.Tensor :
        '''
        [1] upscale the rows and cols dim by a factor of 2 bilinearly
        [2] reduce filters maps by a factor of 2 by channel pooling in case of UNet
        '''

        c0 = self.c0(tensor)
        c1 = self.c1(c0)

        return c1
