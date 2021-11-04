'''

'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from collections import OrderedDict

from unet_blocks import *


class variable_unet_class(nn.Module):
    '''
    class implementation of UNet: https://arxiv.org/abs/1505.04597

    -Modifications to the paper (as approved by TA  )
    [1] addition of batchnorm
    [2] use same padding instead of valid padding
    [3] use upsample/Conv2dTranspose


    '''

    def __init__(self,
               in_channels:int=1,               # channels of input (1 for gray 3 for RGB)
               out_channels:int=1,              # number of classes  (1 for binary classification)
               init_filters:int=64,             # initial number of feature maps in the convolution block 
               block_count:int = 4 ,            # number of blocks in each path
               upsample:bool=True,              # use upsampling layer (True) , use Conv2dTranspose (False)
               conv_block:str ='unet' )->None:  # convolution block choice ( unet, densenet)     

        super().__init__()

        '''
        ######### Contents of BiS800B UNet #########

        **Contractive block consists of 
        [1] (conv->batchnorm->relu) x2
        [2] maxpool


        **Bottleneck block consists of 
        [*] (conv->batchnorm->relu) x2


        **Expansive block consists of 
        [1] Conv2DTranspose  or upscale_class
            - double the size of row and col 
            - halves the size of feature maps
        [2] padding and concatenating
            - pad the smaller tensor(from expansive path) to match the row,col dim of the other tensor(from contractive path)
            - concatenate the two tensors featuremap-wise
        [3] (conv->batchnorm->relu) x2


        ######### Naming convention #########

        *d0_1 => block_number = 0 , operation = (conv->batchnorm->relu) x2
        *d0_2 => block_number = 0 , operation = maxpool previous output

        *u0_1 => expansive block corresponding to block 0 in contractive path , operation = doubling row,col size and halving channels size of previous layer
        *u0_2 => expansive block corresponding to block 0 in contractive path , operation = pad the previous layer from expansive path (u0_1) and concatenate with corresponding layer from contractive path (d0_1)
        *u0_3 => expansive block corresponding to block 0 in contractive path , operation = (conv->batchnorm->relu) x2 of previous layer (u0_2)

        *b0_1 => bottleneck layer
        *f0_1 => final output layer

        '''

        self.in_channels = in_channels
        self.out_channels = out_channels  
        self.init_filters = init_filters
        self.block_count = block_count
        self.upsample = upsample
        self.layers = nn.ModuleDict()

        # preconditions 
        assert block_count >= 4 , f'block count must be >= 4. {block_count} < 4'

        if conv_block == 'unet' :
            conv_block_class = unet_conv_block_class
            
        elif conv_block =='densenet':
            conv_block_class = densenet_conv_block_class
            
        else:
            raise ValueError(f'{conv_block} is not implemented ')

        '''
        contractive path 
        '''

        self.layers['d0_1'] = conv_block_class(1,init_filters)
        self.layers['d0_2'] = nn.MaxPool2d(2)

        for i in range(1,block_count):
            self.layers[f'd{i}_1'] = conv_block_class(init_filters*2**(i-1),init_filters*2**(i))
            self.layers[f'd{i}_2'] = nn.MaxPool2d(2)


        '''
        bottleneck
        '''

        self.layers['b0_1']=conv_block_class(init_filters*2**(block_count-1),init_filters*2**(block_count))

        '''
        expansive path
        '''

        conv_block_class = unet_conv_block_class

        if upsample :

            for i in range(block_count,0,-1):
                self.layers[f'u{i-1}_1'] = upscale_class(in_channels=init_filters*2**(i),out_channels=init_filters*2**(i-1))
                self.layers[f'u{i-1}_3'] = conv_block_class(init_filters*2**(i),init_filters*2**(i-1))

        else:

            for i in range(block_count,0,-1):
                self.layers[f'u{i-1}_1'] = nn.ConvTranspose2d(in_channels=init_filters*2**i,out_channels=init_filters*2**(i-1),kernel_size=2,stride=2)
                self.layers[f'u{i-1}_3'] = conv_block_class(init_filters*2**(i),init_filters*2**(i-1))

        self.layers['f0_1'] = nn.Conv2d(in_channels=init_filters,out_channels=out_channels,kernel_size=1,padding=0)


    def forward(self,tensor:torch.Tensor)->torch.Tensor:


        '''
        Contractive path
        '''

        block_count     =self.block_count

        result = OrderedDict()

        result['d0_1'] = self.layers['d0_1'](tensor)
        result['d0_2'] = self.layers['d0_2'](result['d0_1'])

        for i in range(1,block_count) :
            result[f'd{i}_1'] = self.layers[f'd{i}_1'](result[f'd{i-1}_2'])
            result[f'd{i}_2'] = self.layers[f'd{i}_2'](result[f'd{i}_1']) 

        '''
        bottleneck
        '''

        result['b0_1'] = self.layers['b0_1'](result[f'd{block_count-1}_2'])

        '''
        expansive path 
        '''

        result[f'u{block_count-1}_1'] = self.layers[f'u{block_count-1}_1'](result['b0_1'])
        result[f'u{block_count-1}_2'] = self.pad_and_cat(result[f'u{block_count-1}_1'] , result[f'd{block_count-1}_1'])  # skip connection
        result[f'u{block_count-1}_3'] = self.layers[f'u{block_count-1}_3'](result[f'u{block_count-1}_2'])

        for i in range(block_count-1,0,-1):
            result[f'u{i-1}_1'] = self.layers[f'u{i-1}_1'](result[f'u{i}_3'])
            result[f'u{i-1}_2'] = self.pad_and_cat(result[f'u{i-1}_1'],result[f'd{i-1}_1']) # skip connection
            result[f'u{i-1}_3'] = self.layers[f'u{i-1}_3'](result[f'u{i-1}_2'])

        f0_1 =self.layers['f0_1'](result['u0_3'])

        return f0_1


    @staticmethod
    def pad_and_cat(x1:torch.Tensor,x2:torch.Tensor)->torch.Tensor:
        '''

        Args:
        x1,x2 is in shape of (sample,channels,row,col)

        Objective:
        concatenate the two tensors filter-wise

        Steps:
        1. equalize the row and cols dimension of two tensors , by padding x1(smaller) tensor
        2. concatenate the two tensors filter-wise

        Example :
        >>a,b= torch.rand(1,64,572,572) , torch.rand(1,64,570,570)

        >>pad_and_cat(a,b).shape
        >>Error x1 tensor must be smaller or equal to x2 tensor. torch.Size([1,64, 572, 572]) !< torch.Size([1,64, 570, 570])

        >>pad_and_cat(b,a).shape
        >>Torch.size[1,128,572,572] : 

        '''
        # find the dimension difference
        diff_row = x2.size()[2] - x1.size()[2]
        diff_col = x2.size()[3] - x1.size()[3]

        '''
        preconditions
        '''
        # 1.check length of dims
        assert len(x1.shape)== len(x2.shape) == 4 , f'number of dimensions must equal to each other and equal to 4 . {len(x1.shape)}!=4 or {len(x2.shape)}!=4 or {len(x1.shape)} != {len(x2.shape)}'

        # 2.check that x1>=x2 in row and col dim
        assert diff_row >=0 and diff_col >= 0 ,f'x1 tensor must be smaller or equal to x2 tensor. {x1.shape} !< {x2.shape}'

        # 3.check that x1 ==x2 filters wise (specific to UNet)
        assert x1.shape[1] == x2.shape[1] ,f'filter size of x1 and x2 must be same . {x1.shape[1]} != {x2.shape[1]}'



        # create the padding tuple (before_col,after_col,before_row,after_row)
        pad_tuple = (diff_col // 2, diff_col - diff_col // 2,diff_row // 2, diff_row - diff_row // 2)
        x1 = F.pad(x1, pad_tuple)

        #return the concatenated tensor
        conc = torch.cat([x2, x1], dim=1)

        return conc
