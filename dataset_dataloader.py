
import torch 
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
import numpy as np

import matplotlib.pyplot as plt
import os
from typing import List,Tuple,Callable,Dict
from utility_functions import *
import math


class SegDataset(torch.utils.data.Dataset):
    def __init__(self,root:str,sift:bool  = False,test:bool  = False ,samples:int=-1,parallel:bool=True)->None:
        '''
        **Args :
        
        root    : path of the folder with training images , label images
        test    : boolean to indicate if the dataset is test dataset
        sift    : exclude labels with only zeros
        samples :defines number of samples (all if -1)
        
        
        **Folder structure
        
        root
        |__images
            |__numpy arrays
        |__labels
            |__numpy arrays
                
        '''
        
        self.root = root
        self.sift = sift
        self.test = test
        
        # get all folders path 
        folder_paths  = sorted(os.listdir(root))
        
        # assert "images" and "labels" folders are present under the root foldeer
        assert 'images' in folder_paths , 'please check you folder structure . folder must contain  (images) folder'
        assert 'labels' in folder_paths , 'please check you folder structure . folder must contain  (labels) folder'
        
        # images and labels array paths
        images_paths = sorted(get_file_paths(root,'images','.nii.gz'))
        labels_paths = sorted(get_file_paths(root,'labels','.nii.gz'))
        
        self.sample_count = len(images_paths)
        
        # load the first image , to define the array info
        image_data = load_numpy_arrays(images_paths,parallel=False)
        label_data = load_numpy_arrays(labels_paths,parallel=False)

#         # read numpy arrays 
#         for index,(path_image,path_label) in enumerate(zip(images_paths[1:],labels_paths[1:])) :
            
#             image_numpy = np.load(path_image)
#             label_numpy = np.load(path_label)
            
#             # channels last format
#             image_data = np.concatenate([image_numpy,image_data],axis=2)
#             label_data = np.concatenate([label_numpy,label_data],axis=2)
        
        if sift :
            '''
            exclude all-zeros channels
            
            Note : exclusion before concatenation is better for larger dataset
            
            '''
            
            non_zero_mask_indices = list(find_non_zero_mask_indices(label_data))
            
            image_data = image_data[:,:,non_zero_mask_indices]
            label_data = label_data[:,:,non_zero_mask_indices]
        
        
        '''
        (1) change format from channel last to channel first 
        (2) transform to torch tensor
        '''

        self.image_data = torch.from_numpy(np.rollaxis(image_data,2,0))
        self.label_data = torch.from_numpy(np.rollaxis(label_data,2,0))

        self.images_paths = images_paths
        self.labels_paths = labels_paths
        
        
                                  
    def __getitem__(self,index:int)->Tuple[torch.Tensor,torch.Tensor]:
        '''
        return torch tensor with channel first format [channel,row,col] 
        
        '''
        return self.image_data[[index]],self.label_data[[index]]
                        
    def __len__(self)->int :
        ''' 
        count the total number of channels in all samples after concatenation
        '''
        return len(self.image_data)
    
    
    def __repr__(self)->str :
        string = f'''
        files_count        ={self.sample_count}
        all_channels_count ={len(self.image_data)}
        sample_shape       ={self.image_data[[0]].shape}
        root_path          ={self.root}
        sift               ={self.sift}
        '''
        return string

    

class heart_dataloader:
    
    def __init__(self, root:str,
                       sift:bool            =False,
                       test:bool            =False,
                       batch_size:int       =1,
                       train_val_ratio:float= 0.8)->None:
        
        # root folder (train or test)
        self.root= root
        
        # keep zero masks if False
        self.sift = sift
        
        # split train+validation if False , no split if True
        self.test = test
        
        # minibatch size
        self.batch_size = batch_size
        
        #split ration
        self.train_val_ratio = train_val_ratio
        
        # paths of the loaded images files paths
        self.images_paths = []
        
        # paths of the loaded label files paths
        self.labels_paths = []
        
        self.train_dl = None
        self.val_dl = None
        self.test_dl =None      
        
        self.__get_dataloader()
        

        
        
    def __get_dataloader(self):

        """
        Returns Pytorch dataloaders of Segdataset

        args
            root       : path to root folder
            sift       : boolean to enable only non zero mask
            test       : boolean to return only test data (i,e, no splitting into validation and train)
            batch_size
            log        : boolean to check for loaded file paths
            samples    : number of samples

        """
        sift = self.sift
        root = self.root
        batch_size = self.batch_size
        train_val_ratio = self.train_val_ratio
        test = self.test
        
        if not test :
            
            assert 0<train_val_ratio<1 , f'train to validation ratio must be in (0,1). found {train_val_ratio}'

            dataset = SegDataset(root=root,sift=sift)

            # total_batches = len(dataset) // batch_size
            train_samples = int((len(dataset)*train_val_ratio)//batch_size *batch_size)
            val_samples   = int((len(dataset)*(1-train_val_ratio))//batch_size *batch_size)

            self.images_paths.append(dataset.images_paths)
            self.labels_paths.append(dataset.labels_paths)

            dataset = torch.utils.data.Subset(dataset,range(train_samples + val_samples))

            train_dataset,validation_dataset =  torch.utils.data.dataset.random_split( dataset, [train_samples,val_samples])

            train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
            validation_dataloader = DataLoader(validation_dataset,batch_size=batch_size,shuffle=False)
            
            self.train_dl = train_dataloader
            self.val_dl = validation_dataloader
            # return train_dataloader,validation_dataloader

        else:

            test_dataset = SegDataset(root=root,sift=sift)     

            self.images_paths.append(test_dataset.images_paths)
            self.labels_paths.append(test_dataset.labels_paths)

            test_dataset = torch.utils.data.Subset(test_dataset,range(int(len(test_dataset)//batch_size*batch_size)))
            test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
            
            self.test_dl = test_dataloader
            # return test_dataloader