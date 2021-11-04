
import numpy as np
import numba
from typing import List,Tuple
import os 
import nibabel as nib
from tqdm.notebook import tqdm
'''
Utility functions
'''
    
def get_file_paths(root:str,directory:str,extension_name:str)->List[str]:
    '''
    get full paths of files under certain root folder with a specific extension
    '''
    for dirname, dirnames, filenames in os.walk(root):


        if os.path.join(root,directory) == dirname :
            for filename in sorted(filenames):

                if filename.endswith(extension_name):
                    yield (os.path.join(dirname, filename))

                    
def load_numpy_array(path:str):
    return nib.load(path).get_fdata()

def load_numpy_arrays(paths:List[str],parallel=False):
    
    '''
    load numpy using list of paths
    '''
    #uncomment if you install ray
#     if parallel :
        
#         if not ray.is_initialized():
#             ray.init()
        
#         print('Parallel loading started ...')
        
#         parallel_func = ray.remote(load_numpy_array)
#         works = [parallel_func.remote(path) for path in paths]
#         data = np.array(ray.get(works[0]))

#         for i in tqdm(range(1,len(works))):
#             data = np.concatenate([ray.get(works[i]),data],axis=2)
            
            
#     else :
    data = load_numpy_array(paths[0])

    for i in tqdm(range(1,len(paths))) :
        data = np.concatenate([load_numpy_array(paths[i]),data],axis=2)

    return data

# uncomment if you install numba (~ much faster )

# @numba.njit
def find_non_zero_mask_indices(array) :
    '''
    Args 
        -array = numpy array of shape [rows,cols,channels] 
    
    Objective
        -recieves a numpy array in the shape of [rows,cols,channels] and returns non-zero channel indices
    
    *Fastest implementation using numba+numpy (numpy<=1.20)
    *in case of errors remove the decorator line
    
    '''
    
    assert len(array.shape)==3 , 'please provide an array with [rows,cols,channels] format'
    
    for channel_index in range(array.shape[2]):
        
        #iterate over each item in channel i
        if check_non_zero(array[:,:,channel_index]):
            yield channel_index

# uncomment if you install numba

# @numba.njit
def check_non_zero(array):
    '''
    check if a 2D array contains at least on non zero element (by short circuit)
    
    *Fastest implementation ~ 3 orders of magnitude faster than numpy using numba
    *in case of errors remove the decorator line
    '''
    
    for row_i in range(array.shape[0]):
        for col_i in range(array.shape[1]):
            
            if array[row_i,col_i] != 0 :
                return True
            
    return False
