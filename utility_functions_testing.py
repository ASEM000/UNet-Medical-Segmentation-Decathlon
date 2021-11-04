
import numpy as np
import numba 
from utility_functions import *

'''
Testing utility functions
'''

def test_check_non_zero():
    
    a = np.ones([256,256])
    
    assert check_non_zero(a) == True
    assert check_non_zero(a*0) == False
    
    a = np.array([[0,0,0,0,0,1]])
    assert check_non_zero(a) == True   
    
    a = np.array([[0]])
    assert check_non_zero(a) == False   

def test_find_non_zero_mask_indices():
    a = np.ones([256,256,3])
    b = np.zeros([256,256,3])
    
    ab = np.concatenate([a,b],axis=-1)
    assert list(find_non_zero_mask_indices(ab)) == [0,1,2]
    
    ab = np.concatenate([a,b,b,a,b],axis=-1)  
    assert list(find_non_zero_mask_indices(ab)) == [0,1,2,9,10,11]
    
    ab = np.concatenate([b,b],axis=-1) 
    assert list(find_non_zero_mask_indices(ab)) == []
