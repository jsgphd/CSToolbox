# coding: utf-8
u"""
    Thresholding algorithms
"""
import numpy as np



def indexThresholding(x, k):
 
    desc_idxes  = np.argsort(np.abs(x))[::-1]   # sort indexes in descending order
    return desc_idxes[:k] 



def hardThresholding(x, k):

    x_  = np.zeros(len(x))
    for s in indexThresholding(x, k):
        x_[s] = x[s]
    return x_ 

