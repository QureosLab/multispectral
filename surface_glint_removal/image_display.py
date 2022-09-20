# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:19:09 2019

@author: wonk1
"""
import numpy as np
import matplotlib.pyplot as plt

def imshow_scaled(img_rgb, minmaxvals):
#    minmaxvals = np.zeros(3, 2)
    img_scaled = img_rgb.copy()
    for i in range(3):
#        max_i = np.max(img_rgb[:,:,i])
#        min_i = np.min(img_rgb[:,:,i])
#        range_i = (max_i - min_i)*0.5
#        minmaxvals[i]=max_i
        min_i = minmaxvals[i, 0]
        max_i = minmaxvals[i, 1]
        range_i = max_i - min_i
        img_scaled[:,:,i] = (img_rgb[:,:,i] - min_i)/range_i
    
    ax = plt.imshow(img_scaled)
    
    return ax, img_scaled
        