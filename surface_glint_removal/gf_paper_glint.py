# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:43:25 2022

@author: wonk1
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_band_images(img_wat):
    fig, axs = plt.subplots(2, 2, figsize=(18, 13))
    fig.tight_layout(pad=2.)

    step = 1
    for i in range(2):
        for j in range(2):
            axs[i, j].imshow(img_wat[:,:,step])
            axs[i, j].set_title('Band {:d}'.format(step+1), fontsize=20)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            step += 1


def plot_band_hist(img_wat):
    nrows, ncols,tmp = img_wat.shape 
    nsamp = nrows*ncols
    step = 100
    
    fig, axs = plt.subplots(2, 2, figsize=(22, 16))
    fig.tight_layout(pad=7.)

    band_i = 1
    
    for i in range(2):
        for j in range(2):
            img_i = img_wat[:,:, band_i]
            data_i = img_i.flatten()
            samp_i = data_i[0:nsamp:step]
            ind_valid = ~np.isnan(samp_i)
            samp_i = samp_i[ind_valid]
            
            axs[i,j].hist(samp_i, bins=100, label='bins=30',edgecolor='black')
            axs[i,j].tick_params(axis='x', labelsize=20)
            axs[i,j].tick_params(axis='y', labelsize=20)
            axs[i,j].set_xlabel('Radiance', fontsize=20)
            axs[i,j].set_ylabel('Number', fontsize=20)
            axs[i,j].set_title('Band {:d}'.format(band_i+1), fontsize=20)
            band_i += 1
#            axs[i,j].show()

def perform_glint_correction(img_wat, n_repeat=3, band_k=4, r_correct=0.75):
    # n_repeat : how many times do you want to repeat this whole process
    # band_k  : the reference band from which you derive glint pixels
    #           the longest NIR band is recommended for this
    # n_correct : how much of the glint signal you would correct at one time
    
    nrows, ncols,tmp = img_wat.shape 
    nsamp = nrows*ncols
    step = 100
    
    for k in range(n_repeat):
        

        img_5 = img_wat[:,:,band_k]
        data_5 = img_5.flatten()
        samp_5 = data_5[0:nsamp:step]
        ind_valid = ~np.isnan(samp_5)
        samp_5 = samp_5[ind_valid]
        
        q10=np.quantile(samp_5, 0.1)
        q20=np.quantile(samp_5, 0.2)
        q50=np.quantile(samp_5, 0.5)
        q75=np.quantile(samp_5, 0.75)
        q85=np.quantile(samp_5, 0.85)
        
        # estimate sun glint signal
        ind_low = (img_5 > q10) & (img_5 < q20)
        Ro = np.zeros((5,))
        for j in range(5):
            img_j = img_wat[:,:,j]
            Ro[j] = np.nanmean(img_j[ind_low])
        ind_high = (img_5 > q75) & (img_5 < q85)
        rad_high = np.zeros((5,))
        for j in range(5):
            img_j = img_wat[:,:,j]
            rad_high[j] = np.nanmean(img_j[ind_high])            
        
        Rg = rad_high - Ro

        ind_glint = img_5 > q20
        img_weight = img_5*0. 
        img_weight[ind_glint] = img_5[ind_glint]-q20
        
        img_weight = img_weight/Rg[4]
        img_weight3 = np.dstack([img_weight]*5)
        
        img_Rg3 = np.repeat(Rg, ncols*nrows)
        img_Rg3 = np.reshape(img_Rg3, (5, ncols, nrows))
        img_Rg3 = np.swapaxes(img_Rg3, 0, 2)
    
        img_wat = img_wat - img_weight3*img_Rg3*0.75
        
        return img_wat