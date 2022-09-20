# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:33:33 2022

@author: wonk1
"""

import gf_paper_glint as gf
import rededge_analysis as anal
import matplotlib.pyplot as plt


dir_cwd = "F:\\Dropbox\\000_Code\\PythonProjects\\micasense\\glint_removal\\"
dir_wat = dir_cwd + "data\\"

plt.close('all')
if __name__ == '__main__':

    # (1) Read water radiance 
    fname_wat = dir_wat + "0077_rad_geo.tif"
    scale_factor =  1./65535.
    img_rad_wat = anal.read_tif(fname_wat, scale_factor, 10)
    img_wat = img_rad_wat[:,:, [0,1,2,3,4]]
    
    # (2) Plot initial statistics 
    gf.plot_band_images(img_wat)
    gf.plot_band_hist(img_wat)
    
    
    # (3) Perform glint correction
    img_wat = gf.perform_glint_correction(img_wat, n_repeat=3, band_k=4, r_correct=0.75)

    # (4) Plot after-correction statistics
    gf.plot_band_images(img_wat)
    gf.plot_band_hist(img_wat)

    # (5) Save the corrected image
    filename = dir_cwd + "sg_removed.tif"
    scale_factor = 1.
    anal.write_img_to_tiff(img_wat, filename, scale_factor)     