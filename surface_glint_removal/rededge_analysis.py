# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:30:14 2019

@author: wonk1
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils
from osgeo import gdal, gdal_array
import csv
import os, glob
import micasense.imageset as imageset
import micasense.capture as capture
import rededge_analysis as anal
from micasense.panel import Panel
from micasense.image import Image
import pickle
import math

def align_capture(capture, warp_mode, img_type):
    ## Alignment settings
    match_index = 1 # Index of the band 
    max_alignment_iterations = 50
#    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
#    warp_mode = cv2.MOTION_TRANSLATION
    pyramid_levels = 0 # for images with RigRelatives, setting this to 0 or 1 may improve alignment
    epsilon_threshold=1e-10
    print("Alinging images. Depending on settings this can take from a few seconds to many minutes")
    # Can potentially increase max_iterations for better results, but longer runtimes
    warp_matrices, alignment_pairs = imageutils.align_capture(capture,
                                                              ref_index = match_index,
                                                              max_iterations = max_alignment_iterations,
#                                                              multithreaded=False, 
#                                                              debug = True,
                                                              epsilon_threshold = epsilon_threshold, 
                                                              warp_mode = warp_mode,
                                                              pyramid_levels = pyramid_levels)
    
    print("Finished Aligning, warp matrices={}".format(warp_matrices))
    
    
    cropped_dimensions, edges = imageutils.find_crop_bounds(capture, warp_matrices, warp_mode=warp_mode)
    im_aligned = imageutils.aligned_capture(capture, warp_matrices, warp_mode, cropped_dimensions, match_index, img_type=img_type)
    
    return im_aligned, warp_matrices

def write_img_to_tiff(im_aligned, filename, scale_factor):
    rows, cols, bands = im_aligned.shape
    driver = gdal.GetDriverByName('GTiff')
#    filename = "IMG_" + set_i + "_rad_"+var_ids[0] #blue,green,red,nir,redEdge
    outRaster = driver.Create(filename+".tiff", cols, rows, im_aligned.shape[2], gdal.GDT_UInt16)
   
    for i in range(0,5):
        outband = outRaster.GetRasterBand(i+1)
        outdata = im_aligned[:,:,i]*scale_factor*65535
        outdata[outdata<0] = 0
        outdata[outdata>65535] = 65535
        outband.WriteArray(outdata)
        outband.FlushCache()
    outRaster = None

def swap_4_and_5(im_test):
    im_4 = im_test[:,:,3].copy()
    im_5 = im_test[:,:,4].copy()
    im_new = im_test.copy()
    im_new[:,:,3] = im_5
    im_new[:,:,4] = im_4
    return im_new   

def read_csv(fnames_csv):

    rows = []
    f = open(fnames_csv[0], 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        rows.append(line)
    f.close()    
    
    n_wl = len(rows[0])
    n_sample = int((len(rows)-1)/3)
    
    Ed = np.zeros([n_sample, n_wl])
    Ls = np.zeros([n_sample, n_wl])
    Lw = np.zeros([n_sample, n_wl])
    wl = np.zeros([1, n_wl])
    
    wl[:] = rows[0][:]
    for i in range(0, n_sample):
        Ed[i,:]=rows[i+1][:]
        Ls[i,:]=rows[(n_sample+i+1)][:]
        Lw[i,:]=rows[(n_sample*2+i+1)][:]
    

    rows = []
    f = open(fnames_csv[1], 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        rows.append(line)
    f.close()    
    n_field = len(rows[0])
    n_sample = len(rows)
    ymdhn = np.zeros([n_sample, n_field-1])
    for i in range(0, n_sample):
        ymdhn[i,0:(n_field-1)]=rows[i][0:(n_field-1)]
        
    return Ed, Ls, Lw, wl, ymdhn


def read_tif(filename, scale_factor, nband):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    n_rows, n_cols = arr.shape
    data = np.zeros([n_rows, n_cols, 10])
    for i in range(0,nband):
        band = ds.GetRasterBand(i+1)
        arr = band.ReadAsArray()
        data[:,:,i] = arr
    data = data/scale_factor/65535.
    return data

def dd_to_dms(degs):
    neg = degs < 0
    degs = (-1) ** neg * degs
    degs, d_int = math.modf(degs)
    mins, m_int = math.modf(60 * degs)
    secs        =           60 * mins
    return neg, d_int, m_int, secs

def chl_oc2_re(mbr, ps0, ps1, ps2, ps3, ps4):
    log10_chla = ps0 + ps1*mbr + ps2*np.power(mbr,2) + ps3*np.power(mbr,3) + ps4*np.power(mbr,4)
    return log10_chla

def compute_match(x, y):
    mnb  = np.mean((x-y)/x)
    mnge = np.mean(np.abs((x-y)/x))
    return mnb, mnge

def process_one_station(dir_res, st_id, set_i, img_ids):
    panelNames = None
    useDLS = True
#    dir_res = "F:\\Dropbox\\000_Code\\Python3\\RedEdge\\rededge_read\\analysis\\"
    ref_pan = np.array([50.6, 50.9, 51.1, 51.2, 51.1])*0.01
    
    #=====================================================================================================
    # (1) Load input images
    #=====================================================================================================
    # For a station
#    st_id = 'A4'
#    set_i = '0014'
    imagePath = os.path.join('F:\\','data','CNTP_Rededge', 'TY_2018', set_i+'SET')
#    img_ids = ['0000', '0001', '0003'] # for water, sky, panel
    var_ids = ['wat', 'sky', 'pan']
    
#    # For MRC
#    st_id = 'MRC'
#    set_i = '0022'
#    imagePath = os.path.join('F:\\','data','CNTP_Rededge', 'TY_2018', set_i+'SET')
#    img_ids = ['0001', '0002', '0004'] # for water, sky, panel
#    var_ids = ['wat', 'sky', 'pan']    
    
#    # load panel images
    names_pan = glob.glob(os.path.join(imagePath,'000','IMG_'+ img_ids[2]  +'_*.tif'))
    cap_pan = capture.Capture.from_filelist(names_pan)
#    
    # load water images
    names_wat = glob.glob(os.path.join(imagePath,'000','IMG_'+ img_ids[0]  +'_*.tif'))
    cap_wat = capture.Capture.from_filelist(names_wat)
    
    # load sky images
    names_sky = glob.glob(os.path.join(imagePath,'000','IMG_'+ img_ids[1]  +'_*.tif'))
    cap_sky = capture.Capture.from_filelist(names_sky)

#    # Plot calibration factors  
    cap_wat.plot_raw()
    plt.savefig(dir_res + 'IMG_WAT_RAW_' + set_i + '.png')
    plt.close()
    cap_wat.plot_vignette();
    plt.savefig(dir_res + 'IMG_WAT_VIG_' + set_i + '.png')
    plt.close()
    cap_wat.plot_undistorted_radiance();
    plt.savefig(dir_res + 'IMG_WAT_RAD_' + set_i + '.png')
    plt.close()
#    
    cap_sky.plot_raw()
    plt.savefig(dir_res + 'IMG_SKY_RAW_' + set_i + '.png')
    plt.close()
    cap_sky.plot_vignette();
    plt.savefig(dir_res + 'IMG_SKY_VIG_' + set_i + '.png')
    plt.close()
    cap_sky.plot_undistorted_radiance();
    plt.savefig(dir_res + 'IMG_SKY_RAD_' + set_i + '.png')
    plt.close()
    
    cap_pan.plot_panels()
    plt.savefig(dir_res + 'IMG_PAN_' + set_i + '.png')
    plt.close()
#   
    #=====================================================================================================
    # (2) Get meta-data
    #=====================================================================================================   
    images_dir = os.path.expanduser(os.path.join('F:\\','data','CNTP_Rededge', 'TY_2018', set_i+'SET','000'))
    imgset = imageset.ImageSet.from_directory(images_dir)
    meta_data, columns = imgset.as_nested_lists()
    
    
    #=====================================================================================================
    # (3) Align each capture
    #=====================================================================================================    
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
#    warp_mode = cv2.MOTION_TRANSLATION
    img_type = "radiance"
    
    # Water image
    im_wat, wm_wat = anal.align_capture(cap_wat, warp_mode, img_type)
    im_wat = anal.swap_4_and_5(im_wat)
    
    # Sky image
    im_sky, wm_sky = anal.align_capture(cap_sky, warp_mode, img_type)
    im_sky = anal.swap_4_and_5(im_sky)
    
    
    #=====================================================================================================
    # (4) Get Panel radiance
    #=====================================================================================================     
    # Get panel radiance
    rads_pan = np.zeros([5, 4])
    for i in range(0,5):
        img_pan_i = Image(names_pan[i])
        panel = Panel(img_pan_i)
#        panel.plot()
#        plt.savefig(dir_res + "IMG_PNA_Band" + '{:d}'.format(i+1))
#        plt.close()
        rad_pan_i = panel.radiance()  # mean, std, pixel count, count_saturated
        rads_pan[i,:]=rad_pan_i
    
        
    f = open(dir_res + 'meta_data_' + st_id +'.pckl', 'wb')
    pickle.dump([st_id, set_i, meta_data, rads_pan, ref_pan], f)
    f.close()
    
    #=====================================================================================================
    # (5) Save to files
    #=====================================================================================================   
    # Water image
    im_aligned = im_wat.copy()
    filename = dir_res + "IMG_" + set_i + "_rad_"+var_ids[0] #blue,green,red,nir,redEdge
    scale_factor = 30.
    anal.write_img_to_tiff(im_aligned, filename, scale_factor)
 
    # Sky image
    im_aligned = im_sky.copy()
    filename = dir_res + "IMG_" + set_i + "_rad_"+var_ids[1] #blue,green,red,nir,redEdge
    scale_factor = 5.
    anal.write_img_to_tiff(im_aligned, filename, scale_factor)