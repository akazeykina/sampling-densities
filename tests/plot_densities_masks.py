#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:29:27 2020

@author: Anna
"""

import numpy as np
import matplotlib.pyplot as plt

from densities_calculation.mask import compute_mask
from densities_calculation.s_distribution import avg_s_distribution




img_size = 64
n = img_size**2
wavelet = 'sym4'
level = 3

sparsity = 0.1 # sparsity level: assume that only s = 'sparsity' wavelets coefficients are non zero

sub_sampling_rate = 0.2

####### Distribution of sparsity coefficients

img_s_distrib = "../brain_images/sub-OAS30001_ses-d0129_run-01_T1w.nii"
s_distrib = avg_s_distribution( img_size, img_s_distrib, wavelet, level, sparsity )
print("S distribution:")
print( s_distrib )

######## Compute pi_theta and pi_lambda
#print( "Size of pi:", img_size**2 )
#pi_inf_flattened, pi_th_flattened, pi_th_new_flattened, pi_l_flattened = calculate_pi(
#        img_size, wavelet, level, s_distrib )
#
######## Compute pi radial
#pi_rad = pi_rad( 6, 0.12, np.array( [ img_size, img_size ] ) )
#
#
####### Compute masks
#
#nb_samples = int( sub_sampling_rate * n )
#
#pi_rad_flattened = np.reshape( pi_rad, ( img_size * img_size, ), order = 'C' )
#pi_rad_mask = np.reshape( compute_mask( pi_rad_flattened, n, nb_samples ), ( img_size, img_size ), order = 'C' )
#np.save( "../pi_masks/pi_rad_mask_"+str(img_size)+".npy", pi_rad_mask )
#
#pi_inf_mask = np.reshape( compute_mask( pi_inf_flattened, n, nb_samples ), ( img_size, img_size ), order = 'C' )
#np.save( "../pi_masks/pi_inf_mask_"+str(img_size)+".npy", pi_inf_mask )
#
#pi_th_mask = np.reshape( compute_mask( pi_th_flattened, n, nb_samples ), ( img_size, img_size ), order = 'C' )
#np.save( "../pi_masks/pi_th_mask_"+str(img_size)+".npy", pi_th_mask )
#
#pi_th_new_mask = np.reshape( compute_mask( pi_th_new_flattened, n, nb_samples ), ( img_size, img_size ), order = 'C' )
#np.save( "../pi_masks/pi_th_new_mask_"+str(img_size)+".npy", pi_th_mask )
#
#pi_l_mask = np.reshape( compute_mask( pi_l_flattened, n, nb_samples ), ( img_size, img_size ), order = 'C' )
#np.save( "../pi_masks/pi_l_mask_"+str(img_size)+".npy", pi_l_mask )
#
####### Plot color maps
#
#pi_inf = np.reshape( pi_inf_flattened, ( img_size, img_size ), order = 'C' )
#pi_th = np.reshape( pi_th_flattened, ( img_size, img_size ), order = 'C' )
#pi_th_new = np.reshape( pi_th_new_flattened, ( img_size, img_size ), order = 'C' )
#pi_l = np.reshape( pi_l_flattened, ( img_size, img_size ), order = 'C' )
#
#fig = plt.figure( figsize = ( 18, 10 ) )
#plt.subplot( 231 )
#plt.imshow( pi_rad )
#plt.subplot( 232 )
#plt.imshow( pi_inf )
#plt.subplot( 233 )
#plt.imshow( pi_th )
#plt.subplot( 235 )
#plt.imshow( pi_l )
#plt.subplot( 236 )
#plt.imshow( pi_th_new )
#
#
####### Plots masks
#
#
#fig = plt.figure( figsize = ( 18, 5 ) )
#
#ax = fig.add_subplot(2, 3, 1 )
#plt.imshow(pi_rad_mask, cmap='gray')
#
#ax = fig.add_subplot(2, 3, 2 )
#plt.imshow(pi_inf_mask, cmap='gray')
#
#ax = fig.add_subplot(2, 3, 3 )
#plt.imshow(pi_th_mask, cmap='gray')
#
#ax = fig.add_subplot(2, 3, 5 )
#plt.imshow(pi_l_mask, cmap='gray')
#
#ax = fig.add_subplot(2, 3, 6 )
#plt.imshow(pi_th_new_mask, cmap='gray')
#
#plt.show()
#
####### 3d plots
#
##x = np.arange( img_size )
##y = np.arange( img_size )
##X, Y = np.meshgrid( x, y )
#
##fig = plt.figure( figsize = ( 18, 5 ) )
#
##ax = fig.add_subplot(1, 3, 1, projection = '3d' )
##surf = ax.plot_surface( X, Y, pi_inf )
#
##ax = fig.add_subplot(1, 3, 2, projection = '3d' )
##surf = ax.plot_surface( X, Y, pi_th )
#
##ax = fig.add_subplot(1, 3, 3, projection = '3d' )
##surf = ax.plot_surface( X, Y, pi_l )
#
##plt.show()
#
##fig = plt.figure(); plt.plot( pi_l_flattened )
