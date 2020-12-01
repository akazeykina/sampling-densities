#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:29:27 2020

@author: Anna
"""

import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from densities_calculation.mask import compute_mask
from densities_calculation.s_distribution import avg_s_distribution
from densities_calculation.calculate_densities import pi_rad, calculate_pi_blocks
from densities_calculation.generate_scheme import generate_full_scheme, generate_blocks_list, num_samples
from densities_calculation.utils import extract_images, reduce_img_size




img_size = 32
n = img_size**2
wavelet = 'sym4'
level = 3

scheme_type = 'cartesian'
block_type = 'isolated' # isolated points

# Parameters for calculating the pseudoinverse
reg_type = 'svd'
cond = 0.0
lam = 0.0

sparsity = 0.1 # sparsity level: assume that only s = 'sparsity' wavelets coefficients are non zero

sub_sampling_rate = 0.2

####### Distribution of sparsity coefficients

#img_s_distrib_list = extract_images( "../brain_images/T2w/sub-OAS30008_sess-d0061_acq-TSE_T2w.nii", "nii", "T2w" )
#img_s_distrib_list = extract_images( "../brain_images/T1w/sub-OAS30001_ses-d0129_run-01_T1w.nii", "nii", "T1w" )
img_s_distrib_list = extract_images( "../brain_images/fastmri/file1000265.h5", "h5" )


s_distrib = avg_s_distribution( img_size, img_s_distrib_list, wavelet, level, sparsity )
print("S distribution:")
print( s_distrib )

full_kspace = generate_full_scheme( scheme_type, block_type, img_size )
blocks_list = generate_blocks_list( scheme_type, block_type, img_size )
nb_samples = num_samples( sub_sampling_rate, scheme_type, blocks_list, [], blocks_list )


####### Compute pi radial
pi_rad = pi_rad( 2, 0.2, np.array( [ img_size, img_size ] ) )
#

print( "Calculate pi" )
pi = {}
pi[ "inf" ], pi[ "th_is" ], pi["th_anis"], pi["l"] = calculate_pi_blocks( img_size, scheme_type,
  full_kspace, reg_type, cond, lam, wavelet, level, s_distrib, blocks_list )


pi_fl = { "rad": pi_rad.flatten(),
         "inf": np.zeros( ( full_kspace.shape[ 0 ], ) ),
         "th_is": np.zeros( ( full_kspace.shape[ 0 ], ) ),
        "th_anis": np.zeros( ( full_kspace.shape[ 0 ], ) ),
         "l": np.zeros( ( full_kspace.shape[ 0 ], ) ) }

for j in range( len( blocks_list ) ): 
    block = blocks_list[ j ]
    pi_fl[ "inf" ][ block ] = pi[ "inf" ][ j, : ]
    pi_fl[ "th_is" ][ block ] = pi[ "th_is" ][ j, : ]
    pi_fl[ "th_anis" ][ block ] = pi[ "th_anis" ][ j, : ]
    pi_fl[ "l" ][ block ] = pi[ "l" ][ j, : ]


####### Compute masks

pi_rad_mask = compute_mask( pi_fl[ "rad" ], nb_samples )
#np.save( "../pi_masks/pi_rad_mask_"+str(img_size)+".npy", pi_rad_mask )

#pi_inf_mask = np.reshape( compute_mask( pi_inf_flattened, n, nb_samples ), ( img_size, img_size ), order = 'C' )
#np.save( "../pi_masks/pi_inf_mask_"+str(img_size)+".npy", pi_inf_mask )

#pi_th_mask = np.reshape( compute_mask( pi_th_flattened, n, nb_samples ), ( img_size, img_size ), order = 'C' )
#np.save( "../pi_masks/pi_th_mask_"+str(img_size)+".npy", pi_th_mask )

pi_inf_mask = compute_mask( pi_fl[ "inf" ], nb_samples )

pi_th_is_mask = compute_mask( pi_fl[ "th_is" ], nb_samples )

pi_th_anis_mask = compute_mask( pi_fl[ "th_anis" ], nb_samples )
#np.save( "../pi_masks/pi_th_new_mask_"+str(img_size)+".npy", pi_th_mask )

pi_l_mask = compute_mask( pi_fl[ "l" ], nb_samples )
#np.save( "../pi_masks/pi_l_mask_"+str(img_size)+".npy", pi_l_mask )
#

print( np.sum( pi_l_mask ) / img_size**2 )

###### Plot MRI images
fig = plt.figure( figsize = ( 24, 5 ) )

for i in range( 5 ):
    ax = fig.add_subplot(1, 5, i + 1 )
    plt.imshow( reduce_img_size( img_size, img_s_distrib_list[ i * 2 + 1 ] ), cmap = 'gray' )

plt.show()

###### Plot densities

pi_inf = np.reshape( pi_fl[ "th_is" ], ( img_size, img_size ), order = 'C' )
pi_th_is = np.reshape( pi_fl[ "th_is" ], ( img_size, img_size ), order = 'C' )
pi_th_anis = np.reshape( pi_fl[ "th_anis" ], ( img_size, img_size ), order = 'C' )
pi_l = np.reshape( pi_fl[ "l" ], ( img_size, img_size ), order = 'C' )

val_min = min( np.min( pi_rad ), np.min( pi_inf ), np.min( pi_th_is ), np.min( pi_th_anis ), np.min( pi_l ) )
val_max = max( np.max( pi_rad ), np.max( pi_inf ), np.max( pi_th_is ), np.max( pi_th_anis ), np.max( pi_l ) )

fig = plt.figure( figsize = ( 30, 5 ) )

ax = fig.add_subplot(1, 5, 1 )
#plt.imshow( pi_rad, vmin = val_min, vmax = val_max )
ax.tricontourf( full_kspace[ :, 1 ], full_kspace[ :, 0 ], pi_fl[ "rad" ],
               vmin = val_min, vmax = val_max )
ax.set_aspect( 'equal' )

ax = fig.add_subplot(1, 5, 2 )
#plt.imshow( pi_rad, vmin = val_min, vmax = val_max )
ax.tricontourf( full_kspace[ :, 1 ], full_kspace[ :, 0 ], pi_fl[ "inf" ],
               vmin = val_min, vmax = val_max )
ax.set_aspect( 'equal' )

ax = fig.add_subplot(1, 5, 3 )
#plt.imshow( pi_th_anis, vmin = val_min, vmax = val_max )
ax.tricontourf( full_kspace[ :, 1 ], full_kspace[ :, 0 ], pi_fl[ "th_is" ],
               vmin = val_min, vmax = val_max )
ax.set_aspect( 'equal' )

ax = fig.add_subplot(1, 5, 4 )
#plt.imshow( pi_th_anis, vmin = val_min, vmax = val_max )
ax.tricontourf( full_kspace[ :, 1 ], full_kspace[ :, 0 ], pi_fl[ "th_anis" ],
               vmin = val_min, vmax = val_max )
ax.set_aspect( 'equal' )

ax = fig.add_subplot(1, 5, 5 )
#plt.imshow( pi_l, vmin = val_min, vmax = val_max )
ax.tricontourf( full_kspace[ :, 1 ], full_kspace[ :, 0 ], pi_fl[ "l" ],
               vmin = val_min, vmax = val_max )
ax.set_aspect( 'equal' )

plt.show()
#
#
####### Plots masks
#
#
fig = plt.figure( figsize = ( 30, 5 ) )

ax = fig.add_subplot(1, 5, 1 )
plt.imshow(pi_rad_mask, cmap='gray')

ax = fig.add_subplot(1, 5, 2 )
plt.imshow(pi_inf_mask, cmap='gray')

ax = fig.add_subplot(1, 5, 3 )
plt.imshow(pi_th_is_mask, cmap='gray')

ax = fig.add_subplot(1, 5, 4 )
plt.imshow(pi_th_anis_mask, cmap='gray')

ax = fig.add_subplot(1, 5, 5 )
plt.imshow(pi_l_mask, cmap='gray')

plt.show()
#
