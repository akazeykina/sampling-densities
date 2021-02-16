#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:29:27 2020

@author: Anna
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))

from densities_calculation.s_distribution import avg_s_distribution
from densities_calculation.calculate_densities import pi_rad, calculate_pi_blocks, unravel_pi
from densities_calculation.generate_scheme import generate_full_scheme, generate_blocks_list
from densities_calculation.utils import extract_images, reduce_img_size


img_size = 64
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

dens_type  = [ "rad", "inf", "th_is", "th_anis", "l" ] # types of densities to compute

decay = 4 # decay for pi_rad
cutoff = 0.1 # cutoff for pi_rad

####### Distribution of sparsity coefficients

img_s_distrib_list = extract_images( "../../../../brain_images/T2w/sub-OAS30008_sess-d0061_acq-TSE_T2w.nii", "nii", "T2w" )
#img_s_distrib_list = extract_images( "../brain_images/T1w/sub-OAS30001_ses-d0129_run-01_T1w.nii", "nii", "T1w" )
#img_s_distrib_list = extract_images( "../brain_images/fastmri/file1000265.h5", "h5" )


s_distrib = avg_s_distribution( img_size, img_s_distrib_list, wavelet, level, sparsity )
print("S distribution:")
print( s_distrib )

full_kspace = generate_full_scheme( scheme_type, block_type, img_size )
blocks_list = generate_blocks_list( scheme_type, block_type, img_size )


####### Compute pi radial
pi_rad = pi_rad( decay, cutoff, np.array( [ img_size, img_size ] ) )
#
####### Compute CS densities pi
print( "Calculate pi, vector size:", img_size**2 )
pi = {}
pi[ "rad" ] = pi_rad.flatten()
pi[ "inf" ], pi[ "th_is" ], pi["th_anis"], pi["l"] = calculate_pi_blocks( img_size, scheme_type,
  full_kspace, reg_type, cond, lam, wavelet, level, s_distrib, blocks_list )

##############################################################################

######## Create directory for pictures and for densities

script_dir = os.path.dirname( __file__ )
pics_dir = os.path.join( script_dir, 'pictures/prepare_densities/' )
if not os.path.isdir( pics_dir ):
    os.makedirs( pics_dir )
dens_dir = os.path.join( script_dir, 'pi_dens/' )
if not os.path.isdir( dens_dir ):
    os.makedirs( dens_dir )
        


###### Plot MRI images
fig = plt.figure( figsize = ( 30, 5 ) )

for i in range( 5 ):
    ax = fig.add_subplot(1, 5, i + 1 )
    plt.imshow( reduce_img_size( img_size, img_s_distrib_list[ i * 2 + 1 ] ), cmap = 'gray' )
    
plt.savefig( pics_dir+'oasis_T2w_images.png', bbox_inches='tight')
#plt.show()

###### Prepare densities

pi_fl = unravel_pi( pi, dens_type[ 1: ], blocks_list, full_kspace.shape[ 0 ] )
pi_fl[ "rad" ] = pi[ "rad"].flatten()

######## Save densities
for pi_type in dens_type:
    np.save( "pi_dens/pi_"+pi_type+"_"+str(img_size)+".npy", np.reshape( pi_fl[ pi_type ], 
                ( img_size, img_size ), order = 'C' ) )
        
####### Plot densities

val_min = np.min( np.array( [ pi_fl[ pi_type ] for pi_type in dens_type ] ) )
val_max = np.max( np.array( [ pi_fl[ pi_type ] for pi_type in dens_type ] ) )

fig = plt.figure( figsize = ( 30, 5 ) )

for i, pi_type in enumerate( dens_type ):
    ax = fig.add_subplot(1, 5, i + 1 )
    ax.imshow( np.reshape( pi_fl[ pi_type ], ( img_size, img_size ), order = 'C' ) )
    ax.set_aspect( 'equal' )
    
plt.savefig( pics_dir+'densities.png', bbox_inches='tight' )
#plt.show()

#
