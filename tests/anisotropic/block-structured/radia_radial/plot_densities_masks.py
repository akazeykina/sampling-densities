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

from densities_calculation.mask import compute_mask, fill_det_blocks
from densities_calculation.calculate_densities import unravel_pi
from densities_calculation.generate_scheme import generate_full_scheme, generate_blocks_list, num_samples, split_det_rand


img_size = 32
n = img_size ** 2

wavelet = 'sym4'
level = 3
scheme_type = 'radial'
block_type = 'radia'

reg_type = 'svd'
cond = 0.3
lam = 0.0
    
sparsity = 0.1 # sparsity level: assume that only s = 'sparsity' wavelets coefficients are non zero
#fname = 'brain_phantom/BrainPhantom'+str(n_dim)+'.png' # image for computing s distribution
    
sub_sampling_rate = 0.2


dens_type = [ "un" ]
cs_dens_type = [ "inf", "th_is", "th_anis", "l" ]
dens_type = dens_type + cs_dens_type

###### Creating directory for results

script_dir = os.path.dirname( __file__ )
pics_dir = os.path.join( script_dir, 'pictures/prepare_densities/' )
if not os.path.isdir( pics_dir ):
    os.makedirs( pics_dir )
dens_dir = os.path.join( script_dir, 'pi_dens/' )
if not os.path.isdir( dens_dir ):
    os.makedirs( dens_dir )


######## Generate points of kspace and blocks of points


full_kspace = generate_full_scheme( scheme_type, block_type, img_size )
blocks_list = generate_blocks_list( scheme_type, block_type, img_size )
det_blocks_list, rand_blocks_list = split_det_rand( img_size, blocks_list, 'det_center' )
nb_samples = num_samples( sub_sampling_rate, scheme_type, blocks_list, det_blocks_list, rand_blocks_list )


####### Load densities
pi = {}

for pi_type in dens_type:
    pi[ pi_type ] = np.load( "pi_dens/pi_per_block_"+pi_type+"_"+str(img_size)+".npy" )

####### Unravel pi vectors
pi_fl = unravel_pi( pi, dens_type, rand_blocks_list, full_kspace.shape[ 0 ] )



####### Compute masks
pi_mask = {}
for pi_type in dens_type:
    pi_mask[ pi_type ] = compute_mask( pi[ pi_type ], nb_samples )
#
mask_fl = unravel_pi( pi_mask, dens_type, rand_blocks_list, full_kspace.shape[ 0 ], dtype = bool ) 
mask_fl = fill_det_blocks( mask_fl, dens_type, det_blocks_list )



##############################################################################

######## Create directory for pictures

script_dir = os.path.dirname( __file__ )
results_dir = os.path.join( script_dir, 'pictures/plot_densities_masks/' )
if not os.path.isdir( results_dir ):
    os.makedirs( results_dir )



###### Plot densities

val_min = np.min( np.array( [ pi_fl[ pi_type ] for pi_type in dens_type ] ) )
val_max = np.max( np.array( [ pi_fl[ pi_type ] for pi_type in dens_type ] ) )

fig = plt.figure( figsize = ( 30, 5 ) )

for i, pi_type in enumerate( dens_type ):
    ax = fig.add_subplot(1, 5, i + 1 )
    ax.tricontourf( full_kspace[ :, 1 ], full_kspace[ :, 0 ], pi_fl[ pi_type ],
               vmin = val_min, vmax = val_max )
    ax.set_aspect( 'equal' )
    
plt.savefig( results_dir+'densities.png', bbox_inches='tight')
#plt.show()
#
#
####### Plots masks
#
#
fig = plt.figure( figsize = ( 30, 5 ) )

for i, pi_type in enumerate( dens_type ):
    ax = fig.add_subplot( 1, 5, i + 1 )
    ax.set_facecolor( 'k' )
    ax.plot( full_kspace[ mask_fl[ pi_type ], 1 ], full_kspace[ mask_fl[ pi_type ], 0 ], color = 'w', marker = 's', 
               markersize  = 5 * 32 / img_size, linestyle = '' )
    ax.set_aspect( 'equal' )
    ax.autoscale( tight = True )
    ax.set_xlim( -0.5, 0.5 )
    ax.set_ylim( -0.5, 0.5 )
    
plt.savefig( results_dir+'masks.png', bbox_inches='tight')
#plt.show()

print( fig.dpi )







