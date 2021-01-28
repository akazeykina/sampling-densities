#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:23:11 2021

@author: Anna
"""

import sys, os
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
import time

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from mri.operators import WaveletN

from modopt.opt.proximity import SparseThreshold #, ElasticNet, OrderedWeightedL1Norm
from modopt.opt.linear import Identity
from modopt.math.metrics import ssim

from densities_calculation.mask import compute_indices
from densities_calculation.utils import extract_images, nrmse
from densities_calculation.s_distribution import avg_s_distribution
from densities_calculation.calculate_densities import calculate_pi_blocks, pi_rad, unravel_pi
from densities_calculation.generate_scheme import generate_full_scheme, generate_blocks_list, num_samples
from reconstruction.fourier import masked_fourier_op

img_size = 32
n = img_size ** 2

wavelet = 'sym4'
level = 3
scheme_type = 'cartesian'
block_type = 'isolated'

reg_type = 'svd'
cond = 0.0
lam = 0.0
    
sparsity = 0.1 # sparsity level: assume that only s = 'sparsity' wavelets coefficients are non zero
#fname = 'brain_phantom/BrainPhantom'+str(n_dim)+'.png' # image for computing s distribution

decays = [ 2, 4, 6 ] # decays of pi_radial
cutoffs = [ 0.1, 0.2, 0.3 ] #cutoffs of pi_radial
    
sub_sampling_rate = 0.2

num_runs = 5 # number of runs of reconstruction algorithm
num_imgs = 5 # number of images over which the result of reconstruction is averaged

gammas = np.logspace( 0, -3, 4 ) # regularisation parameter of the reconstruction algorithm
#mus = [ 1e1 ]

dens_type  = [ "rad_"+str(decay)+"_"+str(cutoff) for decay in decays for cutoff in cutoffs ] # types of densities to compute
cs_dens_type = [ "inf", "th_is", "th_anis", "l" ]
dens_type = dens_type + cs_dens_type

img_list = extract_images( "../../../brain_images/fastmri/file1000001.h5", "h5", 
                          img_size = img_size, num_images = num_imgs ) # images to reconstruct

img_s_distrib_list = extract_images( "../../../brain_images/fastmri/file1000265.h5", "h5", 
                                    img_size = img_size ) # images for calculation of s

linear_op = WaveletN( wavelet_name = wavelet, nb_scale = level, padding_mode = 'periodization' )

###### Creating directory for results

script_dir = os.path.dirname( __file__ )
pics_dir = os.path.join( script_dir, 'pictures/prepare_densities/' )
if not os.path.isdir( pics_dir ):
    os.makedirs( pics_dir )
dens_dir = os.path.join( script_dir, 'pi_dens/' )
if not os.path.isdir( dens_dir ):
    os.makedirs( dens_dir )


start_time = time.time()
######## Generate points of kspace and blocks of points


full_kspace = generate_full_scheme( scheme_type, block_type, img_size )
blocks_list = generate_blocks_list( scheme_type, block_type, img_size )
nb_samples = num_samples( sub_sampling_rate, scheme_type, blocks_list, [], blocks_list )

####### Distribution of sparsity coefficients

s_distrib = avg_s_distribution( img_size, img_s_distrib_list, wavelet, level, sparsity )
print("S distribution:")
print( s_distrib )


####### Compute CS densities pi
print( "Calculate pi, vector size:", img_size**2 )
pi = {}

# Calculate pi_rad
for decay in decays:
    for cutoff in cutoffs:
        pi_radial = pi_rad( decay, cutoff, np.array( [ img_size, img_size ] ) )
        pi[ "rad_"+str(decay)+"_"+str(cutoff) ] = pi_radial.flatten()


pi[ "inf" ], pi[ "th_is" ], pi["th_anis"], pi["l"] = calculate_pi_blocks( img_size, scheme_type,
  full_kspace, reg_type, cond, lam, wavelet, level, s_distrib, blocks_list )

######## Save densities
for pi_type in dens_type:
    np.save( "pi_dens/pi_per_block_"+pi_type+"_"+str(img_size)+".npy", pi[ pi_type ] )
    

print( "Density construction time:", time.time() - start_time )



