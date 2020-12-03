#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 08:58:29 2020

@author: Anna
"""
import sys, os
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from mri.operators import WaveletN
from mri.reconstructors import SingleChannelReconstructor

from modopt.opt.proximity import SparseThreshold #, ElasticNet, OrderedWeightedL1Norm
from modopt.opt.linear import Identity
from modopt.math.metrics import ssim

from densities_calculation.mask import compute_indices
from densities_calculation.utils import extract_images
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
    
sub_sampling_rate = 0.2

num_runs = 2 # number of runs of reconstruction algorithm
num_imgs = 3 # number of images over which the result of reconstruction is averaged

mus = np.logspace( 1, 0, 2 ) # regularisation parameter of the reconstruction algorithm
#mus = [ 1e1 ]

dens_type  = [ "rad", "inf", "th_is", "th_anis", "l" ] # types of densities to compute

img_list = extract_images( "../brain_images/fastmri/file1000001.h5", "h5", 
                          img_size = img_size, num_images = num_imgs ) # images to reconstruct

img_s_distrib_list = extract_images( "../brain_images/fastmri/file1000265.h5", "h5", 
                                    img_size = img_size ) # images for calculation of s

linear_op = WaveletN( wavelet_name = wavelet, nb_scale = level, padding_mode = 'periodization' )

regularizer_op = SparseThreshold( Identity(), 2e-7, thresh_type = "soft" )


######## Generate points of kspace and blocks of points


full_kspace = generate_full_scheme( scheme_type, block_type, img_size )
blocks_list = generate_blocks_list( scheme_type, block_type, img_size )
nb_samples = num_samples( sub_sampling_rate, scheme_type, blocks_list, [], blocks_list )

####### Distribution of sparsity coefficients

s_distrib = avg_s_distribution( img_size, img_s_distrib_list, wavelet, level, sparsity )
print("S distribution:")
print( s_distrib )


####### Compute pi radial
pi_rad = pi_rad( 2, 0.2, np.array( [ img_size, img_size ] ) )
#
####### Compute CS densities pi
print( "Calculate pi, vector size:", img_size**2 )
pi = {}
pi[ "rad" ] = pi_rad.flatten()
pi[ "inf" ], pi[ "th_is" ], pi["th_anis"], pi["l"] = calculate_pi_blocks( img_size, scheme_type,
  full_kspace, reg_type, cond, lam, wavelet, level, s_distrib, blocks_list )

####### Unravel pi vectors
pi_fl = unravel_pi( pi, dens_type[ 1: ], blocks_list, full_kspace.shape[ 0 ] )
pi_fl[ "rad" ] = pi[ "rad"].flatten()


####### Initialize variables to keep track of ssim, mu, num_points
good_ssim, good_mu, num_points = {}, {}, {}
for pi_type in dens_type:
    good_ssim[ pi_type ] = np.zeros( num_runs )
    good_mu[ pi_type ] = np.zeros( num_runs )
    #num_points[ pi_type ] = []


####### Reconstruction

print( "Reconstruction" )

start_time = time.time()
for pi_type in dens_type:
    
    for i in range( num_runs ):
        pi_mask = compute_indices( pi[ pi_type ], nb_samples )
        
        fourier_op = masked_fourier_op( img_size, full_kspace, [], 
                                       blocks_list[ 0: ], pi_fl[ pi_type ], pi_mask, normalize = False )

        # Setup Reconstructor
        reconstructor = SingleChannelReconstructor(
            fourier_op = fourier_op,
            linear_op = linear_op,
            regularizer_op = regularizer_op,
            gradient_formulation = 'synthesis',
            num_check_lips = 0,
            verbose = 0,
        )
    
    
        cur_mu = 0
        cur_ssims = []
        
        for j in range( num_imgs ):
        
            img = img_list[ j ]
            kspace_obs = fourier_op.op( img )
        
            cur_ssims.append( 0 )
                
            for mu in mus:
                #print(mu)
                reconstructor.prox_op.weights = mu
            
            
                x_final, costs, metric = reconstructor.reconstruct(
                        kspace_data = kspace_obs,
                        optimization_alg = 'fista',
                        num_iterations = 150,
                        #metrics=metrics,
                        #lambda_update_params = { "restart_strategy":"greedy", "s_greedy":1.1, "xi_restart":0.96 }
                )
                
                if ssim( x_final, img ) > cur_ssims[ -1 ]:
                    cur_ssims[ -1 ] = ssim( x_final, img )

        good_mu[ pi_type ][ i ] = 0
        good_ssim[ pi_type ][ i ] = np.mean( cur_ssims )


####### Display results
        
print( "Reconstruction time:", time.time() - start_time )

for pi_type in dens_type:
    print( "Pi type:", pi_type, ", SSIM mean:", np.mean( good_ssim[ pi_type ] ),
          ", SSIM std:", np.std( good_ssim[ pi_type ] ) )
    print( "Mu mean:", np.mean( good_mu[ pi_type ] ), "mu std:", np.std( good_mu[ pi_type ] )  )

