#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:11:21 2021

@author: Anna
"""

kspace_loc = np.load( "kpoints/sparkling_"+pi_type+"_"+str(img_size)+".npy" )

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
from densities_calculation.utils import extract_images, nrmse
from densities_calculation.s_distribution import avg_s_distribution
from densities_calculation.calculate_densities import calculate_pi_blocks, pi_rad, unravel_pi
from densities_calculation.generate_scheme import generate_full_scheme, generate_blocks_list, num_samples
from reconstruction.fourier import masked_fourier_op

img_size = 64
n = img_size ** 2

wavelet = 'sym4'
level = 3

decays = [ 2, 4, 6 ] # decays of pi_radial
cutoffs = [ 0.1, 0.2, 0.3 ] #cutoffs of pi_radial

num_runs = 3 # number of runs of reconstruction algorithm
num_imgs = 3 # number of images over which the result of reconstruction is averaged

mus = np.logspace( -4, -6, 3 ) # regularisation parameter of the reconstruction algorithm
#mus = [ 1e1 ]

dens_type  = [ "rad_"+str(decay)+"_"+str(cutoff) for decay in decays for cutoff in cutoffs ] # types of densities to compute
cs_dens_type = [ "inf", "th_is", "th_anis", "l" ]
dens_type = dens_type + cs_dens_type

img_list = extract_images( "../brain_images/T2w/sub-OAS30007_sess-d0061_acq-TSE_T2w.nii", "nii", "T2w", 
                          img_size = img_size, num_images = num_imgs ) # images to reconstruct
                                    


linear_op = WaveletN( wavelet_name = wavelet, nb_scale = level, padding_mode = 'periodization' )

regularizer_op = SparseThreshold( Identity(), 2e-7, thresh_type = "soft" )

        

####### Initialize variables to keep track of ssim, mu, num_points
good_ssim, good_nrmse, good_mu, num_points = {}, {}, {}, {}
for pi_type in dens_type:
    good_ssim[ pi_type ] = np.zeros( num_runs )
    good_nrmse[ pi_type ] = np.zeros( num_runs )
    good_mu[ pi_type ] = np.zeros( num_runs )
    #num_points[ pi_type ] = []


####### Reconstruction

print( "Reconstruction" )

start_time = time.time()
for pi_type in dens_type:
    
    for i in range( num_runs ):
        
        fourier_op = NonCartesianFFT( samples = kspace_loc, shape = ( img_size, img_size ), implementation = 'cpu' )

        # Setup Reconstructor
        reconstructor = SingleChannelReconstructor(
            fourier_op = fourier_op,
            linear_op = linear_op,
            regularizer_op = regularizer_op,
            gradient_formulation = 'synthesis',
            num_check_lips = 0,
            verbose = 0,
        )
    
    
        cur_mu = [] # stores mu corresponding to the best ssim for every image
        cur_ssims = [] # stores best ssim for every image
        cur_nrmse = []
        
        for j in range( num_imgs ):
        
            img = img_list[ j ]
            kspace_obs = fourier_op.op( img )
        
            cur_ssims.append( 0 )
            cur_mu.append( mus[ 0 ] )
            cur_nrmse.append( 1.0 )
                
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
                    cur_mu[ -1 ] = mu
                if nrmse( x_final, img ) < cur_nrmse[ -1 ]:
                    cur_nrmse[ -1 ] = nrmse( x_final, img )

        good_mu[ pi_type ][ i ] = np.mean( cur_mu )
        good_ssim[ pi_type ][ i ] = np.mean( cur_ssims )
        good_nrmse[ pi_type ][ i ] = np.mean( cur_nrmse )


####### Display results
        
print( "Reconstruction time:", time.time() - start_time )

for pi_type in dens_type:
    print( "Pi type:", pi_type, ", SSIM mean:", np.mean( good_ssim[ pi_type ] ),
          ", SSIM std:", np.std( good_ssim[ pi_type ] ) )
    print( "NRMSE mean:", np.mean( good_nrmse[ pi_type ] ), "NRMSE std:", np.std( good_nrmse[ pi_type ] )  )
    print( "Mu mean:", np.mean( good_mu[ pi_type ] ), "mu std:", np.std( good_mu[ pi_type ] )  )
