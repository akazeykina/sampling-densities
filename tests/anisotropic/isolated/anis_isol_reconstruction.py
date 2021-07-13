#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:35:27 2021

@author: Anna
"""

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
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from mri.operators import WaveletN
from mri.reconstructors import SingleChannelReconstructor

from modopt.opt.proximity import SparseThreshold #, ElasticNet, OrderedWeightedL1Norm
from modopt.opt.linear import Identity
from modopt.math.metrics import ssim

from densities_calculation.mask import compute_indices
from densities_calculation.utils import extract_images, nrmse
from densities_calculation.calculate_densities import unravel_pi
from densities_calculation.generate_scheme import generate_full_scheme, generate_blocks_list, num_samples, split_det_rand
from reconstruction.fourier import masked_fourier_op

img_size = 128
n = img_size ** 2

wavelet = 'sym4'
level = 3
scheme_type = 'radial'
block_type = 'isolated'
    
sub_sampling_rate = 0.2

num_runs = 5 # number of runs of reconstruction algorithm
num_imgs = 5 # number of images over which the result of reconstruction is averaged

#mus = np.logspace( -4, -6, 4 ) # regularisation parameter of the reconstruction algorithm
#mus = [ 1e1 ]

dens_type = [ "un", "inf", "th_is", "th_anis", "l" ]

img_list = extract_images( "../../../brain_images/fastmri/file1000561.h5", "h5", 
                          img_size = img_size, num_images = num_imgs ) # images to reconstruct

linear_op = WaveletN( wavelet_name = wavelet, nb_scale = level, padding_mode = 'periodization' )

regularizer_op = SparseThreshold( Identity(), 2e-7, thresh_type = "soft" )


######## Generate points of kspace and blocks of points


full_kspace = generate_full_scheme( scheme_type, block_type, img_size )
blocks_list = generate_blocks_list( scheme_type, block_type, img_size )
det_blocks_list, rand_blocks_list = split_det_rand( img_size, blocks_list, 'det_center' )
nb_samples = num_samples( sub_sampling_rate, scheme_type, blocks_list, det_blocks_list, rand_blocks_list )

det_indices = [item for sublist in det_blocks_list for item in sublist]

####### Load densities
pi = {}

for pi_type in dens_type:
    pi[ pi_type ] = np.load( "pi_dens/pi_per_block_"+pi_type+"_"+str(img_size)+".npy" )

####### Unravel pi vectors
pi_fl = unravel_pi( pi, dens_type, rand_blocks_list, full_kspace.shape[ 0 ] )

        

####### Initialize variables to keep track of ssim, mu, nrmse
meas_val = { 'SSIM': {}, 'NRMSE': {}, 'MU': {} }
for meas in [ 'SSIM', 'NRMSE', 'MU' ]:
    for pi_type in dens_type:
        meas_val[ meas ][ pi_type ] = [] #np.zeros( num_runs )
        
        
####### Setting up regularisation parameter
mus = {}
for pi_type in dens_type:
    if pi_type == 'inf':
        mus[ pi_type ] = np.logspace( -3, -6, 5 )
    else:
        mus[ pi_type ] = np.logspace( -4, -6, 4 )
            
####### Reference and reconstructed images for plotting      
ref_imgs = {}
reconstr_imgs = {}

###### Creating directory for results

script_dir = os.path.dirname( __file__ )
rec_img_dir = os.path.join( script_dir, 'rec_imgs/' )
if not os.path.isdir( rec_img_dir ):
    os.makedirs( rec_img_dir )        


####### Reconstruction

print( "Reconstruction" )

start_time = time.time()
for pi_type in dens_type:
    
    for i in range( num_runs ):
        pi_mask = compute_indices( pi[ pi_type ], nb_samples )
        
        fourier_op = masked_fourier_op( img_size, full_kspace, det_indices, 
                                       rand_blocks_list, pi_fl[ pi_type ], pi_mask, normalize = False )

        # Setup Reconstructor
        reconstructor = SingleChannelReconstructor(
            fourier_op = fourier_op,
            linear_op = linear_op,
            regularizer_op = regularizer_op,
            gradient_formulation = 'synthesis',
            num_check_lips = 0,
            verbose = 0,
        )
    
    
        
        for j in range( num_imgs ):
        
            img = img_list[ j ]
            kspace_obs = fourier_op.op( img )
        
            cur_ssim = 0
            cur_mu = mus[ pi_type ][ 0 ]
            cur_nrmse = 1.0
                
            for mu in mus[ pi_type ]:
                #print(mu)
                reconstructor.prox_op.weights = mu
            
            
                x_final, costs, metric = reconstructor.reconstruct(
                        kspace_data = kspace_obs,
                        optimization_alg = 'fista',
                        num_iterations = 150,
                        #metrics=metrics,
                        #lambda_update_params = { "restart_strategy":"greedy", "s_greedy":1.1, "xi_restart":0.96 }
                )
                
                if ssim( x_final, img ) > cur_ssim:
                    cur_ssim = ssim( x_final, img )
                    cur_mu = mu
                    cur_rec_img = x_final
                if nrmse( x_final, img ) < cur_nrmse:
                    cur_nrmse = nrmse( x_final, img )
            
            if ( j == num_imgs // 2 ) and ( i == 0 ):
                ref_imgs[ pi_type ] = img
                reconstr_imgs[ pi_type ] = cur_rec_img 
            
            meas_val[ 'MU' ][ pi_type ].append( cur_mu )
            meas_val[ 'SSIM' ][ pi_type ].append( cur_ssim )
            meas_val[ 'NRMSE' ][ pi_type ].append( cur_nrmse )


####### Display results
        
print( "Reconstruction time:", time.time() - start_time )

data = []

for meas in [ 'SSIM', 'NRMSE', 'MU' ]:
    for pi_type in dens_type:
        for i in range( num_runs * num_imgs ):
            data.append( [ meas, pi_type, meas_val[ meas ][ pi_type ][ i ] ] )
                
            
df = pd.DataFrame( data = data, columns = [ 'meas', 'pi_type', 'val' ] )
df.to_csv( 'out_data_'+str(img_size)+'.csv' )   


######## Save reference and reconstructed images
for pi_type in dens_type:
    np.save( "rec_imgs/ref_img_"+pi_type+"_"+str(img_size)+".npy", ref_imgs[ pi_type ] )
    np.save( "rec_imgs/reconstr_img_"+pi_type+"_"+str(img_size)+".npy", reconstr_imgs[ pi_type ] )
