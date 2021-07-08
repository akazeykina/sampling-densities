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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))

from mri.operators import WaveletN, NonCartesianFFT
from mri.reconstructors import SingleChannelReconstructor

from modopt.opt.proximity import SparseThreshold #, ElasticNet, OrderedWeightedL1Norm
from modopt.opt.linear import Identity
from modopt.math.metrics import ssim

from densities_calculation.utils import extract_images, nrmse


img_size = 256
n = img_size ** 2

wavelet = 'sym4'
level = 3
scheme_type = 'cartesian'
block_type = 'isolated'
    
#decay = 4
#cutoff = 0.1
#decays = [ 2, 4, 6 ] # decays of pi_radial
#cutoffs = [ 0.1, 0.2, 0.3 ] #cutoffs of pi_radial
    
#sub_sampling_rate = 0.2

#num_runs = 5 # number of runs of reconstruction algorithm
num_imgs = 5 # number of images over which the result of reconstruction is averaged

mus = np.logspace( -4, -6, 4 ) # regularisation parameter of the reconstruction algorithm
#mus = [ 1e1 ]

#dens_type  = [ "rad_"+str(decay)+"_"+str(cutoff) for decay in decays for cutoff in cutoffs ] # types of densities to compute
#cs_dens_type = [ "inf", "th_is", "th_anis", "l" ]
#dens_type = dens_type + cs_dens_type
dens_type = [ "rad", "inf", "th_is", "th_anis", "l" ]
#cs_dens_type = [ "l" ]

img_list = extract_images( "../../../../brain_images/T2w/sub-OAS30008_sess-d0061_acq-TSE_T2w.nii", "nii", "T2w",
                          img_size = img_size, num_images = num_imgs ) # images to reconstruct

linear_op = WaveletN( wavelet_name = wavelet, nb_scale = level, padding_mode = 'periodization' )

regularizer_op = SparseThreshold( Identity(), 2e-7, thresh_type = "soft" ) 

######## Generate points of kspace and blocks of points


#full_kspace = generate_full_scheme( scheme_type, block_type, img_size )
#blocks_list = generate_blocks_list( scheme_type, block_type, img_size )
#nb_samples = num_samples( sub_sampling_rate, scheme_type, blocks_list, [], blocks_list )

        

####### Initialize variables to keep track of ssim, mu, nrmse
meas_val = { 'SSIM': {}, 'NRMSE': {}, 'MU': {} }
for meas in [ 'SSIM', 'NRMSE', 'MU' ]:
    for pi_type in dens_type:
        meas_val[ meas ][ pi_type ] = [] #np.zeros( num_runs )


####### Reconstruction

print( "Reconstruction" )

start_time = time.time()
for pi_type in dens_type:
    
    kspace_loc = np.load( "kpoints/sparkling_"+pi_type+"_"+str(img_size)+".npy" ) 
    
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


    for j in range( num_imgs ):
    
        img = img_list[ j ]
        kspace_obs = fourier_op.op( img )
    
        cur_ssim = 0
        cur_mu = mus[ 0 ]
        cur_nrmse = 1.0
            
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
            
            if ssim( x_final, img ) > cur_ssim:
                    cur_ssim = ssim( x_final, img )
                    cur_mu = mu
            if nrmse( x_final, img ) < cur_nrmse:
                    cur_nrmse = nrmse( x_final, img )
                    
        meas_val[ 'MU' ][ pi_type ].append( cur_mu )
        meas_val[ 'SSIM' ][ pi_type ].append( cur_ssim )
        meas_val[ 'NRMSE' ][ pi_type ].append( cur_nrmse )


####### Display results
        
print( "Reconstruction time:", time.time() - start_time )

data = []

for meas in [ 'SSIM', 'NRMSE', 'MU' ]:
    for pi_type in dens_type:
        for i in range( num_imgs ):
            data.append( [ meas, pi_type, meas_val[ meas ][ pi_type ][ i ] ] )
                
            
df = pd.DataFrame( data = data, columns = [ 'meas', 'pi_type', 'val' ] )
df.to_csv( 'out_data_'+str(img_size)+'.csv' )   

