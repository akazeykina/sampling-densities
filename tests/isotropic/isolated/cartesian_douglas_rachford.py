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

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from mri.operators import WaveletN

from modopt.math.metrics import ssim

from densities_calculation.mask import compute_indices
from densities_calculation.utils import extract_images, nrmse
from densities_calculation.calculate_densities import unravel_pi
from densities_calculation.generate_scheme import generate_full_scheme, generate_blocks_list, num_samples
from reconstruction.fourier import masked_fourier_op
from reconstruction.douglas_rachford import dr1

img_size = 32
n = img_size ** 2

wavelet = 'sym4'
level = 3
scheme_type = 'cartesian'
block_type = 'isolated'
    

decays = [] # [ 2, 4, 6 ] # decays of pi_radial
cutoffs = [] # [ 0.1, 0.2, 0.3 ] #cutoffs of pi_radial
    
sub_sampling_rate = 0.2

num_runs = 1 # number of runs of reconstruction algorithm
num_imgs = 1 # number of images over which the result of reconstruction is averaged

mus = np.logspace( -6, -9, 4 ) # regularisation parameter of the reconstruction algorithm
#mu = [ 1e-2 ]

#dens_type  = [ "rad_"+str(decay)+"_"+str(cutoff) for decay in decays for cutoff in cutoffs ] # types of densities to compute
#cs_dens_type = [ "inf", "th_is", "th_anis", "l" ]
#dens_type = dens_type + cs_dens_type
dens_type = [ "l" ]
cs_dens_type = [ "l" ]

img_list = extract_images( "../../../brain_images/fastmri/file1000001.h5", "h5", 
                          img_size = img_size, num_images = num_imgs ) # images to reconstruct

linear_op = WaveletN( wavelet_name = wavelet, nb_scale = level, padding_mode = 'periodization' )
linear_op.op( img_list[ 0 ] )


######## Generate points of kspace and blocks of points


full_kspace = generate_full_scheme( scheme_type, block_type, img_size )
blocks_list = generate_blocks_list( scheme_type, block_type, img_size )
nb_samples = num_samples( sub_sampling_rate, scheme_type, blocks_list, [], blocks_list )

####### Load densities
pi = {}

for pi_type in dens_type:
    pi[ pi_type ] = np.load( "pi_dens/pi_per_block_"+pi_type+"_"+str(img_size)+".npy" )

####### Unravel pi vectors
pi_fl = unravel_pi( pi, cs_dens_type, blocks_list, full_kspace.shape[ 0 ] )
for decay in decays:
    for cutoff in cutoffs:
        pi_fl[ "rad_"+str(decay)+"_"+str(cutoff) ] = pi[ "rad_"+str(decay)+"_"+str(cutoff) ]
        

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
        pi_mask = compute_indices( pi[ pi_type ], nb_samples )
        
        fourier_op = masked_fourier_op( img_size, full_kspace, [], 
                                       blocks_list[ 0: ], pi_fl[ pi_type ], pi_mask, normalize = False )
        A = lambda x: fourier_op.op( linear_op.adj_op( x ) )
        At = lambda y: linear_op.op( fourier_op.adj_op( y ) )
        
    
        cur_mu = [] # stores mu corresponding to the best ssim for every image
        cur_ssims = [] # stores best ssim for every image
        cur_nrmse = []
        
        for j in range( num_imgs ):
        
            img = img_list[ j ]
            kspace_obs = fourier_op.op( img )
            num_obs = kspace_obs.size
            
            z = At( kspace_obs )
            print( At( kspace_obs ).shape )
            print( A( z ).shape )
        
            cur_ssims.append( 0 )
            cur_mu.append( mus[ 0 ] )
            cur_nrmse.append( 1.0 )
                
            for mu in mus:
            
                
                coef_final = dr1( kspace_obs, A, At, 
                                 niter = 150, gamma = mu, plot_error = True )
                x_final = linear_op.adj_op( coef_final )
                
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

