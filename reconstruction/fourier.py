#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:48:49 2020

@author: Anna
"""

import numpy as np

from mri.operators import NonCartesianFFT

def masked_fourier_op( img_size, full_kspace, det_indices, rand_blocks_list, pi_fl, pi_mask, normalize = True ):
    """
    Compute masked Fourier operator
    
    Parameters
    ----------
    img_size: int
        Size of image (power of 2)
    full_kspace: ndarray
        Points of full kspace (array of size (N, 2))
    det_indices: list
        List of indices of points in kspace to be sampled deterministically
    rand_blocks_list: list
        List of sublists of indices of points in kspace to be sampled randomly
    pi_fl: ndarray
        Vector of probabilities (of size N)
    pi_mask: ndarray
        Indices of blocks in rand_blocks_list to be sampled
    normalize: bool
        If True then apply normalization as in [Adcock, Boyer, Brugiapaglia, 2018]
        
    Returns
    -------
    mri.operators.fourier.non_cartesian.NonCartesianFFT
        Masked Fourier operator
    """
    
    full_mask = np.zeros( ( full_kspace.shape[ 0 ], ), dtype = 'bool' )
    
    if det_indices:
        full_mask[ det_indices ] = True
    
    for ind in pi_mask:
        full_mask[ rand_blocks_list[ ind ] ] = True
        
    kspace_loc = full_kspace[ full_mask, : ]
      
    #num_obs = np.sum( full_mask )
    #num_points[ pi_type ].append( num_obs )
    rand_indices = [item for sublist in rand_blocks_list for item in sublist]
    
    if normalize:
        div = np.zeros( ( full_kspace.shape[ 0 ], ) )
        div[ det_indices ] = 1
        div[ rand_indices ] = np.sqrt( pi_fl[ rand_indices ] * len( rand_blocks_list ) )
        
        fourier_op = NonCartesianFFT( samples = kspace_loc, shape = ( img_size, img_size ), implementation = 'cpu' )
        #fourier_op = FFT( samples = kspace_loc, shape = cartesian_image.shape )
        norm_fourier_op = NonCartesianFFT( samples = kspace_loc, shape = ( img_size, img_size ), implementation = 'cpu' )
        norm_fourier_op.op = lambda x: fourier_op.op( x ) / div[ full_mask ]
        norm_fourier_op.adj_op = lambda x: fourier_op.adj_op( x / div[ full_mask ] )
    else: 
        norm_fourier_op = NonCartesianFFT( samples = kspace_loc, shape = ( img_size, img_size ), implementation = 'cpu' )
        
    return norm_fourier_op








