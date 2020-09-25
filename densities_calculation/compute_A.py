#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:19:11 2020

@author: Anna
"""

import numpy as np

from mri.operators import WaveletN, NonCartesianFFT 

def A_matrix_anisotropic( img_size, wavelet, level, kspace_loc ):
    """
    Calculate the measurement matrix A = F Psi^*
    
    Parameters
    ----------
    img_size: integer
        Image size, a power of 2
    wavelet: string
        The type of wavelet; the wavelet transform should be an orthogonal transform
    level: integer
        The level of wavelet transform
    kspace_loc: (N, 2) np.ndarray
        The kspace locations of the full scheme
        
    Returns
    -------
    np.ndarray
        The measurement matrix of size (N,img_size**2)
    
    """
    
    n = img_size ** 2
    num_kspace_loc = kspace_loc.shape[ 0 ]
    A = np.zeros( ( num_kspace_loc, n ), dtype = 'complex128' )
    
    linear_op = WaveletN( wavelet_name = wavelet, nb_scale = level, padding_mode = 'periodization' )
    
    x = np.zeros( ( num_kspace_loc, ) )
    
    for i in range( num_kspace_loc ):
        if i%1000 == 0:
            print( "Calculating A, iteration:", i )
        
        x[ i ] = 1    
        image = x
        
        fourier_op = NonCartesianFFT( samples = kspace_loc, 
                                     shape = (img_size, img_size), implementation='cpu')
        adj_four_coef = fourier_op.adj_op( image )
        A[ i, : ] = np.conj( linear_op.op( adj_four_coef ) )
        x[ i ] = 0
    
    return A

    
    
    
    
    
    
    
    
    
    
    
    
    