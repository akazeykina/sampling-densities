#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:19:11 2020

@author: Anna
"""

import numpy as np
import scipy

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

def compute_pseudoinv_A( A, scheme_type, cond = 0.0, lam = 0.0 ):
    """
    Calculate the pseudoinverse of A
    
    Parameters
    ----------
    A: ndarray
        2d matrix
    scheme_type: string
        type of sampling scheme (if 'cartesian' then the calculation of pseudoinverse is simplified)
    cond: float
        parameter cond of scipy.linalg.pinv2 (for using regularisation via discarding small svd values)
    lam: float
        parameter of Tikhonov regularisation for inverting A^*A
        
    Returns
    -------
    np.ndarray
        The pseudoinverse of A
    
    """
    
    print("Calculating pseudoinverse")
    if scheme_type == 'cartesian':
        pseudo_inv_A = np.conj( A.T )
    else:
        pseudo_inv_A = scipy.linalg.pinv2( A )  
        #A_conj = np.conj( A.T )
        #inv = scipy.linalg.pinvh( np.dot( A_conj, A ) )
        #pseudo_inv_A = np.dot( inv, A_conj )
    print("End of calculation of pseudoinverse")
    
    return pseudo_inv_A
    
    
    
    
    
    
    
    
    