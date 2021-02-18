#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:19:11 2020

@author: Anna
"""

import numpy as np
import scipy
from tqdm import tqdm

from mri.operators import WaveletN, NonCartesianFFT 

from densities_calculation.generate_scheme import generate_full_scheme

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
    
    for i in tqdm( range( num_kspace_loc ) ):
        
        x[ i ] = 1    
        image = x
        
        fourier_op = NonCartesianFFT( samples = kspace_loc, 
                                     shape = (img_size, img_size), implementation='cpu')
        adj_four_coef = fourier_op.adj_op( image )
        A[ i, : ] = np.conj( linear_op.op( adj_four_coef ) )
        x[ i ] = 0
    
    return A

def A_block_isotropic( img_size, wavelet, level, kspace_loc, block ):
    """
    Calculate the measurement matrix A = F Psi^*
    
    Parameters
    ----------
    img_size: integer
        Image size, a power of 2
    wavelet: string
        Type of wavelet; the wavelet transform should be an orthogonal transform
    level: integer
        Level of wavelet transform
    kspace_loc: (N, 2) np.ndarray
        The kspace locations of the full scheme
    block: list
        List of indices of rows to calculate
        
    Returns
    -------
    np.ndarray
        Block of the measurement matrix of size corresponding to the row indices in block (N,img_size**2)
    
    """
    
    n = img_size ** 2
    num_kspace_loc = kspace_loc.shape[ 0 ]
    A_block = np.zeros( ( len( block ), n ), dtype = 'complex128' )
    
    linear_op = WaveletN( wavelet_name = wavelet, nb_scale = level, padding_mode = 'periodization' )
    
    x = np.zeros( ( num_kspace_loc, ) )
    
    for i, row_ind in enumerate( block ):
        
        x[ row_ind ] = 1    
        image = x
        
        fourier_op = NonCartesianFFT( samples = kspace_loc, 
                                     shape = (img_size, img_size), implementation='cpu')
        adj_four_coef = fourier_op.adj_op( image )
        A_block[ i, : ] = np.conj( linear_op.op( adj_four_coef ) )
        x[ row_ind ] = 0
    
    return A_block

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

def compute_A_by_column( img_size, wavelet, level, kspace_loc ):
    """Compute matrix A column by column.
    Same parameters as A_matrix_anisotropic"""
    
    n = img_size ** 2
    num_kspace_loc = kspace_loc.shape[ 0 ]
    A = np.zeros( ( num_kspace_loc, n ), dtype = 'complex128' )
    
    linear_op = WaveletN( wavelet_name = wavelet, nb_scale = level, padding_mode = 'periodization' )
    
    x = np.zeros( ( n, ) )
    linear_op.op( np.reshape( x, ( img_size, img_size ) ) )
    
    for i in range( n ):
        if i%1000 == 0:
            print( "Calculating A, iteration:", i )
        
        x[ i ] = 1    
        image = x
        
        fourier_op = NonCartesianFFT( samples = kspace_loc, 
                                     shape = (img_size, img_size), implementation='cpu')
        adj_wav_coef = linear_op.adj_op( image )
        A[ :, i ] = fourier_op.op( adj_wav_coef )
        x[ i ] = 0
    return A

if __name__ == "__main__":
    
    img_size = 8
    wavelet = 'sym4'
    level = 3
    
    scheme_type = 'cartesian'
    block_type = 'isolated'
    
    kspace_loc = generate_full_scheme( scheme_type, block_type, img_size )
    
    A_by_line = A_matrix_anisotropic( img_size, wavelet, level, kspace_loc )
    A_by_column = compute_A_by_column( img_size, wavelet, level, kspace_loc )
    
    print( np.allclose( A_by_line, A_by_column ) )
    
    
    
    
    
    
    
    
    