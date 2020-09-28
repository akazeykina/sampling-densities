#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:57:28 2020

@author: Anna
"""

import numpy as np

def wav_coef_array_to_matrix( coefs, level ):
    """
    Convert the array of wavelet coefficients to a matrix
    
    Parameters
    ----------
    coefs: ndarray
        array of wavelet coefficients produced by WaveletN.op
    level: integer
        level of the wavelet transform
        
    Returns
    -------
    ndarray
        matrix of the wavelet coefficients
    """

    img_size = int( np.sqrt( coefs.size ) )
    J = int( np.log2( img_size ) )
    
    coef_matrix = np.zeros( ( img_size, img_size ), dtype = 'complex' )
    
    cA = np.reshape( coefs[ : 2**( 2 * ( J - level ) ) ], ( 2**( J - level ), 2**( J - level ) ) )
    coef_matrix[ : 2**( J - level ), : 2**( J - level ) ] = cA
    
    for j in reversed( range( 1, level + 1 ) ):
        
        cH = np.reshape( coefs[ 2**( 2 * ( J - j ) ) : 2**( 2 * ( J - j ) + 1 ) ], 
                        ( 2**( J - j ), 2**( J - j ) ) )
        cV = np.reshape( coefs[ 2**( 2 * ( J - j ) + 1 ) : 3 * 2**( 2 * ( J - j ) ) ], 
                        ( 2**( J - j ), 2**( J - j ) ) )
        cD = np.reshape( coefs[ 3 * 2**( 2 * ( J - j ) ) : 4 * 2**( 2 * ( J - j ) ) ], 
                        ( 2**( J - j ), 2**( J - j ) ) )
        
        coef_matrix[ : 2**( J - j ), 2**( J - j ) : 2**( J - j + 1 ) ] = cH
        coef_matrix[ 2**( J - j ) : 2**( J - j + 1 ), : 2**( J - j ) ] = cV
        coef_matrix[ 2**( J - j ) : 2**( J - j + 1 ), 2**( J - j ) : 2**( J - j + 1 ) ] = cD
            
    return coef_matrix

def nrmse( sim, obs ):
    """
    Compute nrmse metrics
    
    Parameters
    ----------    
    sim: ndarray
        predicted values (real or complex)
    obs: ndarray
        observed values (real)
        
    Returns
    -------
    float
        nrmse metrics
    
    """
    
    o_max = np.max( obs )
    o_min = np.min( obs )
    metrics = np.sqrt( np.mean( np.abs( sim - obs ) **2 ) ) / ( o_max - o_min )
    
    return metrics

def reduce_img_size( img_size, image ):
    """
    Reduce the size of the image (keep only low Fourier frequencies)
    
    Parameters
    ----------    
    img_size: int
        desired size of the image (typically power of 2)
    image: 2D ndarray
        the image to shrink
        
    Returns
    -------
    2D ndarray
        image of size img_size x img_size
    
    """
    
    mid = image.shape[ 0 ] // 2
    
    f_coef = np.fft.fftshift( np.fft.fft2( image ) )
    
    reduced_coef = f_coef[ mid - img_size // 2: mid + img_size // 2, mid - img_size // 2: mid + img_size // 2 ]
    
    reduced_image = np.fft.ifft2( np.fft.fftshift( reduced_coef ) )
    
    return np.abs( reduced_image )