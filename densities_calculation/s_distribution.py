#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:17:37 2020

@author: Anna
"""

from skimage import io
import numpy as np
import nibabel as nib


from mri.operators import WaveletN

from densities_calculation.utils import wav_coef_array_to_matrix, reduce_img_size

def s_distribution( img_size, im, wavelet, level, sparsity ):
    """
    Calculate the vector s of the numbers of nonzero wavelet coefficients per subband
    
    Parameters
    ----------
    img_size: integer
        size of the image (power of 2)
    im: string or ndarray
        name of the file containing the image from which the vector s is computed or
        the image itself
    wavelet: string
        wavelet type (orthogonal transform)
    level: integer
        level of the wavelet transform
    sparsity: float
        portion of coefficients that are considered to be nonzero (between 0.0 and 1.0)
        
    Returns
    -------
    ndarray
        a tri-diagonal matrix of size (level+1, level+1) containing the numbers of nonzero 
        wavelet coefficients per subband;
        s[0,0] corresponds to the approximation part, the lower diagonal to the vertical details, 
        the upper diagonal to the horizontal details, the main diagonal to the diagonal details
    """

    J = int( np.log2( img_size ) )

    linear_op = WaveletN( wavelet_name = wavelet, nb_scale = level, padding_mode = "periodization" )
    
    matr = np.zeros( ( img_size, img_size ), dtype = "complex128" )
    
    if type( im ) == str:
        image = io.imread( im, as_gray = True )
    else:
        image = im
            
    coeffs = linear_op.op( image )
    matr = wav_coef_array_to_matrix( coeffs, level )
    

    S = np.zeros( ( level + 1, level + 1 ) )
    eps = np.quantile( np.abs( matr ), 1 - sparsity )
    
    cA = matr[ : 2**( J - level ), : 2**( J - level ) ]
    
    S[ 0, 0 ] = np.sum( np.abs( cA ) > eps ) 
    
    for j in reversed( range( 1, level + 1 ) ):
        cH = matr[ : 2**( J - j ), 2**( J - j ) : 2**( J - j + 1 ) ]
        cV = matr[ 2**( J - j ) : 2**( J - j + 1 ), : 2**( J - j ) ]
        cD = matr[ 2**( J - j ) : 2**( J - j + 1 ), 2**( J - j ) : 2**( J - j + 1 ) ]
        
        k = level - j
        S[ k, k + 1 ] = np.sum( np.abs( cH ) > eps )
        S[ k + 1, k ] = np.sum( np.abs( cV ) > eps )
        S[ k + 1, k + 1 ] = np.sum( np.abs( cD ) > eps )
    
    return S

def avg_s_distribution( img_size, img_list, wavelet, level, sparsity ):
    """
    Calculate the averaged vector s of the numbers of nonzero wavelet coefficients per subband
    
    Parameters
    ----------
    img_size: integer
        size of the image (power of 2)
    img_list: string
        list of images from which the vector s is computed
    wavelet: string
        wavelet type (orthogonal transform)
    level: integer
        level of the wavelet transform
    sparsity: float
        portion of coefficients that are considered to be nonzero (between 0.0 and 1.0)
        
    Returns
    -------
    ndarray
        a tri-diagonal matrix of size (level+1, level+1) containing the number of nonzero 
        wavelet coefficients per subband;
        s[0,0] corresponds to the approximation part, the lower diagonal to the vertical details, 
        the upper diagonal to the horizontal details, the main diagonal to the diagonal details
    """
    
    
    S = np.zeros( ( level + 1, level + 1 ) )
    
    for img in img_list:

        S += s_distribution( img_size, img, wavelet, level, sparsity ) 
        
    S /= 10
    
    return S