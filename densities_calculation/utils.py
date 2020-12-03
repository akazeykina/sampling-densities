#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:57:28 2020

@author: Anna
"""

import numpy as np
import nibabel as nib
import h5py

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
        
        cV = np.reshape( coefs[ 2**( 2 * ( J - j ) ) : 2**( 2 * ( J - j ) + 1 ) ], 
                        ( 2**( J - j ), 2**( J - j ) ) )
        cH = np.reshape( coefs[ 2**( 2 * ( J - j ) + 1 ) : 3 * 2**( 2 * ( J - j ) ) ], 
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
        image to shrink
        
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

def extract_images( fname, fextension, data_type = None, num_images = 10 ):
    """
    Extract images from a given file
    
    Parameters
    ----------    
    fname: string
        name of the file
    fextension: string
        file extension: "nii" or "h5"
    data_type: string
        "T1w" or "T2w"
    num_images: int
        Number of images to extract
    Returns
    -------
    list
        list of ten images extracted from the file
    """
    if fextension == "nii":
        images = nib.load( fname )
        images = images.get_fdata()
        if data_type == "T1w":
            num_img = images.shape[ 0 ]
            img_list = [ images[ num_img // 2 - 50 + 100 // num_images * j, :, : ] for j in range( num_images ) ]
        elif data_type == "T2w":
            num_img = images.shape[ 2 ]
            img_list = [ images[ :, :, num_img // 2 - 10 + 20 // num_images * j ] for j in range( num_images ) ]
    elif fextension == "h5":
        with h5py.File( fname, 'r' ) as h5_obj:
            images = h5_obj['reconstruction_esc'][:]
        num_img = images.shape[ 0 ]
        img_list = [ images[ num_img // 2 - 10 + 20 // num_images * j, :, : ] for j in range( num_images ) ]
    
    
    return img_list


def split_det_rand( img_size, blocks_list, spl_type = None, det_portion = 0.0 ):
    """
    Split the blocks into deterministically chosen and randomly sampled
    
    Parameters
    ----------    
    img_size: int
        desired size of the image (typically power of 2)
    blocks_list: list
        list of blocks of points
    spl_type: string
        'none': choose deterministically the first several blocks, 
        'det_center': for radial and spiral schemes choose the central point deterministically
        'det_central_line': for cartesian scheme choose the central line deterministically 
    det_portion: float
        portion of points to be chosen deterministically (between 0.0 and 1.0)
        
    Returns
    -------
    det_blocks_list: list
        list of deterministically chosen blocks
    rand_blocks_list: list
        list of randomly sampled blocks
    """
    
    if spl_type == 'det_center':
        det_blocks_list = [ blocks_list[ 0 ] ]
        rand_blocks_list = blocks_list[ 1 : ]
    elif spl_type == 'det_central_line':
        bl_list_copy = blocks_list.copy()
        det_blocks_list = [ bl_list_copy.pop( img_size //2 ) ]
        rand_blocks_list = bl_list_copy
    elif spl_type == 'none':
        det_last_ind = int( det_portion * len( blocks_list ) )
        det_blocks_list = blocks_list[ : det_last_ind ]
        rand_blocks_list = blocks_list[ det_last_ind : ]
    
    return det_blocks_list, rand_blocks_list

def blocks_list_size( blocks_list ):
    """
    Calculate the overall number of points in a list of blocks
    
    Parameters
    ----------    
    blocks_list: list
        list of blocks of points
        
    Returns
    -------
    int
        number of points in the blocks of blocks_list
    """
    
    tot_size = 0
    
    for block in blocks_list:
        tot_size += len( block )
        
    return tot_size