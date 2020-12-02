#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:48:13 2020

@author: Anna
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy

from densities_calculation.utils import wav_coef_array_to_matrix
from densities_calculation.compute_A import A_matrix_anisotropic, compute_pseudoinv_A, A_block_isotropic

def calculate_pi_blocks( img_size, scheme_type, full_kspace, reg_type, cond, lam,
                        wavelet, level, s_distrib, blocks_list ):
    """
    Calculate pi_theta, pi_lambda in the block-structured anisotropic setting
    
    Parameters
    ----------
    img_size: integer
        size of the image (power of 2)
    scheme_type: string
        "cartesian", "radial" or "spiral"
    full_kspace: ndarray
        The kspace locations of the full scheme, matrix of size (N,2) with columns k_y, k_x
    reg_type: string
        Type of regularisation used to calculate the pseudoinverse of A; 'svd' or 'tikh'
    cond: float
        Condition number for svd regularisation
    lam: float
        Parameter lambda for Tikhonov regularisation
    wavelet: string
        wavelet type (orthogonal transform)
    level: integer
        level of the wavelet transform
    s_distrib: ndarray
        tridiagonal matrix of size (level+1, level+1) containing the values of numbers of nonzeros
        wavelet coefficients per subband
    blocks_list: list
        list of lists of row numbers corresponding to blocks of measurements
    
    Returns
    -------
    pi_inf: ndarray
        pi_inf, array of size len( blocks_list )
    pi_th_is: ndarray
        pi_theta_isotropic, array of size len( blocks_list )
    pi_th_anis: ndarray
        pi_theta_anisotropic, array of size len( blocks_list )
    pi_l: ndarray
        pi_lambda, array of size len( blocks_list )
    """

    J = int( np.log2( img_size ) )
    
    pi_inf = np.zeros( ( len( blocks_list ), 1 ) )
    pi_th_is = np.zeros( ( len( blocks_list ), 1 ) )
    pi_th_anis = np.zeros( ( len( blocks_list ), 1 ) )
    pi_l = np.zeros( ( len( blocks_list ), 1 ) )
    
    ############## Calculate matrix A and its pseudoinverse
    
    if scheme_type != 'cartesian':
        print( "Calculate matrix A" )
        st_time = time.time()
        A = A_matrix_anisotropic( img_size, wavelet, level, full_kspace )
        print( "Matrix A calculation time:", time.time() - st_time ) 


        print("Calculating pseudoinverse")
        st_time = time.time()
        if reg_type == "svd":
            pseudo_inv_A = scipy.linalg.pinv2( A, cond )
        else:
            pseudo_inv_A = compute_pseudoinv_A( A, lam, parallel = True )
        print( "Pseudoinverse calculation time:", time.time() - st_time )
        print("End of calculation of pseudoinverse")
        
    ###################################
    
    block_num = 0
    for block in blocks_list:
        #if block_num%10 == 0:
            #print( "Calculating pi, iteration:", block_num )
            
        if scheme_type == 'cartesian':
            a_block = A_block_isotropic( img_size, wavelet, level, full_kspace, block )
            ps_a_block = np.conj( a_block.T )
        else:
           a_block = A[ block, : ]
           ps_a_block = pseudo_inv_A[ :, block ]
        
        pi_inf[ block_num, : ] = np.linalg.norm( a_block.flatten(), ord = np.inf ) ** 2
        pi_th_is[ block_num, : ] = np.linalg.norm( a_block.flatten(), ord = np.inf )
        pi_th_anis[ block_num, : ] = np.sqrt( 
                 np.linalg.norm( a_block.flatten(), ord = np.inf ) * \
                np.linalg.norm( a_block.flatten(), ord = 1 ) )
        
        subsum_th1 = 0
        subsum_th2 = 0
        subsum_l1 = 0
        subsum_l2 = 0
        
        for i in range( len( block ) ):
        
            ps_a_col = ps_a_block[ :, i ]
            a_line = a_block[ i, : ]
        
            ps_a_matr = wav_coef_array_to_matrix( ps_a_col, level )
            a_matr = wav_coef_array_to_matrix( a_line, level )
    
            term_th1 = 0
            term_th2 = 0
            term_l1 = 0
            term_l2 = 0
        
            pcAm = np.max( np.abs( ps_a_matr[ : 2**( J - level ), : 2**( J - level ) ] ) )
            cAm = np.max( np.abs( a_matr[ : 2**( J - level ), : 2**( J - level ) ] ) )
            subsum_th1 = subsum_th1 + pcAm  * s_distrib[ 0, 0 ]
            if s_distrib[ 0, 0 ] != 0:
                subsum_th2 = max( subsum_th2, pcAm )
            subsum_l1 = subsum_l1 + pcAm**2 * s_distrib[ 0, 0 ]
            subsum_l2 = subsum_l2 + cAm**2 * s_distrib[ 0, 0 ]
    
            for j in reversed( range( 1, level + 1 ) ):
                pcHm = np.max( np.abs( ps_a_matr[ : 2**( J - j ), 2**( J - j ) : 2**( J - j + 1 ) ] ) )
                pcVm = np.max( np.abs( ps_a_matr[ 2**( J - j ) : 2**( J - j + 1 ), : 2**( J - j ) ] ) )
                pcDm = np.max( np.abs( ps_a_matr[ 2**( J - j ) : 2**( J - j + 1 ), 2**( J - j ) : 2**( J - j + 1 ) ] ) ) 
            
                cHm = np.max( np.abs( a_matr[ : 2**( J - j ), 2**( J - j ) : 2**( J - j + 1 ) ] ) )
                cVm = np.max( np.abs( a_matr[ 2**( J - j ) : 2**( J - j + 1 ), : 2**( J - j ) ] ) )
                cDm = np.max( np.abs( a_matr[ 2**( J - j ) : 2**( J - j + 1 ), 2**( J - j ) : 2**( J - j + 1 ) ] ) ) 
            
                k = level - j
                term_th1 = pcHm * s_distrib[ k, k + 1 ] + pcVm * s_distrib[ k + 1, k ] + pcDm * s_distrib[ k + 1, k + 1 ]
                if s_distrib[ k, k+1 ] != 0:
                    term_th2 = pcHm
                if s_distrib[ k+1, k ] != 0:
                    term_th2 = max( term_th2, pcVm )
                if s_distrib[ k+1, k+1 ] != 0:
                    term_th2 = max( term_th2, pcDm )
                term_l1 = pcHm**2 * s_distrib[ k, k + 1 ] + pcVm**2 * s_distrib[ k + 1, k ] + \
                    pcDm**2 * s_distrib[ k + 1, k + 1 ]
                term_l2 =  cHm**2 * s_distrib[ k, k + 1 ] + cVm**2 * s_distrib[ k + 1, k ] + \
                    cDm**2 * s_distrib[ k + 1, k + 1 ]
            
                subsum_th1 = subsum_th1 + term_th1
                subsum_th2 = max( subsum_th2, term_th2 )
                subsum_l1 = subsum_l1 + term_l1
                subsum_l2 = subsum_l2 + term_l2
        
        pi_th_is[ block_num, : ] = pi_th_is[ block_num, : ] * subsum_th1
        pi_th_anis[ block_num, : ] = pi_th_anis[ block_num, : ] * np.sqrt( subsum_th1 ) * np.sqrt( subsum_th2 )
        pi_l[ block_num, : ] = np.sqrt( subsum_l1 ) * np.sqrt( subsum_l2 )
        
        block_num += 1
        
    pi_inf = pi_inf / np.sum( pi_inf )
    pi_th_is = pi_th_is / np.sum( pi_th_is )
    pi_th_anis = pi_th_anis / np.sum( pi_th_anis )
    pi_l = pi_l / np.sum( pi_l )

    return pi_inf, pi_th_is, pi_th_anis, pi_l

def pi_rad( decay, cutoff, im_size ):
    """
    Calculate radial density that is a power decay function |x|^{-d} with cutoff 
    Code adapted from CSMRI_sparkling generate_density function; works for square images only
    
    Parameters
    ----------
    decay: real
        d (positive)
    cutoff: real
        percentage of kspace where pi_rad is constant (between 0.0 and 1.0)
    im_size: ndarray
        array np.array( [ img_size, img_size ] ) where img_size is the size of the image
    
    Returns
    -------
    pi_rad: ndarray
        radial density, array of shape im_size
    """
        
    grid_lines = [ np.linspace( -size / 2 + 0.5, size / 2 - 0.5, size ) for size in im_size ]
    
    grid = np.meshgrid( *grid_lines, indexing = 'ij' )
    
    norm = np.sqrt( np.sum( np.power( grid, 2 ), axis = 0 ) )
    
    density = np.power( norm, -decay )
    
    mid_im_size = im_size / 2
    
    
    ind = tuple( im_size // 2 + ( im_size * cutoff // 2 ).astype( 'int' ) - 1 )
    
    density[ norm < norm[ ind ] ] = density[ ind ] 
    
    density[ norm > mid_im_size[ 0 ] ] = 0 
    
    density = density / ( np.sum( density ) )
    
    return density

def unravel_pi( pi, dens_type, blocks_list, num_kpoints ):
    """
    Turn a vector of probabilities associated with blocks to a vector of probabilities
    defined for each point of kspace (for plotting)
    
    Parameters
    ----------
    pi: dict
        Dictionary of vectors of probabilities defined for blocks
    dens_type: list
        Density type (e.g. "rad", "inf", "th_is", "th_anis", "l"); keys of pi dictionary
    blocks_list: list
        List of sublists of row indices corresponding to blocks of points of kspace
    num_kpoints: int
        Number of kspace points
    
    Returns
    -------
    pi_fl: dict
        Unraveled pi
    """
    pi_fl = {}
    for pi_type in dens_type:
        pi_fl[ pi_type ] = np.zeros( ( num_kpoints, ) )

    for j, block in enumerate( blocks_list ): 
        for pi_type in dens_type:
            pi_fl[ pi_type ][ block ] = pi[ pi_type ][ j, : ]
            
    return pi_fl

#def pi_blocks_to_matr( pi_fl, dens_type, img_size ):
#
#    pi_matr = {}
#    for pi_type in dens_type:
#        pi_matr[ pi_type ] = np.reshape( pi_fl[ pi_type ], ( img_size, img_size ), order = 'C' )
#    
#    return pi_matr

