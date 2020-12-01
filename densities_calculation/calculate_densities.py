#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:48:13 2020

@author: Anna
"""

import numpy as np
import matplotlib.pyplot as plt

from densities_calculation.utils import wav_coef_array_to_matrix
from densities_calculation.s_distribution import s_distribution
from densities_calculation.compute_A import A_matrix_anisotropic, compute_pseudoinv_A
from densities_calculation.mask import compute_mask

def calculate_pi_blocks( img_size, A, pseudo_inv_A, level, s_distrib, blocks_list ):
    """
    Calculate pi_theta, pi_lambda in the block-structured anisotropic setting
    
    Parameters
    ----------
    img_size: integer
        size of the image (power of 2)
    A: ndarray
        matrix of measurements (complex-valued) A = F Psi^*
    pseudo_inv_A: ndarray
        pseudoinverse of A (complex-valued) psinv(A) = ( A^* A )^{-1} A^*
    level: integer
        level of the wavelet transform
    s_distrib: ndarray
        tridiagonal matrix of size (level+1, level+1) containing the values of numbers of nonzeros
        wavelet coefficients per subband
    blocks_list: list
        list of lists of row numbers corresponding to blocks of measurements
    
    Returns
    -------
    pi_th_anis: ndarray
        pi_theta_anisotropic, array of size len( blocks_list )
    pi_l: ndarray
        pi_lambda, array of size len( blocks_list )
    """

    J = int( np.log2( img_size ) )
    
    pi_th_is = np.zeros( ( len( blocks_list ), 1 ) )
    pi_th_anis = np.zeros( ( len( blocks_list ), 1 ) )
    pi_l = np.zeros( ( len( blocks_list ), 1 ) )
    
    block_num = 0
    for block in blocks_list:
        #if block_num%10 == 0:
            #print( "Calculating pi, iteration:", block_num )
           
        a_block = A[ block, : ]
        ps_a_block = pseudo_inv_A[ :, block ]
        
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
        
    pi_th_is = pi_th_is / np.sum( pi_th_is )
    pi_th_anis = pi_th_anis / np.sum( pi_th_anis )
    pi_l = pi_l / np.sum( pi_l )

    return pi_th_is, pi_th_anis, pi_l

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

