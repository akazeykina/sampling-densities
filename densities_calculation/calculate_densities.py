#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:48:13 2020

@author: Anna
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

from densities_calculation.utils import wav_coef_array_to_matrix
from densities_calculation.s_distribution import s_distribution
from densities_calculation.compute_A import A_matrix_anisotropic
from densities_calculation.mask import compute_mask

def calculate_pi_blocks( img_size, A, level, s_distrib, blocks_list, scheme_type ):
    """
    Calculate pi_theta, pi_lambda in the block-structured anisotropic setting
    
    Parameters
    ----------
    img_size: integer
        size of the image (power of 2)
    A: ndarray
        matrix of measurements (complex-valued) A = F Psi^*
    level: integer
        level of the wavelet transform
    s_distrib: ndarray
        tridiagonal matrix of size (level+1, level+1) containing the values of numbers of nonzeros
        wavelet coefficients per subband
    blocks_list: list
        list of lists of row numbers corresponding to blocks of measurements
    scheme_type: string
        type of sampling scheme (if 'cartesian' then the calculation of pseudoinverse is simplified)
    
    Returns
    -------
    pi_th: ndarray
        pi_theta, array of size len( blocks_list )
    pi_l: ndarray
        pi_lambda, array of size len( blocks_list )
    """

    J = int( np.log2( img_size ) )

    pi_th = np.zeros( ( len( blocks_list ), 1 ) )
    pi_l = np.zeros( ( len( blocks_list ), 1 ) )
    
    print("Calculating pseudoinverse")
    if scheme_type == 'cartesian':
        pseudo_inv_A = np.conj( A.T )
    else:
        pseudo_inv_A = scipy.linalg.pinv2( A )  
        #A_conj = np.conj( A.T )
        #inv = scipy.linalg.pinvh( np.dot( A_conj, A ) )
        #pseudo_inv_A = np.dot( inv, A_conj )
    print("End of calculation of pseudoinverse")
    
    block_num = 0
    for block in blocks_list:
        #if block_num%10 == 0:
            #print( "Calculating pi, iteration:", block_num )
           
        a_block = A[ block, : ]
        ps_a_block = pseudo_inv_A[ :, block ]
        pi_th[ block_num, : ] = np.sqrt( 
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
        
        pi_th[ block_num, : ] = pi_th[ block_num, : ] * np.sqrt( subsum_th1 ) * np.sqrt( subsum_th2 )
        pi_l[ block_num, : ] = np.sqrt( subsum_l1 ) * np.sqrt( subsum_l2 )
        
        block_num += 1
        
    pi_th = pi_th / np.sum( pi_th )
    pi_l = pi_l / np.sum( pi_l )

    return pi_th, pi_l

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

if __name__ == "__main__":
    
    img_size = 64
    wavelet = 'sym4'
    level = 3
    scheme_type = 'cartesian'
    
    sparsity = 0.1 # sparsity level: assume that only s = 'sparsity' wavelets coefficients are non zero
    fname = '../brain_images/BrainPhantom'+str(img_size)+'.png' # image for computing s distribution
    
    sub_sampling_rate = 0.4
####### Distribution of sparsity coefficients

    s_distrib = s_distribution( img_size, fname, wavelet, level, sparsity )
    #print( s_distrib )
    
#    s_distrib = np.zeros( ( level + 1, level + 1 ) )  
#    s_distrib[ 0, 0 ] = 0.0125
#    s_distrib[ 0, 1 ] = 0.0175
#    s_distrib[ 1, 0 ] = 0.0175
#    s_distrib[ 1, 1 ] = 0.025
    
    
    # Construct the kspace_loc
    if scheme_type == "cartesian":
        x = np.linspace( -0.5, 0.5, img_size, endpoint = False )
        y = np.linspace( -0.5, 0.5, img_size, endpoint = False )
        X, Y = np.meshgrid( x, y )
        #Y, X = np.meshgrid( y, x )
        X = X.flatten('C')
        Y = Y.flatten('C')
        
        kspace_loc = np.stack( ( X, Y ), axis = 1 )  
        blocks_list = [ [ img_size * i + j for j in range( img_size ) ] for i in range( img_size )  ]
        #blocks_list = [ [ i ] for i in range( img_size * img_size )  ]
        nb_samples = int( sub_sampling_rate * len(blocks_list) )
        
    elif scheme_type == 'radial':
        n_rad = int( img_size / 2 )
        n_ang = int( img_size * np.pi ) 
        phi = np.linspace( 0, 2 * np.pi, n_ang, endpoint = False )
        rad = np.linspace( 0.5 / n_rad, 0.5, (n_rad-1), endpoint = False )
        Phi, Rad = np.meshgrid( phi, rad )
        #Rad, Phi = np.meshgrid( rad, phi )
        
        X = ( Rad * np.cos( Phi ) ).flatten( 'C' )
        Y = ( Rad * np.sin( Phi ) ).flatten( 'C' )
        
        kspace_loc = np.stack( ( X, Y ), axis = 1 )  
        kspace_loc = np.vstack( ( np.array( [ 0, 0 ] ), kspace_loc ) )
        blocks_list = [ [ (n_ang) * i + j + 1 for j in range( n_ang ) ] \
                         for i in range( n_rad-1 )  ] # for Phi, Rad
        #blocks_list = [ [ (n_rad-1) * i + j + 1 for j in range( n_rad-1 ) ] \
                         #for i in range( n_ang )  ] # for Rad, Phi
        #blocks_list = [ [ i + 1 ] \
                         #for i in range( n_ang * ( n_rad - 1 ) )  ] # for Rad, Phi
        blocks_list.insert( 0, [ 0 ] )
        nb_samples = int( sub_sampling_rate * len(blocks_list) * 2 / np.pi )
    elif scheme_type == 'spiral':
        n_c = int( img_size / 2 )
        n_s = int( img_size * np.pi )
        b = 0.5 / np.pi
        phi = np.linspace( 0, 2 * np.pi, n_s, endpoint = False )
        theta = np.linspace( np.pi / n_c, np.pi, ( n_c -  1 ), endpoint = False )
        Phi, Theta = np.meshgrid( phi, theta )
        #Theta, Phi = np.meshgrid( theta, phi )
        R = b * Theta
        Ang = Theta + Phi
        X = ( R * np.cos( Ang ) ).flatten( 'C' )
        Y = ( R * np.sin( Ang ) ).flatten( 'C' )
        
#        fig = plt.figure()
#        plt.scatter( X, Y, s = 1 )
#        plt.show()
        
        kspace_loc = np.stack( ( X, Y ), axis = 1 )  
        kspace_loc = np.vstack( ( np.array( [ 0, 0 ] ), kspace_loc ) )
        blocks_list = [ [ (n_s) * i + j + 1 for j in range( n_s ) ] \
                         for i in range( n_c-1 )  ] # for Phi, Theta
        blocks_list = [ [ (n_c-1) * i + j + 1 for j in range( n_c-1 ) ] \
                         for i in range( n_s )  ] # for Theta, Phi
        #blocks_list = [ [ i + 1 ] \
                         #for i in range( n_s * ( n_c - 1 ) )  ] # for isolated
        blocks_list.insert( 0, [ 0 ] )
        nb_samples = int( sub_sampling_rate * len(blocks_list) * 2 / np.pi )
    
    ####### Compute pi_theta and pi_lambda
    print( "Size of pi:", img_size**2 )    
    
    A = A_matrix_anisotropic( img_size, wavelet, level, kspace_loc )
    pi_th, pi_l = calculate_pi_blocks( img_size, A, level, s_distrib, blocks_list, scheme_type )


    ###### Compute masks
    pi_th_mask = compute_mask( pi_th, nb_samples )
    pi_l_mask = compute_mask( pi_l, nb_samples )

    pi_th_fl = np.zeros( ( kspace_loc.shape[ 0 ], 1 ) )
    pi_l_fl = np.zeros( ( kspace_loc.shape[ 0 ], 1 ) )
    
    pi_th_mask_fl = np.zeros( ( kspace_loc.shape[ 0 ], ), dtype = 'bool' )
    pi_l_mask_fl = np.zeros( ( kspace_loc.shape[ 0 ], ), dtype = 'bool' )
    
    ctr = 0
    for i in range( len( blocks_list ) ):
        block = blocks_list[ i ]
        len_block = len( block )
        pi_th_fl[ ctr : ctr + len_block, : ] = pi_th[ i, : ]
        pi_l_fl[ ctr : ctr + len_block, : ] = pi_l[ i, : ]
        if i in pi_th_mask:
            pi_th_mask_fl[ ctr : ctr + len_block ] = True
        if i in pi_l_mask:
            pi_l_mask_fl[ ctr : ctr + len_block ] = True
        ctr += len_block
    print( np.sum( pi_th_mask_fl ) )
    
    ###### Save densities and masks
    
    #np.save( "pi_dens/pi_th_"+str(img_size)+".npy", pi_th_fl )
    #np.save( "pi_dens/pi_l_"+str(img_size)+".npy", pi_l_fl )
    #np.save( "pi_masks/pi_th_"+str(img_size)+".npy", pi_th_mask_fl )
    #np.save( "pi_masks/pi_l_"+str(img_size)+".npy", pi_l_mask_fl )
    
    
    ##### Plots
    
    th_min = np.min( pi_th )
    th_max = np.max( pi_th )
    l_min = np.min( pi_l )
    l_max = np.max( pi_l )
    val_min = np.min( [ th_min, l_min ] )
    val_max = np.max( [ th_max, l_max ] )
    
    ###### Plot densities
    fig = plt.figure( )

    ax1 = fig.add_subplot(1, 2, 1 )
    tcf1 = ax1.tricontourf( kspace_loc[ :, 0 ], kspace_loc[ :, 1 ], np.squeeze( pi_th_fl ),
                    vmin = val_min, vmax = val_max )
    ax1.set_aspect( 'equal' )
    ax1.set_title( "pi_theta" )

    ax2 = fig.add_subplot(1, 2, 2 )
    tcf2 = ax2.tricontourf( kspace_loc[ :, 0 ], kspace_loc[ :, 1 ], np.squeeze( pi_l_fl ),
               vmin = val_min, vmax = val_max )
    ax2.set_aspect( 'equal' )
    ax2.set_title( "pi_lambda" )
    
    fig.subplots_adjust(right=0.8)
    cbar_ax2 = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(tcf2, cax=cbar_ax2)

    plt.show()
    
    ###### Plot masks
    fig = plt.figure( )

    ax1 = fig.add_subplot(1, 2, 1 )
    ax1.scatter( kspace_loc[ :, 0 ], kspace_loc[ :, 1 ], s = 1, c = 'r' )
    ax1.scatter( kspace_loc[ pi_th_mask_fl, 0 ], kspace_loc[ pi_th_mask_fl, 1 ], s = 1, c = 'k' )
    ax1.set_aspect( 'equal' )
    ax1.set_title( "mask for pi_theta" )

    ax2 = fig.add_subplot(1, 2, 2 )
    ax2.scatter( kspace_loc[ :, 0 ], kspace_loc[ :, 1 ], s = 1, c = 'r' )
    ax2.scatter( kspace_loc[ pi_l_mask_fl, 0 ], kspace_loc[ pi_l_mask_fl, 1 ], s = 1, c = 'k' )
    ax2.set_aspect( 'equal' )
    ax2.set_title( "mask for pi_lambda" )

    plt.show()

