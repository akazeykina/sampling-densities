#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:55:46 2020

@author: Anna
"""
import numpy as np

def generate_full_scheme( scheme_type, block_type, img_size, num_revolutions = 1 ):
    """
    Generate points of the full scheme in k-space
    
    Parameters
    ----------
    scheme_type: string
        "cartesian", "radial" or "spiral"
    block_type: string
        The type of blocks of points: "isolated" for all schemes; "hor_lines" or "ver_lines" for cartesian scheme,
        "circles" or "radia" for radial scheme, "circles" or "spokes" for spiral.
    img_size: integer
        Image size, a power of 2
    num_revolutions: integer
        The number of revolutions for the spiral scheme
        
    Returns
    -------
    np.ndarray
        The kspace locations of the full scheme, matrix of size (N,2) with columns k_x, k_y
    
    """
    if scheme_type == "cartesian":
        x = np.linspace( -0.5, 0.5, img_size, endpoint = False )
        y = np.linspace( -0.5, 0.5, img_size, endpoint = False )
        if ( block_type == "hor_lines" ) or ( block_type == "isolated" ):
            X, Y = np.meshgrid( x, y )
        elif block_type == "vert_lines":
            Y, X = np.meshgrid( y, x )
        X = X.flatten('C')
        Y = Y.flatten('C')
            
        full_kspace = np.stack( ( X, Y ), axis = 1 )  
            
    elif scheme_type == 'radial':
        n_rad = int( img_size / 2 )
        n_ang = int( img_size * np.pi )
        phi = np.linspace( 0, 2 * np.pi, n_ang, endpoint = False )
        rad = np.linspace( 0.5 / n_rad, 0.5, (n_rad-1), endpoint = False )
        if ( block_type == "circles" ) or ( block_type == "isolated" ):
            Phi, Rad = np.meshgrid( phi, rad )
        elif block_type == "radia":
            Rad, Phi = np.meshgrid( rad, phi )
            
        X = ( Rad * np.cos( Phi ) ).flatten( 'C' )
        Y = ( Rad * np.sin( Phi ) ).flatten( 'C' )
            
        full_kspace = np.stack( ( X, Y ), axis = 1 )  
        full_kspace = np.vstack( ( np.array( [ 0, 0 ] ), full_kspace ) )
    
    elif scheme_type == 'spiral':
        n_c = int( img_size / 2 )
        n_s = int( img_size * np.pi )
        b = 0.5 / ( np.pi * num_revolutions ) 
        phi = np.linspace( 0, 2 * np.pi, n_s, endpoint = False )
        theta = np.linspace( np.pi / n_c, np.pi * num_revolutions, ( n_c -  1 ), endpoint = False )
        if ( block_type == "circles" ) or ( block_type == "isolated" ):
            Phi, Theta = np.meshgrid( phi, theta )
        elif ( block_type == "spokes" ):
            Theta, Phi = np.meshgrid( theta, phi )
        R = b * Theta
        Ang = Theta + Phi
        X = ( R * np.cos( Ang ) ).flatten( 'C' )
        Y = ( R * np.sin( Ang ) ).flatten( 'C' )
            
        full_kspace = np.stack( ( X, Y ), axis = 1 )  
        full_kspace = np.vstack( ( np.array( [ 0, 0 ] ), full_kspace ) )
        
    return full_kspace
    
def generate_blocks_list():
    if scheme_type == 'cartesian':
        blocks_list = [ [ img_size * i + j for j in range( img_size ) ] for i in range( img_size )  ]
        #blocks_list = [ [ i ] for i in range( img_size * img_size )  ]
        
        #Split into random and deterministic blocks
        det_blocks_list, rand_blocks_list = split_det_rand( img_size, blocks_list, spl_type, det_portion )
        
        nb_samples = int( sub_sampling_rate * len(blocks_list) ) - len( det_blocks_list )
        
    elif scheme_type == 'radial':
        #blocks_list = [ [ (n_ang) * i + j + 1 for j in range( n_ang ) ] \
                             #for i in range( n_rad-1 )  ] # for Phi, Rad
        blocks_list = [ [ (n_rad-1) * i + j + 1 for j in range( n_rad-1 ) ] \
                             for i in range( n_ang )  ] # for Rad, Phi
        #blocks_list = [ [ i + 1 ] \
                             #for i in range( n_ang * ( n_rad - 1 ) )  ] # for Rad, Phi
        blocks_list.insert( 0, [ 0 ] )
        
        #Split into random and deterministic blocks
        det_blocks_list, rand_blocks_list = split_det_rand( img_size, blocks_list, spl_type,
                                                           det_portion )
        
        nb_samples = int( sub_sampling_rate * len(rand_blocks_list) * 2 / np.pi )
    elif scheme_type == 'spiral':
        #blocks_list = [ [ (n_s) * i + j + 1 for j in range( n_s ) ] \
                             #for i in range( n_c-1 )  ] # for Phi, Theta
        #blocks_list = [ [ (n_c-1) * i + j + 1 for j in range( n_c-1 ) ] \
                             #for i in range( n_s )  ] # for Theta, Phi
        blocks_list = [ [ i + 1 ] \
                             for i in range( n_s * ( n_c - 1 ) )  ] # for isolated
        blocks_list.insert( 0, [ 0 ] )
        
        #Split into random and deterministic blocks
        det_blocks_list, rand_blocks_list = split_det_rand( img_size, blocks_list, spl_type, det_portion )
        
        nb_samples = int( sub_sampling_rate * len(rand_blocks_list) * 2 / np.pi )
    return None

if __name__ == "__main__":
    full_kspace = generate_full_scheme( 'radial', 'isolated', 64 )
    print( full_kspace[ 10, : ] )