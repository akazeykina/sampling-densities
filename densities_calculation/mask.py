#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:46:49 2020

@author: Anna
"""

import numpy as np

def compute_indices( pi, nb_samples ):
    """
    Compute mask from a given probability density without replacement
    
    Parameters
    ----------
    pi: ndarray
        probability density
    nb_samples: integer
        the number of samples to be drawn
        
    Returns
    -------
    ndarray
        Indices of the sampled points
    """
    
    ind = np.arange( pi.size )
    sampled_points = np.random.choice( ind, size = nb_samples, replace = False, p = np.squeeze( pi ) )
    
    return sampled_points

def compute_mask( pi, nb_samples ):
    """
    Compute mask (a matrix of 0 and 1) from the indices of sampled points 
    
    Parameters
    ----------
    pi: ndarray
        probability density
    nb_samples: integer
        the number of samples to be drawn
    
    Returns 
    -------
    ndarray
        Matrix of shape (img_size, img_size) of 0 and 1 with 1 corresponding to sampled points
    """
    mask = np.zeros( pi.shape )
    
    indices = compute_indices( pi, nb_samples )
    mask[ indices, : ] = 1
    
    return mask


def fill_det_blocks( mask, dens_type, det_blocks_list ):
    """
    Fill the elements of mask corresponding to deterministically sampled blocks with True
    
    Parameters
    ----------
    mask: dict
        Dictionary of masks (of size full_kspace.shape[0])
    dens_type: list
        Density type (e.g. "rad", "inf", "th_is", "th_anis", "l"); keys of mask dictionary
    det_blocks_list: list
        List of sublists of row indices corresponding to blocks of points of kspace that are deterministically sampled
    
    Returns
    -------
    pi_fl: dict
        Unraveled pi
    """

    for block in det_blocks_list: 
        for pi_type in dens_type:
            mask[ pi_type ][ block ] = 1
            
    return mask


