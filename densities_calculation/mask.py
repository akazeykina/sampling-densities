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





