#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:46:49 2020

@author: Anna
"""

import numpy as np

def compute_mask( pi, nb_samples ):
    """
    Compute mask from a given probability density without replacement
    
    Parameters
    ----------
    pi: ndarray
        probability density
    nb_samples: integer
        the number of samples to be drawn
    """
    
    ind = np.arange( pi.size )
    sampled_points = np.random.choice( ind, size = nb_samples, replace = False, p = np.squeeze( pi ) )
    
    return sampled_points





