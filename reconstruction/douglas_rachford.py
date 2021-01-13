#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:27:56 2021

@author: Anna
"""

import numpy as np
import matplotlib.pyplot as plt

def prox_l1( x, tau ):
    """
    Proximal operator of l1 norm: prox_{tau ||.||_1}( x )
    
    Parameters
    ----------
    x: ndarray
        argument of prox
    tau: float
        parameter
        
    Returns
    -------
    ndarray
        the value of prox_{tau ||.||_1}( x )
    """
    thr = np.where( np.abs( x ) > 1e-15, np.abs( x ), 1e-15 )
    term = 1 - tau / thr
    return np.where( term > 0, term, 0 ) * x

def proj_set( x, y, A, At ):
    """
    Projection on the set y = Ax
    
    Parameters
    ----------
    x: ndarray
        vector of size n
    y: ndarray
        vector of observations of size m
    A: function
        observation operator; argument is ndarray of size n, output is ndarray of size m
    At: function
        operator corresponding to the pseudoinverse of A: At = A^* ( A A^* )^{-1}; for a unitary A we have At = A^*.
        
    Returns
    -------
    ndarray
        the projected value
    """
    return x + At( y - A( x ) )

def dr1( y, A, At, niter, gamma, pr ):
    """
    Implementation of the Douglas-Rachford algorithm to solve min ||x||_1 under constraint y = Ax
    
    Parameters
    ----------
    y: ndarray
        vector of observations of size m
    A: function
        observation operator; argument is ndarray of size n, output is ndarray of size m
    At: function
        operator corresponding to the pseudoinverse of A: At = A^* ( A A^* )^{-1}; for a unitary A we have At = A^*.
    niter: int
        number of iterations of DR algorithm
    gamma: float
        parameter of DR algorithm
    plot_error: bool
        if True then plot error and L1 norm of x
        
    Returns
    -------
    ndarray
        vector x an approximate solution of the minimization problem
    
    """
    
    norm_y = np.linalg.norm( y )
    y = y / norm_y
    
    #Parameters of Douglas Rachford 
    lam = 1#1.5

    z = np.zeros( ( At( y ) ).shape ) # z is of size nx1
    L1 = np.zeros( niter )
    err = np.zeros( niter )

    for i in range( niter ):
        x = proj_set( z, y, A, At )
        z = z + lam * ( prox_l1( 2 * x - z, gamma ) - x )
        
        L1[ i ] = np.sum( np.abs( x ) )
        err[ i ] = np.linalg.norm( y - A( x ) )

    if pr:
        plt.figure()
        plt.plot( L1 )
        plt.show()
        plt.figure()
        plt.plot( err )
        plt.show()
    
    rec = x * norm_y
    
    return rec

if __name__ == "__main__":
    
    np.random.seed( 0 )
    n = 400
    p = round( n / 4 )
    A_matr = np.random.randn( p, n ) / np.sqrt( p )
    AH = np.conj( A_matr.T )
    A = lambda x: np.dot( A_matr, x ) 
    At = lambda y: np.dot( AH, np.dot( np.linalg.inv( np.dot( A_matr, AH ) ), y ) )
    
    s = 17
    sel = np.random.permutation( n )
    x0 = np.zeros( ( n, 1 ) )
    x0[ sel[ 0 : s ] ] = 1
    
    y = np.dot( A_matr, x0 )
    
    niter = 500
    
    x_final = dr1( y, A, At, niter, 1e-2, True )
    
    fig = plt.figure()
    plt.plot( x0, 'b' )
    plt.plot( x_final, 'r' )
    plt.show()

