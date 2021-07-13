#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:10:08 2020

@author: Anna
"""

import numpy as np
import os, copy
#import matplotlib.pyplot as plt

from sparkling import Run
from sparkling.parameters.initializations import INITIALIZATION_2D, use_cpu
from sparkling.utils.shots import convert_NCxNSxD_to_NCNSxD
from sparkling.utils.argparse import fix_n_check_params

from mri.operators.utils import normalize_frequency_locations


def get_sparkling( INITIALIZATION, verbose = 0 ):
    runObj = Run( **INITIALIZATION, verbose = verbose )
    runObj.initialize_shots()
    while runObj.current[ 'decim' ] >= 1:
        runObj.start_optimization()
        runObj.update_params_decim( do_decimation = True )
    shots = convert_NCxNSxD_to_NCNSxD(runObj.current[ 'shots' ] )
    return normalize_frequency_locations( shots )

im_size = 256
n = im_size ** 2

dens_type = [ "rad", "inf", "th_is", "th_anis", "l" ]

nc = 32 # number of shots
ns = 1024 # number of samples per shot; nc = 13, ns = 1024 correspond to s/s factor 0.2 for img_size = 256

######## Create directory for pictures

script_dir = os.path.dirname( __file__ )
pics_dir = os.path.join( script_dir, 'pictures/sparkling_trajectories/' )
if not os.path.isdir( pics_dir ):
    os.makedirs( pics_dir )
kpoints_dir = os.path.join( script_dir, 'kpoints/' )
if not os.path.isdir( kpoints_dir ):
    os.makedirs( kpoints_dir )



init = copy.deepcopy(INITIALIZATION_2D)
init = fix_n_check_params(init) #comment for the old version of sparkling
init = use_cpu(init) #comment for the old version of sparkling


init['dist_params']['mask'] = False
init['traj_params']['num_shots'] = nc
init['traj_params']['num_samples_per_shot'] = ns
init['traj_params']['initialization'] = 'RadialIO'
init['recon_params']['img_size'] = ( im_size, im_size ) # replace ( im_size, im_size ) with im_size for old version
init['algo_params']['max_grad_iter'] = 100
init['algo_params']['max_proj_iter'] = 100
init['algo_params']['start_decim' ] = 16

inObj = Run( **init, verbose = 0 )
inObj.initialize_shots()
in_kspace_loc = np.pi * convert_NCxNSxD_to_NCNSxD( inObj.traj_params.init_shots )

#fig = plt.figure()
#ax = fig.add_subplot( 1, 1, 1 )
#ax.scatter( in_kspace_loc[ :, 0 ], in_kspace_loc[ :, 1 ], s = 1 )
#plt.savefig( pics_dir+'initial_kspace_points.png', bbox_inches='tight')
#plt.show()

kspace_loc = {}

for pi_type in dens_type:
    pi_density = np.load( "pi_dens/pi_"+pi_type+"_"+str(im_size)+".npy" )
    init['dist_params']['density'] = pi_density
    init['dist_params']['cutoff'] = None
    init['dist_params']['decay'] = None
    kspace_loc[ pi_type ] = get_sparkling( init, verbose = 10 )
    np.save( "kpoints/sparkling_"+pi_type+"_"+str(im_size)+".npy", kspace_loc[ pi_type ] )
    
    
####### Plot the resulting trajectories
    
#fig = plt.figure( figsize = ( 30, 5 ) )

#for i, pi_type in enumerate( dens_type ):
#    ax = fig.add_subplot(1, len( dens_type ), i + 1 )
#    ax.scatter( kspace_loc[ pi_type ][ :, 1 ], kspace_loc[ pi_type ][ :, 0 ], s = 1 )
#    ax.set_aspect( 'equal' )
    
#plt.savefig( pics_dir+'trajectories.png', bbox_inches='tight')
    
    
    
    
    
    
    
    
    
