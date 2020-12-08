#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:10:08 2020

@author: Anna
"""

import numpy as np
import os, copy
import matplotlib.pyplot as plt

from sparkling import Run
from sparkling.parameters.initializations import INITIALIZATION_2D
from sparkling.utils.shots import convert_NCxNSxD_to_NCNSxD

#from modopt.opt.proximity import SparseThreshold
#from modopt.opt.linear import Identity
#from modopt.math.metrics import ssim

#from mri.operators import NonCartesianFFT, WaveletN
from mri.operators.utils import normalize_frequency_locations
#from mri.reconstructors import SingleChannelReconstructor


def get_sparkling( INITIALIZATION, verbose = 0 ):
    runObj = Run( **INITIALIZATION, verbose = verbose )
    runObj.initialize_shots()
    while runObj.current[ 'decim' ] >= 1:
        runObj.start_optimization()
        runObj.update_params_decim( do_decimation = True )
    shots = convert_NCxNSxD_to_NCNSxD(runObj.current[ 'shots' ] )
    return normalize_frequency_locations( shots )

img_size = 256
n = img_size ** 2

nc = 26 # number of shots
ns = 512 # number of samples per shot; nc = 13, ns = 1024 correspond to s/s factor 0.2 for img_size = 256

######## Create directory for pictures

script_dir = os.path.dirname( __file__ )
results_dir = os.path.join( script_dir, 'pictures/sparkling_trajectories/' )
if not os.path.isdir( results_dir ):
    os.makedirs( results_dir )

init = copy.deepcopy(INITIALIZATION_2D)
init['dist_params']['mask'] = False
init['traj_params']['num_shots'] = nc
init['traj_params']['num_samples_per_shot'] = ns
init['traj_params']['initialization'] = 'RadialIO'
init['recon_params']['img_size'] = img_size
init['algo_params']['max_grad_iter'] = 100
init['algo_params']['max_proj_iter'] = 100
init['algo_params']['start_decim' ] = 16

inObj = Run( **init, verbose = 0 )
inObj.initialize_shots()
in_kspace_loc = np.pi * convert_NCxNSxD_to_NCNSxD( inObj.traj_params.init_shots )

fig = plt.figure()
ax = fig.add_subplot( 1, 1, 1 )
ax.scatter( in_kspace_loc[ :, 0 ], in_kspace_loc[ :, 1 ], s = 1 )
plt.savefig( results_dir+'initial_kspace_points.png', bbox_inches='tight')
#plt.show()
