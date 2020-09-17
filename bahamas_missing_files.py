#!/usr/bin/python

"""Find which Bahamas clusters are missing from the folder"""

# Import the necessary libraries
import sys
import numpy as np
import pyfits
import os

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
import scipy.optimize as optimization
import scipy.stats as stats
import modules_EG as utils

#path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_all/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
#path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_isolated_strong/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_isolated_new/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
catnum = 515

print('Missing clusters:')
for c in range(catnum):
    # Import the BAHAMAS profiles
    profname = '%s/PROFILES/cluster_%i_Menclosed_profile_types.dat'%(path_cat, c)
    
    try:
        profile = np.loadtxt(profname).T
    except:
        print(c)
        print(profname)
        print()
