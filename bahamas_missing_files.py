#!/usr/bin/python

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

path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
catnum = 1039

print('Missing clusters:')
for c in range(catnum):
    # Import the BAHAMAS profiles
    profname = '%s/PROFILES/cluster_%i_Menclosed_profile.dat'%(path_cat, c)
    
    try:
        profile = np.loadtxt(profname).T
    except:
        print(c)

