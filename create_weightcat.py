#!/usr/bin/python
"""
# This script is used to create lists for weighting individual lenses.
"""
import astropy.io.fits as pyfits
import gc
import numpy as np
import sys
import os
import time
from glob import glob

import modules_EG as utils
from astropy import constants as const, units as u

# Import lens catalog
cat = 'gama'
h=0.7
fields, path_lenscat, lenscatname, lensRA, lensDEC, lensZ, lensDc,\
rmag, rmag_abs, logmstar = utils.import_lenscat(cat, h)

# Calculate the total baryonic mass (stars + cold gas)
fcold = 10.**(-0.69*logmstar + 6.63)
mstar = 10.** logmstar
mbar = mstar * (1 + fcold)
logmbar = np.log10(mbar) 

filename = '/data2/brouwer/shearprofile/EG_project/baryonic_mass_catalog.fits'
galIDlist = np.arange(len(logmstar))+1
outputnames = ['logmstar', 'logmbar']
output = [logmstar, logmbar]

utils.write_catalog(filename, galIDlist, outputnames, output)
