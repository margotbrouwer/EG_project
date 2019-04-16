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
from astropy.cosmology import LambdaCDM

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Import lens catalog
cat = 'gama'

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)
print(logmstar)

# Calculate the total baryonic mass (stars + cold gas)
fcold = 10.**(-0.69*logmstar + 6.63)
mstar = 10.** logmstar
mbar = mstar * (1 + fcold)
logmbar = np.log10(mbar) 

filename = '/data/users/brouwer/LensCatalogues/baryonic_mass_catalog_%s.fits'%cat

outputnames = ['logmstar', 'logmbar']
formats = ['D', 'D']
output = [logmstar, logmbar]

utils.write_catalog(filename, outputnames, formats, output)
