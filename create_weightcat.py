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

# Import lens catalog
cat = 'mice'

# Constants
h = 0.7
if 'mice' in cat:
    O_matter = 0.25
    O_lambda = 0.75
else:
    O_matter = 0.2793
    O_lambda = 0.7207

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Calculate the total baryonic mass (stars + cold gas)
def calc_logmbar(logmstar):
    fcold = 10.**(-0.69*logmstar + 6.63)
    mstar = 10.** logmstar
    mbar = mstar * (1 + fcold)
    logmbar = np.log10(mbar)
    return logmbar

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

# Mean difference with the GAMA masses (log(M_ANN)-log(M_G))
if ('kids' in cat) or ('matched' in cat):
    #diff_GL = -0.10978165582547783
    diff_GL = -0.056
else:
    diff_GL = 0.

bias = 0.2
logmstar_GL = logmstar - diff_GL
logmstar_min = logmstar_GL-bias
logmstar_max = logmstar_GL+bias

logmbar, logmbar_GL, logmbar_min, logmbar_max = \
        [calc_logmbar(b) for b in [logmstar, logmstar_GL, logmstar_min, logmstar_max]]

output = [lensID, logmstar, logmbar, logmstar_GL, logmbar_GL, \
        logmstar_min, logmbar_min, logmstar_max, logmbar_max]
outputnames = ['ID', 'logmstar', 'logmbar', 'logmstar_GL', 'logmbar_GL', \
        'logmstar_min', 'logmbar_min', 'logmstar_max', 'logmbar_max']

filename = '/data/users/brouwer/LensCatalogues/baryonic_mass_catalog_%s.fits'%cat

formats = ['D']*len(outputnames)

utils.write_catalog(filename, outputnames, formats, output)
