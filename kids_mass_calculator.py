#!/usr/bin/python

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import LambdaCDM
import astropy.io.fits as pyfits

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

import modules_EG as utils

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

## Configuration

# Data selection
cat = 'kids' # Select the lens catalogue (kids/gama/mice)

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, imag, logML =\
utils.import_lenscat(cat, h, cosmo)
lensDc = lensDc.to('pc').value

lensDl = lensDc * (1+lensZ)
rmag_abs = rmag - 5 * (np.log10(lensDl) - 1.)
lensLum = 10.**(-0.4 * rmag_abs)

logmstar = np.log10(lensLum * 10.**logML)

print('M/L:', 10.**logML)
print('logmstar:', logmstar)

plt.hist(logmstar)
plt.show()
