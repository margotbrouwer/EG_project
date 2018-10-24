#!/usr/bin/python

"Module to compute the optimal lens binning."

# Import the necessary libraries
import astropy.io.fits as pyfits
import gc
import numpy as np
import sys
import os
import time
from glob import glob

from astropy import constants as const, units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM
from collections import Counter

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

import modules_EG as utils
import treecorr

# Import constants
G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')
inf = np.inf
   
h, O_matter, O_lambda = [0.7, 0.325, 0.685]
cosmo = LambdaCDM(H0=h*100, Om0=O_matter, Ode0=O_lambda)

## Configuration

# Data selection
cat = 'gama' # Select the lens catalogue (kids/gama/mice)

# Import lens catalog
fields, path_lenscat, lenscatname, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

# Binning parameter
binname = 'logmstar'
binvals = logmstar
Nbins = 4

# Lens selection
paramnames = np.array(['logmstar'])
maskvals = np.array([ [8.5,12.] ])

# Path to shear catalog
path_sheardata = '/data2/brouwer/shearprofile/Lensing_results/EG_results_Oct18'
path_catalog = 'catalogs/results_shearcatalog'
path_cosmo = 'shearcatalog_ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins20_30_3000_kpc'
path_filename = 'shearcatalog'

shearcatname = '%s/%s/%s/%s.fits'%(path_sheardata, path_catalog, path_cosmo, path_filename)



## Pipeline


shearcat = pyfits.open(shearcatname, memmap=True)[1].data

gammat = np.sum(shearcat['gammat_A'], axis=1)
wk2 = np.sum(shearcat['lfweight_A*k^2'], axis=1)
w2k2 = np.sum(shearcat['lfweight_A^2*k^2'], axis=1)
srcm = np.sum(shearcat['bias_m_A'], axis=1)
variance = (shearcat['variance(e[A,B,C,D])'])[0,0]

# Creating the lens mask
lensmask, filename_var = utils.define_lensmask(paramnames, maskvals, path_lenscat, lenscatname, h)

# Mask binning values and shear catalog
binvals, gammat, wk2, w2k2, srcm = binvals[lensmask], gammat[lensmask], wk2[lensmask], w2k2[lensmask], srcm[lensmask]

# Sort shear catalog by binning parameter
binsort = np.argsort(binvals)
gammat_sortsum, wk2_sortsum, w2k2_sortsum, srcm_sortsum = \
np.cumsum(gammat[binsort]), np.cumsum(wk2[binsort]), np.cumsum(w2k2[binsort]), np.cumsum(srcm[binsort])

# Calculate the cumulative S/N ratio
ESD_tot = gammat_sortsum / wk2_sortsum # Final Excess Surface Density (tangential comp.)
error_tot = (w2k2_sortsum / wk2_sortsum**2 * variance)**0.5 # Final error
bias_tot = (1 + (srcm_sortsum / wk2_sortsum)) # Final multiplicative bias (by which the signal is to be divided)

SN_ratio = np.abs(ESD_tot)/(error_tot*bias_tot)
binvals_sorted = binvals[binsort]

SN_min, SN_max = [np.amin(SN_ratio), np.amax(SN_ratio)]
SN_bins = np.linspace(0., SN_max, Nbins+1)
idx = [(np.abs(SN_ratio - value)).argmin() for value in SN_bins]
binning_values = binvals_sorted[idx]

binvals_print = ','.join(['%g'%np.round(i,1) for i in binning_values])

print('Bin name: %s'%binname)
print('Binning values: %s'%binvals_print)



plt.plot(binvals_sorted, ESD_tot/(error_tot*bias_tot), color='black')
plt.xlabel('Binning parameter (%s)'%binname)
plt.ylabel('Cumulative S/N of the ESD profile')
[plt.axvline(x=binning_values[i]) for i in xrange(Nbins+1)]
[plt.axhline(y=SN_bins[i]) for i in xrange(Nbins+1)]


#plt.xscale('log')
#plt.yscale('log')
plt.show()






