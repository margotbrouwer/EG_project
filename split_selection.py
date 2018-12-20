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

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

import modules_EG as utils

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
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

# Binning parameter
binname = 'logmstar'
binvals = logmstar
Nbins = 4

# Lens selection
isocatname = 'gama_isolated_galaxies_perc_h70.fits'

paramnames = np.array(['logmstar', 'nQ', 'dist0p2perc'])
maskvals = np.array([ [8.5,12.], [3, inf], [4, inf] ])
lenscatnames = np.array([lenscatname, lenscatname, isocatname])

# Path to shear catalog
path_sheardata = '/data/users/brouwer/Lensing_results/EG_results_Nov18'
path_catalog = 'catalogs/results_shearcatalog'
path_cosmo = 'shearcatalog_ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_30_3000_kpc'
path_filename = 'shearcatalog'

shearcatname = '%s/%s/%s/%s.fits'%(path_sheardata, path_catalog, path_cosmo, path_filename)



## Pipeline


shearcat = pyfits.open(shearcatname, memmap=True)[1].data

gammat = (shearcat['gammat_A'])
wk2 = shearcat['lfweight_A*k^2']
w2k2 = shearcat['lfweight_A^2*k^2']
srcm = shearcat['bias_m_A']
variance = (shearcat['variance(e[A,B,C,D])'])[0,0]

# Creating the lens mask
lensmask, filename_var = utils.define_lensmask(paramnames, maskvals, path_lenscat, lenscatnames, h)

# Masking the binning values and shear catalog
binvals, gammat, wk2, w2k2, srcm = binvals[lensmask], gammat[lensmask], wk2[lensmask], w2k2[lensmask], srcm[lensmask]

# Sorting shear catalog by binning parameter
binsort = np.argsort(binvals)#[::-1]
gammat_sortsum, wk2_sortsum, w2k2_sortsum, srcm_sortsum = \
np.cumsum(gammat[binsort], 0), np.cumsum(wk2[binsort], 0), np.cumsum(w2k2[binsort], 0), np.cumsum(srcm[binsort], 0)

#"""
# Calculate the cumulative S/N ratio
ESD_tot = gammat_sortsum / wk2_sortsum # Final Excess Surface Density (tangential comp.)
error_tot = (w2k2_sortsum / wk2_sortsum**2 * variance)**0.5 # Final error
bias_tot = (1 + (srcm_sortsum / wk2_sortsum)) # Final multiplicative bias (by which the signal is to be divided)

SN_ratio = np.mean((ESD_tot/(error_tot*bias_tot)), 1)**2 # Compute the average cumulative S/N per lens
"""
SN_ratio = np.mean(gammat_sortsum, 1)
"""

binvals_sorted = binvals[binsort]
SN_ratio, binvals_sorted = [SN_ratio[np.isfinite(SN_ratio)], binvals_sorted[np.isfinite(SN_ratio)]] # Removing NaN from list


SN_min, SN_max = [np.amin(SN_ratio), np.amax(SN_ratio)]
SN_bins = np.linspace(SN_min, SN_max, Nbins+1)
idx = [(np.abs(SN_ratio - value)).argmin() for value in SN_bins]
binning_values = binvals_sorted[idx]

#print('SN bins:', SN_bins)

binvals_print = ','.join(['%g'%np.round(i,1) for i in binning_values])

print('Bin name: %s'%binname)
print('Binning values: %s'%binvals_print)
print('S/N per bin: %g'%np.sqrt(SN_bins[1]))


plt.plot(binvals_sorted, SN_ratio, color='black')
plt.xlabel('Binning parameter (%s)'%binname)
plt.ylabel('Cumulative $|S/N|^2$ of the ESD profile')
[plt.axvline(x=binning_values[i]) for i in range(Nbins+1)]
[plt.axhline(y=SN_bins[i]) for i in range(Nbins+1)]


#plt.xscale('log')
#plt.yscale('log')
plt.show()
