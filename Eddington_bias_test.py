#!/usr/bin/python
"""
# This script is used to estimate the size of the Eddington bias on the stellar masses
"""

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

# Import lens catalog
cat = 'kids'
splittype = 'sersic'
Noffset = 1 # Minimum 1

# Constants
h = 0.7
if 'mice' in cat:
    O_matter = 0.25
    O_lambda = 0.75
else:
    O_matter = 0.2793
    O_lambda = 0.7207

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)


## Import "true" and "offset" equal mass selection catalogues

# Import "true" equal mass selection catalogue
path_lenscat = '/data/users/brouwer/LensCatalogues'
lenscatname = 'mass_selection_catalog_%s-offsetx%s_%s.fits'%(splittype, Noffset-1, cat)
lenscatfile = '%s/%s'%(path_lenscat, lenscatname)
lenscat = pyfits.open(lenscatfile, memmap=True)[1].data

logmstar = lenscat['logmstar']

# Define spiral and elliptical galaxies by sersic index or color
if 'sersic' in splittype:
    typename = 'n_2dphot'
    typelist = lenscat[typename]
    splitlim = 2.
    
if 'color' in splittype:
    typename = 'MAG_GAAP_u-r'
    typelist = lenscat[typename]
    splitlim = 2.5

mask_ell = (typelist > splitlim)
mask_spir = (typelist < splitlim)

print('Imported lens catalogue:', lenscatname)

# Import isolation criterion
isocatname = '%s_isolated_galaxies_perc_h70.fits'%cat
isocatfile = '%s/%s'%(path_lenscat, isocatname)
isocat = pyfits.open(isocatfile, memmap=True)[1].data

R_iso = isocat['dist0p1perc'] # Distance to closest satellite
mask_iso = R_iso > 3. # Should be larger than 3 Mpc

# Import "offset" equal mass selection catalogue
selcatname = 'mass_selection_catalog_%s-offsetx%s_%s.fits'%(splittype, Noffset, cat)
selcatfile = '%s/%s'%(path_lenscat, selcatname)
selcat = pyfits.open(selcatfile, memmap=True)[1].data
mask_sel = (selcat['selected'] == 1.)


## Creating the stellar mass histograms

# Creating the stellar mass bins
Nbins = 100
binedges = np.linspace(8., 11., Nbins+1)
bincenters = binedges[0:-1] + np.diff(binedges)/2.

# Creating stellar mass histograms for *true* ellipticals and spirals
N_ell, foo, foo = plt.hist(logmstar[mask_ell*mask_iso], bins=binedges, histtype='step', \
    label='Isolated Ellipticals', color='red')
N_spir, foo, foo = plt.hist(logmstar[mask_spir*mask_iso], bins=binedges, histtype='step', \
    label='Isolated Spirals', color='blue')

# Creating stellar mass histograms for *selected* ellipticals and spirals (incl. Eddington bias)
N_ell_sel, foo, foo = plt.hist(logmstar[mask_ell*mask_sel], bins=binedges, histtype='step', \
    label='Selected Ellipticals (incl. Eddington bias)', color='purple')
N_spir_sel, foo, foo = plt.hist(logmstar[mask_spir*mask_sel], bins=binedges, histtype='step', \
    label='Selected Spirals (incl. Eddington bias)', color='green')

xlabel = r'Stellar mass log($M_*$)'
plt.xlabel(xlabel, fontsize=14)

plt.legend(loc='upper left')

# Save plot
plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Jul20/Plots/Eddington_bias_hist'
for ext in ['pdf']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext)
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf

# Determine the mean difference between the Elliptical and Spiral masses
mstar_mean_ell = np.mean(10.**logmstar[mask_ell*mask_sel])
mstar_mean_spir = np.mean(10.**logmstar[mask_spir*mask_sel])

f_edd = mstar_mean_ell/mstar_mean_spir

print('Eddington bias (offset x %i):'%(Noffset))
print('Fraction:', f_edd)
print('Dex:', np.log10(f_edd))
