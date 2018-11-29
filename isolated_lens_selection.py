#!/usr/bin/python

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import LambdaCDM
from astropy.io import fits

import modules_EG as utils

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

## Configuration

# Data selection
cat = 'mice' # Select the lens catalogue (kids/gama/mice)

# Import lens catalog
fields, path_lenscat, lenscatname, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

# This list will contain all satellite distances
logmstarlist = logmstar
dist0p5dex = np.zeros(len(lensRA)) * u.Mpc

# Remove all galaxies with logmstar=NAN
nanmask = np.isfinite(logmstar)
lensRA, lensDEC, lensDc, logmstar = [lensRA[nanmask], lensDEC[nanmask], lensDc[nanmask], logmstar[nanmask]]

# Creating gama coordinates
lenscoords = SkyCoord(ra=lensRA*u.deg, dec=lensDEC*u.deg, distance=lensDc)

# Define the galaxy mass bins
logm_min = 7.
logm_max = 13.

Nlogmbins = int(1e3)
logmlims = np.linspace(logm_min, logm_max, Nlogmbins+1)
dlogm = np.diff(logmlims)[0]
logmbins = logmlims[0:-1] + dlogm

cendistlist = np.zeros(len(lenscoords)) * u.Mpc

# The nearby galaxies should not be heaver than X times the galaxy
massratio = 1./10**0.5 # ... where X=0.5 dex

for m in np.arange(Nlogmbins):

    # Masking the centrals and satellites
    massmask_cen = (logmlims[m] < logmstar) & (logmstar <= logmlims[m+1])
    massmax_sat = np.log10(np.mean(10**logmstar[massmask_cen]) * massratio) # Satellites with X times the mean central mass
    massmask_sat = (logmstar > massmax_sat) # ... are "too heavy" for these centrals
    cencoords, satcoords = [lenscoords[massmask_cen], lenscoords[massmask_sat]]
    
    print('%g < logmstar < %g: %i galaxies'%(logmlims[m], logmlims[m+1], len(cencoords)))
    print('Max(logm) satellite: %g'%massmax_sat)


    if (len(cencoords)>0) & (len(satcoords)>0):
        
        # Calculate the distance to the nearest satellite that is too heavy
        idx, sep2d, cendist3d = cencoords.match_to_catalog_3d(satcoords, nthneighbor=2)
        cendistlist[massmask_cen] = cendist3d
        #print(cendist3d)
        
print('logmbins:', logmlims)
print('dlogm:', dlogm)

dist0p5dex[nanmask] = cendistlist

# Plot the results to a fits table
filename = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_h%i'%(cat, h*100.)

outputnames = ['logmstar', 'dist0p5dex']
formats = ['D', 'D']
output = [logmstarlist, dist0p5dex]

utils.write_catalog('%s.fits'%filename, outputnames, formats, output)

