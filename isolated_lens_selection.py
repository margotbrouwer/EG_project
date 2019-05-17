#!/usr/bin/python

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import LambdaCDM
import astropy.io.fits as pyfits

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
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

"""
if cat == 'kids':
    lensDl = lensDc.to('pc').value * (1+lensZ)
    rmag_abs = rmag - 5. * (np.log10(lensDl) - 1.)
    lensLum = 10.**(-0.4 * rmag_abs)
    logmstar = np.log10(lensLum * 10.**logmstar)
"""
logmstarlist = logmstar

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


# The nearby galaxies should not be heavier than X times the galaxy
"""
rationame = 'dex'
dexvalues = [0.5, 0.6, 0.7, 0.8]
rationames = [('%s'%d).replace('.', 'p') for d in dexvalues]
massratios = [1./10**d for d in dexvalues] # ... where X=0.5 dex
"""
rationame = 'perc'
massratios = [0.3, 0.25, 0.2, 0.15, 0.1]
#massratios = [0.2, 0.1]
rationames = [('%s'%d).replace('.', 'p') for d in massratios]

# This list will contain all satellite distances
distmdex = np.zeros([len(massratios), len(logmstarlist)]) * u.Mpc

for d in range(len(massratios)):
    print('Satellites lighter then: %g %s * Mcen'%(massratios[d], rationame))

    cendistlist = np.zeros(len(lenscoords)) * u.Mpc
    for m in np.arange(Nlogmbins):
    
        #print(d, m)
    
        # Masking the centrals and satellites
        massmask_cen = (logmlims[m] < logmstar) & (logmstar <= logmlims[m+1])
        massmax_sat = np.log10(np.mean(10**logmstar[massmask_cen]) * massratios[d]) # Satellites with X times the mean central mass
        massmask_sat = (logmstar > massmax_sat) # ... are "too heavy" for these centrals
        cencoords, satcoords = [lenscoords[massmask_cen], lenscoords[massmask_sat]]
        
        #print('%g < logmstar < %g: %i galaxies'%(logmlims[m], logmlims[m+1], len(cencoords)))
        #print('Max(logm) satellite: %g'%massmax_sat)
        #print('%g percent of satellites selected'%(np.sum(massmask_sat)/float(len(massmask_sat))*100.))
        #print(len(cencoords), len(satcoords))
        
        if (len(cencoords)>0) & (len(satcoords)>0):
            
            # Calculate the distance to the nearest satellite that is too heavy
            idx, sep2d, cendist3d = cencoords.match_to_catalog_3d(satcoords, nthneighbor=2)
            cendistlist[massmask_cen] = cendist3d
            #print(cendist3d)
    
    (distmdex[d])[nanmask] = cendistlist

print('logmbins:', logmlims)
print('dlogm:', dlogm)

# Write the results to a fits table
filename = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_perc_h%i'%(cat, h*100.)

outputnames = np.append(['logmstar'], ['dist%s%s'%(n,rationame) for n in rationames])
formats = np.append(['D'], ['D']*len(massratios))
output = np.append([logmstarlist], distmdex, axis=0)
print(outputnames, formats, output)

utils.write_catalog('%s.fits'%filename, outputnames, formats, output)

