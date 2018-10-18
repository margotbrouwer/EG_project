#!/usr/bin/python

import numpy as np
import pyfits
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import LambdaCDM

import modules_EG as utils

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

## Configuration

# Data selection
cat = 'gama' # Select the lens catalogue (kids/gama/mice)

# Import lens catalog
fields, path_lenscat, lenscatname, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

# Creating gama coordinates
lenscoords = SkyCoord(ra=lensRA, dec=lensDEC, distance=lensDc)

# Define Rmax, the maximum radius around each galaxy to which the signal is measured
Rmax2 = 2. * u.Mpc
Rmax3 = 3. * u.Mpc
Rmax4 = 4. * u.Mpc

# Define the lens redshift bins
massmin = 8.5
massmax = np.amax(logmstar)
Nmassbins = 20
masslims = np.linspace(massmin, massmax, Nmassbins+1)
print('logmbins:', masslims, 'dm:', (np.amax(masslims)-np.amin(masslims))/Nmassbins)

distlist = np.zeros(len(gamacoords)) * u.Mpc

for m in np.arange(Nlogmbins)
    massmask_high = (masslims[m] < logmstar) & (logmstar <= masslims[m+1])
    massmask_log = (masslims[m] < logmstar) & (logmstar <= masslims[m+1])
    

    # Calculate the distance to the nearest central
    
    
    idx, sep2d, cendist3d = cencoords.match_to_catalog_3d(lenscoords[massmask], nthneighbor=2)

    cendistlist[massmask] = cendist3d



"""

iso2 = np.ones(len(galIDlist))
iso3 = np.ones(len(galIDlist))
iso4 = np.ones(len(galIDlist))

iso2[distlist < Rmax2] = 0
iso3[distlist < Rmax3] = 0
iso4[distlist < Rmax4] = 0


iso = np.array([iso2, iso3, iso4])

Niso = [ float(np.sum(iso[i])) / float(len(galIDlist)) * 100. for i in xrange(len(iso)) ]
print Niso, 'percent'

# Plot the results to a fits table
filename = 'gama_isolated_galaxies_h%i'%(h*100.)

outputnames = ['logmstar', 'RankBCG', 'BCGdist', 'cendist', 'isoBCG2', 'isocen2', 'isoBCG3', 'isocen3', 'isoBCG4', 'isocen4']
output = [logmstarlist, ranklist, BCGdistlist, cendistlist, isoBCG2, isocen2, isoBCG3, isocen3, isoBCG4, isocen4]

utils.write_catalog('%s.fits'%filename, galIDlist, outputnames, output)
