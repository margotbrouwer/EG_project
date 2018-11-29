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

# Importing the GAMA catalogues
path_gamacat = '/data/users/brouwer/LensCatalogues/GAMACatalogue_2.0.fits'
gamacat = pyfits.open(path_gamacat, ignore_missing_end=True)[1].data

print('Importing GAMA catalogue:', path_gamacat)

# Importing and correcting log(Mstar)
galIDlist = gamacat['ID']

logmstarlist = gamacat['logmstar']
ranklist = gamacat['RankBCG']

RAlist = gamacat['RA']*u.degree
DEClist = gamacat['DEC']*u.degree
zlist = gamacat['Z']

# Calculating galaxy distances
Dcllist = cosmo.comoving_distance(zlist)

# Creating gama coordinates
gamacoords = SkyCoord(ra=RAlist, dec=DEClist, distance=Dcllist)

cenmask = (ranklist <= 1)
BCGmask = (ranklist == 1)
isomask = (ranklist == -999)

cencoords = gamacoords[cenmask]
BCGcoords = gamacoords[BCGmask]
isocoords = gamacoords[isomask]

# Define Rmax, the maximum radius around each galaxy to which the signal is measured
Rmax2 = 2. * u.Mpc
Rmax3 = 3. * u.Mpc
Rmax4 = 4. * u.Mpc

# Calculate the distance to the nearest central
cendistlist = np.zeros(len(gamacoords)) * u.Mpc
idx, d2d, cendist = cencoords.match_to_catalog_3d(cencoords, nthneighbor=2)
cendistlist[cenmask] = cendist

# Calculate the distance to the nearest BCG
BCGdistlist = np.zeros(len(gamacoords)) * u.Mpc

# For BCGs
idx, d2d, BCGdist = BCGcoords.match_to_catalog_3d(BCGcoords, nthneighbor=2)
BCGdistlist[BCGmask] = BCGdist

# For non-grouped galaxies
idx, d2d, isodist = isocoords.match_to_catalog_3d(BCGcoords, nthneighbor=1)
BCGdistlist[isomask] = isodist


isoBCG2 = np.ones(len(galIDlist))
isocen2 = np.ones(len(galIDlist))
isoBCG3 = np.ones(len(galIDlist))
isocen3 = np.ones(len(galIDlist))
isoBCG4 = np.ones(len(galIDlist))
isocen4 = np.ones(len(galIDlist))

isoBCG2[BCGdistlist < Rmax2] = 0
isocen2[cendistlist < Rmax2] = 0
isoBCG3[BCGdistlist < Rmax3] = 0
isocen3[cendistlist < Rmax3] = 0
isoBCG4[BCGdistlist < Rmax4] = 0
isocen4[cendistlist < Rmax4] = 0


iso = np.array([isoBCG2, isocen2, isoBCG3, isocen3, isoBCG4, isocen4])

Niso = [ float(np.sum(iso[i])) / float(len(galIDlist)) * 100. for i in range(len(iso)) ]
print(Niso, 'percent')

# Plot the results to a fits table
filename = 'gama_isolated_galaxies_h%i_Brouwer17'%(h*100.)

outputnames = ['logmstar', 'RankBCG', 'BCGdist', 'cendist', 'isoBCG2', 'isocen2', 'isoBCG3', 'isocen3', 'isoBCG4', 'isocen4']
formats = ['D', 'J', 'D', 'D', 'J', 'J', 'J', 'J', 'J', 'J']
output = [logmstarlist, ranklist, BCGdistlist, cendistlist, isoBCG2, isocen2, isoBCG3, isocen3, isoBCG4, isocen4]

utils.write_catalog('%s.fits'%filename, outputnames, formats, output)

"""
# For each GAMA galaxy...
for l in range(len(galIDlist)):
#for l in range(2000):
    
    if l%1000 == 0:
        print(l)

    # Remove all neighbours that are far away from our galaxy
    #closemask = (RAlist[l]-Dmax[l] < RAlist) & (RAlist < RAlist[l]+Dmax[l]) & (DEClist[l]-Dmax[l] < DEClist) & (DEClist < DEClist[l]+Dmax[l])
    
    # Calculate the projected separation to our galaxy (in kpc)
    sep = (gamacoords[l].separation(gamacoords) * deg_kpc)
    BCGsep = (gamacoords[l].separation(gamacoords[BCGmask]) * deg_kpc[BCGmask])
    
    BCGdist[l] = np.amin(BCGsep).value
    
    
    # Check wether there are galaxies closer than Rmax(galaxy) + Rmax(neighbour)
    closegalmask = (sep < Rmax)
    closeBCGmask = (BCGsep < Rmax)
    
    closegals[l] = np.sum(closegalmask)
    closeBCGs[l] = np.sum(closeBCGmask)
"""
