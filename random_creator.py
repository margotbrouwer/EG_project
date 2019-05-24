#!/usr/bin/python

# Import the necessary libraries
import sys

import numpy as np
import pyfits
import os
import random
from astropy.coordinates import SkyCoord

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
import scipy.optimize as optimization
from scipy import stats
import modules_EG as utils

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

from matplotlib import gridspec

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})


# Import lens catalogue

# Data selection
cat = 'kids' # Select the lens catalogue (kids/gama/mice)
Ngal = 1.83e3 # Number of galaxies per square degree

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)
lenscoords = SkyCoord(ra=lensRA*u.deg, dec=lensDEC*u.deg)

# Define KiDS field coordinates
fieldRAs = np.array([[128., 142], [149., 239.], [0., 54.2], [328.9, 360.]])
fieldDECs = np.array([[-2., 3.], [-4., 3.], [-35.7, -25.5], [-35.7, -26.9]])
fieldAreas = np.array([(fieldRAs[i,1]-fieldRAs[i,0])*(fieldDECs[i,1]-fieldDECs[i,0]) for i in range(len(fieldRAs)) ])

gridRA = np.array([])
gridDEC = np.array([])

## Creating a grid to measure the galaxy density
for f in range(len(fieldRAs)):
    Ngal_field = int(fieldAreas[f]*Ngal)
    gridRA = np.append(gridRA, np.random.uniform(fieldRAs[f,0], fieldRAs[f,1], Ngal_field))
    gridDEC = np.append(gridDEC, np.random.uniform(fieldDECs[f,0], fieldDECs[f,1], Ngal_field))
    
print('Number of gridpoints:', len(gridRA))

#plt.scatter(gridRA, gridDEC, marker='.')

# Masking areas where there are no lenses
gridcoords = SkyCoord(ra=gridRA*u.deg, dec=gridDEC*u.deg)

# Distances of grid points to the nearest galaxy
idx, d2d, d3d = gridcoords.match_to_catalog_sky(lenscoords)

# Find grid points that are outside the field
gridmask = (d2d < 0.05*u.deg) # Points that lie outside the source field

# Define new grid coordinates
gridRA, gridDEC = [gridRA[gridmask], gridDEC[gridmask]]
Ngrid = len(gridRA)
print('Number of randoms:', Ngrid)

# Create the random ID's and redshifts
gridID = np.arange(Ngrid)

Nmult = int(np.ceil(Ngrid/len(lensZ))) # Multiplier to have enough redshift values
print(Ngrid/len(lensZ))
print(Nmult)

gridZ = np.array(random.sample(list(lensZ)*Nmult, Ngrid))

## Write the results to a fits table
filename = '/data/users/brouwer/LensCatalogues/%s_random_catalogue'%(cat)

output = np.array([gridID, gridRA, gridDEC, gridZ])
outputnames = np.array(['ID', 'RA', 'DEC', 'Z'])
formats = np.array(['D']*len(outputnames))

utils.write_catalog('%s.fits'%filename, outputnames, formats, output)

#plt.scatter(gridRA, gridDEC, marker='.')
#plt.show()
#plt.clf()
