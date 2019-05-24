#!/usr/bin/python

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import LambdaCDM
import astropy.io.fits as pyfits
from matplotlib import pyplot as plt
import modules_EG as utils

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

## Configuration

# Data selection
cat = 'mice-faint' # Select the lens catalogue (kids/gama/mice or mice-faint)

# Import lens catalog
fields, path_galcat, galcatname, galID, galRA, galDEC, galZ, galDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)
galDa = galDc/(1.+galZ)

# Creating the galaxy coordinates
galcoords = SkyCoord(ra=galRA*u.deg, dec=galDEC*u.deg, distance=galDa)
Ngal = len(galcoords)

# Find the closest galaxy around each lens
if 'faint' not in cat:
    idx, d2d, d3d = galcoords.match_to_catalog_3d(galcoords, nthneighbor=2)
else:
    faintID, faintRA, faintDEC, faintZ, faintDc, faint_rmag, faint_rmag_abs, faint_e1, faint_e2, faint_logmstar =\
    utils.import_micecat('/data/users/brouwer/LensCatalogues', 'mice2_faint_catalog_400deg2.fits', h)
    
    faintmask = (faint_rmag<21.)
    faintRA, faintDEC, faintZ, faintDc = [faintRA[faintmask], faintDEC[faintmask], faintZ[faintmask], faintDc[faintmask]]
    faintDa = faintDc/(1.+faintZ)
    
    faintcoords = SkyCoord(ra=faintRA*u.deg, dec=faintDEC*u.deg, distance=faintDa)
    idx, d2d, d3d = galcoords.match_to_catalog_3d(faintcoords, nthneighbor=2)

rsat = d3d.to(u.Mpc).value

print('rsat:', rsat)
print()

# Write the results to a fits table
filename = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_h%i'%(cat, h*100.)

if 'faint' not in cat:
    outputnames = ['riso']
else:
    outputnames = ['riso_faint']

formats = ['D']
output = [rsat]

utils.write_catalog('%s.fits'%filename, outputnames, formats, output)

print('Isolated galaxies (3D):', np.sum(rsat > 3.), np.sum(rsat > 3.)/Ngal*100., 'percent')
