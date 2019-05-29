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
cat = 'kids' # Select the lens catalogue (kids/gama/mice)
r_iso = 3 # Maximum satellite finding radius (in Mpc)

# Import lens catalog
fields, path_lenscat, lenscatname, galID, galRA, galDEC, galZ, galDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)
galDa = galDc/(1.+galZ)

logmstar_cat = logmstar
satellite_masses_cat = np.ones(len(galID))*-999
satellite_fractions_cat = np.ones(len(galID))*-999
cat_index = np.arange(len(galID))

# Remove all galaxies with logmstar=NAN
nanmask = (np.isfinite(logmstar))&(logmstar>0.)
galRA, galDEC, galDa, logmstar, rmag, cat_index = \
[galRA[nanmask], galDEC[nanmask], galDa[nanmask], logmstar[nanmask], rmag[nanmask], cat_index[nanmask]]

# Creating the galaxy coordinates
galcoords = SkyCoord(ra=galRA*u.deg, dec=galDEC*u.deg, distance=galDa)
gal_index = np.arange(len(galcoords))

# Creating the lens sample
lensmask = (7.<logmstar)&(logmstar<13.)
lenscoords = galcoords[lensmask]
lens_mstar = 10.**logmstar[lensmask]
lens_index = gal_index[lensmask]
cat_index = cat_index[lensmask]

# Find all galaxies around each lens
idxlens, idxgal, d2d, d3d = galcoords.search_around_3d(lenscoords, r_iso*u.Mpc)
unique_lenses, satellite_counts = np.unique(idxlens, return_counts=True)
print('Satellite count within %g Mpc:'%r_iso, satellite_counts, len(unique_lenses))

Nlenses = len(unique_lenses)
#Nlenses = int(1e3)

# Compute the total satellite mass (fraction) around each lens
satellite_masses = np.zeros(Nlenses)
satellite_fractions = np.zeros(Nlenses)

for i in unique_lenses[0:Nlenses]:
    satellite_index = idxgal[idxlens==i] # Find all galaxies around the lens
    satellite_index = satellite_index[satellite_index != lens_index[i]] # Remove the lens itself

    satellite_mass = np.sum(10**(logmstar[satellite_index])) # Compute the total satellite mass
    if satellite_mass > 0:
        # Add log(Msat) to the list
        satellite_masses[i] = np.log10(satellite_mass)
        
        # Compute satellite mass compared to galaxy mass
        satellite_fractions[i] = satellite_mass/lens_mstar[i]

    if i % 1e4 == 0:
        print(i/len(unique_lenses)*100., r'\% finished')


print('Isolated galaxies (percentage):', np.sum(satellite_fractions < 1.)/Nlenses*100.)

#plt.hist(satellite_fractions[0:Ngal], np.logspace(-1.,4.,100))
#plt.xscale('log')
#plt.hist(satellite_fractions, np.linspace(0.,100.,100))
#plt.show()


# Write the results to a fits table
riso_name = str(r_iso).replace('.','p')
filename = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_Msat-%sMpc_h%i'%(cat, riso_name, h*100.)

outputnames = ['ID', 'logmstar', 'logMsat%sMpc'%riso_name, 'fsat%sMpc'%riso_name]
formats = ['D']*4

satellite_masses_cat[cat_index] = satellite_masses
satellite_fractions_cat[cat_index] = satellite_fractions
output = [galID, logmstar_cat, satellite_masses_cat, satellite_fractions_cat]

utils.write_catalog('%s.fits'%filename, outputnames, formats, output)

