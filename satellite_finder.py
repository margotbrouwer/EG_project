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
cat = 'gama' # Select the lens catalogue (kids/gama/mice)
Rmax = 3 # Maximum satellite finding radius (in Mpc)

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)
logmstarlist = logmstar

# Remove all galaxies with logmstar=NAN
nanmask = np.isfinite(logmstar)
lensRA, lensDEC, lensDc, logmstar, rmag = \
[lensRA[nanmask], lensDEC[nanmask], lensDc[nanmask], logmstar[nanmask], rmag[nanmask]]

# Creating bright galaxy coordinates
lenscoords = SkyCoord(ra=lensRA*u.deg, dec=lensDEC*u.deg, distance=lensDc)
brightmask = (rmag<17.3)&(8.<logmstar)&(logmstar<13.)
lenscoords_bright = lenscoords[brightmask]
bright_index = np.arange(len(lenscoords))[brightmask]
bright_mstar = 10.**logmstar[brightmask]

print('Number of bright galaxies:', len(lenscoords_bright))

# Find all galaxies around each bright lens
idxbright, idxall, d2d, d3d = lenscoords.search_around_3d(lenscoords_bright, Rmax*u.Mpc)

unique, satellite_counts = np.unique(idxbright, return_counts=True)
print('Satellite count within %g Mpc:'%Rmax, satellite_counts, len(unique))

# Compute the total satellite mass (fraction) around each lens
satellite_masses = np.zeros(len(bright_index))
satellite_fractions = np.zeros(len(bright_index))
for i in unique:
    satellite_index = idxall[idxbright==i] # Find all galaxies around the lens
    satellite_index = satellite_index[satellite_index != bright_index[i]] # Remove the lens itself
    
    satellite_mass = np.sum(10**(logmstar[satellite_index])) # Compute the total satellite mass
    if satellite_mass > 0:
        # Add log(Msat) to a list
        satellite_masses[i] = satellite_mass
        
        # Compute satellite mass compared to galaxy mass
        satellite_fractions[i] = satellite_mass/bright_mstar[i]

## Plot the results

#"""


# Dividing the values into bins of logmstar
logmstarbins = np.linspace(7., 13., 20)
binindex_logmstar = np.digitize(logmstar[brightmask], logmstarbins)

# Find the mean satellite mass per logmstar-bin
satellite_mass_mean = np.array([ np.mean(satellite_masses[binindex_logmstar==m]) \
        for m in np.arange(20)])
mstar_mean = np.array([ np.mean(bright_mstar[binindex_logmstar==m]) \
        for m in np.arange(20)])

plt.plot(np.log10(mstar_mean), satellite_mass_mean/mstar_mean)

plt.yscale('log')
plt.show()

#plt.xscale('log')


"""
plt.hist2d((logmstar[brightmask])[satellite_fractions<5.], \
        (satellite_fractions)[satellite_fractions<5.], bins=50)
plt.axhline(y=1., color='black')

plt.hist2d((logmstar[brightmask])[satellite_masses>7.], \
        (satellite_masses)[satellite_masses>7.], bins=50)
plt.plot(np.arange(8,15), np.arange(8,15), color='black')


xlabel = r'logmstar'
ylabel = r'$log_{10}(M_*)$ of galaxies within 3 Mpc'

plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)

plt.show()



# Write the results to a fits table
filename = '/data/users/brouwer/LensCatalogues/%s_satellites_h%i'%(cat, h*100.)

outputnames = np.append(['logmstar'], ['dist%s%s'%(n,rationame) for n in rationames])
formats = np.append(['D'], ['D']*len(massratios))
output = np.append([logmstarlist], distmdex, axis=0)
print(outputnames, formats, output)

utils.write_catalog('%s.fits'%filename, outputnames, formats, output)
"""
