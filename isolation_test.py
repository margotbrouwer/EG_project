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


cat = 'gama' # Select the lens catalogue (kids/gama/mice)
rationame = 'perc'
massratios = [0.3, 0.25, 0.2, 0.15, 0.1]
massratio_names = [str(p).replace('.','p') for p in massratios]
massratio_num = 4
print(massratios[massratio_num])

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)
logmstarlist = logmstar

# Remove all galaxies with logmstar=NAN
nanmask = np.isfinite(logmstar)
lensRA, lensDEC, lensDc, logmstar, rmag = \
    [lensRA[nanmask], lensDEC[nanmask], lensDc[nanmask], logmstar[nanmask], rmag[nanmask]]


# Import isolation catalog
isocatfile = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_%s_h%i.fits'%(cat, rationame, h*100.)
isocat = pyfits.open(isocatfile, memmap=True)[1].data

#distcat = np.array([ isocat['dist%s%s'%(p, rationame)] for p in massratio_names])
isodist = (isocat['dist%s%s'%(massratio_names[massratio_num], rationame)])[nanmask]
isomask = isodist > 3e6 # There are no galaxies within X pc

# Define the galaxy mass bins
rmag_min, rmag_max = [13., 20.]
Nrmagbins = 100.
rmagbins = np.linspace(rmag_min, rmag_max, Nrmagbins)
rmagcenters = rmagbins[0:-1] + 0.5*np.diff(rmagbins)

isohist, foo = np.histogram(rmag[isomask], rmagbins)
rmaghist, foo = np.histogram(rmag, rmagbins)

maglim = -2.5 * np.log10(massratios[massratio_num])

print(maglim, 19.8-maglim)

plt.plot(rmagcenters, isohist/1e4, label='Isolated galaxies')
plt.plot(rmagcenters, rmaghist/1e4, label='All galaxies')
plt.plot(rmagcenters, isohist/rmaghist, label='Ratio (Isolated/All)')
#plt.plot(rmagcenters, np.cumsum(isohist)/np.cumsum(rmaghist))

plt.axvline(x=19.8-maglim, color='green', ls='--')
plt.axvline(x=19.8, color='black', ls='--')

xlabel = r'r-band apparent magnitude $m_{\rm r}$'
ylabel = r'Number of galaxies (x10.000)'

plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)


#plt.yscale('log')
plt.legend()

plt.show()


"""
plt.scatter(rmag, isodist, marker='.', alpha=0.1)
plt.show()
"""