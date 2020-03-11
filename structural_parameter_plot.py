#!/usr/bin/python

"Plot the galaxy morphology samples based on their structural parameters."

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

#colors = ['#0571b0', '#d7191c', 'grey', 'darkgrey']
#colors = ['#0571b0', '#92c5de']*2

# Dark blue, light blue, red, orange
colors = ['#0571b0', '#92c5de', '#d7191c', '#fdae61']*2

# Import constants
G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')
inf = np.inf
   
h, O_matter, O_lambda = [0.7, 0.2793, 0.7207]
cosmo = LambdaCDM(H0=h*100, Om0=O_matter, Ode0=O_lambda)

## Configuration

# Data selection
cat = 'kids' # Select the lens catalogue (kids/gama/mice)

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

# Lens selection
isocatname = '%s_isolated_galaxies_perc_h70.fits'%cat
masscatname = 'baryonic_mass_catalog_%s.fits'%cat

# Binning parameters

if 'gama' in cat:
    paramnames = np.array(['logmstar', 'nQ', 'dist0p1perc'])
    maskvals = np.array([[8.5,12.], [3, inf], [4, inf]])
    lenscatnames = np.array([lenscatname, lenscatname, isocatname])

if 'kids' in cat:
    paramnames = np.array(['z_ANNZ_KV', 'dist0p1perc', 'n_2dphot', 'logmstar_GL'])
    lenscatnames = np.array([lenscatname, isocatname, lenscatname, masscatname])
    maskvals = np.array([ \
                        [[0., 0.5], [3, inf], [2.,inf], [8., 11.]], \
                        [[0., 0.5], [3, inf], [0., 2.], [8., 11.]], \
                        [[0., 0.5], [3, inf], [2., inf], [11., 13.]], \
                        [[0., 0.5], [3, inf], [0., 2.], [11., 13.]], \
                        ])


# Importing the lens and mass catalogues
masscat = pyfits.open('%s/%s'%(path_lenscat, masscatname), memmap=True)[1].data
logmstar_GL = masscat['logmstar_GL']

lenscat = pyfits.open('%s/%s'%(path_lenscat, lenscatname), memmap=True)[1].data
mu0 = lenscat['mu0_2dphot']
color_gr = lenscat['MAG_GAAP_g'] - lenscat['MAG_GAAP_r']

## Plot central surface brightness

plt.figure(figsize=(7,7))
valpha = 0.07

for m in range(len(maskvals)):
    
    print('Sample:', m+1)
    
    # Creating the lens mask
    lensmask, filename_var = utils.define_lensmask(paramnames, maskvals[m], path_lenscat, lenscatnames, h)

    # Masking the binning values and shear catalog
    if m==0:
        plt.scatter(logmstar_GL[lensmask], mu0[lensmask], marker='.', alpha=valpha, color=colors[m], label='Ellipticals (n>2)')
    else:
        if m==1:
            plt.scatter(logmstar_GL[lensmask], mu0[lensmask], marker='.', alpha=valpha, color=colors[m], label='Spirals (n<2)')
        else:
            plt.scatter(logmstar_GL[lensmask], mu0[lensmask], marker='.', alpha=valpha, color=colors[m])

plt.axvline(x=11., color='black')

plt.axvline(x=10.7, color='black', ymin=5./11.)
plt.axhline(y=10., color='black', xmax=10./13.)

plt.text(10., 20., r'1', fontsize=15)
plt.text(10.8, 20., r'2', fontsize=15)
plt.text(10.5, 13., r'3', fontsize=15)
plt.text(10.8, 13., r'4', fontsize=15)
plt.text(10.8, 6., r'5', fontsize=15)


plt.xlabel(r'Stellar mass log($M_*$) (${\rm M_\odot}/h_{70}^2$)')
plt.ylabel(r'Central surface brightness')# $\mu_0$ (mag/arcsec$^2$)')

plt.xlim([8.5, 11.75])
plt.ylim([-2.5, 25.])

leg = plt.legend(loc='best')

for lh in leg.legendHandles: 
    lh.set_alpha(1)

plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Mar20/Plots/galaxy_morphology_mu0_samples'

# Save plot
for ext in ['pdf', 'png']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf()


## Plot colour


valpha = 0.03
for m in range(len(maskvals)):
    
    print('Sample:', m+1)
    
    # Creating the lens mask
    lensmask, filename_var = utils.define_lensmask(paramnames, maskvals[m], path_lenscat, lenscatnames, h)

    # Masking the binning values and shear catalog
    if m==0:
        plt.scatter(logmstar_GL[lensmask], color_gr[lensmask], marker='.', alpha=valpha, color=colors[m], label='Ellipticals (n>2)')
    else:
        if m==1:
            plt.scatter(logmstar_GL[lensmask], color_gr[lensmask], marker='.', alpha=valpha, color=colors[m], label='Spirals (n<2)')
        else:
            plt.scatter(logmstar_GL[lensmask], color_gr[lensmask], marker='.', alpha=valpha, color=colors[m])

plt.axvline(x=11., color='black')

plt.xlabel(r'Stellar mass log($M_*$) (${\rm M_\odot}/h_{70}^2$)')
plt.ylabel(r'Colour (g-r)')

plt.xlim([8.5, 11.75])
plt.ylim([0., 2.])
#plt.ylim([1., 6.])

leg = plt.legend(loc='best')

for lh in leg.legendHandles: 
    lh.set_alpha(1)

plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Mar20/Plots/galaxy_morphology_color_g-r'

# Save plot
for ext in ['pdf', 'png']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf()
