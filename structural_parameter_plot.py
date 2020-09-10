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
from matplotlib.lines import Line2D

import modules_EG as utils

# Make use of TeX
rc('text',usetex=True)
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})

# Red, light blue, dark grey, grey
colors = ['#d7191c', '#0571b0', 'darkgrey', 'grey']

# Dark blue, light blue, red, orange
#colors = ['#0571b0', '#92c5de', '#d7191c', '#fdae61']*2

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
mue = lenscat['mue_2dphot']
color_gr = lenscat['MAG_GAAP_u'] - lenscat['MAG_GAAP_r']


plt.figure(figsize=(6,5))

"""
## Plot central surface brightness

valpha = 0.07

for m in range(len(maskvals)):
    
    print('Sample:', m+1)
    
    # Creating the lens mask
    lensmask, filename_var = utils.define_lensmask(paramnames, maskvals[m], path_lenscat, lenscatnames, h)

    # Masking the binning values and shear catalog
    if m==0:
        plt.scatter(logmstar_GL[lensmask], mue[lensmask], marker='.', alpha=valpha, color=colors[m], label='Ellipticals (n>2)')
    else:
        if m==1:
            plt.scatter(logmstar_GL[lensmask], mue[lensmask], marker='.', alpha=valpha, color=colors[m], label='Spirals (n<2)')
        else:
            plt.scatter(logmstar_GL[lensmask], mue[lensmask], marker='.', alpha=valpha, color=colors[m])

plt.axvline(x=11., color='black')

#plt.axvline(x=10.7, color='black')#, ymin=5./11.)
#plt.axhline(y=10., color='black', xmax=10./13.)

#plt.text(10., 20., r'1', fontsize=15)
#plt.text(10.8, 20., r'2', fontsize=15)
#plt.text(10.5, 13., r'3', fontsize=15)
#plt.text(10.8, 13., r'4', fontsize=15)
#plt.text(10.8, 6., r'5', fontsize=15)


plt.xlabel(r'Stellar mass log($M_*$) (${\rm M_\odot}/h_{70}^2$)')
plt.ylabel(r'Effective surface brightness $\mu_e$ (mag/arcsec$^2$)')

plt.xlim([8.5, 12.])
plt.ylim([16., 28.])

leg = plt.legend(loc='best')

for lh in leg.legendHandles: 
    lh.set_alpha(1)

plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Mar20/Plots/galaxy_morphology_mue_samples'

# Save plot
for ext in ['pdf', 'png']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf()
"""

## Plot colour

valpha = 0.01
for m in range(len(maskvals)):
    
    print('Sample:', m+1)
    
    # Creating the lens mask
    lensmask, filename_var = utils.define_lensmask(paramnames, maskvals[m], path_lenscat, lenscatnames, h)

    # Masking the binning values and shear catalog
    plt.plot(logmstar_GL[lensmask], color_gr[lensmask], marker='.', ls='', alpha=valpha, color=colors[m], rasterized=True)


plt.axvline(x=11., color='black')
plt.axhline(y=2.5, color='black', ls='--')

"""
# Elliptical and spiral mass bins
plt.axvline(x=10.5, ymax=4./9., color='black', ls='--')
plt.axvline(x=10.8, ymin=4./9., color='black', ls='--')
plt.text(10.6, 3., r'1', fontsize=15)
plt.text(10.85, 3., r'2', fontsize=15)
plt.text(10.25, 2., r'3', fontsize=15)
plt.text(10.7, 2., r'4', fontsize=15)
"""

plt.xlabel(r'Stellar mass log($M_*$) (${\rm M_\odot}/h_{70}^2$)', fontsize=15)
plt.ylabel(r'Colour ($u-r$)', fontsize=15)

plt.xlim([8.5, 11.75])
plt.ylim([0.5, 5.])

plt.tick_params(labelsize='14')

legend_elements = [Line2D([0], [0], marker='.', ls = '', color=colors[0], label=r'Ellipticals (n$>2$)'), \
                   Line2D([0], [0], marker='.', ls = '', color=colors[1], label=r'Spirals (n$<2$)')]

plt.legend(handles=legend_elements, loc='best', fontsize=15)


plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Mar20/Plots/galaxy_morphology_color_u-r'

# Save plot
for ext in ['pdf', 'png']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf()

