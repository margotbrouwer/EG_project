#!/usr/bin/python
"""
# This script is used to make samples of ellipticals and spirals with equal mass distributions
"""

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

# Make use of TeX
rc('text',usetex=True)
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})

# Colours

# Dark blue, light blue, red, orange
colors = ['#0571b0', '#92c5de', '#d7191c', '#fdae61']

# Calculate the total baryonic mass (stars + cold gas)
def calc_logmbar(logmstar):
    fcold = 10.**(-0.69*logmstar + 6.63)
    mstar = 10.** logmstar
    mbar = mstar * (1 + fcold)
    logmbar = np.log10(mbar)
    return logmbar

# Import lens catalog
cat = 'kids' # kids / gama / matched
splittype = 'color' #sersic / color
Noffset = 0 # Minimum 0
masstype = 'GL' # GL for GAMA-like, or var for varying IMF

plot=False # Choose True for the plots, False for the catalogue

# Constants
h = 0.7
if 'mice' in cat:
    O_matter = 0.25
    O_lambda = 0.75
else:
    O_matter = 0.2793
    O_lambda = 0.7207

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)
Nlenses = len(lensZ) # Total number of lenses

# Full directory & name of the corresponding lens catalogue
lenscatfile = '%s/%s'%(path_lenscat, lenscatname)
lenscat = pyfits.open(lenscatfile, memmap=True)[1].data

# Define spiral and elliptical galaxies by sersic index or color
if 'sersic' in splittype:
    typename = 'n_2dphot'
    typelist = lenscat['n_2dphot']
    splitlim = 2.
    splitnames = ['bulge', 'disc']
        
if 'color' in splittype:
    if 'kids' in cat:
        typename = 'MAG_GAAP_u-r'
        typelist = lenscat['MAG_GAAP_u'] - lenscat['MAG_GAAP_r']
        splitlim = 2.5
        splitnames = ['red', 'blue']
    if 'mice' in cat:
        typename = 'sdss_u-r_true'
        typelist = lenscat['sdss_u_true'] - lenscat['sdss_r_true']
        splitlim = 2.5
        splitnames = ['red', 'blue']

mask_ell = (typelist > splitlim)
mask_spir = (typelist < splitlim)

## Stellar mass corrections

# Adding GAMA-like correction to the masses
diff_GL = -0.056
logmstar_GL = logmstar - diff_GL

if 'var' in masstype:
    # Adding a 0.09 offset to the early-type galaxies
    logmstar_GL[mask_ell] = logmstar_GL[mask_ell] + 0.09
    Noffset = 0 # Set offset to 0
    print('Varying IMF: Adding 0.09 to early-type galaxies; setting offset to 0.')

# Calculate the total baryonic mass (stars + cold gas)
logmbar_GL = calc_logmbar(logmstar_GL)

# Import mask
if 'kids' in cat:
    masked = lenscat['masked']

# If we are creating an offset catalogue
if Noffset > 1:
    # Import masses from the catalogue that is one offset less than the one we want to make
    lenscatname = 'mass_selection_catalog_%s-offsetx%s_%s.fits'%(splittype, Noffset-1, cat)
    lenscatfile = '%s/%s'%(path_lenscat, lenscatname)
    lenscat = pyfits.open(lenscatfile, memmap=True)[1].data
    logmstar = lenscat['logmstar']
    logmstar_GL = logmstar - diff_GL

if Noffset > 0:
    # Add a random offset to each stellar mass to estimate the Eddington bias
    Sigma_M = [0.29]*len(logmstar)
    dMlist = np.random.normal(loc=0., scale=Sigma_M, size=len(Sigma_M))
    logmstar = logmstar+dMlist
    logmstar_GL = logmstar - diff_GL
    print()
    print('Adding offset to lens masses: x%i'%(Noffset))
    print()

# Isolated galaxy catalogue
isocatname = '%s_isolated_galaxies_perc_h70.fits'%cat
isocatfile = '%s/%s'%(path_lenscat, isocatname)
isocat = pyfits.open(isocatfile, memmap=True)[1].data

# Import isolation criterion
R_iso = isocat['dist0p1perc'] # Distance to closest satellite
# Should have no satellites within 3 Mpc, masked and within 0.1 < z < 0.5
if 'kids' in cat:
    mask_iso = (R_iso > 3.) & (masked==0.) & (0.1<lensZ)&(lensZ<0.5)
if 'mice' in cat:
    mask_iso = (R_iso > 3.) & (0.1<lensZ)&(lensZ<0.5)

print('Galaxies, isolated:', sum(mask_iso))
print('Ellipticals:', sum(mask_ell*mask_iso))
print('Spirals:', sum(mask_spir*mask_iso))

# Creating the stellar mass bins
Nbins = 100
binedges = np.linspace(8., 11., Nbins+1)
bincenters = binedges[0:-1] + np.diff(binedges)/2.

# Creating stellar mass histograms for ellipticals and spirals
N_ell, foo, foo = plt.hist(logmstar_GL[mask_ell*mask_iso], bins=binedges, histtype='step', \
    label=r'Isolated %s galaxies'%splitnames[0], color=colors[2])
N_spir, foo, foo = plt.hist(logmstar_GL[mask_spir*mask_iso], bins=binedges, histtype='step', \
    label=r'Isolated %s galaxies'%splitnames[1], color=colors[0])

# Finding the smallest number of galaxies in each M-bin
Nmin = np.array([np.amin(np.array([N_ell[x], N_spir[x]])) for x in np.arange(Nbins)])

if plot:
    
    # Plot the resulting histograms
    plt.fill_between(bincenters, Nmin, 0., \
        label=r'Selected %s \& %s galaxies'%(splitnames[0], splitnames[1]), color=colors[1], alpha=0.3)

    xlabel = r'Stellar mass log($M_*$) (${\rm M_\odot}/h_{70}^2$)'
    ylabel = r'Number of galaxies'
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
        
    plt.tick_params(labelsize='14')
    plt.legend(loc='upper left', fontsize=14)

    # Save plot
    plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Jul20/Plots/mass_range_selection_offsetx%i'%(Noffset)
    for ext in ['pdf']:
        plotname = '%s.%s'%(plotfilename, ext)
        plt.savefig(plotname, format=ext)
        
    print('Written: ESD profile plot:', plotname)
    
    plt.show()
    plt.clf
    
    print('Creating the catalogue takes very long. If you want that, set: plot=False.')
    quit()
else:
    pass

# This list will contain the selected lenses
selection_list = np.zeros(Nlenses)

print()
print('# galaxies in Mstar-bin:')
print('Mstar Min. Ell. Spir.')

# For each M-bin, select Nmin random galaxies from each galaxy type
for m in np.arange(Nbins):
    mask_mstar = (binedges[m] < logmstar_GL) & (logmstar_GL <= binedges[m+1])
    
    # Number of isolated ellipticals and spirals in each mass bin
    Nell_bin, Nspir_bin = \
        [sum(mask_iso*mask_ell*mask_mstar), sum(mask_iso*mask_spir*mask_mstar)]
    
    # The least amount of lenses Nmin in this mass bin
    Nmin_bin = np.amin(np.array([Nell_bin, Nspir_bin]))
    
    print((binedges[m]+binedges[m+1])/2., Nmin_bin, Nell_bin, Nspir_bin)
    
    # Create two lists of Nmin selected galaxies, one for each type
    arr_ell = np.array([1]*int(Nmin_bin) + [0]*int(Nell_bin-Nmin_bin))
    arr_spir = np.array([1]*int(Nmin_bin) + [0]*int(Nspir_bin-Nmin_bin))
    
    # Shuffle the two arrays to randomize the selected galaxies
    np.random.shuffle(arr_ell)
    np.random.shuffle(arr_spir)
    
    # Assign these lists to the final list of selected lenses
    selection_list[mask_iso*mask_ell*mask_mstar] = arr_ell
    selection_list[mask_iso*mask_spir*mask_mstar] = arr_spir
    
    output = [lensID, logmstar_GL, logmbar_GL, typelist, selection_list]
    outputnames = ['ID', 'logmstar_GL', 'logmbar_GL', typename, 'selected']

# Create the output catalogue
filename = '/data/users/brouwer/LensCatalogues/logmstar_%s_selection_catalog_%s-offsetx%i_%s.fits'%(masstype, splittype, Noffset, cat)

formats = ['D']*len(outputnames)

utils.write_catalog(filename, outputnames, formats, output)
