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
O_matter = 0.2793
O_lambda = 0.7207

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})


# Define number of random catalogues
Nrandoms = 100

## Import lens catalogue
path_lenscat = '/data/users/brouwer/LensCatalogues'

"""
# Data selection
cat = 'gama' # Select the lens catalogue (kids/gama/mice)
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)
"""

cat = 'kids'
lenscatfile = '%s/photozs.DR4_trained-on-GAMAequ_ugri+KV_version0.9.fits'%(path_lenscat)
lenscat = pyfits.open(lenscatfile, memmap=True)[1].data

lensZ = lenscat['z_ANNZ_KV']
masked = lenscat['masked']

masscatfile = '%s/baryonic_mass_catalog_kids.fits'%(path_lenscat)
masscat = pyfits.open(masscatfile, memmap=True)[1].data

logmstar = masscat['logmstar_GL']
logmbar = masscat['logmbar_GL']

# Mask the redshifts
lensmask = (0.<lensZ)&(lensZ<0.5)&(0.<logmstar)&(logmstar<11.)&(masked==0)

lensZ = lensZ[lensmask]
logmstar = logmstar[lensmask]
logmbar = logmbar[lensmask]

# Define the number of galaxies per catalogue, and ID's
Ngals = len(lensZ)
IDcat = np.arange(Ngals)


print('Creating %i random catalogues with %g galaxies'%(Nrandoms, Ngals))

## Import MJ's random catalogue

randomcatfile = '%s/Randoms/new_DR4_9bandnoAwr_uniform_randoms.fits'%(path_lenscat)
randomcat = pyfits.open(randomcatfile, memmap=True)[1].data

# List of the observables of all lenses in the GAMA catalogue
RAlist = randomcat['RA']
DEClist = randomcat['DEC']
#randomJK = randomcat['JK_LABEL']


## Create the random redshifts and stellar masses

# Create the redshift histogram
Zhist, Zbin_edges = np.histogram(lensZ, int(1e3))
Zbins = Zbin_edges[0:-1]+np.diff(Zbin_edges)

# Create the stellar mass histogram
Mhist, Mbin_edges = np.histogram(logmbar, int(1e3))
Mbins = Mbin_edges[0:-1]+np.diff(Mbin_edges)

# Compute the random values
Nmult = int(np.ceil(Ngals/len(lensZ))) # Multiplier to have enough redshift values

# This list will contain a long list of random values, to be selected from
Zlist = np.array([])
Mlist = np.array([])
for b in range(len(Zbins)):
    Zbin = np.random.uniform(Zbin_edges[b], Zbin_edges[b+1], Nmult*Zhist[b])
    Zlist = np.append(Zlist, Zbin)
    
    Mbin = np.random.uniform(Mbin_edges[b], Mbin_edges[b+1], Nmult*Mhist[b])
    Mlist = np.append(Mlist, Mbin)

## Create the random catalogues

for r in range(Nrandoms):
        
    # Add the right number of galaxies to the catalogue
    RAcat = RAlist[r*Ngals:(r+1)*Ngals]
    DECcat = DEClist[r*Ngals:(r+1)*Ngals]
    
    # Select redshifts for the catalogue
    Zcat = np.random.choice(Zlist, Ngals, replace=False)
    Mcat = np.random.choice(Mlist, Ngals, replace=False)
    
    # Write everything to a catalogue
    outputnames = np.array(['ID', 'RA', 'DEC', 'Z', 'logmbar'])
    formats = np.array(['D']*len(outputnames))
    output = np.array([IDcat, RAcat, DECcat, Zcat, Mcat])

    filename = '%s/Randoms/randoms-MJ_%s-Z_%i'%(path_lenscat, cat, r)
    utils.write_catalog('%s.fits'%filename, outputnames, formats, output)


