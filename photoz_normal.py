#!/usr/bin/python
"""
# Create histograms to implement the lens photo-z errors into the GGL-pipeline

Model 1:
ugri photo-zs: error(zphot) = Gauss(sigma=0.026) [formally there's
also mu but it's ~1e-3]
KV photo-zs: error(zphot) = Gauss(sigma=0.022) [mu is even smaller
here than for ugri]

Model 2:
ugri photo-zs: error(zphot) = Gauss(sigma=0.021*(1+zphot))
KV photo-zs: error(zphot) = Gauss(sigma=0.018*(1+zphot))
[mu's again negligible]
"""

import astropy.io.fits as pyfits
import gc
import numpy as np
import sys
import os
import time
from glob import glob

import modules_EG as utils
from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685

G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')

# Create LCDM cosmology
cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

## Defining the lens and source samples

# Creating a normalized gaussian distribution
def calc_gaussian(x, mu, sigma):
    a = 1/(sigma * np.sqrt(2*np.pi))
    gaussian = a * np.exp(-0.5*((x-mu)/sigma)**2.)
    return gaussian

# Source redshift/distance bins
zsbins = np.linspace(0.05, 1.2, 70)
Dcsbins = (cosmo.comoving_distance(zsbins).to('pc')).value

# Source redshift PDF
srcZ = 0.6
srcSigma = 0.2
srcPZ = calc_gaussian(zsbins, srcZ, srcSigma)

# Lens redshifts
galZlist = np.linspace(0.2, 0.5, 10)
Dcls = (cosmo.comoving_distance(galZlist).to('pc')).value
Dals = Dcls/(1.+galZlist)

# Redshift barrier
z_epsilon = 0.2
Dc_epsilon = (cosmo.comoving_distance(z_epsilon).to('pc')).value


# Plot the lens/source p(z)'s
plt.plot(zsbins, srcPZ, label='Source P(z)')
[plt.axvline(zl, label='Lens P(z)') for zl in galZlist]
plt.show()


## Calculating SigmaCrit for each lens

# Calculate the values of Dls/Ds for all lens/source-redshift-bin pair
Dcls, Dcsbins = np.meshgrid(Dcls, Dcsbins)
DlsoDs = (Dcsbins-Dcls)/Dcsbins
print('DlsoDs:', np.shape(DlsoDs))

# Mask all values with Dcl=0 (Dls/Ds=1) and Dcl>Dcsbin (Dls/Ds<0)
DlsoDs[np.logical_not(((Dc_epsilon/Dcsbins) < DlsoDs) & (DlsoDs < 1.))] = 0.0

# Matrix multiplication that sums over P(z), to calculate <Dls/Ds> for each lens-source pair
DlsoDs = np.dot(srcPZ, DlsoDs).T
print('Source P(zs):', np.shape(srcPZ))
print('Integrated DlsoDs:', np.shape(DlsoDs))
print(DlsoDs)

# Calculate the values of k (=1/Sigmacrit)
#Dals = np.reshape(Dals,[len(Dals),1])
k = 1 / ((c.value**2)/(4*np.pi*G.value) * 1/(Dals*DlsoDs)) # k = 1/Sigmacrit
print('k:', k)

plt.plot(galZlist, k)
plt.show()
