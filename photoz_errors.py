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
O_matter = 0.2793
O_lambda = 0.7207

G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')

# Create LCDM cosmology
cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Creating a normalized gaussian distribution
def calc_gaussian(x, mu, sigma):
    a = 1/(sigma * np.sqrt(2*np.pi))
    gaussian = a * np.exp(-0.5*((x-mu)/sigma)**2.)
    return gaussian

## Defining the lens and source samples

# Source redshift/distance bins
zsbins = np.linspace(0.05, 1.2, 70)
dZs = np.diff(zsbins)[0]
Dcsbins = (cosmo.comoving_distance(zsbins).to('pc')).value

# Source redshift PDF
srcZ = 0.6
srcSigma = 0.2
srcPZ = calc_gaussian(zsbins, srcZ, srcSigma)

# Lens redshift/distance bins
zlensbins = np.linspace(0.05, 0.7, 100)
dZl = np.diff(zlensbins)[0]
Dclbins = (cosmo.comoving_distance(zlensbins).to('pc')).value
Dalbins = Dclbins/(1+zlensbins)

# Lens redshift PDF's
galZlist = np.linspace(0.2, 0.5, 10)
galSigma = [0.026]*len(galZlist) #0.021*(1+galZlist) # 0.026
galPZ = np.array([calc_gaussian(zlensbins, galZlist[z], galSigma[z]) for z in range(len(galZlist))])

# Redshift barrier
z_epsilon = 0.2
Dc_epsilon = (cosmo.comoving_distance(z_epsilon).to('pc')).value

print(np.sum(srcPZ*dZs))
print(np.sum(galPZ*dZl,1))

# Plot the lens/source p(z)'s
plt.plot(zsbins, srcPZ, label='Source P(z)')
plt.plot(np.array([zlensbins]*3).T, galPZ.T, label='Lens P(z)')
plt.show()


## Calculating SigmaCrit for each lens

# Calculate the values of Dls/Ds for all lens/source redshift-bin pair
Dclbins, Dcsbins = np.meshgrid(Dclbins, Dcsbins)
DlsoDs = (Dcsbins-Dclbins)/Dcsbins

# Mask all values with Dcl=0 (Dls/Ds=1) and Dcl>Dcsbin (Dls/Ds<0)
DlsoDs[np.logical_not(((Dc_epsilon/Dcsbins) < DlsoDs) & (DlsoDs < 1.))] = 0.0
"""
plt.imshow(DlsoDs)
plt.colorbar()
plt.show()
"""

print('DlsoDs:', np.shape(DlsoDs))
print('Source P(zs):', np.shape(srcPZ))

# Matrix multiplication that sums over P(zs), to calculate <Dls/Ds> for each Zlens-bin
DlsoDs = np.dot(srcPZ, DlsoDs)
print('Integrated DlsoDs:', np.shape(DlsoDs))
print('DlsoDs', DlsoDs)
print()
#print('Dalbins:', np.shape(Dalbins))
#print('Lens P(zl):', np.shape(galPZ))


# Matrix multiplication that sums over P(zl), to calculate <Da*Dls/Ds> for each lens
DaDlsoDs = np.dot(galPZ*dZl, Dalbins*DlsoDs)
print('Integrated Da*DlsoDs:', np.shape(DaDlsoDs))
print('DaDlsoDs:', DaDlsoDs)
print()

# Calculate the values of k (=1/Sigmacrit)
k = 1 / ((c.value**2)/(4*np.pi*G.value) * 1/(DaDlsoDs)) # k = 1/Sigmacrit
print('k:', k)

plt.plot(galZlist, k)
plt.show()

"""

withsigma = np.array([0.00173103, 0.01502663, 0.02573743, 0.03245891, 0.03623504, 0.03839803, 0.03936853, 0.03945987, 0.03799865, 0.03467676, 0.03033093, 0.02639618, 0.02383567, 0.02111101, 0.01530048, 0.00302056, 0.00015602, 2.1929862e-06, 1.00036402e-08, 1.92882622e-11])
withoutsigma = np.array([2.52592542e-06, 0.0001088, 0.00018257, 0.00022654, 0.00024493, 0.00025936, 0.0002663, 0.00027123, 0.0002638, 0.0002503, 0.00021967, 0.00016453, 0.00015867, 0.00014945, 0.00018567, 0.00017652, 0.00015016, 0.00011551, 5.50189749e-05, 2.47846097e-05])

zlist = np.arange(0.001, 1., 0.05)
print(zlist)

plt.plot(zlist, withsigma, label='with sigma')
plt.plot(zlist, withoutsigma*100, label='without sigma')

plt.legend()
plt.show()
"""
