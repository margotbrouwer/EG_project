#!/usr/bin/python

"""Plot the Bahamas and MICE stellar-to-halo-mass relations, and other mass tests"""

# Import the necessary libraries
import sys
import numpy as np
import pyfits
import os

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
import scipy.optimize as optimization
import scipy.stats as stats
import modules_EG as utils

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams
from matplotlib import gridspec

# Import constants
pi = np.pi
G = const.G.to('pc3 / (M_sun s2)').value
c = const.c.to('m/s').value
pc_to_m = 3.08567758e16

# Define cosmology
O_matter = 0.2793
O_lambda = 0.7207
h = 0.7
H0 = h * 100 * (u.km/u.s)/u.Mpc
H0 = H0.to('s-1').value

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Make use of TeX
rc('text',usetex=True)
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})

# Dark blue, light blue, red, orange
colors = ['#0571b0', '#92c5de', '#d7191c', '#fdae61']*2


## Import Bahamas catalogues

profbins = 40
catnum = 515
lenslist = np.arange(catnum)

path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_isolated_new/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
catname = '%s/catalog.dat'%path_cat
catalog = np.loadtxt(catname).T[:,lenslist]

# Import galaxy observables (M200, r200, logmstar)
logM200_bhm = catalog[3] # M200 of each galaxy
logM500_bhm = catalog[1] # M500 of each galaxy
r200_bhm = catalog[4] * 1e6 # r200 of each galaxy (in pc)
logmstar_bhm = catalog[5] # Stellar mass of each lens galaxy

## Import MICE catalogue

fields, path_micecat, micecatname, lensID_mice, lensRA_mice, lensDEC_mice, \
    lensZ_mice, lensDc_mice, rmag_mice, rmag_abs_mice, logmstar_mice = \
    utils.import_lenscat('mice', h, cosmo)
lensDc_mice = lensDc_mice.to('pc').value
lensDa_mice = lensDc_mice/(1.+lensZ_mice)

# Full directory & name of the catalogue
micecatfile = '%s/%s'%(path_micecat, micecatname)
micecat = pyfits.open(micecatfile, memmap=True)[1].data

logmhalo_mice = micecat['lmhalo']

"""
# Plot Bahamas mass histogram
massbins = np.arange(12., 15., 0.25)
plt.hist(M200_bhm, bins=massbins)

# Define the labels for the plot
xlabel = r'Halo mass log($M_{200}/M_\odot$)'
ylabel = r'Number of haloes'

"""

massbins = np.arange(9.,15.,0.1)


plt.plot(logmstar_bhm, logM200_bhm, ls='', marker='.', color='red', alpha=0.5, label='Bahamas (M200)')
plt.plot(logmstar_bhm, logM500_bhm, ls='', marker='.', color='orange', alpha=0.5, label='Bahamas (M500)')

plt.plot(0., 0., ls='', marker='.', color='blue', alpha=0.5, label='MICE (FoF halo mass)')
plt.hist2d(logmstar_mice, logmhalo_mice, bins=[massbins, massbins], \
    cmin=1, cmap='Blues', zorder=0)

xlabel = r'Stellar mass log($M_*/M_\odot$)'
ylabel = r'Halo mass log($M_{200}/M_\odot$)'

# Create the plot
plt.legend()
plt.xlabel(xlabel, fontsize=16)
plt.ylabel(ylabel, fontsize=16)

plt.xlim(9., 12)
plt.ylim(11., 14.)

plt.show()
plt.clf()
