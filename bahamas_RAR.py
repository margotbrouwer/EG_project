#!/usr/bin/python

# Import the necessary libraries
import sys

import numpy as np
import pyfits
import os

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
import scipy.optimize as optimization
import modules_EG as utils

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

from matplotlib import gridspec

def gobs_mond(gbar, g0=1.2e-10):
    gobs = gbar / (1 - np.exp( -np.sqrt(gbar/g0) ))
    return gobs

# Import constants
G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')
inf = np.inf
h=0.7

## Define projected distance bins R

# Creating the Rbins
#Runit, Nbins, Rmin, Rmax = ['Mpc', 20, 0.03, 3.] # R-bins
Runit, Nbins, Rmin, Rmax = ['mps2', 10, 1e-15, 5e-12] # gbar-bins
Rbins, Rcenters, Rmin_pc, Rmax_pc, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins, True)
print('Rbins: %i bins between %g and %g %s'%(Nbins, Rmin, Rmax, Runit))

gbar_mond = np.logspace(-15, -8, 50)
gbar_ext = np.logspace(-15, -12, 30)
gbar_uni = np.logspace(-15, -8, 50)

## Import Bahamas ESD catalog

path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
catfile = '%s/ESD/ESD_profiles_Rbins-%i_%g-%g%s.fits'%(path_cat, Nbins, Rmin, Rmax, Runit)
print('Imported:', catfile)

# Import Mstar
catname = '%s/catalog.dat'%path_cat
catalog = np.loadtxt(catname).T
logmstarlist = catalog[5] # Stellar mass of each lens galaxy

# Import the ESD
cat = pyfits.open(catfile, memmap=True)[1].data
ESD_list = cat['ESD']
Rbins_list = cat['Rbins_pc']
catnum = len(ESD_list)

print(ESD_list)

#data_x, data_y, error_h, error_l = utils.read_esdfiles(esdfiles)
rar_list = ESD_list * 4.*G*3.08567758e16 # Convert ESD (Msun/pc^2) to acceleration (m/s^2)

## Ploting the result

for i in range(catnum):
    plt.plot(Rcenters, rar_list[i])
#    Rbins_centers = (Rbins_list[i])[0:-1] + np.diff(Rbins_list[i])/2.
#    plt.plot(Rbins_centers/1e6, ESD_list[i])

plt.plot(gbar_mond, gobs_mond(gbar_mond), ls='--')
plt.plot(gbar_mond, gbar_mond, ls=':')

# Define the labels for the plot
xlabel = r'Expected baryonic acceleration [$m/s^2$]'
ylabel = r'Observed radial acceleration [$m/s^2$]'
plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)

plt.xscale('log')
plt.yscale('log')

plt.show()




plt.show()
