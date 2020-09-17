#!/usr/bin/python

"""Plot the true Bahamas density profiles against Kyles density fits"""

# Import the necessary libraries
import sys
import numpy as np
import pyfits
import os

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
from scipy import integrate
import scipy.optimize as optimization
import scipy.stats as stats
import modules_EG as utils
from deproject.piecewise_powerlaw import esd_to_rho

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams
from matplotlib import gridspec


# Bahamas simulation parameters
Zlens = 0.25
O_matter = 0.2793
O_lambda =  0.7207
sigma8 = 0.821
h=0.7
cosmo = LambdaCDM(H0=h*100, Om0=O_matter, Ode0=O_lambda)
rho_crit = cosmo.critical_density(Zlens).to('Msun/pc3').value
print(rho_crit)

## Define projected distance bins R

# Creating the Rbins
Runit, Nbins, Rmin, Rmax = ['Mpc', 16, -999, 999] # Same R-bins as r from PROFILES
Rbins, Rcenters, Rmin_pc, Rmax_pc, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins, True)
print('R-bins: %i bins between %g and %g %s'%(Nbins, Rmin, Rmax, Runit))

## Import galaxy observables from Bahamas catalog

# Define the list of 'used' galaxies
catnum = 515
lenslist = np.arange(catnum)
#lenslist = np.delete(lenslist, [322,326,648,758,867])
catnum = len(lenslist)
print('Number of Bahamas lenses:', catnum)


# Path to the Bahamas catalog
path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_isolated_new/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
catname = '%s/catalog.dat'%path_cat
catalog = np.loadtxt(catname).T[:,lenslist]

# Import galaxy observables (M200, r200, logmstar)
M200list = 10.**catalog[3] # M200 of each galaxy
r200list = catalog[4] * 1e6/xvalue # r200 of each galaxy (in Xpc)
logmstarlist = catalog[5] # Stellar mass of each lens galaxy
mstarlist = np.reshape(10.**logmstarlist[0:catnum], [catnum,1])


## Import true density profiles

profiles_radius = np.zeros([catnum, Nbins])
profiles_rho = np.zeros([catnum, Nbins])
for c in range(catnum):
    profname = '%s/PROFILES/cluster_%i_rho_profile.dat'%(path_cat, lenslist[c])
    profile_c = np.loadtxt(profname).T
    profiles_radius[c] = profile_c[0,0:Nbins] * r200list[c] # in Xpc
    profiles_rho[c] = profile_c[1,0:Nbins] * rho_crit # in Msun/pc^3

profiles_radius_mean = np.mean(profiles_radius,0)
profiles_rho_mean = np.mean(profiles_rho,0)

"""
## Import Kyle's rho(r) fits

fitsfile = '%s/ESD/deproject_bahamas.npy'%(path_cat)
fitscat = np.load(fitsfile)

# Remove rows with NaN
profiles_rho = profiles_rho[~np.isnan(fitscat).any(axis=1)]
profiles_radius = profiles_radius[~np.isnan(fitscat).any(axis=1)]
fitscat_rho = fitscat[~np.isnan(fitscat).any(axis=1)] / 1e6
catnum = len(fitscat_rho) # Assign new number to catnum

fitscat_rho_mean = np.mean(fitscat_rho,0)

# Compute the difference
difference = (fitscat_rho - profiles_rho) / profiles_rho
mean_diff = np.mean(np.abs(difference))
chi2 = stats.chisquare(fitscat_rho, f_exp=profiles_rho, axis=None)

print('chi^2:', chi2)
print('difference:', difference)
print('mean difference:', mean_diff)

"""

## Plot the results

for i in range(catnum):
    #plt.plot(profiles_radius[i], fitscat_rho[i], color='blue', marker='.', alpha=0.03)
    plt.plot(profiles_radius[i], profiles_rho[i], color='red', marker='.', alpha=0.03)


#plt.plot(profiles_radius_mean, fitscat_rho_mean, color='lightblue', marker='.', label='From density maps (numerical)')
plt.plot(profiles_radius_mean, profiles_rho_mean, color='pink', marker='.', label='True density profiles')

# Define axis labels and legend
xlabel = r'Radius (Mpc)'
ylabel = r'Density $\rho$ $(M_{\odot}/pc^3)$'

plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)
plt.legend()

plt.xscale('log')
plt.yscale('log')

#plt.ylim([1e-13, 1e-9])
#plt.xlim([1e-15, 1e-10])

plt.tight_layout()


plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Feb19/Plots/bahamas_rho_test_numerical'

# Save plot
for ext in ['pdf', 'png']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)


plt.show()
plt.clf

