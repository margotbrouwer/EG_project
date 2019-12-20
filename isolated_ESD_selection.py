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

cat = 'kids'
path_ESDcat = '/data/users/brouwer/Lensing_results/EG_results_Nov19'


if 'kids' in cat:
    ESDcatname = 'catalogs/results_shearcatalog/shearcatalog_ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc/shearcatalog.fits'
if 'gama' in cat:
    ESDcatname = 'catalogs/results_shearcatalog/shearcatalog_ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc/shearcatalog_gama.fits'


# Full directory & name of the corresponding ESD catalogue
ESDcatfile = '%s/%s'%(path_ESDcat, ESDcatname)
ESDcat = pyfits.open(ESDcatfile, memmap=True)[1].data

# List of the observables of all sources in the KiDS catalogue
Rcenter = ESDcat['Rcenter']
wk2 = ESDcat['lfweight_A*k^2']
w2k2 = ESDcat['lfweight_A^2*k^2']
gamma_t = ESDcat['gammat_A']

# Import isolated galaxy catalogue
isocatfile = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_perc_h70.fits'%cat
isocat = pyfits.open(isocatfile, memmap=True)[1].data
dist0p1perc = isocat['dist0p1perc']
logmstar = isocat['logmstar']

nRbins = len(Rcenter[0])
Nlens = len(Rcenter)
print('Number of lenses:', Nlens)

# Select the isolated galaxies
iso_index = np.ndarray.flatten(np.argwhere((dist0p1perc > 3.) & (logmstar < 11.)))

# Find the radius where satellites dominate (2-halo term)
sat_index = np.amin(np.argwhere(Rcenter[0] > 0.5))
# Select 1-halo term
gamma_1h = gamma_t[:, 0:sat_index]
wk2_1h = wk2[:, 0:sat_index]
# Select 2-halo term
gamma_2h = gamma_t[:, sat_index:nRbins]
wk2_2h = wk2[:, sat_index:nRbins]

# Calculate the sum of the 2-halo term for each lens
satsum = np.nansum(gamma_2h, 1)
print('satsum', satsum, np.shape(satsum))

# Sort the lenses by the strength of their 2-halo term
satsort = np.sort(satsum) # Sorted 2-halo terms

# Give a percentage number to each chunk, based on its 2-halo term
satpercindex = np.searchsorted(satsort, satsum)
satperclist = np.linspace(0.,1.,len(satsum))
satperc = satperclist[satpercindex] # Percentage value corresponding to each lens

perc_zero = satperclist[np.argmax(satsort>0.)]

print('satperc', satperc, np.shape(satperc))
print('2-halo value is zero at:', perc_zero)

plt.plot(satperclist, satsort)
#plt.plot(satperc[percmask], satsum[percmask], marker='.', ls='')
plt.axhline(y=0, ls=':')

plt.xlabel('Percentage')
plt.ylabel('2-halo value')

filename_output = '%s/Plots/2halo_distribution'%path_ESDcat

# Save plot
for ext in ['png']:
    plotname = '%s.%s'%(filename_output, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf()

# Select the galaxies based on their percentage
perc_max = np.arange(0.9, 0.99, 0.02)
perc_min = [0.]*len(perc_max)

print(perc_max)

for p in range(len(perc_max)):
    percmask = (perc_min[p] < satperc) & (satperc < perc_max[p])
    lenssel_index = np.ndarray.flatten(np.argwhere(percmask)) # The indices of the selected lenses

    ESDprofile_iso = np.nansum(gamma_t[lenssel_index], 0)/np.nansum(wk2[lenssel_index], 0)
    #ESDprofile_iso = np.nansum(gamma_t[iso_index], 0)/np.nansum(wk2[iso_index], 0)

    plt.plot(Rcenter[0], ESDprofile_iso, marker='.', ls=':', label=r'2-halo signal: %g - %g%%'%(perc_min[p]*100., perc_max[p]*100.))
    print('ESD iso', ESDprofile_iso)

#print('lenssel_index', len(lenssel_index)/Nlens)
#print('satperc[lenssel_index]', satperc[lenssel_index])
#print('satsum[lenssel_index]', satsum[lenssel_index])



ESDprofile = np.nansum(gamma_t, 0)/np.nansum(wk2, 0)
plt.plot(Rcenter[0], ESDprofile, marker='.', ls=':', label=r'All galaxies (100 percent)')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Radius R (Mpc)')
plt.ylabel('ESD')

plt.ylim(1e-1, 1e2)
plt.legend()

filename_output = '%s/Plots/isolated_ESD_selection'%path_ESDcat

# Save plot
for ext in ['png']:
    plotname = '%s.%s'%(filename_output, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
