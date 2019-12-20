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

cat = 'gama'
path_ESDcat = '/data/users/brouwer/Lensing_results/EG_results_Nov19'

# Select the galaxies based on their percentage
perc_min = 0.
perc_max = 0.25

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
variance = 8.3111e-02

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
w2k2_1h = w2k2[:, 0:sat_index]
# Select 2-halo term
gamma_2h = gamma_t[:, sat_index:nRbins]
wk2_2h = wk2[:, sat_index:nRbins]
w2k2_2h = w2k2[:, sat_index:nRbins]

# Calculate the sum of the 2-halo term for each lens
satsum = np.nansum(gamma_2h, 1)
wk2sum = np.nansum(wk2_2h, 1)
w2k2sum = np.nansum(w2k2_2h, 1)

#"""
# Find the number of chunks in which to divide the lenses
if 'kids' in cat:
    Lchunk = 5000 # Chunk corresponding to S/N=2
if 'gama' in cat:
    Lchunk = 4200 # Chunk corresponding to S/N=2
"""

# Find the number of chunks in which to divide the lenses
if 'kids' in cat:
    Lchunk = 6600 # Chunk corresponding to S/N=2
if 'gama' in cat:
    Lchunk = 5400 # Chunk corresponding to S/N=2
"""

Nchunk = int(Nlens/Lchunk)
print('Chunck length:', Lchunk)
print('Number of chunks:', Nchunk)

# Sort lenses by stellar mass and chunk them together
#mstarsort_index = np.argsort(satsum)
#satsum_mstarsort = satsum[mstarsort_index]
satsum_split = np.array_split(satsum, Nchunk)
satsum_chunked = np.array([np.sum(satsum_split[c]) for c in range(Nchunk)])

wk2sum_split = np.array_split(wk2sum, Nchunk)
wk2sum_chunked = np.array([np.sum(wk2sum_split[c]) for c in range(Nchunk)])

w2k2sum_split = np.array_split(w2k2sum, Nchunk)
w2k2sum_chunked = np.array([np.sum(w2k2sum_split[c]) for c in range(Nchunk)])

Lchunks = np.array([len(satsum_split[c]) for c in range(Nchunk)])

# Calculate the shear/ESD of the 2-halo term of each chunk
satsum_chunked = satsum_chunked/wk2sum_chunked
noise_chunked = (w2k2sum_chunked / wk2sum_chunked**2. * variance)**0.5

print('Mean S/N:', np.mean(satsum_chunked/noise_chunked))

# Sort the chuncks by the strength of their 2-halo signal
satsort_chunked = np.sort(satsum_chunked) # Sorted 2-halo terms

# Give a percentage number to each chunk, based on its 2-halo signal
satpercindex_chunked = np.searchsorted(satsort_chunked, satsum_chunked)
satperclist_chunked = np.linspace(0.,1.,len(satsum_chunked))
satperc_chunked = satperclist_chunked[satpercindex_chunked] # Percentage value corresponding to each lens

percmask_chunked = (perc_min < satperc_chunked) & (satperc_chunked < perc_max)
perc_zero = satperclist_chunked[np.argmax(satsort_chunked>0.)]

#print('satperc_chunked', satperc_chunked, np.shape(satperc_chunked))
print('2-halo value is zero at:', perc_zero)

plt.plot(satperclist_chunked, satsort_chunked)
plt.plot(satperc_chunked[percmask_chunked], satsum_chunked[percmask_chunked], marker='.', ls='')
plt.axhline(y=0, ls=':')

plt.xlabel('Percentage')
plt.ylabel('2-halo value')

filename_output = '%s/Plots/2halo_distribution_%s_chunksize-%i'%(path_ESDcat, cat, Lchunk)

# Save plot
for ext in ['png']:
    plotname = '%s.%s'%(filename_output, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf()


# Assign the appropriate percentage number to each lens
satperc = np.array([])

for c in range(Nchunk):
    satperc = np.append(satperc, [satperc_chunked[c]] * Lchunks[c])

#satperc = np.ndarray.flatten(np.array([[satperc_chunked[c]]*int(Lchunks[c]) for c in range(Nchunk)]))
#mstarsort_reverse = np.searchsorted(satsum, satsum_mstarsort)
#satperc = satperc_Nlens[mstarsort_reverse]

percmask = (perc_min < satperc) & (satperc < perc_max)

lenssel_index = np.ndarray.flatten(np.argwhere(percmask)) # The indices of the selected lenses

ESDprofile_sat = np.nansum(gamma_t[lenssel_index], 0)/np.nansum(wk2[lenssel_index], 0)
errors_sat = (np.nansum(w2k2[lenssel_index], 0) / np.nansum(wk2[lenssel_index], 0)**2. * variance)**0.5

ESDprofile_iso = np.sum(gamma_t[iso_index], 0)/np.sum(wk2[iso_index], 0)
errors_iso = (np.sum(w2k2[iso_index], 0) / np.sum(wk2[iso_index], 0)**2. * variance)**0.5
print(errors_iso)

plt.errorbar(Rcenter[0]*1.05, ESDprofile_sat, errors_sat, marker='.', ls=':', label=r'Lowest 2-halo term: %g - %g'%(perc_min, perc_max))
plt.errorbar(Rcenter[0]*1.1, ESDprofile_iso, errors_iso, marker='.', ls=':', label=r'Isolated galaxies')


print('ESD iso', ESDprofile_iso)


ESDprofile = np.nansum(gamma_t, 0)/np.nansum(wk2, 0)
errors = (np.sum(w2k2, 0)/np.sum(wk2, 0)**2. * variance)**0.5
plt.errorbar(Rcenter[0], ESDprofile, errors, marker='.', ls=':', label=r'All galaxies')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Radius R (Mpc)')
plt.ylabel('ESD')

plt.ylim(1e-1, 1e2)
plt.legend()

filename_output = '%s/Plots/isolated_ESD_selection_%s_chunksize-%i'%(path_ESDcat, cat, Lchunk)

# Save plot
for ext in ['png']:
    plotname = '%s.%s'%(filename_output, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
