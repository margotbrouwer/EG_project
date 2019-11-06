#!/usr/bin/python

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

def gobs_mond(gbar, g0=1.2e-10):
    gobs = gbar / (1 - np.exp( -np.sqrt(gbar/g0) ))
    return gobs

# Import constants
G = const.G.to('pc3/Msun s2').value # in pc3/Msun s2
c = const.c.to('pc/s').value # in pc/s
inf = np.inf
pi = np.pi
pc_to_m = 3.08567758e16 # meters

# Bahamas simulation parameters
Zlens = 0.25
O_matter = 0.2793
O_lambda =  0.7207
sigma8 = 0.821
h=0.7


## Define projected distance bins R

# Creating the Rbins
#Runit, Nbins, Rmin, Rmax = ['Mpc', 15, 0.03, 3.] # Fixed Rbins
#Runit, Nbins, Rmin, Rmax = ['Mpc', 16, -999, 999] # Same R-bins as r from PROFILES
Runit, Nbins, Rmin, Rmax = ['mps2', 15, 1e-15, 5e-12] # gbar-bins

Rbins, Rcenters, Rmin_pc, Rmax_pc, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins, True)
print('R-bins: %i bins between %g and %g %s'%(Nbins, Rmin, Rmax, Runit))

plotunit = 'mps2'
method = 'SIS' # SIS or numerical

gbar_mond = np.logspace(-16, -8, 40)
gbar_ext = np.logspace(-16, -12, 30)
gbar_uni = np.logspace(-16, -8, 50)
gbar_prof = np.logspace(-16, -10, 50)

## Import galaxy observables from Bahamas catalog

# Define the list of 'used' galaxies
catnum = 515
lenslist = np.arange(catnum)
#lenslist = np.delete(lenslist, [322,326,648,758,867])
catnum = len(lenslist)

# Path to the Bahamas catalog
#path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_isolated_new/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
catname = '%s/catalog.dat'%path_cat
catalog = np.loadtxt(catname).T[:,lenslist]

# Import galaxy observables (M200, r200, logmstar)
M200list = 10.**catalog[3] # M200 of each galaxy
r200list = catalog[4] * 1e6/xvalue # r200 of each galaxy (in Xpc)
logmstarlist = catalog[5] # Stellar mass of each lens galaxy
mstarlist = np.reshape(10.**logmstarlist[0:catnum], [catnum,1])

## Import measured Bahamas ESD profiles

# Path to the ESD catalog
catfile = '%s/ESD/Bahamas_ESD_profiles_Rbins-%i_%g-%g%s_isolated.fits'%(path_cat, Nbins, Rmin, Rmax, Runit)
cat = pyfits.open(catfile, memmap=True)[1].data
print('Imported:', catfile)

# Import the ESD profiles
ESD_list = cat['ESD'][0:catnum]
Rbins_list = cat['Rbins'][0:catnum]


## Import true enclosed mass profiles
profbins = 40

profiles_radius = np.zeros([catnum, profbins])
profiles_Menclosed = np.zeros([catnum, profbins])
for c in range(catnum):
    profname = '%s/PROFILES/cluster_%i_Menclosed_profile_types.dat'%(path_cat, lenslist[c])
    profile_c = np.loadtxt(profname).T
    profiles_radius[c] = profile_c[0] * 1e6 # * r200list[c] # in pc # profile_c[0,0:Nbins]
    profiles_Menclosed[c] = profile_c[1]# * M200list[c] # in Msun # profile_c[1,0:Nbins]

# Calculate true gbar and gobs from enclosed mass profiles
profiles_gbar = (G * mstarlist) / (profiles_radius*xvalue)**2. * pc_to_m # in m/s^2
profiles_gobs = (G * profiles_Menclosed) / (profiles_radius*xvalue)**2. * pc_to_m # in m/s^2

gbar_profiles_mean, gobs_profiles_mean, mock_y_std = utils.mean_profile(profiles_gbar, profiles_gobs, profbins, True)

## Calculate gbar and gobs from the ESD profiles

# Calculate gobs using the SIS assumption
if method == 'SIS':
    gobs_list = ESD_list * 4.*G * pc_to_m # Convert ESD (Msun/pc^2) to acceleration (m/s^2)

    # Calculate gbar from R
    gbar_bins = (G * mstarlist) / (Rbins_list*xvalue)**2. * pc_to_m # in m/s^2
    gbar_list = np.array([(gbar_bins[i])[0:-1] + np.diff(gbar_bins[i])/2. for i in range(catnum)])
    
    print(np.shape(gobs_list))
    print(np.nanmean(gobs_list, 0))
    
# Calculate gobs using Kyle's numerical integration method
if method == 'numerical':
    
    # Import Kyle's M(<r) fits
    
    fitsfile = '%s/ESD/mencl_bahamas.npy'%(path_cat)
    fitscat = np.load(fitsfile)
    
    Menc_radii = fitscat[0,:,0:Nbins] # in pc
    Menc_values = fitscat[1,:,0:Nbins] # in Msun
    
    # Remove rows with NaN and/or inf
    profiles_gbar = profiles_gbar[np.isfinite(Menc_values).any(axis=1)]
    profiles_gobs = profiles_gobs[np.isfinite(Menc_values).any(axis=1)]
    profiles_radius = profiles_radius[np.isfinite(Menc_values).any(axis=1)]
    Rbins_list = Rbins_list[np.isfinite(Menc_values).any(axis=1)]
    Menc_radii = Menc_radii[np.isfinite(Menc_values).any(axis=1)]
    mstarlist = mstarlist[np.isfinite(Menc_values).any(axis=1)]

    Menc_values = Menc_values[np.isfinite(Menc_values).any(axis=1)]
    catnum = len(Menc_values) # Assign new number to catnum
    
    print('Menclosed:', Menc_values)
    print()
    print('Radius:', Menc_radii)
    
    gbar_list = (G * mstarlist) / (Menc_radii*xvalue)**2. * pc_to_m # in m/s^2
    gobs_list = (G * Menc_values) / (Menc_radii*xvalue)**2. * pc_to_m # in m/s^2
    
    """
    print('Performing analitical method')
    ## density profile, not shallower than -1 in outer part!
    
    # Create a SIS for the 'guess' ESD(gbar)
    sigma_v = 300.e4 / pc_to_m # velocity dispersion in pc/s
    
    plt.plot()
    
    # Fitting setup
    extrapolate_outer = False
    extrapolate_inner = False
    inner_extrapolation_type = 'extrapolate'  # or 'flat'
    
    for c in [1]:
    
        ESD = ESD_list[c]
        R = Rbins_list[c] * xvalue # 'real' values, not log
        r = profiles_radius[c] * xvalue # 'real' values, not log
        
        print(ESD, len(ESD))
        print(R, len(R))
        print(r, len(r))
        print()
        
        guess = sigma_v**2. / (2.*G*pi*r**2.)
        startstep = np.min(-np.diff(np.log(guess))) / 3.  # probably reasonable
        minstep = .001  # sets tolerance in fit in terms of Delta log(DeltaSigma)
        
        print(ESD/guess)
        
        rho = esd_to_rho(ESD, guess, r, R,
            extrapolate_inner=extrapolate_inner,
            extrapolate_outer=extrapolate_outer,
            inner_extrapolation_type=inner_extrapolation_type,
            startstep=startstep, minstep=minstep, verbose=True)
        
        plt.xscale('log')
        plt.yscale('log')
        
        plt.plot(r, ESD)
        plt.plot(r, guess)
        plt.plot(r, rho)
        plt.show()
    """
    
## Plotting the result

# Define the labels for the plot
if 'pc' in plotunit:
    for i in range(catnum):
        Rbins_centers = (Rbins_list[i])[0:-1] + np.diff(Rbins_list[i])/2.
        plt.plot(Rbins_centers, ESD_list[i])
        
    xlabel = r'Radius R [%s]'%Runit
    ylabel = r'Excess Surface Density [$M_\odot/pc^2$]'

else:
    for i in range(catnum):
    
        plt.plot(gbar_list[i], gobs_list[i], color='blue', marker='.', alpha=0.03)
        plt.plot(profiles_gbar[i], profiles_gobs[i], color='red', marker='.', alpha=0.03)
    
    # Mean gobs from the density maps
    gbar_maps_mean = np.nanmean(gbar_list, 0) 
    gobs_maps_mean = np.nanmean(gobs_list, 0)

    plt.plot(gbar_maps_mean, gobs_maps_mean, color='darkblue', marker='.', label='From density maps (%s)'%method)
    plt.plot(gbar_profiles_mean, gobs_profiles_mean, color='darkred', marker='.', label='Calculated from mass profiles')
    
    plt.plot(gbar_uni, gbar_uni, ls=':', color='grey')
    plt.plot(gbar_mond, gobs_mond(gbar_mond), ls='--', color='black')
    
    """
    #chi2 = np.sum((gobs_list - profiles_gobs)**2. / profiles_gobs)
    difference = (gobs_list - profiles_gobs) / profiles_gobs
    mean_diff = np.mean(np.abs(difference))
    chi2 = stats.chisquare(gobs_list, f_exp=profiles_gobs, axis=None)
    
    print('chi^2:', chi2)
    print('difference:', difference)
    print('mean difference:', mean_diff)
    """

# Define axis labels and legend
xlabel = r'Expected baryonic acceleration [$m/s^2$]'
ylabel = r'Observed radial acceleration [$m/s^2$]'

plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)
plt.legend()

plt.xscale('log')
plt.yscale('log')

plt.xlim([5e-16, 1e-9])
plt.ylim([1e-14, 1e-9])

plt.tight_layout()

plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Sep19/Plots/bahamas_RAR_test_%s'%method

# Save plot
for ext in ['pdf', 'png']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written plot:', plotname)

plt.show()
plt.clf

"""
## Plot enclosed radii

for i in range(catnum):

    plt.plot(Menc_radii[i], Menc_values[i], color='blue', marker='.', alpha=0.03)
    plt.plot(profiles_radius[i], profiles_Menclosed[i], color='red', marker='.', alpha=0.03)

# Define axis labels and legend
xlabel = r'Radius [Mpc]'
ylabel = r'Enclosed mass [$M_{\odot}$]'

# Mean gobs from the density maps
Menc_radii_mean = np.mean(Menc_radii, 0)
Menc_values_mean = np.mean(Menc_values, 0) 

profiles_radius_mean = np.mean(profiles_radius, 0)
profiles_Menclosed_mean = np.mean(profiles_Menclosed, 0) 

plt.plot(Menc_radii_mean, Menc_values_mean, color='blue', marker='.', label='From density maps (%s)'%method)
plt.plot(profiles_radius_mean, profiles_Menclosed_mean, color='red', marker='.', label='Calculated from mass profiles')

plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)

plt.xscale('log')
plt.yscale('log')
plt.legend()

plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Feb19/Plots/bahamas_Menclosed_test_%s'%method

# Save plot
for ext in ['pdf']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written plot:', plotname)

plt.show()
plt.clf
"""
