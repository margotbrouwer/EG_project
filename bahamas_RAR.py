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
pc_to_meter = 3.086e16 # meters

# Bahamas simulation parameters
Zlens = 0.25
O_matter = 0.2793
O_lambda =  0.7207
sigma8 = 0.821
h=0.7


## Define projected distance bins R

# Creating the Rbins
#Runit, Nbins, Rmin, Rmax = ['Mpc', 20, 0.03, 3.] # Fixed Rbins
Runit, Nbins, Rmin, Rmax = ['Mpc', 16, -999, 999] # Same R-bins as r from PROFILES
#Runit, Nbins, Rmin, Rmax = ['mps2', 20, 1e-15, 5e-12] # gbar-bins

Rbins, Rcenters, Rmin_pc, Rmax_pc, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins, True)
print('R-bins: %i bins between %g and %g %s'%(Nbins, Rmin, Rmax, Runit))

plotunit = 'mps2'
method = 'numerical' # SIS or numerical

gbar_mond = np.logspace(-16, -8, 40)
gbar_ext = np.logspace(-16, -12, 30)
gbar_uni = np.logspace(-16, -8, 50)


## Import galaxy observables from Bahamas catalog

# Define the list of 'used' galaxies
catnum = 402 #1039
lenslist = np.arange(catnum)
lenslist = np.delete(lenslist, [322,326,648,758,867])
catnum = len(lenslist)

# Path to the Bahamas catalog
path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
catname = '%s/catalog.dat'%path_cat
catalog = np.loadtxt(catname).T[:,lenslist]

# Import galaxy observables (M200, r200, logmstar)
M200list = 10.**catalog[3] # M200 of each galaxy
r200list = catalog[4] * 1e6/xvalue # r200 of each galaxy (in Xpc)
logmstarlist = catalog[5] # Stellar mass of each lens galaxy
mstarlist = np.reshape(10.**logmstarlist[0:catnum], [catnum,1])


## Import measured Bahamas ESD profiles

# Path to the ESD catalog
catfile = '%s/ESD/ESD_profiles_Rbins-%i_%g-%g%s.fits'%(path_cat, Nbins, Rmin, Rmax, Runit)
cat = pyfits.open(catfile, memmap=True)[1].data
print('Imported:', catfile)

# Import the ESD profiles
ESD_list = cat['ESD'][0:catnum]
Rbins_list = cat['Rbins_pc'][0:catnum]


## Import true enclosed mass profiles

profiles_radius = np.zeros([catnum, Nbins])
profiles_Menclosed = np.zeros([catnum, Nbins])
for c in range(catnum):
#for c in range(320, 326):
    profname = '%s/PROFILES/cluster_%i_Menclosed_profile.dat'%(path_cat, lenslist[c])
    profile_c = np.loadtxt(profname).T
    profiles_radius[c] = profile_c[0,0:Nbins] * r200list[c] # in Xpc
    profiles_Menclosed[c] = profile_c[1,0:Nbins] * M200list[c] # in Msun

# Calculate true gbar and gobs from enclosed mass profiles
profiles_gbar = (G * mstarlist) / (profiles_radius*xvalue)**2. * 3.08567758e16 # in m/s^2
profiles_gobs = (G * profiles_Menclosed) / (profiles_radius*xvalue)**2. * 3.08567758e16 # in m/s^2

gbar_profiles_mean = np.mean(profiles_gbar, 0)
gobs_profiles_mean = np.mean(profiles_gobs, 0)


## Calculate gbar and gobs from the ESD profiles

# Calculate gbar from R
gbar_list = (G * mstarlist) / (Rbins_list*xvalue)**2. * 3.08567758e16 # in m/s^2
#gbar_centers = np.array([(gbar_list[i])[0:-1] + np.diff(gbar_list[i])/2. for i in range(catnum)])
#gbar_maps_mean = np.mean(gbar_centers, 0)


# Calculate gobs using the SIS assumption
if method == 'SIS':
    gobs_list = ESD_list * 4.*G*3.08567758e16 # Convert ESD (Msun/pc^2) to acceleration (m/s^2)

# Calculate gobs using Kyle's numerical integration method
if method == 'numerical':
    
    # Import Kyle's rho(r) fits
    
    fitsfile = '%s/ESD/deproject_bahamas.npy'%(path_cat)
    fitscat = np.load(fitsfile)
    
    # Remove rows with NaN
    profiles_gbar = profiles_gbar[~np.isnan(fitscat).any(axis=1)]
    profiles_gobs = profiles_gobs[~np.isnan(fitscat).any(axis=1)]
    profiles_radius = profiles_radius[~np.isnan(fitscat).any(axis=1)]
    Rbins_list = Rbins_list[~np.isnan(fitscat).any(axis=1)]
    fitscat = fitscat[~np.isnan(fitscat).any(axis=1)]
    catnum = len(fitscat) # Assign new number to catnum
    
    # Calculate the enclosed mass using numerical integration
    #Mobs_list = 4.*pi * np.array([integrate.cumtrapz(fitscat[c]*profiles_radius[c]**2., \
    #            profiles_radius[c], initial=0.) for c in range(catnum)])
    
    Mbins_list = np.array([fitscat[c] * 4.*pi*profiles_radius[c]**2.*np.diff(Rbins_list[c]) \
                 for c in range(catnum)])
                 
    Mobs_list = np.cumsum(Mbins_list, 1)
    gobs_list = (G * Mobs_list) / (profiles_radius)**2. * 3.08567758e16 # in m/s^2
    
    print(fitscat)
    print()
    print(Mobs_list)
    
    """
    print('Performing analitical method')
    ## density profile, not shallower than -1 in outer part!
    
    # Create a SIS for the 'guess' ESD(gbar)
    sigma_v = 300.e4 / pc_to_meter # velocity dispersion in pc/s
    
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
    
        plt.plot(profiles_gbar[i], gobs_list[i], color='blue', marker='.', alpha=0.03)
        plt.plot(profiles_gbar[i], profiles_gobs[i], color='red', marker='.', alpha=0.03)
    
    # Mean gobs from the density maps
    gobs_maps_mean = np.mean(gobs_list, 0)    
    
    plt.plot(gbar_profiles_mean, gobs_maps_mean, color='blue', marker='.', label='From density maps (%s)'%method)
    plt.plot(gbar_profiles_mean, gobs_profiles_mean, color='red', marker='.', label='Calculated from mass profiles')
    
    plt.plot(gbar_uni, gbar_uni, ls=':', color='grey')
    plt.plot(gbar_mond, gobs_mond(gbar_mond), ls='--', color='black')
        
    #chi2 = np.sum((gobs_list - profiles_gobs)**2. / profiles_gobs)
    difference = (gobs_list - profiles_gobs) / profiles_gobs
    mean_diff = np.mean(np.abs(difference))
    chi2 = stats.chisquare(gobs_list, f_exp=profiles_gobs, axis=None)
    
    print('chi^2:', chi2)
    print('difference:', difference)
    print('mean difference:', mean_diff)


# Define axis labels and legend
xlabel = r'Expected baryonic acceleration [$m/s^2$]'
ylabel = r'Observed radial acceleration [$m/s^2$]'

plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)
plt.legend()

plt.xscale('log')
plt.yscale('log')

plt.ylim([1e-13, 1e-9])
plt.xlim([1e-15, 1e-10])

plt.tight_layout()

plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Feb19/Plots/bahamas_RAR_test_%s'%method

# Save plot
for ext in ['pdf', 'png']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf

