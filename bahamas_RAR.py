#!/usr/bin/python

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

def gobs_mond(gbar, g0=1.2e-10):
    gobs = gbar / (1 - np.exp( -np.sqrt(gbar/g0) ))
    return gobs

# Import constants
G = const.G.to('pc3/Msun s2').value # in pc3/Msun s2
c = const.c.to('pc/s').value # in pc/s
inf = np.inf
pi = np.pi

Zlens = 0.25
O_matter = 0.2793
O_lambda =  0.7207
sigma8 = 0.821
h=0.7
#H0 = h * 100 * 3.08567758e19 # in (km/s)/Mpc
#H = H0 * np.sqrt(O_matter**-3. + O_lambda)
#rho_crit = (3*H**2.)/(8*pi*G)

## Define projected distance bins R

# Creating the Rbins
#Runit, Nbins, Rmin, Rmax = ['Mpc', 20, 0.03, 3.] # Fixed Rbins
Runit, Nbins, Rmin, Rmax = ['Mpc', 14, -999, 999] # Same R-bins as PROFILES
#Runit, Nbins, Rmin, Rmax = ['mps2', 20, 1e-15, 5e-12] # gbar-bins

Rbins, Rcenters, Rmin_pc, Rmax_pc, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins, True)
print('R-bins: %i bins between %g and %g %s'%(Nbins, Rmin, Rmax, Runit))

plotunit = 'mps2'

gbar_mond = np.logspace(-16, -8, 40)
gbar_ext = np.logspace(-16, -12, 30)
gbar_uni = np.logspace(-16, -8, 50)

## Import Bahamas ESD catalog

path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
catfile = '%s/ESD/ESD_profiles_Rbins-%i_%g-%g%s.fits'%(path_cat, Nbins, Rmin, Rmax, Runit)
print('Imported:', catfile)

catnum = 402 #1039
lenslist = np.arange(catnum)
lenslist = np.delete(lenslist, [322,326,648,758,867])
catnum = len(lenslist)

# Import galaxy observables
catname = '%s/catalog.dat'%path_cat
catalog = np.loadtxt(catname).T[:,lenslist]
print(np.shape(catalog))
M200list = 10.**catalog[3] # M200 of each galaxy
r200list = catalog[4] * 1e6/xvalue # r200 of each galaxy (in Xpc)
logmstarlist = catalog[5] # Stellar mass of each lens galaxy

# Import density profiles
profiles_radius = np.zeros([catnum, Nbins])
profiles_Menclosed = np.zeros([catnum, Nbins])
for c in range(catnum):
#for c in range(320, 326):
    profname = '%s/PROFILES/cluster_%i_Menclosed_profile.dat'%(path_cat, lenslist[c])
    profile_c = np.loadtxt(profname).T
    profiles_radius[c] = profile_c[0,0:Nbins] * r200list[c] # in Xpc
    profiles_Menclosed[c] = profile_c[1,0:Nbins] * M200list[c] # in Msun

mstarlist = np.reshape(10.**logmstarlist[0:catnum], [catnum,1])
profiles_gbar = (G * mstarlist) / (profiles_radius*xvalue)**2. * 3.08567758e16 # in m/s^2
profiles_gobs = (G * profiles_Menclosed) / (profiles_radius*xvalue)**2. * 3.08567758e16 # in m/s^2

# Import the ESD
cat = pyfits.open(catfile, memmap=True)[1].data
ESD_list = cat['ESD'][0:catnum]
Rbins_list = cat['Rbins_pc'][0:catnum]
gbar_list = (G * mstarlist) / (Rbins_list*xvalue)**2. * 3.08567758e16 # in m/s^2
gbar_centers = np.array([(gbar_list[i])[0:-1] + np.diff(gbar_list[i])/2. for i in range(catnum)])

#data_x, data_y, error_h, error_l = utils.read_esdfiles(esdfiles)
gobs_list = ESD_list * 4.*G*3.08567758e16 # Convert ESD (Msun/pc^2) to acceleration (m/s^2)

## Ploting the result

# Define the labels for the plot
if 'pc' in plotunit:
    for i in range(catnum):
        Rbins_centers = (Rbins_list[i])[0:-1] + np.diff(Rbins_list[i])/2.
        plt.plot(Rbins_centers, ESD_list[i])
        
    xlabel = r'Radius R [%s]'%Runit
    ylabel = r'Excess Surface Density [$M_\odot/pc^2$]'

else:
    for i in range(catnum):
    
        plt.plot(gbar_centers[i], gobs_list[i], color='blue', marker='.', alpha=0.03)
        plt.plot(profiles_gbar[i], profiles_gobs[i], color='red', marker='.', alpha=0.03)
        
    #plt.hist2d(np.ndarray.flatten(profiles_gbar), np.ndarray.flatten(profiles_gobs), [gbar_mond, gbar_mond])
    #plt.hist2d(gbar_centers, gobs_list, [gbar_mond, gbar_mond])
    
    
    gbar_profiles_mean = np.mean(profiles_gbar, 0)
    gobs_profiles_mean = np.mean(profiles_gobs, 0)
    
    gbar_maps_mean = np.mean(gbar_centers, 0)
    gobs_maps_mean = np.mean(gobs_list, 0)
    
    plt.plot(gbar_maps_mean, gobs_maps_mean, color='blue', marker='.', label='From density maps (SIS assumption)')
    plt.plot(gbar_profiles_mean, gobs_profiles_mean, color='red', marker='.', label='Calculated from mass profiles')
    
    plt.plot(gbar_uni, gbar_uni, ls=':', color='grey')
    plt.plot(gbar_mond, gobs_mond(gbar_mond), ls='--', color='black')
    
    xlabel = r'Expected baryonic acceleration [$m/s^2$]'
    ylabel = r'Observed radial acceleration [$m/s^2$]'
    
    #chi2 = np.sum((gobs_list - profiles_gobs)**2. / profiles_gobs)
    difference = (gobs_maps_mean - gobs_profiles_mean) / gobs_profiles_mean
    mean_diff = np.mean(np.abs((gobs_maps_mean - gobs_profiles_mean) / gobs_profiles_mean))
    chi2 = stats.chisquare(gobs_list, f_exp=profiles_gobs, axis=None)
    
    print('chi^2:', chi2)
    print('difference:', difference)
    print('mean difference:', mean_diff)


plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)
plt.legend()

plt.xscale('log')
plt.yscale('log')

plt.ylim([1e-13, 1e-9])
plt.xlim([1e-15, 1e-10])

plt.tight_layout()

plotfilename = 'bahamas_SIS_test'

# Save plot
for ext in ['pdf']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf

