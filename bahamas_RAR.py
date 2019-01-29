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
Runit, Nbins, Rmin, Rmax = ['Mpc', 16, -999, 999] # Same R-bins as PROFILES
#Runit, Nbins, Rmin, Rmax = ['mps2', 20, 1e-15, 5e-12] # gbar-bins

Rbins, Rcenters, Rmin_pc, Rmax_pc, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins, True)
print('R-bins: %i bins between %g and %g %s'%(Nbins, Rmin, Rmax, Runit))

plotunit = 'mps2'

gbar_mond = np.logspace(-15, -8, 50)
gbar_ext = np.logspace(-15, -12, 30)
gbar_uni = np.logspace(-15, -8, 50)

## Import Bahamas ESD catalog

path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
catfile = '%s/ESD/ESD_profiles_Rbins-%i_%g-%g%s.fits'%(path_cat, Nbins, Rmin, Rmax, Runit)
print('Imported:', catfile)


# Import galaxy observables
catname = '%s/catalog.dat'%path_cat
catalog = np.loadtxt(catname).T
M200list = 10.**catalog[3] # M200 of each galaxy
r200list = catalog[4] # r200 of each galaxy
logmstarlist = catalog[5] # Stellar mass of each lens galaxy


# Import density profiles
catnum = 3 #1039
profiles_radius = np.zeros([catnum, 20])
profiles_Menclosed = np.zeros([catnum, 20])
for c in range(catnum):
#for c in lenslist:
    profname = '%s/PROFILES/cluster_%i_Menclosed_profile.dat'%(path_cat, c)
    profile_c = np.loadtxt(profname).T
    profiles_radius[c] = profile_c[0] * r200list[c] * 1e6 # in pc
    profiles_Menclosed[c] = profile_c[1] * M200list[c] # in Msun

mstarlist = np.reshape(10.**logmstarlist[0:catnum], [catnum,1])
profiles_gbar = (G * mstarlist) / profiles_radius**2. * 3.08567758e16 # in m/s^2
profiles_gobs = (G * profiles_Menclosed) / profiles_radius**2. * 3.08567758e16 # in m/s^2

# Import the ESD
cat = pyfits.open(catfile, memmap=True)[1].data
ESD_list = cat['ESD'][0:catnum]
print(ESD_list)
Rbins_list = cat['Rbins_pc'][0:catnum]
print(Rbins_list)
gbar_list = (G * mstarlist) / Rbins_list**2. * 3.08567758e16 # in m/s^2

print('ESD_list:', ESD_list)

#data_x, data_y, error_h, error_l = utils.read_esdfiles(esdfiles)
gobs_list = ESD_list * 4.*G*3.08567758e16 # Convert ESD (Msun/pc^2) to acceleration (m/s^2)

## Ploting the result

# Define the labels for the plot
if 'pc' in plotunit:
    for i in range(catnum):
        #Rbins_centers = (Rbins_list[i])[0:-1] + np.diff(Rbins_list[i])/2.
        #rho = ESD_list[i] * 4. * Rbins_centers^2.
        Rbins_centers = (Rbins_list[i])[1:Nbins]
        plt.plot(ESD_list[i])
        #plt.plot(Rbins_centers/1e6, ESD_list[i])
        
    xlabel = r'Radius R [%s]'%Runit
    ylabel = r'Excess Surface Density [$M_\odot/pc^2$]'

else:
    for i in range(catnum):
        
        plt.plot(gbar_list[i], gobs_list[i], color='blue')
        plt.plot(profiles_gbar[i], profiles_gobs[i], color='red')
    
    """
    gbar_profiles_mean = np.mean(profiles_gbar, 0)
    gobs_profiles_mean = np.mean(profiles_gobs, 0)
    
    gbar_maps_mean = Rcenters
    gobs_maps_mean = np.mean(gobs_list, 0)
    
    plt.plot(gbar_maps_mean, gobs_maps_mean)
    plt.plot(gbar_profiles_mean, gobs_profiles_mean)
    """
    
    plt.plot(gbar_mond, gobs_mond(gbar_mond), ls='--')
    plt.plot(gbar_mond, gbar_mond, ls=':')
    
    xlabel = r'Expected baryonic acceleration [$m/s^2$]'
    ylabel = r'Observed radial acceleration [$m/s^2$]'
    

plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)

plt.xscale('log')
plt.yscale('log')

plt.show()




plt.show()
