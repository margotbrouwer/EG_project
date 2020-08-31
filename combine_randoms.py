#!/usr/bin/python

# Import the necessary libraries
import sys

import numpy as np
import os
import random
from astropy.coordinates import SkyCoord

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
import scipy.optimize as optimization
from scipy import stats
import modules_EG as utils

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

from matplotlib import gridspec

#Import constants
pi = np.pi
G = const.G.to('pc3 / (M_sun s2)').value
c = const.c.to('m/s').value
h = 0.7

H0 = h * 100 * (u.km/u.s)/u.Mpc
H0 = H0.to('s-1').value
pc_to_m = 3.08567758e16

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})

#shear = [0.83473222, 1.29898205, 1.78349961, 2.6023282, 3.14144787, 3.96727933, 6.23174432, \
#7.40010758, 11.02089698, 11.98303072, 16.84342999, 19.96224538, 26.28309872, 32.03966264, 37.37300951]

# Define path to the random catalogues
Nrandoms = 50
Runit = 'mps2'
shear_path = '/data/users/brouwer/Lensing_results/EG_results_Aug20'
shear_cosmo = 'ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2/shearcovariance'

# Importing the shear profile (for comparison)
shear_name = 'No_bins/dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0p1_0p5_lw-logmbar_GL'

shearfile = '%s/%s/%s/No_bins_C.txt'%(shear_path, shear_name, shear_cosmo)
sheardata = np.loadtxt(shearfile).T
shear_gammat = sheardata[1]

# Importing the randoms
random_names = ['Randoms/No_bins_random_%i/Z_0p1_0p5_lw-logmbar'%n for n in range(Nrandoms)]

gammat_tot = np.zeros(15)
gammax_tot = np.zeros(15)
error_tot = np.zeros(15)

# For each random file...
for n in range(Nrandoms):
    
    # Load the random shear and errors
    randomfile = '%s/%s/%s/No_bins_C.txt'%(shear_path, random_names[n], shear_cosmo)
    randomdata = np.loadtxt(randomfile).T

    random_Rcenters = randomdata[0]    
    random_gammat = randomdata[1]
    random_gammax = randomdata[2]
    random_error = randomdata[3]
    
    #plt.errorbar(random_Rcenters, random_gammat, yerr = random_error)
    
    gammat_tot = gammat_tot + random_gammat
    gammax_tot = gammax_tot + random_gammax
    error_tot = error_tot + random_error**2.
    
    # Load the random covariance matrix
    #covfile = '%s/%s/No_matrix_C.txt'%(shear_path, random_names[n])
    
gammat_tot = gammat_tot/Nrandoms
gammax_tot = gammax_tot/Nrandoms
error_tot = np.sqrt(error_tot)/Nrandoms

random_bias = randomdata[4]
random_variance = randomdata[5]

print('Random gamma_t:', random_gammat)
print('Random gamma_t error:', random_error)
print()

random_perc = gammat_tot/shear_gammat*100.

print('Percentage of ESD:', random_perc)
print('Mean percentage of ESD:', np.mean(np.abs(random_perc)))

plt.xscale('log')
plt.yscale('log')

plt.errorbar(random_Rcenters, gammat_tot, yerr = error_tot, color='black')
#plt.plot(random_Rcenters, shear_gammat)

plt.show()

# Convert ESD into acceleration
#random_gobs, random_gobs_x, random_gerror = [4.*G*pc_to_m*gammat_tot, 4.*G*pc_to_m*gammax_tot, 4.*G*pc_to_m*error_tot]

#print('Random g_obs:', random_gobs)
#print('Random g_obs error:', random_gerror)

filename = '%s/combined_randoms.txt'%shear_path 

foo = np.zeros(15)
utils.write_stack(filename, random_Rcenters, Runit, gammat_tot, gammax_tot, \
    error_tot, random_bias, foo, foo, foo, random_variance, 0.7)
