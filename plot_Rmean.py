#!/usr/bin/python

# Import the necessary libraries
import sys

import numpy as np
import pyfits
import os

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

def bins_to_name(binlims):
    binname = str(binlims).replace(', ', '_')
    binname = str(binname).replace('[', '')
    binname = str(binname).replace(']', '')
    binname = str(binname).replace('.', 'p')
    return(binname)

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})

path_sheardata = '/data/users/brouwer/Lensing_results/EG_results_Mar19'
path_lenssel = np.array([['No_bins_GAMAII/Z_0_0p5-dist0p1perc_3_inf-nQ_3_inf']])
path_cosmo = np.array([['ZB_0p1_0p9-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']])
path_filename = np.array([['shearcatalog/No_bins_A']])


esdfiles = np.array([['%s/%s/%s/%s.txt'%\
	(path_sheardata, path_lenssel[i,j], path_cosmo[i,j], path_filename[i,j]) \
	for j in np.arange(np.shape(path_lenssel)[1])] for i in np.arange(np.shape(path_lenssel)[0]) ])

Nbins = np.shape(esdfiles)
Nsize = np.size(esdfiles)
esdfiles = np.reshape(esdfiles, [Nsize])

print('Plots, profiles:', Nbins)

# Importing the shearprofiles and lens IDs
data_x, Rsrc, data_y, error_h, error_l, Nsrc = utils.read_esdfiles(esdfiles)
data_x, Rsrc, data_y, error_h, error_l, Nsrc = data_x[0], Rsrc[0], data_y[0], error_h[0], error_l[0], Nsrc[0]
difference = np.abs(data_x - Rsrc/1e6)/data_x

print('Rsources:', Rsrc)
print('Rcenter:', data_x)
print('Difference: %g percent'%(np.mean(difference)*100.))

plt.plot((Rsrc/1e6)/data_x)
plt.ylim([0.99, 1.01])

#plt.xscale('log')
#plt.yscale('log')

plt.show()
