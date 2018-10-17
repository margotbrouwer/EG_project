#!/usr/bin/python

# Import the necessary libraries
import sys

import numpy as np
import pyfits
import os

from astropy import constants as const, units as u
import scipy.optimize as optimization
import modules_EG as utils

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

from matplotlib import gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#import matplotlib.style
#matplotlib.style.use('classic')

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['Computer Modern']})


#Import constants
G = const.G.to('pc3/Msun s2').value
c = const.c.to('pc/s')
h=0.7
inf = np.inf

# Colours
# Blue, green, turquoise, cyan
blues = ['#332288', '#44AA99', '#117733', '#88CCEE']

# Light red, Red, light pink, pink
reds = ['#CC6677', '#882255', '#CC99BB', '#AA4499']
#colors = np.array([reds,blues])

colors = ['#0571b0', '#92c5de', '#d7191c', '#fdae61']

# Defining the paths to the data
blind = 'A'

#Import constants
Runit = 'mps2'
Rlog = True
plot = True


filename_output = 'mice_RAR_ESD'
path_sheardata = '/data2/brouwer/shearprofile/Lensing_results/EG_results_Oct18'

# Input lens selections
#path_lenssel = np.array(['No_bins/logmstar_8p5_11_lw-logmbar', 'No_bins/logmstar_8p5_11_lw-logmbar'])# 'No_bins/lmstellar_8p5_11_lw-lmstellar'])
#path_lenssel = np.array(['No_bins/logmstar_8p5_11', 'No_bins/lmstellar_8p5_11'])
path_lenssel = np.array(['No_bins/logmstar_8p5_11_lw-logmbar', 'No_bins/lmstellar_8p5_11_lw-lmstellar'])


#path_cosmo = np.array(['Rbins-20-0.2-20-Mpc_Zbins-%i'%i for i in [1,20,100]])
#path_cosmo = np.array(['%s_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_20_2000_kpc'%z for z in ['ZB', 'zcgal']])
path_cosmo = np.array(['%s_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins20_1em15_5em12_mps2'%z for z in ['ZB', 'zcgal']])
#path_cosmo = np.array(['ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins20_1em15_5em12_%s'%z for z in ['pc', 'mps2']])


#esdfiles = np.array(['%s/%s_%s.txt'%(path_sheardata, path_lenssel, path_cosmo[i]) for i in np.arange(len(path_cosmo))])
esdfiles = np.array(['%s/%s/%s/shearcovariance/No_bins_A.txt'%\
(path_sheardata, path_lenssel[i], path_cosmo[i]) for i in np.arange(len(path_lenssel))])

print('Plotting:', esdfiles)

#datalabels = np.array(['Zbins=%s'%i for i in [1,20,100]])
datalabels = [r'KiDS-GAMA ($M_*<10^{11} M_{\odot}$)', r'MICE ($M_*<10^{11} M_{\odot}$)']
#datalabels = [r'KiDS-GAMA (New)', r'KiDS-GAMA (Old)']


# Importing the shearprofiles and lens IDs
data_x, data_y, error_h, error_l = utils.read_esdfiles(esdfiles)
#if Runit == 'mps2':
#    data_y, error_h, error_l = 4. * G * 3.08567758e16 *\
#        np.array([data_y, error_h, error_l]) # Convert ESD (Msun/pc^2) to acceleration (m/s^2)


path_output = '%s/%s'%(path_sheardata, filename_output)
print('Writing plot:', filename_output)


utils.write_plot(data_x, data_y, None, error_h, datalabels, path_output, Runit, Rlog, plot, h)
