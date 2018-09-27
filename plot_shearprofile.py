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
Runit = 'Mpc'
h=0.7
Rlog = True
plot = True

path_sheardata = '/data2/brouwer/shearprofile/EG_results_Sep18/mice'

# Input lens selections
path_lenssel = 'lenssel-abs_mag_r-m20-m19p5'
#path_cosmo = np.array(['Rbins-20-0.2-20-Mpc_Zbins-%i'%i for i in [1,20,100]])
path_cosmo = np.array(['Rbins-20-0.2-20-Mpc_Zbins-20%s'%i for i in ['', '_lenssplit']])

esdfiles = np.array(['%s/%s_%s.txt'%(path_sheardata, path_lenssel, path_cosmo[i]) for i in np.arange(len(path_cosmo))])

#datalabels = np.array(['Zbins=%s'%i for i in [1,20,100]])
datalabels = np.array(['Source bins', 'Lens bins'])



# Importing the shearprofiles and lens IDs
data_x, data_y, error_h, error_l = utils.read_esdfiles(esdfiles)
#data_y, error_h, error_l = 4. * G * 3.08567758e16 *\
#    np.array([data_y, error_h, error_l]) # Convert ESD (Msun/pc^2) to acceleration (m/s^2)

filename_output = 'Zbins_test'
path_output = '%s/%s.txt'%(path_sheardata, filename_output)
print('Writing plot:', filename_output)

print(data_y)
print(error_h)

utils.write_plot(data_x, data_y, None, error_h, datalabels, path_output, Runit, Rlog, plot, h)
