#!/usr/bin/python

import astropy.io.fits as pyfits
import gc
import numpy as np
import sys
import os
import time
from glob import glob

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

from matplotlib import gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid.inset_locator import inset_axes

from scipy.stats import chi2, norm
import scipy.optimize as optimization

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['Computer Modern']})

#colors = ['red', 'orange', 'cyan', 'blue', 'green']
colors = ['#d7191c', '#fdae61', '#92c5de', '#0571b0', '#0ba52d']

show = True

# Cosmological parameters
O_matter = 0.315
O_lambda = 1.-O_matter
O_k = 0.
h = 1.

# Constants
G = 4.5171e-30 # pc3 / (Msun s2)
c = 9.7156e-09 # pc/s
H0 = 70 # km/s/Mpc
Cd = (c * H0) / (G * 6) # Msun/pc^2

r = 10e3 # pc
Mb = 1e11 # Msun

# Define redshift range
zrange = np.linspace(0.,3.,100)

# Define hubble parameter
Hz = H0 * np.sqrt(O_matter*(1+zrange)**3 + O_k*(1+zrange)**2 + O_lambda) # km/s/Mpc

Cd = (c * Hz*3.240779289469756e-20 ) / (G * 6) # Msun/pc^2

#Md_r = np.sqrt(Cd * np.gradient(Mb_r * r, r)) * r
Md = np.sqrt(Cd * Mb) * r



# Plot result
fig = plt.figure(figsize=(5,4))

plt.plot(zrange, np.log10(1 + Md/Mb), label='Mtot/M_*')
#plt.plot(zrange, Hz, label='Hubble parameter [m/s/Mpc]')

# Define the labels for the plot

plt.xlabel(r'Redshift', fontsize=14)
#plt.ylabel(r'Hubble parameter (m/s/Mpc)', fontsize=14)

plt.ylim([0., 1.])
#plt.axis([2.,100.,6e-5,1e-3])

#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')

path_filename = '/data2/brouwer/shearprofile/EG_results'
filename = 'hubble_afo_redshift'

for ext in ['pdf']:

    plotfilename = '%s/Plots/%s'%(path_filename, filename)
    plotname = '%s.%s'%(plotfilename, ext)

    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written plot:', plotname)
if show:
    plt.show()
plt.close()
