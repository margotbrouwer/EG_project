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

# Import constants
G = const.G.to('m3/Msun s2').value
m_per_pc = 3.08567758e16
inf = np.inf
h=0.7

print(G)

Rnames = [30,200,3000] # kpc
#Rrange = np.logspace(np.log10(Rmin), np.log10(Rmax), 50)

Rrange = np.array(Rnames)*1e3*m_per_pc # Convert kpc to meter
Mrange = np.logspace(8.5,12,10)

def calc_gbar(R, Mbar):
    gbar = (G * Mbar)/R**2. 
    return gbar

for r in range(len(Rrange)):
    gbar = calc_gbar(Rrange[r], Mrange)
    plt.plot(Mrange, gbar, label=r'$R=%g$ kpc'%Rnames[r])

plt.axhline(y=1e-12, label=r'$g_{\rm bar,min}$ (McGaugh+2016)', color='black')

plt.legend(loc='best')
plt.xscale('log')
plt.yscale('log')

# Define the labels for the plot
xlabel = r'Galaxy baryonic mass [$h^{-2}_{%g} \, M_{\odot}$]'%(h*100)
ylabel = r'Baryonic radial acceleration $g_{\rm bar}$ [${\rm h_{%g} \, m/s^2}$]'%(h*100)

plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)

plt.show()
