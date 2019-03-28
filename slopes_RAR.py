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

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})

# Colours
# Blue, green, turquoise, cyan
blues = ['#332288', '#44AA99', '#117733', '#88CCEE']

# Light red, Red, light pink, pink
reds = ['#CC6677', '#882255', '#CC99BB', '#AA4499']
#colors = np.array([reds,blues])

#colors = ['#0571b0', '#92c5de', '#d7191c']*2#, '#fdae61']
colors = ['#0571b0', '#92c5de', '#d7191c', '#fdae61']*2


# Defining the paths to the data
blind = 'A'

#Import constants
pi = np.pi
G = const.G.to('pc3/Msun s2').value
c = const.c.to('m/s').value
H0 = 2.269e-18 # 70 km/s/Mpc in 1/s
h=0.7

def calc_gobs_mond(gbar, g0=1.2e-10):
    gobs_mond = gbar / (1 - np.exp( -np.sqrt(gbar/g0) ))
    return gobs_mond

def gobs_verlinde(gbar):
    gobs = gbar + np.sqrt((c*H0)/6) * np.sqrt(gbar)
    return gobs

gbar_mcgaugh = np.logspace(-15, -9, 50)
gbar_ext = np.logspace(-15, -12, 30)
gbar_short = np.logspace(-13, -8, 50)


# rho = const
def calc_gobs_0(gbar):
    gD_0 = np.sqrt((c*H0)/6) * np.sqrt(4*gbar)
    gobs_0 = gbar + gD_0
    return gD_0, gobs_0

# rho = r^{-1} ?
def calc_gobs_1(gbar):
    gD_1 = np.sqrt((c*H0)/6) * np.sqrt(3*gbar)
    gobs_1 = gbar + gD_1
    return gD_1, gobs_1

# rho = r^{-2}?
def calc_gobs_2(gbar):
    gD_2 = np.sqrt((c*H0)/6) * np.sqrt(2*gbar)
    gobs_2 = gbar + gD_2
    return gD_2, gobs_2

# rho = r^{-2}
def calc_gobs_3(gbar):
    gD_3 = np.sqrt((c*H0)/6) * np.sqrt(gbar)
    gobs_3 = gbar + gD_3
    return gD_3, gobs_3

def calc_gobs_point(gbar):
    gD_point = np.sqrt((c*H0)/6) * np.sqrt(gbar)
    gobs_point = gbar + gD_point
    return gD_point, gobs_point

gD_0, gobs_0 = calc_gobs_0(gbar_mcgaugh)
gD_1, gobs_1 = calc_gobs_1(gbar_mcgaugh)
gD_2, gobs_2 = calc_gobs_2(gbar_mcgaugh)
gD_3, gobs_3 = calc_gobs_3(gbar_mcgaugh)
gD_point, gobs_point = calc_gobs_point(gbar_mcgaugh)
gobs_mond = calc_gobs_mond(gbar_mcgaugh)

gD = np.array([gD_0, gD_2, gD_point])
gobs = np.array([gobs_0, gobs_2, gobs_point, gobs_mond])
labels = np.array([r'Flat ($\rho(r)$ = const.)', r'SIS ($\rho(r)\sim 1/r^2$)', \
    r'Point mass ($M_b(r)$=const.)', 'MOND relation (McGaugh et al. 2016)'])

plt.xscale('log')
plt.yscale('log')

plt.plot(gbar_short, gbar_short)

for i in np.arange(len(gobs)):
    #plt.plot(gbar_mcgaugh, gD[i])
    plt.plot(gbar_mcgaugh, gobs[i], label=labels[i])

plt.legend()
    
plt.show()
