#!/usr/bin/python

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import LambdaCDM
import astropy.io.fits as pyfits

import modules_EG as utils

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

# Constants
h = 0.7
O_matter = 0.2793
O_lambda = 0.7207

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)


# Constants
G = const.G.to('pc3 / (M_sun s2)')
c = const.c.to('pc/s')
H0 = 100 * (u.km/u.s)/u.Mpc
H0 = H0.to('s-1')

print('G =', G)
print('c =', c)
print('H0 =', H0)

G = G.value
c = c.value
H0 = H0.value

# Creating the galaxy projected distance bins
nRbins = 20
Rmin = 0.1e3 # in pc
Rmax = 10e6 # in pc

Rbins = 10.**np.linspace(np.log10(Rmin), np.log10(Rmax), nRbins)
Rbins = np.append(Rbins,Rmax)
dR = np.diff(Rbins)
Rcenters = Rbins[0:-1]+0.5*dR


# Define line-of-sight distance bins
zbins = Rbins
zcenters = Rcenters
dz = dR

nxbins = 30
xmin = 0.1e3 # in pc
xmax = 10e6 # in pc
xbins = 10.**np.linspace(np.log10(xmin), np.log10(xmax), nxbins)
xbins = np.append(xbins,xmax)
dx = np.diff(xbins)
xcenters = xbins[0:-1]+0.5*dx
dx = np.diff(xbins)

Alist = pi * Rbins**2
dAlist = np.diff(Alist)



