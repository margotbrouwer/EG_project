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

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685
cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

cat = 'kids'

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

# Mask the redshifts
Zmask = (0.<lensZ)&(lensZ<0.5)
lensZ = lensZ[Zmask]

# Create the redshift histogram
Zhist, Zbin_edges = np.histogram(lensZ, 100)
Zbins = Zbin_edges[0:-1]+np.diff(Zbin_edges)

# Compute the random redshifts
randomZ = np.array([])
for z in range(len(Zbins)):
    randombin = np.random.uniform(Zbin_edges[z], Zbin_edges[z+1], Zhist[z])
    randomZ = np.append(randomZ, randombin)

# Plot the redshift histograms
Zhist_random, Zbin_edges = np.histogram(randomZ, 100)

#plt.plot(Zbins, Zhist)
#plt.plot(Zbins, Zhist_random)

plt.scatter(lensZ, randomZ)

plt.show()
