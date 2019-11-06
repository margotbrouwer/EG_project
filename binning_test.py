#!/usr/bin/python

# Import the necessary libraries
import sys
import numpy as np
import pyfits
import os

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
from scipy import integrate
import scipy.optimize as optimization
import scipy.stats as stats
import modules_EG as utils
from deproject.piecewise_powerlaw import esd_to_rho

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams
from matplotlib import gridspec

windspeed = 8 * np.random.rand(500)
boatspeed = .3 * windspeed**.5 + .2 * np.random.rand(500)
bin_means, bin_edges, binnumber = stats.binned_statistic(windspeed, boatspeed, statistic='median', bins=[1,2,3,4,5,6,7])

plt.figure()
plt.plot(windspeed, boatspeed, 'b.', label='raw data')
plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
            label='binned statistic of data')
plt.legend()
