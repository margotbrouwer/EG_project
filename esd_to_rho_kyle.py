#!/usr/bin/python
"""
# This code converts the projected Excess Surface Density profile to the observed radial acceleration
"""
import astropy.io.fits as pyfits
import gc
import numpy as np
import sys
import os
import time
from glob import glob

from astropy import constants as const, units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM
from deproject.piecewise_powerlaw import esd_to_rho

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams
colors = ['#0571b0', '#92c5de', '#d7191c', '#fdae61']

import modules_EG as utils

# Import constants
G = const.G.to('pc3/Msun s2').value
c = const.c.to('pc/s').value
pi = np.pi
inf = np.inf

pc_to_meter = 3.086e16 # meters
h, O_matter, O_lambda = [0.7, 0.325, 0.685]

# Define the Singular Isothermal Sphere profile
def calc_sis(sigma_v, r, R):
    
    gobs_sis = (2.*sigma_v**2.) / r
    Mobs_sis = (2.*sigma_v**2.*r) / G
    rho_sis = sigma_v**2. / (2.*G*pi*r**2.)
    
    sigma_sis = sigma_v**2. / (2.*G*R)
    avsigma_sis = sigma_v**2. / (G*R)
    esd_sis = sigma_v**2. / (2.*G*R)
    
    return gobs_sis, Mobs_sis, rho_sis, sigma_sis, avsigma_sis, esd_sis


# Define the projected (R) and spherical (r) distance bins
Nbins_R = int(20)
Nbins_r = int(20)
Runit = 'kpc'
Rlog = True
Rmin, Rmax = np.array([30., 3000.]) # in pc
rmin, rmax = np.array([30., 3000.]) # in pc

Rbins, Rcenters, Rmin, Rmax, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins_R, Rlog)
rbins, rcenters, rmin, rmax, xvalue = utils.define_Rbins(Runit, rmin, rmax, Nbins_r, Rlog)

Rbins, Rcenters = np.array([Rbins, Rcenters])*xvalue
rbins, rcenters = np.array([rbins, rcenters])*xvalue

# Create a mock Singular Isothermal Sphere profile
sigma_v = 200.e3 / pc_to_meter # to m/s to pc/s
gobs_sis, Mobs_sis, rho_sis, sigma_sis, avsigma_sis, esd_sis = calc_sis(sigma_v, rcenters, Rcenters)

## density profile, not shallower than -1 in outer part!

# Fitting setup
r = rcenters  # 'real' values, not log
R = Rbins  # 'real' values, not log
obs = esd_sis
guess = rho_sis  # density profile, not shallower than -1 in outer part!

print()
print('Rbins: (%i):'%len(R), R)
print('ESD_obs: (%i):'%len(obs), obs)
print()
print('rcenters (%i):'%len(r), r)
print('rho_guess: (%i):'%len(guess), guess)
print()

extrapolate_outer = True
extrapolate_inner = True

inner_extrapolation_type = 'extrapolate'  # or 'flat'
startstep = np.min(-np.diff(np.log(guess))) / 3.  # probably reasonable
minstep = .001  # sets tolerance in fit in terms of Delta log(DeltaSigma)

rho = esd_to_rho(
    obs, guess, r, R,
    extrapolate_inner=extrapolate_inner,
    extrapolate_outer=extrapolate_outer,
    inner_extrapolation_type=inner_extrapolation_type,
    startstep=startstep,
    minstep=minstep,
    verbose=True,
    testwith_rho=rho_sis
)

# Plot the results
for r in range(len(results_num)):
    plt.plot(Rcenters/xvalue, results_an[r], ls='--', color=colors[r], label=labels_an[r])
    plt.plot(Rcenters/xvalue, results_num[r], color=colors[r], label=labels_num[r])
    
    print()
    print('Difference:', labels_an[r], labels_num[r])
    print(np.mean(1.-results_num[r]/results_an[r]))

plt.axvline(x=rmin/xvalue, ls=':', color='black')
plt.axvline(x=rmax/xvalue, ls=':', color='black')

plt.legend()
plt.xscale('log')
plt.yscale('log')

xlabel = r'Radius $R$ (%s/h$_{%g}$)'%(Runit, h*100)
ylabel = r'ESD $\langle\Delta\Sigma\rangle$ [h$_{%g}$ M$_{\odot}$/pc$^2$]'%(h*100)

plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.show()
