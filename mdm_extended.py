#!/usr/bin/python

# Import the necessary libraries

import numpy as np
import pyfits
import os
from matplotlib import pyplot as plt
from scipy import stats

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

import scipy.constants as const
from scipy.integrate import cumtrapz, trapz, quad
from scipy import special as sp

from astropy.modeling.models import Sersic1D

#import astropy.units as u
#import astropy.constants as const


# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['Computer Modern']})

O_matter = 0.315
O_lambda = 1.-O_matter
Ok = 0.
h = 1.


# Constants
G = 4.5171e-30 # pc3 / (Msun s2)
c = 9.7156e-09 # pc/s
H0 = 70 * 3.240779289469756e-20 # km/s/Mpc -> 1/s
Cd = (c * H0) / (G * 6)

pi = np.pi
inf = np.inf

Mb = 10**10 # Msun

# Creating the galaxy projected distance bins
nRbins = 1000
Rmin = 0.1e3 # in pc
Rmax = 1000e3 # in pc

Rbins = 10.**np.linspace(np.log10(Rmin), np.log10(Rmax), nRbins)
Rbins = np.append(Rbins,Rmax)
Rcenters = np.array([(Rbins[r] + Rbins[r+1])/2 for r in xrange(nRbins)])
dR = np.diff(Rbins)

"""
# Power law mass distribution
def power_Mb(Mb, n, r):

    int_rho = trapz(r**n * r**2, x = r)	
    rho_0 = Mb / (4. * pi * int_rho)
    rhob_r = rho_0 * r**n

    Mb_r = cumtrapz(4. * pi * rhob_r * r**2, x = r)
    Mb_r = np.append(Mb_r, Mb_r[-1])

    return rho_0, rhob_r, Mb_r

# Jaffe mass distribution
def jaffe_Mb(Mb, r0, r):

    int_rho = trapz((r/r0)**-2 * (1+r/r0)**-2 / (4 * pi) * r**2, x = r)	
    rho_0 = Mb / (4. * pi * int_rho)

    rhob_r = rho_0 * (r/r0)**-2 * (1+r/r0)**-2 / (4 * pi)
    Mb_r = cumtrapz(4. * pi * rhob_r * r**2, x = r)
    Mb_r = np.append(Mb_r, Mb_r[-1])

    return rho_0, rhob_r, Mb_r

# Sersic mass distribution
def calc_Mb(Mb, r, r_e, n, b_n): # Masses in Msun, radii in pc

    # the density rho_e is defined such that half the mean M_* lies within r_e
    #    relist = np.logspace(np.log10(Rmin), np.log10(r_e), 100)
    #    int_rho = trapz(np.exp(-b_n * ((relist/r_e)**(1/n) - 1)) * relist**2, x = relist)
    #    rho_e = (Mb/2.) / (4 * pi * int_rho)

    int_rho = trapz(np.exp(-b_n * ((r/r_e)**(1/n) - 1)) * r**2, x = r)
    rho_e = Mb / (4. * pi * int_rho)

    rhob_r = rho_e * np.exp(-b_n * ((r/r_e)**(1/n) - 1))
    Mb_r = cumtrapz(4. * pi * rhob_r * r**2, x = r)
    Mb_r = np.append(Mb_r, Mb_r[-1])

    return rho_e, rhob_r, Mb_r
"""

def calc_Md(Mb_r, r, dr):

    H = H0# * np.sqrt(O_matter*(1+z)**3 + O_lambda)
    Cd = (c * H) / (G * 6)
    Md_r = (Cd * np.gradient(Mb_r * r, dr))**0.5 * r
    rhod_r = np.gradient(Md_r, dr) / (4. * pi * r**2)

    return Md_r, rhod_r
    
def calc_radvel(M_r, r):

    vel_r = (G * M_r / r)**0.5
    acc_r = G * M_r / r**2
    
    return vel_r, acc_r

Nlist = [-2, -1]
print(Nlist)

for N in Nlist:
    
    n, b_n, r_e = [0.6, , 2.2e3]
    Mb_r = calc_Mb(Mb, r, r_e, n, b_n)

    Md_r, rhod_r = calc_Md(Mb_r, Rcenters, dR)
    Md_an = (Cd * Mb_an * Rcenters**2 * (N+4))**0.5

    vb_r, ab_r = calc_radvel(Mb_r, Rcenters)
    vd_r, ad_r = calc_radvel(Md_r, Rcenters)
    vd_an, ad_an = calc_radvel(Md_an, Rcenters)

    # Plot the result

    #plt.autoscale(enable=False, axis='both', tight=None)
    #plt.axis([2e1,2e3,-1e-1,5e2])
    #plt.ylim(1e-1,5e2)
    #        plt.ylim(0,1e2)



    # Translate R -> kpc

    plt.plot(Rcenters/1e3, rhob_r, label=r'$\rho_b$(r)')

    #plt.plot(Rcenters/1e3, Mb_an, label=r'M$_b$(r) (Analytical)')
    #plt.plot(Rcenters/1e3, Mb_r, label=r'M$_b$(r) (Computational)')

    #plt.plot(Rcenters/1e3, Md_an, label=r'M$_D$(r) (Analytical)')
    #plt.plot(Rcenters/1e3, Md_r, label=r'M$_D$(r) (Computational)')
    
    plt.plot(Rcenters/1e3, vb_r, label=r'v$_b$(r)')
    plt.plot(Rcenters/1e3, vd_r, label=r'v$_D$(r) (Computational)')
    #plt.plot(Rcenters/1e3, vd_an, label=r'v$_D$(r) (Analytical)')

plt.xscale('log')
plt.yscale('log')

xlabel = r'Radius r (kpc)$'
ylabel = r'Density $\rho(r)$ (M$_\odot$/pc$^3$)'
plt.xlabel(xlabel)
plt.ylabel(ylabel)

#plt.tight_layout()
#lgd = plt.legend(bbox_to_anchor=(1.5, 0.5))
plt.legend(loc='center right')

plt.show()
