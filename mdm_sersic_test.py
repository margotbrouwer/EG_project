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

pi = np.pi
inf = np.inf


def Sersic(Mb, n, r_eff, r):
    
    b = 2.*n - 1./3. + 0.009876/n
    
    A = (b**(2.*n)) / (2.*pi*n*sp.gamma(2.*n))
    #A=1.
    rhob_R = Mb * A * np.exp(-b*((r/r_eff)**(1/n))) / r_eff**2.
    
    print('n=%g -> b=%g'%(n, b))
    
    alpha = 1. - 1.188/(2.*n) + 0.22/(4.*n**2.)
    rhob_r = Mb * A * (r/r_eff)**-alpha * np.exp(-b*((r/r_eff)**(1./n))) / r_eff**2.
    
    return rhob_R, rhob_r



# Creating the galaxy projected distance bins
nRbins = int(1e5)
Rmin = 1e-2 # in pc
Rmax = 1e5 #18.6e3 # in pc

R = 10.**np.linspace(np.log10(Rmin), np.log10(Rmax), nRbins)
#Rbins = np.append(Rbins,Rmax)
#R = np.array([(Rbins[r] + Rbins[r+1])/2 for r in xrange(nRbins)])

M_test = 1.
n_test = 4.
r_test = 1e3

nlist = [0.6, 1., 2., 3., 4.]

for i in range(len(nlist)):

    n_test = nlist[i]

    # Sersic profile density and mass
    rhob_R, rhob_r = Sersic(M_test, n_test, r_test, R)
    Mb_R = 2.*pi * cumtrapz(rhob_R * R, x = R, initial=0.)


    # Projected Sersic profile
    s_vD = Sersic1D(amplitude=M_test, r_eff=r_test, n=n_test) # Van Dokkum et al. 2018
    rhob_Rpy = s_vD(R)
    Mb_Rpy = 2.*pi * cumtrapz(rhob_Rpy * R, x = R, initial=0.)

    # Adjust the amplitude of rhob and Mb according to Mtot
    #Mb_last = Mb_Rpy[-1]
    #rhob_Rpy = rhob_Rpy * Mb_vD/Mb_last
    #Mb_Rpy = Mb_Rpy * Mb_vD/Mb_last


    # Plot Sersic profile results
    """
    plt.plot(R, rhob_R, ls='-', label=r'$\rho_{\rm b}$(r) for n=%g (projected)'%n_test)
    plt.plot(R, rhob_Rpy, ls='-', label=r'$\rho_{\rm b}$(r) for n=%g (python)'%n_test)
    plt.plot(R, np.abs((rhob_R-rhob_Rpy)/(rhob_R+rhob_Rpy)/2.), ls=':', label=r'Difference')
    """

    plt.plot(R, Mb_R, ls='--', label=r'$M_{\rm b}$(r) for n=%g (projected)'%n_test)
    #plt.plot(R, Mb_Rpy, ls=':', label=r'$M_{\rm b}$(r) for n=%g (python)'%n_test)
    #plt.plot(R, np.abs((Mb_R-Mb_Rpy)/(Mb_R+Mb_Rpy)/2.), ls=':', label=r'Difference')


plt.xscale('log')
plt.yscale('log')

xlabel = r'Radius r (kpc)$'
#ylabel = r'Mass [M$_\odot$]'# / 
ylabel = r'Density'# [M$_\odot$/pc$^3$]'

plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.xlim(1e0,1e5)
#plt.ylim(1e0, 1e10)
#plt.ylim(1e7, 1e10)


plt.show()
