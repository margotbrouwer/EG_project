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

colors = ['#d7191c', '#fdae61', '#92c5de', '#0571b0', '#0ba52d']

# Cosmological parameters
O_matter = 0.315
O_lambda = 1.-O_matter
Ok = 0.
h = 1.

# Constants
G = 4.5171e-30 # pc3 / (Msun s2)
c = 9.7156e-09 # pc/s
H0 = 70 * 3.240779289469756e-20 # km/s/Mpc -> 1/s
Cd = (c * H0) / (G * 6) # Msun/pc^2

pi = np.pi
inf = np.inf

# Creating the galaxy projected distance bins
nRbins = int(1e3)
Rmin = 0.01e3 # in pc
Rmax = 100e3 #18.6e3 # in pc

R = 10.**np.linspace(np.log10(Rmin), np.log10(Rmax), nRbins)
#Rbins = np.append(Rbins,Rmax)
#R = np.array([(Rbins[r] + Rbins[r+1])/2 for r in xrange(nRbins)])

def Sersic(Mb, n, r_eff, r):
    
    b = 2*n - 1/3 + 0.009876/n
    A = b**(2*n) / (2*pi*n*sp.gamma(2*n))
    alpha = 1 - 1.188/(2*n) + 0.22/(4*n**2.)
    print(b, A, alpha)
    
    rhob_r = Mb * A * (r/r_eff)**-alpha * np.exp(-b*(r/r_eff)**(1/n))
    
    return rhob_r

def calc_Md(Mb_r, r):

    H = H0# * np.sqrt(O_matter*(1+z)**3 + O_lambda)
    Cd = (c * H) / (G * 6)
    Md_r = np.sqrt(Cd * np.gradient(Mb_r * r, r)) * r
    rhod_r = np.gradient(Md_r, r) / (4. * pi * r**2)
    
    return Md_r, rhod_r
    
def calc_radvel(M_r, r):

    vel_r = (G * M_r / r)**0.5
    acc_r = G * M_r / r**2
    
    return vel_r, acc_r

Mb_vD = 2.2e8 # in Msun
Rout_vD = 7.6e3 # in pc
Mtot_vD = 3.4e8 # in Msun
Re_vD = 3.1e3 # in pc
Me_vD = 3.2e8 # in Msun

# Compute Mb(r) and Md(r) for multiple Sersic indices
nlist = [0.6, 1, 2]
rlist = [2.2e3, 1e3, 0.5e3]

#nlist = [1]
#rlist = [1e3]

for i in range(len(nlist)):

    """
    # Projected Sersic profile
    s_vD = Sersic1D(amplitude=1, r_eff=1e3, n=nlist[i]) # Van Dokkum et al. 2018
    rhob_r = s_vD(R)
    Mb_r = 4.*pi * cumtrapz(rhob_r * R**2, x = R, initial=0.)
    Mb_last = Mb_r[-1]

    # Adjust the amplitude of rhob and Mb according to Mtot
    rhob_r = rhob_r * Mb_vD/Mb_last
    Mb_r = Mb_r * Mb_vD/Mb_last
    """

    """
    # Test stuff
    Mb_an = 4.*pi*Mb_vD/Mb_last * np.log(R) # Test mass
    Md_an = np.sqrt(Cd * 4.*pi*Mb_vD/Mb_last * (np.log(R)+1.)) * R
    Md_an = np.sqrt(Cd * np.gradient(Mb_an * R, R)) * R
    Mtot_an = Mb_an + Md_an

    plt.plot(R, rhob_r)
    plt.plot(R, Mb_r)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    quit()
    """

    # Sersic profile density and mass
    rhob_r = Sersic(Mb_vD, nlist[i], rlist[i], R)
    Mb_r = 2.*pi * cumtrapz(rhob_r * R/rlist[i], x = R/rlist[i], initial=0.)
    
    # Calculate the baryonic and DM distributions

    Md_r, rhod_r = calc_Md(Mb_r, R)
    Mtot_r = Mb_r+Md_r
    Md_point = np.sqrt(Cd*Mb_r)*R
    Md_add = np.sqrt(Cd * np.gradient(Mb_r, R))

    # Find the value of Me for Re
    idx = (np.abs(R - Re_vD)).argmin()
    Me_eg = Mtot_r[idx]
    sigma_e = np.sqrt((G*Me_eg)/(4*Re_vD)) * 3.086e13 # pc to km

    # Find the value of Mout for Rout
    idx = (np.abs(R - Rout_vD)).argmin()
    Mout_eg = Mtot_r[idx]
    sigma_out= np.sqrt((G*Mout_eg)/(1.9*Rout_vD)) * 3.086e13 # pc to km

    print('At Re=%g kpc: EG mass=%g Msun, sigma=%g km/s'%((Re_vD/1e3), Me_eg, sigma_e))
    print('At Rout=%g kpc: EG mass=%g Msun, sigma=%g km/s'%((Rout_vD/1e3), Mout_eg, sigma_out))

    #vb_r, ab_r = calc_radvel(Mb_r, R)
    #vd_r, ad_r = calc_radvel(Md_r, R)

    # Plot Sersic profile results

    """
    #plt.plot(R/1e3, rhob_r, label=r'$\rho_{\rm b}$(r)')
    plt.plot(R/1e3, Mb_r, color=colors[i], label=r'$M_{\rm b}$(\textless r) (Sersic profile)')
    #plt.plot(R/1e3, Md_r, label=r'M$_{\rm D}$(\textless r) (Sersic profile)')
    plt.plot(R/1e3, Mtot_r, color=colors[i], label=r'M$_{\rm tot,EG}$(\textless r) (Sersic profile)')
    """
    
    #plt.plot(R/1e3, rhob_r*1e6, ls='--', color=colors[i], label=r'$\rho_{\rm b}$(r) for n=%g'%nlist[i])
    plt.plot(R/1e3, Mb_r, ls='--', color=colors[i], label=r'M$_{\rm b}$(\textless r) for n=%g'%nlist[i])
    #plt.plot(R/1e3, Md_add, ls=':', color=colors[i])
    #plt.plot(R/1e3, Mtot_r, ls='-', color=colors[i])
    #plt.plot(R/1e3, Md_r, label=r'M$_{\rm D}$(\textless r) (Sersic profile)')
    #plt.plot(R/1e3, Mtot_r, color=colors[i])


# Plot point mass results
Md_point = np.sqrt(Cd*Mb_vD)*R
plt.plot(R/1e3, [Mb_vD]*len(R), ls='--', label=r'$M$_{\rm b}$(\textless r) (Point mass)')
#plt.plot(R/1e3, Mb_vD+Md_point, ls='--', label=r'M$_{\rm tot,EG}$(\textless r) (Point mass)')

#plt.axvline(x=Re_vD/1e3, ls=':', color='red', label=r'r$_{\rm 1/2}$ (=%g kpc)'%(Re_vD/1e3))
#plt.axhline(y=Me_vD, ls=':', color='red', label=r'M$_{\rm obs,1/2}$ (=%g M$_\odot$)'%Me_vD)

#plt.axvline(x=Rout_vD/1e3, ls=':', label=r'r$_{\rm out}$ (=%g kpc)'%(Rout_vD/1e3))
#plt.axhline(y=Mtot_vD, ls=':', label=r'M$_{\rm obs,out}$ (=%g M$_\odot$)'%Mtot_vD)

#plt.plot(R/1e3, vb_r, label=r'v$_b$(r)')
#plt.plot(R/1e3, vd_r, label=r'v$_D$(r) (Computational)')

plt.xscale('log')
plt.yscale('log')

xlabel = r'Radius r (kpc)$'
ylabel = r'Mass [M$_\odot$]'# / 
#ylabel = r'Density'# [M$_\odot$/pc$^3$]'
plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.xlim(0.1,10)
#plt.ylim(1e3, 1e10)
#plt.ylim(1e7, 1e10)


plt.tight_layout()
#lgd = plt.legend(bbox_to_anchor=(1.5, 0.5))
plt.legend()

for ext in ['pdf']:

    plotfilename = 'sersic_amplitude_EG'
    plotname = '%s.%s'%(plotfilename, ext)
    
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written plot:', plotname)

plt.show()
plt.close()
