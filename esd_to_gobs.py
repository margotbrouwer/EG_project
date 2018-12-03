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

def calc_sis(sigma_v, r, R):
    
    gobs_sis = (2.*sigma_v**2.) / r
    Mobs_sis = (2.*sigma_v**2.*r) / G
    rho_sis = sigma_v**2. / (2.*G*pi*r**2.)
    
    sigma_sis = sigma_v**2. / (2.*G*R)
    avsigma_sis = sigma_v**2. / (G*R)
    esd_sis = sigma_v**2. / (2.*G*R)
    
    return gobs_sis, Mobs_sis, rho_sis, sigma_sis, avsigma_sis, esd_sis

# Define the projected (R) and spherical (r) distance bins
Nbins_R = 20
Nbins_r = 20
Runit = 'kpc'
Rlog = True
Rmin, Rmax = np.array([30., 3000.]) # in pc
rmin, rmax = np.array([30., 3000.]) # in pc

Rbins, Rcenters, Rmin, Rmax, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins_R, Rlog)
rbins, rcenters, rmin, rmax, xvalue = utils.define_Rbins(Runit, rmin, rmax, Nbins_r, Rlog)

Rbins, Rcenters = np.array([Rbins, Rcenters])*xvalue
rbins, rcenters = np.array([rbins, rcenters])*xvalue

# Create a mock Singular Isothermal Sphere profile
sigma_v = 200.e3 / pc_to_meter # to km/s to pc/s
gobs_sis, Mobs_sis, rho_sis, sigma_sis, avsigma_sis, esd_sis = calc_sis(sigma_v, rcenters, Rcenters)


def calc_Cmn(R, r, m, n):
  
    if r[n+1] < R[m]: # If outer r < inner R: ))||
        integ = 0.
    elif (r[n+1] < R[m+1]): # If outer r < outer R: )|)| or |))|
        if (r[n] < R[m]):
            integ = r[n+1]**3. - R[m]**3. - (r[n+1]**2.-R[m]**2.)**(3./2.)
        else:
            integ = (r[n]**2.-R[m]**2.)**(3./2.) - (r[n+1]**2.-R[m]**2.)**(3./2.) \
            + r[n+1]**3. - r[n]**3.
    elif r[n] < R[m]: # If inner r < inner R: )||)
        integ = (r[n+1]**2.-R[m+1]**2.)**(3./2.) - (r[n+1]**2.-R[m]**2.)**(3./2.) \
        + R[m+1]**3. - R[m]**3.
    elif r[n] < R[m+1]: # If inner r < outer R: |)|)
        integ = (r[n+1]**2.-R[m+1]**2.)**(3./2.) - (r[n+1]**2.-R[m]**2.)**(3./2.) \
        + (r[n]**2.-R[m]**2.)**(3./2.) + R[m+1]**3. - r[n]**3.
    else: # If inner r > outer R: ||))
        integ = (r[n+1]**2.-R[m+1]**2.)**(3./2.) - (r[n+1]**2.-R[m]**2.)**(3./2.) \
        - (r[n]**2.-R[m+1]**2.)**(3./2.) + (r[n]**2.-R[m]**2.)**(3./2.)

    fact = -((4.*pi)/3.) / (R[m+1]**2. - R[m]**2.)
    C_mn = fact * integ
    
    #print('r[n+1]<R[m]:', r[n+1]<R[m])
    #print('r[n+1]<R[m+1]:', r[n+1]<R[m+1])
    #print('r[n]<R[m]:', r[n]<R[m])
    #print('r[n]<R[m+1]:', r[n]<R[m+1])    
    #print()

    return C_mn


# Matrix Bmn that calculates the average surface density avgsigma(<R)
B_mn = np.zeros([Nbins_R, Nbins_r])
for m in range(Nbins_R):
    for n in range(Nbins_r):
        for p in range(m-1):
            A_p = pi * (Rbins[p+1]**2. - Rbins[p]**2.)
            A_m = pi * Rbins[m]**2.
            B_mn[m,n] = B_mn[m,n] + \
                (A_p * calc_Cmn(Rbins, rbins, p, n) / A_m)

# Matrix Cmn that calculates the surface density Sigma(R) from rho(r)
C_mn = np.zeros([Nbins_R, Nbins_r])
for m in range(Nbins_R):
    for n in range(Nbins_r):
        C_mn[m,n] = calc_Cmn(Rbins, rbins, m, n)

# Matrix Kmn calculates the ESD(R) = avsigma(<R) - sigma(R)
K_mn = B_mn - C_mn

# Calculate test quantities from rho(r)
sigma_m = np.dot(C_mn, rho_sis) # OK
avsigma_m = np.dot(B_mn, rho_sis) # OK
esd_m = np.dot(K_mn, rho_sis)


#"""
# Plot the mock ESD profiles
labels_an = [r'Sigma (analytical)', r'avSigma (analytical)', r'ESD (analytical)']
labels_num = [r'Sigma (numerical)', r'avSigma (numerical)', r'ESD (numerical)']
results_an = [sigma_sis, avsigma_sis, esd_sis]
results_num = [sigma_m, avsigma_m, esd_m]
#"""

"""
print(C_mn)
# Invert matrices
Cmn_inv = np.linalg.inv(C_mn)
#Bmn_inv = np.linalg.inv(B_mn)
#Kmn_inv = np.linalg.inv(K_mn)
#print(Cmn_inv)

#sigma_sis = np.matrix(np.reshape(sigma_sis, [Nbins_R, 1]))

# Calculate test quantities from rho(r)
rho_n1 = np.dot(Cmn_inv, sigma_m)
rho_n2 = np.dot(Cmn_inv, sigma_sis)

#rho_n = np.dot(Bmn_inv, avsigma_sis)
#rho_n = np.dot(Kmn_inv, esd_sis)
#print(rho_n)


# Plot the mock ESD profiles
labels_an = [r'rho1 (analytical)', r'rho2 (analytical)']
labels_num = [r'rho1 (numerical)', r'rho2 (numerical)']
results_an = [rho_sis, rho_sis]
results_num = [rho_n1, rho_n2]
"""

# Plot the results

for r in range(len(results_num)):
    plt.plot((Rcenters*(1+r/20.))/xvalue, results_an[r], marker='o', ls='', color=colors[r], label=labels_an[r])
    plt.plot((Rcenters*(1+r/20.))/xvalue, results_num[r], color=colors[r], label=labels_num[r])
    
    print()
    print('Difference:', labels_an[r], labels_num[r])
    print(np.mean(1.-results_num[r]/results_an[r]))

#plt.axvline(x=rmin/xvalue)
#plt.axvline(x=rmax/xvalue)

plt.legend()
plt.xscale('log')
plt.yscale('log')

xlabel = r'Radius $R$ (%s/h$_{%g}$)'%(Runit, h*100)
ylabel = r'ESD $\langle\Delta\Sigma\rangle$ [h$_{%g}$ M$_{\odot}$/pc$^2$]'%(h*100)

plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.show()
