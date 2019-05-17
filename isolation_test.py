#!/usr/bin/python

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import LambdaCDM
import astropy.io.fits as pyfits

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

import modules_EG as utils

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
colors = ['#d7191c', '#0571b0', '#92c5de', '#fdae61']


# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685
cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)


cat = 'kids' # Select the lens catalogue (kids/gama/mice)
#rationame = 'perc'
massratios = [0.3, 0.25, 0.2, 0.15, 0.1]
massratio_names = [str(p).replace('.','p') for p in massratios]
massratio_num = 4
distval = 3 # in Mpc

if cat=='kids':
    magmax=20.
if cat=='gama':
    magmax=19.8

print()
#print('Testing isolation criterion: D(perc>%g)>%g Mpc'%(massratios[massratio_num],distval))
print('Testing isolation criterion: r_iso=%g Mpc'%distval)

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)
logmstarlist = logmstar

# Remove all galaxies with logmstar=NAN
nanmask = np.isfinite(logmstar)# & (rmag < 17.3)
lensRA, lensDEC, lensDc, logmstar, rmag = \
    [lensRA[nanmask], lensDEC[nanmask], lensDc[nanmask], logmstar[nanmask], rmag[nanmask]]


# Import isolation catalog
#isocatfile = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_%s_h%i.fits'%(cat, rationame, h*100.)
isocatfile = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_h%i.fits'%(cat, h*100.)
isocat = pyfits.open(isocatfile, memmap=True)[1].data
print('Imported:', isocatfile)

#distcat = np.array([ isocat['dist%s%s'%(p, rationame)] for p in massratio_names])
#isodist = (isocat['dist%s%s'%(massratio_names[massratio_num], rationame)])[nanmask]
isodist = (isocat['riso'])[nanmask]
isomask = isodist > distval # There are no galaxies within X pc

# Define the galaxy mass bins
rmag_min, rmag_max = [13., 20.]
Nrmagbins = 100.
rmagbins = np.linspace(rmag_min, rmag_max, Nrmagbins)
rmagcenters = rmagbins[0:-1] + 0.5*np.diff(rmagbins)

isohist, foo = np.histogram(rmag[isomask], rmagbins)
rmaghist, foo = np.histogram(rmag, rmagbins)

maglim = -2.5 * np.log10(massratios[massratio_num])
magmask = (rmag < magmax-maglim)

Ngal_tot = len(magmask)
Ngal_bright = np.sum(magmask)
Ngal_faint = Ngal_tot - Ngal_bright
Niso_tot = np.sum(isomask)
Niso_bright = np.sum(isomask*magmask)
Niso_faint = Niso_tot - Niso_bright

Niso_false = Niso_faint - Ngal_faint*(Niso_bright/Ngal_bright)

print()
print('Testing: Perc:', massratios[massratio_num], ', Distance (Mpc):', distval)
print('Maglim:', maglim, magmax-maglim)
print()
print('Isolated galaxies:')
print('Ngal_bright:', Ngal_bright, Ngal_bright/Ngal_tot*100., '%')
print('Niso_bright:', Niso_bright, Niso_bright/Ngal_bright*100., '%')
print('Niso_faint:', Niso_faint, Niso_faint/Ngal_faint*100., '%')
print('Niso_tot:', Niso_tot, Niso_tot/Ngal_tot*100., '%')
print('Niso_false:', Niso_false, Niso_false/Niso_tot*100., '%')

## Plot the result

plt.figure(figsize=(4.7,3.7))
plotscale = 1.e3 # Scale the number of galaxies

plt.plot(rmagcenters, rmaghist/plotscale, color=colors[1], label=r'All galaxies')
plt.plot(rmagcenters, isohist/plotscale, color=colors[2], label=r'Isolated galaxies (r$_{\rm iso}=3$)')
plt.plot(rmagcenters, isohist/rmaghist, color=colors[0], label=r'Fraction of isolated galaxies')
plt.text(17.7, 3.e-2, r'f$_{\rm L}=0.1$', fontsize=12)

#plt.plot(rmagcenters, np.cumsum(isohist)/np.cumsum(rmaghist))

plt.axvline(x=magmax-maglim, color='black', ls='--')
#plt.axvline(x=magmax, color='black', ls='--')

xlabel = r'Apparent magnitude $m_{\rm r}$'
ylabel = r'Number of galaxies (x$%g$)'%plotscale

plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)

plt.xlim([14., magmax])
plt.ylim([1.e-2, 1.e2])


plt.yscale('log')

plt.legend(loc='best', fontsize=10)
plt.tight_layout()

# Save plot
plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Mar19/Plots/isolation_test'
for ext in ['pdf', 'png']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf
