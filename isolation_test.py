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

# Dark blue, orange, red, light blue
colors = ['#0571b0', '#fdae61', '#d7191c', '#92c5de']

# Select the lens catalogue (kids/gama/mice)
cat = 'kids'

# Constants
h = 0.7
if 'mice' in cat:
    O_matter = 0.25
    O_lambda = 0.75
else:
    O_matter = 0.2793
    O_lambda = 0.7207
cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

rationame = 'perc'
massratios = [0.3, 0.25, 0.2, 0.15, 0.1]
massratio_names = [str(p).replace('.','p') for p in massratios]
massratio_num = 4
distval = 3 # in Mpc

if cat=='gama':
    magmin=16.
    magmax=19.8
else:
    magmin=16.
    magmax=20.

print()
print('Testing isolation criterion: r_sat(f_M*>%g)>%g Mpc/h'%(massratios[massratio_num],distval))
#print('Testing isolation criterion: r_iso=%g Mpc'%distval)

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)
logmstarlist = logmstar

lenscat = pyfits.open('%s/%s'%(path_lenscat,lenscatname), memmap=True)[1].data
if 'kids' in cat:
    kidsmask = lenscat['masked']

# Remove all galaxies with logmstar=NAN and masked
if 'kids' in cat:
    nanmask = np.isfinite(logmstar) & (kidsmask==0)
else:
    nanmask = np.isfinite(logmstar)
    
lensRA, lensDEC, lensDc, logmstar, rmag = \
    [lensRA[nanmask], lensDEC[nanmask], lensDc[nanmask], logmstar[nanmask], rmag[nanmask]]


# Import isolation catalog
isocatfile = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_%s_h%i.fits'%(cat, rationame, h*100.)
#isocatfile = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_h%i.fits'%(cat, h*100.)
isocat = pyfits.open(isocatfile, memmap=True)[1].data
print('Imported:', isocatfile)

#distcat = np.array([ isocat['dist%s%s'%(p, rationame)] for p in massratio_names])
isodist = (isocat['dist%s%s'%(massratio_names[massratio_num], rationame)])[nanmask]
#isodist = (isocat['riso'])[nanmask]
isomask = isodist > distval # There are no galaxies within X pc

# Define the galaxy mass bins

Nrmagbins = 100.
rmagbins = np.linspace(magmin, magmax, Nrmagbins)
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
print('Ngal_bright:', Ngal_bright, Ngal_bright/Ngal_tot*100., '% of total sample')
print('Niso_tot:', Niso_tot, Niso_tot/Ngal_tot*100., '% of total sample')
print('Niso_bright:', Niso_bright, Niso_bright/Ngal_bright*100., '% of bright sample')
print('Niso_faint:', Niso_faint, Niso_faint/Ngal_faint*100., '% of faint sample')
print('Niso_false:', Niso_false, Niso_false/Niso_tot*100., '% of isolated sample')


## Plot the result

plt.figure(figsize=(4.7,3.7))
plotscale = 1.e3 # Scale the number of galaxies

plt.plot(rmagcenters, rmaghist/plotscale, color=colors[2], label=r'All KiDS-bright lens galaxies')
plt.plot(rmagcenters, isohist/plotscale, color=colors[1], \
    label=r'Isolated: $r_{\rm sat}(f_{\rm M_*}>%g)>%g$ Mpc$/h_{70}$'%(massratios[massratio_num], distval))
    
    
plt.plot(rmagcenters, isohist/rmaghist, color=colors[0], label=r'Fraction (isolated/all galaxies)')
plt.text(17.6, 7.e-2, r'f$_{\rm L}=%g$'%(massratios[massratio_num]), fontsize=12)

#plt.plot(rmagcenters, np.cumsum(isohist)/np.cumsum(rmaghist))

plt.axvline(x=magmax-maglim, color='grey', ls='--')
#plt.axvline(x=magmax, color='black', ls='--')

xlabel = r'Apparent magnitude $m_{\rm r}$'
ylabel = r'Number of galaxies ($\times %g$)'%plotscale

plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)

plt.xlim([magmin, magmax])
plt.ylim([3.e-2, 3.e2])


plt.yscale('log')

plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()

# Save plot
plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Mar19/Plots/isolation_test_%s_%s%s-%gMpc'\
                                                %(cat, rationame, massratio_names[massratio_num], distval)
for ext in ['png','pdf']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written plot:', plotname)

plt.show()
plt.clf
