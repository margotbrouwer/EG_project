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


## Configuration

cat = 'kids' # Data selection: select the lens catalogue (kids/gama/mice)
purpose = 'isolated' # Purpose of this run (isolated or mstar)
plot = False # To plot or not to plot


# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

path_isocat = '/data/users/brouwer/LensCatalogues'
plotfile_path = '/data/users/brouwer/Lensing_results/EG_results_Sep19/Plots'

# Import isolated galaxy catalog
isocatname = '%s_isolated_galaxies_perc_h70.fits'%cat
isocatfile = '%s/%s'%(path_isocat, isocatname)
isocat = pyfits.open(isocatfile, memmap=True)[1].data
Riso =  isocat['dist0p1perc']

# Import offset isolated galaxy catalog
if 'kids' in cat:
    offset = ''
else:
    offset = '-offsetZ'

isoffcatname = '%s%s_isolated_galaxies_perc_h70.fits'%(cat, offset)
isoffcatfile = '%s/%s'%(path_isocat, isoffcatname)

isoffcat = pyfits.open(isoffcatfile, memmap=True)[1].data
Risoff = isoffcat['dist0p1perc%s'%offset.replace('-', '_')]

# Remove NAN from stellar masses
nanmask = np.isfinite(logmstar)
lensRA, lensDEC, lensZ, lensDc, logmstar, Riso, Risoff = \
    [lensRA[nanmask], lensDEC[nanmask], lensZ[nanmask], lensDc[nanmask], logmstar[nanmask], Riso[nanmask], Risoff[nanmask]]

lensDa = lensDc/(1.+lensZ)
mstar = 10.**logmstar

lenscoords = SkyCoord(ra=lensRA*u.deg, dec=lensDEC*u.deg, distance=lensDa)
lenscoords = lenscoords

# Creating the spherical distance bins
nrbins = 100
rmin = 0.03 # in Mpc
rmax = 3 # in Mpc

rbins = 10.**np.linspace(np.log10(rmin), np.log10(rmax), nrbins)
dr = np.diff(rbins)
rcenters = rbins[0:-1]+0.5*dr

if 'mstar' in purpose:
    offset = ''

    # Create histogram for different stellar mass bins

    logmstarbins = np.array([8.5,10.3,10.6,10.8,11.])
    masks = [(logmstarbins[m]<=logmstar)&(logmstar<logmstarbins[m+1]) \
                for m in range(len(logmstarbins)-1)]
    
    masknames = ['%g < log(M*/Msun) < %g'%(logmstarbins[m], logmstarbins[m+1]) \
                    for m in range(len(logmstarbins)-1)]
    outputnames = ['rcenters']
    massnames = [['M%s_bin%i'%(s, m+1) for s in ['sat', 'tot']] for m in range(len(logmstarbins)-1)]
    outputnames = np.append(outputnames, massnames)
    
    print(outputnames)
    
if 'iso' in purpose:
    ## Create histogram for different isolated samples

    # Create isolated masks
    all_mask = (logmstar < 11.)
    iso_mask = (Riso > 3.)&(logmstar < 11.)
    isoff_mask = (Risoff > 3.)&(logmstar < 11.)

    if 'kids' in cat:
        masks = [all_mask, iso_mask]
        masknames = ['All galaxies (log(M*/Msun)<11)', 'Isolated']
        outputnames = np.array(['rcenters', 'Msat_all', 'Mtot_all', \
                    'Msat_iso', 'Mtot_iso'])
    else:
        masks = [all_mask, iso_mask, isoff_mask]
        masknames = ['All galaxies (log(M*/Msun)<11)', 'Isolated', 'Isolated, offset']
        outputnames = np.array(['rcenters', 'Msat_all', 'Mtot_all', \
                    'Msat_iso', 'Mtot_iso', 'Msat_isoff', 'Mtot_isoff'])

output = np.array([rcenters])

print()
for m in range(len(masknames)):

    # Defining the central and satellite sample
    catalog = lenscoords
    c = lenscoords[masks[m]]
    
    N_lenses = float(len(c))
    lensmass_tot = np.sum(mstar[masks[m]])
    lensmass_mean = lensmass_tot/N_lenses
    
    print(masknames[m])    
    print('Lenses:', N_lenses)
    print('Mean lens mass:', np.log10(lensmass_mean))
    
    idxcatalog, idxc, d2d, d3d = c.search_around_3d(catalog, rmax*u.Mpc)
    
    # The total mass distribution of satellites
    sat_hist = np.histogram(d3d, bins=rbins, weights=mstar[idxcatalog])[0]
    Msat = np.cumsum(sat_hist)
    
    # The total mass distribution of satellites + lenses
    tot_hist = sat_hist + lensmass_tot
    Mtot = Msat + lensmass_tot
    
    # The mean mass distribution of satellites per lens
    sat_hist_mean, Msat_mean = [sat_hist/N_lenses, Msat/N_lenses]
    tot_hist_mean, Mtot_mean = [tot_hist/N_lenses, Mtot/N_lenses]
    
    # Compute the maximum radius that can be trusted
    rlim = rcenters[(Msat_mean > lensmass_mean)]
    if len(rlim) > 0:
        rlim = rlim[0]
    else:
        rlim = rmax
    
    print('Radius where Msat>Mlens:', rlim)
    print()
    
    if plot:
        #plt.plot(rcenters, sat_hist_mean, label='Density %s'%masknames[m])
        plt.plot(rcenters, Msat_mean, label='Mass %s, satellites'%masknames[m])
        plt.plot(rcenters, Mtot_mean, label='Mass %s, total'%masknames[m])
        #plt.axhline(y=lensmass_mean)
        
    
    output = np.vstack([output, Msat_mean, Mtot_mean])

filename = '%s/satellite_histogram_%s_%s%s.fits'%(plotfile_path, cat, purpose, offset)
formats = np.array(['D']*len(outputnames))

utils.write_catalog(filename, outputnames, formats, output)

if plot:
    # Create plot

    plt.xscale('log')
    plt.yscale('log')

    xlabel = r'Radius r (Mpc/h)'
    ylabel = r'Stellar mass $M_*$ of satellite galaxies'
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    plt.legend()
    plotfilename = '%s/satellite_histogram_%s_%s%s'%(plotfile_path, cat, purpose, offset)

    # Save plot
    for ext in ['png']:
        plotname = '%s.%s'%(plotfilename, ext)
        plt.savefig(plotname, format=ext)
        
    print('Written plot:', plotname)

    plt.show()
    plt.clf()

