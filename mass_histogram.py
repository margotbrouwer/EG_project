#!/usr/bin/python
"""
# Create histograms to test the galaxy stellar masses
"""
import astropy.io.fits as pyfits
import gc
import numpy as np
import sys
import os
import time
from glob import glob

import modules_EG as utils
from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)
path_lenscat = '/data/users/brouwer/LensCatalogues'
plot_path = 'Users/users/brouwer/Documents/scp_files'
plot=True

## Import GAMA catalogue

# Full directory & name of the corresponding GAMA catalogue
gamacatname = 'GAMACatalogue_2.0.fits'
gamacatfile = '%s/%s'%(path_lenscat, gamacatname)
gamacat = pyfits.open(gamacatfile, memmap=True)[1].data

# List of the observables of all lenses in the GAMA catalogue
galZ_gama = gamacat['Z']
rmag_gama = gamacat['Rpetro']
rmag_abs_gama = gamacat['absmag_r']
logmstar_gama = gamacat['logmstar']

# Fluxscale, needed for absolute magnitude and stellar mass correction
fluxscale = gamacat['fluxscale']
rmag_abs_gama = rmag_abs_gama + 5*np.log10(h/0.7)
logmstar_gama = logmstar_gama - 2.*np.log10(h/0.7)

## Import KiDS catalogue

# Full directory & name of the corresponding KiDS catalogue
kidscatname = 'photozs.DR4_GAMAequ_ugri_beta_100ANNs_masses.fits'
kidscatfile = '%s/%s'%(path_lenscat, kidscatname)
kidscat = pyfits.open(kidscatfile, memmap=True)[1].data

# List of the observables of all sources in the KiDS catalogue
galZ_kids = kidscat['zANNz2ugri']

rmag_kids = kidscat['MAG_AUTO']
rmag_gaap_kids = kidscat['MAG_GAAP_r']
rmag_abs_kids = kidscat['MAG_ABS_r']

logmstar_kids = kidscat['MASS_BEST']
logmstar_kids = logmstar_kids + (rmag_gaap_kids-rmag_kids)/2.5
mass_med = kidscat['MASS_MED']


## Import matched catalogue

# Full directory & name of the corresponding GAMA catalogue
matchcatname = 'Matched_gama_kids_mass_catalogue.fits'
matchcatfile = '%s/%s'%(path_lenscat, matchcatname)
matchcat = pyfits.open(matchcatfile, memmap=True)[1].data

# Matched redshifts
galZ_gama_matched = matchcat['Z']
galZ_kids_matched = matchcat['zANNz2ugri']

# Matched masses
logmstar_gama_matched = matchcat['logmstar']
fluxscale_matched = matchcat['fluxscale']
logmstar_gama_matched = logmstar_gama_matched - 2.*np.log10(h/0.7)

logmstar_kids_matched = matchcat['MASS_BEST']
rmag_kids_matched = matchcat['MAG_AUTO']
rmag_gaap_kids_matched = matchcat['MAG_GAAP_r']
logmstar_kids_matched = logmstar_kids_matched + (rmag_gaap_kids_matched-rmag_kids_matched)/2.5


## Select data for plotting

gamamask = (8.<logmstar_gama)&(logmstar_gama<12.)
kidsmask = (8.<logmstar_kids)&(logmstar_kids<12.)
massmedmask = (mass_med>0.)

massmask_matched = (8.<logmstar_kids_matched)&(logmstar_kids_matched<12.)& \
                (8.<logmstar_gama_matched)&(logmstar_gama_matched<12.)

Zmask_matched = (0.0<galZ_kids_matched)&(galZ_kids_matched<0.5)& \
                (0.0<galZ_gama_matched)&(galZ_gama_matched<0.5)

Zmask = (0.0<galZ_kids_matched)


logmstar_gama = logmstar_gama[gamamask]
logmstar_kids = logmstar_kids[kidsmask]

#logmstar_gama_matched = logmstar_gama_matched[massmask_matched]
#logmstar_kids_matched = logmstar_kids_matched[massmask_matched]

galZ_gama_matched = galZ_gama_matched[Zmask_matched]
galZ_kids_matched = galZ_kids_matched[Zmask_matched]

if plot:
    ## Plot mass histograms
    plt.hist(logmstar_gama_matched[massmask_matched*Zmask], label=r'GAMA (matched)', bins=50, histtype='step')
    #plt.hist(logmstar_kids, label=r'KiDS', bins=50, histtype='step')
    plt.hist(logmstar_kids_matched[massmask_matched], label=r'KiDS (matched)', bins=50, histtype='step')
    plt.hist(logmstar_kids_matched[massmask_matched*Zmask], label=r'KiDS (Z$>$0.1)', bins=50, histtype='step')
    #plt.hist(logmstar_massmed, label=r'KiDS (MASS$_{\rm MED}>0$)', bins=50, histtype='step', normed=1)

    # Define the labels for the plot
    xlabel = r'Galaxy stellar mass [log(${\rm M/h_{%g}^{-2}M_{\odot}})$]'%(h*100)
    ylabel = r'Number of galaxies (normalized)'
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plt.legend(loc='upper left')

    #plt.xlim([7., 12.])
    #plt.ylim(ylims)

    plt.tight_layout()

    plotfilename = '/%s/KiDS_GAMA_mass_histogram_matched'%plot_path

    # Save plot
    for ext in ['png']:
        plotname = '%s.%s'%(plotfilename, ext)
        plt.savefig(plotname, format=ext, bbox_inches='tight')
        
    print('Written plot:', plotname)

    plt.show()
    plt.clf


    ## Plot 2D histogram
    massline = np.linspace(8, 12, 50)

    plt.hist2d(logmstar_gama_matched[massmask_matched*Zmask], logmstar_kids_matched[massmask_matched*Zmask], bins=100)#, cmin=1, cmap='Blues')
    plt.plot(massline, massline, color='black', ls='--')
    plt.plot(massline, massline+0.2, color='grey', ls='--')
    plt.plot(massline, massline-0.2, color='grey', ls='--')

    # Define the labels for the plot
    xlabel = r'GAMA-II stellar mass [log(${\rm M/h_{%g}^{-2}M_{\odot}})$]'%(h*100)
    ylabel = r'KiDS-1000 stellar mass [log(${\rm M/h_{%g}^{-2}M_{\odot}})$]'%(h*100)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plotfilename = '/%s/KiDS_GAMA_mass_2Dhist_matched'%plot_path

    # Save plot
    for ext in ['png']:
        plotname = '%s.%s'%(plotfilename, ext)
        plt.savefig(plotname, format=ext, bbox_inches='tight')
        
    print('Written plot:', plotname)

    plt.show()
    plt.clf


    ## Plot 2D redshift histogram
    Zline = np.linspace(0, 0.5, 50)

    plt.hist2d(galZ_gama_matched, galZ_kids_matched, bins=100)#, cmin=1, cmap='Blues')
    plt.plot(Zline, Zline, color='black', ls='--')
    plt.plot(Zline, Zline+0.02, color='grey', ls='--')
    plt.plot(Zline, Zline-0.02, color='grey', ls='--')

    # Define the labels for the plot
    xlabel = r'GAMA-II spectroscopic redshift'
    ylabel = r'KiDS-1000 ANNZ redshift'

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plotfilename = '/%s/KiDS_GAMA_redshift_2Dhist_matched'%plot_path

    # Save plot
    for ext in ['png']:
        plotname = '%s.%s'%(plotfilename, ext)
        plt.savefig(plotname, format=ext, bbox_inches='tight')
        
    print('Written plot:', plotname)

    plt.show()
    plt.clf


print('specZ GAMA:', np.mean(galZ_gama_matched))
print('ANNZ KiDS:', np.mean(galZ_kids_matched))

print('Diff. Fraction Z:', np.mean((galZ_kids_matched-galZ_gama_matched)/(1.+galZ_gama_matched)))
print('Stand. Dev. Z:', np.std((galZ_kids_matched-galZ_gama_matched)/(1.+galZ_gama_matched)))

print('Diff. Mstar:', np.mean((logmstar_kids_matched[massmask_matched*Zmask]-logmstar_gama_matched[massmask_matched*Zmask])))
print('Stand. Dev. Mstar:', np.std((logmstar_kids_matched[massmask_matched*Zmask]-logmstar_gama_matched[massmask_matched*Zmask])))
