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
O_matter = 0.2793
O_lambda = 0.7207

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)
path_lenscat = '/data/users/brouwer/LensCatalogues'

# Plotting information
plot=False
#plot_path = 'Users/users/brouwer/Documents/scp_files'
plot_path = '/data/users/brouwer/Lensing_results/EG_results_Jul20'


z_min = 0.1
z_max = 0.5
logmstar_min = 8.
logmstar_max = 12.

splittype = 'none' # sersic / color / none

## Import GAMA catalogue

# Full directory & name of the corresponding GAMA catalogue
gamacatname = 'GAMACatalogue_2.0.fits'
gamacatfile = '%s/%s'%(path_lenscat, gamacatname)
gamacat = pyfits.open(gamacatfile, memmap=True)[1].data

# List of the observables of all lenses in the GAMA catalogue
Z_gama = gamacat['Z']
rmag_gama = gamacat['Rpetro']
rmag_abs_gama = gamacat['absmag_r']
logmstar_gama = gamacat['logmstar']
nQ_gama = gamacat['nQ']

# Fluxscale, needed for absolute magnitude and stellar mass correction
fluxscale = gamacat['fluxscale']
rmag_abs_gama = rmag_abs_gama + 5*np.log10(h/0.7)
logmstar_gama = logmstar_gama - 2.*np.log10(h/0.7)

## Import KiDS catalogue

# Full directory & name of the corresponding KiDS catalogue
kidscatname = 'photozs.DR4.1_bright_ugri+KV_struct.fits'

kidscatfile = '%s/%s'%(path_lenscat, kidscatname)
kidscat = pyfits.open(kidscatfile, memmap=True)[1].data

# List of the observables of all sources in the KiDS catalogue
#Z_kids = kidscat['zANNz2ugri']
Z_kids = kidscat['Z_ANNZ_KV']

rmag_kids_auto = kidscat['MAG_AUTO_CALIB']
rmag_kids_gaap = kidscat['MAG_GAAP_r']
rmag_kids_sersic = kidscat['MAG_AUTO_2dphot']
rmag_abs_kids = kidscat['MAG_ABS_r']

logmstar_kids = kidscat['MASS_BEST']
logmstar_kids = logmstar_kids + (rmag_kids_gaap-rmag_kids_auto)/2.5
mass_med = kidscat['MASS_MED']
masked_kids = kidscat['masked']


## Import matched catalogue

# Full directory & name of the corresponding matched KiDS-GAMA catalogue
matchcatname = 'photozs.DR4.1_bright_ugri+KV_matched.fits'
matchcatfile = '%s/%s'%(path_lenscat, matchcatname)
matchcat = pyfits.open(matchcatfile, memmap=True)[1].data

# Matched redshifts
Z_gama_matched = matchcat['Z']
Z_kids_matched = matchcat['Z_ANNZ_KV']

# Matched masses
logmstar_gama_matched = matchcat['logmstar']
fluxscale_matched = matchcat['fluxscale']
logmstar_gama_matched = logmstar_gama_matched - 2.*np.log10(h/0.7)
masked_matched = matchcat['masked']
nQ_matched = matchcat['nQ']

# Matched magnitudes
rmag_gama_matched = matchcat['Rpetro']
rmag_kids_auto_matched = matchcat['MAG_AUTO_CALIB']
rmag_kids_gaap_matched = matchcat['MAG_GAAP_r']

logmstar_kids_matched = matchcat['MASS_BEST']
logmstar_kids_matched = logmstar_kids_matched + \
    (rmag_kids_gaap_matched-rmag_kids_auto_matched)/2.5 - 2.*np.log10(h/0.7)

if 'none' not in splittype:
    
    # Matched spiral and elliptical galaxies by sersic index or color
    if 'sersic' in splittype:
        typelist = matchcat['n_2dphot']
        splitlim = 2.
        
    if 'color' in splittype:
        typelist = matchcat['MAG_GAAP_u'] - matchcat['MAG_GAAP_r']
        splitlim = 2.5

    # Matched equal mass selection catalogue
    selcatname = 'mass_selection_catalog_%s-offsetx0_matched.fits'%splittype
    selcatfile = '%s/%s'%(path_lenscat, selcatname)
    selcat = pyfits.open(selcatfile, memmap=True)[1].data
    mask_sel = (selcat['selected'] == 1.)
    logmstar_matched = selcat['logmstar']

## Select data for plotting

kidsmask = (logmstar_min<logmstar_kids)&(logmstar_kids<logmstar_max)&(masked_kids==0.)
gamamask = (logmstar_min<logmstar_gama)&(logmstar_gama<logmstar_max)&(nQ_gama<3.)
massmedmask = (mass_med>0.)

massmask_matched = (logmstar_min<logmstar_kids_matched)&(logmstar_kids_matched<logmstar_max)&(masked_matched==0)& \
                (logmstar_min<logmstar_gama_matched)&(logmstar_gama_matched<logmstar_max)&(nQ_matched>=3.)

Zmask_matched = (z_min<Z_kids_matched)&(Z_kids_matched<z_max)&(masked_matched==0)& \
                (z_min<Z_gama_matched)&(Z_gama_matched<z_max)&(nQ_matched>=3.)

Zmask = (0.<Z_kids_matched)

# Masking the data
logmstar_gama = logmstar_gama[gamamask]
logmstar_kids = logmstar_kids[kidsmask]

rmag_kids_auto_matched = rmag_kids_auto_matched[massmask_matched*Zmask]
rmag_kids_gaap_matched = rmag_kids_gaap_matched[massmask_matched*Zmask]
rmag_gama_matched = rmag_gama_matched[massmask_matched*Zmask]

Zmask_gama = (z_min<Z_gama)&(Z_gama<z_max)&(nQ_gama<3.)
Zmask_kids = (z_min<Z_kids)&(Z_kids<z_max)&(masked_kids==0.)

if 'none' in splittype:
    
    Z_gama_matched = Z_gama_matched[Zmask_matched]
    Z_kids_matched = Z_kids_matched[Zmask_matched]
    
    logmstar_gama_matched = logmstar_gama_matched[massmask_matched*Zmask]
    logmstar_kids_matched = logmstar_kids_matched[massmask_matched*Zmask]
    
else:
# Apply the elliptical and spiral masks to both KiDS and GAMA
    
    Z_gama_matched = Z_gama_matched[Zmask_matched*mask_sel]
    Z_kids_matched = Z_kids_matched[Zmask_matched*mask_sel]
    
    mask_ell = (typelist[Zmask_matched*mask_sel] > splitlim)
    mask_spir = (typelist[Zmask_matched*mask_sel] < splitlim)
    
    Z_gama_ell = Z_gama_matched[mask_ell]
    Z_kids_ell = Z_kids_matched[mask_ell]
    Z_gama_spir = Z_gama_matched[mask_spir]
    Z_kids_spir = Z_kids_matched[mask_spir]
    
    logmstar_gama_matched = logmstar_gama_matched[massmask_matched*mask_sel]
    logmstar_kids_matched = logmstar_kids_matched[massmask_matched*mask_sel]
    
    mask_ell = (typelist[massmask_matched*mask_sel] > splitlim)
    mask_spir = (typelist[massmask_matched*mask_sel] < splitlim)
    
    logmstar_gama_ell = logmstar_gama_matched[mask_ell]
    logmstar_kids_ell = logmstar_kids_matched[mask_ell]
    logmstar_gama_spir = logmstar_gama_matched[mask_spir]
    logmstar_kids_spir = logmstar_kids_matched[mask_spir]
    
if plot:
    
    # Define the guiding lines
    massline = np.linspace(logmstar_min, logmstar_max, 50)
    Zline = np.linspace(z_min, z_max, 50)


    ## Plot mass histograms
    if 'none' in splittype:
        plt.hist(logmstar_gama_matched, label=r'GAMA (matched)', bins=50, histtype='step', normed=1)
        plt.hist(logmstar_kids_matched, label=r'KiDS (matched)', bins=50, histtype='step', normed=1)
        plt.hist(logmstar_kids, label=r'KiDS', bins=50, histtype='step', normed=1)
    else:
        plt.hist(logmstar_gama_ell, label=r'GAMA (elliptical)', bins=50, histtype='step', normed=1)
        plt.hist(logmstar_gama_spir, label=r'GAMA (spiral)', bins=50, histtype='step', normed=1)
        plt.hist(logmstar_kids_ell, label=r'KiDS (elliptical)', bins=50, histtype='step', normed=1)
        plt.hist(logmstar_kids_spir, label=r'KiDS (spiral)', bins=50, histtype='step', normed=1)
    
    # Define the labels for the plot
    xlabel = r'Galaxy stellar mass [log(${\rm M/h_{%g}^{-2}M_{\odot}})$]'%(h*100)
    ylabel = r'Number of galaxies (normalized)'
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plt.legend(loc='upper left')

    plt.tight_layout()

    plotfilename = '/%s/KiDS_GAMA_mass_histogram_matched'%plot_path

    # Save plot
    for ext in ['png']:
        plotname = '%s.%s'%(plotfilename, ext)
        plt.savefig(plotname, format=ext, bbox_inches='tight')
        
    print('Written plot:', plotname)

    plt.show()
    plt.clf

    if 'none' in splittype:
        ## Plot 2D mass histogram

        plt.hist2d(logmstar_gama_matched, logmstar_kids_matched, bins=100)#, cmin=1, cmap='Blues')
        plt.plot(massline, massline, color='black', ls='--')
        plt.plot(massline, massline+0.2, color='grey', ls='--')
        plt.plot(massline, massline-0.2, color='grey', ls='--')

        # Define the labels for the plot
        xlabel = r'GAMA-II stellar mass [log(${\rm M/h_{%g}^{-2}M_{\odot}})$]'%(h*100)
        ylabel = r'KiDS-bright stellar mass [log(${\rm M/h_{%g}^{-2}M_{\odot}})$]'%(h*100)

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

        plt.hist2d(Z_gama_matched, Z_kids_matched, bins=100)#, cmin=1, cmap='Blues')
        plt.plot(Zline, Zline, color='black', ls='--')
        plt.plot(Zline, Zline+0.02, color='grey', ls='--')
        plt.plot(Zline, Zline-0.02, color='grey', ls='--')

        # Define the labels for the plot
        xlabel = r'GAMA-II spectroscopic redshift'
        ylabel = r'KiDS-bright ANNZ redshift'

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
    
    else:
        
        ## Plot 2D redshift histogram of ellipticals and spirals
        
        plt.scatter(Z_gama_spir, Z_kids_spir, marker='.', alpha=0.01, color='blue')
        
        plt.scatter(Z_gama_ell, Z_kids_ell, marker='.', alpha=0.01, color='red')

        plt.plot(Zline, Zline, color='black', ls='--')
        plt.plot(Zline, Zline+0.02, color='grey', ls='--')
        plt.plot(Zline, Zline-0.02, color='grey', ls='--')

        # Define the labels for the plot
        xlabel = r'GAMA-II spectroscopic redshift'
        ylabel = r'KiDS-bright ANNZ redshift'

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        
        plt.xlim([z_min, z_max])
        plt.ylim([z_min, z_max])
        
        plotfilename = '/%s/KiDS_GAMA_redshift_2Dhist_galtypes'%plot_path

        # Save plot
        for ext in ['png']:
            plotname = '%s.%s'%(plotfilename, ext)
            plt.savefig(plotname, format=ext, bbox_inches='tight')
            
        print('Written plot:', plotname)

        plt.show()
        plt.clf
        
        ## Plot 2D mass histogram of ellipticals and spirals

        plt.scatter(logmstar_gama_spir, logmstar_kids_spir, marker='.', alpha=0.01, \
                                            color='blue', label = 'Spirals')
        plt.scatter(logmstar_gama_ell, logmstar_kids_ell, marker='.', alpha=0.01, \
                                            color='red', label='Ellipticals')

        plt.plot(massline, massline, color='black', ls='--')
        plt.plot(massline, massline+0.2, color='grey', ls='--')
        plt.plot(massline, massline-0.2, color='grey', ls='--')

        # Define the labels for the plot
        xlabel = r'GAMA-II stellar mass [log(${\rm M/h_{%g}^{-2}M_{\odot}})$]'%(h*100)
        ylabel = r'KiDS-bright stellar mass [log(${\rm M/h_{%g}^{-2}M_{\odot}})$]'%(h*100)

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)

        plt.xlim([logmstar_min, logmstar_max])
        plt.ylim([logmstar_min, logmstar_max])
        
        plotfilename = '/%s/KiDS_GAMA_mass_2Dhist_galtypes'%plot_path
        
        # Save plot
        for ext in ['png']:
            plotname = '%s.%s'%(plotfilename, ext)
            plt.savefig(plotname, format=ext, bbox_inches='tight')
            
        print('Written plot:', plotname)

        plt.show()
        plt.clf
    
    
## Calculate the differences between the GAMA and KiDS redshifts/masses

# Mean redshifts of KiDS and GAMA
Zmean_gama = np.mean(Z_gama[Zmask_gama])
Zmean_kids = np.mean(Z_kids[Zmask_kids])
print()
print('specZ GAMA (mean):', Zmean_gama)
print('ANNZ KiDS (mean):', Zmean_kids)

if 'none' in splittype:
    
    # Redshift offset and standard deviation between KiDS and GAMA
    diff_Z = (Z_kids_matched-Z_gama_matched)
    std_diff_Z = np.std(diff_Z)
    print()
    print('Diff. Fraction Z:', np.mean(diff_Z/(1.+Z_gama_matched)))
    print('Stand. Dev. Z:', std_diff_Z)
    print( 'std(z)/(1+z):', std_diff_Z / (1.+np.mean(Z_kids_matched)) )

    # Stellar mass offset and standard deviation between KiDS and GAMA
    diff_logmstar = (logmstar_kids_matched-logmstar_gama_matched)
    std_diff_logmstar = np.std(diff_logmstar)
    print()
    print('Diff. Mstar:', np.mean(diff_logmstar))
    print('Stand. Dev. Mstar:', std_diff_logmstar)
    print('std(M)/M:', np.std( \
        10.**logmstar_kids_matched-10.**logmstar_gama_matched)/np.mean(10.**logmstar_gama_matched))

    # Estimate the mass uncertainty caused by the redshift uncertainty
    dist_kids_matched = (1.+Z_kids_matched) * cosmo.comoving_distance(Z_kids_matched)
    dist_gama_matched = (1.+Z_gama_matched) * cosmo.comoving_distance(Z_gama_matched)
    std_diff_dist = np.std(dist_kids_matched - dist_gama_matched)
    std_diff_lum = np.std(dist_kids_matched**2 - dist_gama_matched**2)
    print()
    print('dD/D:', std_diff_dist / np.mean(dist_gama_matched) )
    print('L+dL/L = %s dex'%(np.log10(std_diff_lum/np.mean(dist_gama_matched**2)+1.)) )

    # Estimate the mass uncertainty caused by the magnitude uncertainty
    magmask_kids = (0.<rmag_kids_sersic)&(rmag_kids_sersic<30.)&(0.<rmag_kids_auto)&(rmag_kids_auto<20.)
    diff_rmag = rmag_kids_sersic[Zmask_kids*magmask_kids] - rmag_kids_auto[Zmask_kids*magmask_kids]
    
    std_ratio_flux = 10.**(0.4*np.std(diff_rmag))
    mean_ratio_flux = 10.**(0.4*np.mean(diff_rmag))
    
    print()
    print('Mag offset: mean(dm)=', np.mean(diff_rmag))
    print('Mag uncertainty: std(dm)=', np.std(diff_rmag))
    print(std_ratio_flux)
    print('Flux offset: mean(F+dF/F)= %g dex'%np.log10(std_ratio_flux))
    print('Flux uncertainty: std(F+dF/F)= %g dex'%np.log10(mean_ratio_flux))

    #plt.hist(logmstar_gama_matched, label=r'GAMA (matched)', bins=50, histtype='step', normed=1)

else:
    # Redshift offset of ellipticals and spirals
    diff_Z_ell = (Z_kids_ell-Z_gama_ell)
    diff_Z_spir = (Z_kids_spir-Z_gama_spir)
    mean_diff_Z_ell = np.mean(diff_Z_ell)
    mean_diff_Z_spir = np.mean(diff_Z_spir)
    print()
    print('dZ for ellipticals: %g'%( mean_diff_Z_ell ))
    print('dZ for spirals: %g'%( mean_diff_Z_spir ))
    #print('Difference in dZ: %g dex'%( np.log10(mean_diff_Z_ell/mean_diff_Z_spir) ))
    print('dZe+dZs/Z:', (np.abs(mean_diff_Z_ell)+np.abs(mean_diff_Z_spir)) / \
        np.mean(Z_gama_matched))
    
    # Stellar mass offset of ellipticals and spirals
    diff_mstar_ell = (10.**logmstar_kids_ell-10.**logmstar_gama_ell)
    diff_mstar_spir = (10.**logmstar_kids_spir-10.**logmstar_gama_spir)
    diff_logmstar_ell = (logmstar_kids_ell-logmstar_gama_ell)
    diff_logmstar_spir = (logmstar_kids_spir-logmstar_gama_spir)
    mean_diff_mstar_ell = np.mean(diff_mstar_ell)
    mean_diff_mstar_spir = np.mean(diff_mstar_spir)
    mean_diff_mstar_gama = np.mean(10**logmstar_gama_ell) / np.mean(10**logmstar_gama_spir)
    mean_diff_mstar_kids = np.mean(10**logmstar_kids_ell) / np.mean(10**logmstar_kids_spir)
    
    print()
    print('M_ell/M_spir (GAMA): %g (%g dex)'%(mean_diff_mstar_gama, np.log10(mean_diff_mstar_gama) ))
    print('M_ell/M_spir (KiDS): %g (%g dex)'%(mean_diff_mstar_kids, np.log10(mean_diff_mstar_kids) ))
    print('dlogM for ellipticals: %g'%( np.mean(diff_logmstar_ell) ))
    print('dlogM for spirals: %g'%( np.mean(diff_logmstar_spir) ))
    #print('Difference in dlogM: %g dex'%( np.log10(mean_diff_logmstar_ell/mean_diff_logmstar_spir) ))
    print('dMe+dMs/M:', (np.abs(mean_diff_mstar_ell)+np.abs(mean_diff_mstar_spir)) / \
        np.mean(10.**logmstar_gama_matched))
