#!/usr/bin/python

"Module to compute the KiDS shear/ESD profiles of a lens selection."

# Import the necessary libraries
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
from collections import Counter

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

import modules_EG as utils
import treecorr

# Import constants
G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')
inf = np.inf
   
h, O_matter, O_lambda = [0.7, 0.325, 0.685]
cosmo = LambdaCDM(H0=h*100, Om0=O_matter, Ode0=O_lambda)


## Configuration

# Data selection
cat = 'mice' # Select the lens catalogue (kids/gama/mice)

"""
# Profile selection
Runit = 'arcmin' # Select distance unit (shear: arcsec/arcmin/degrees/hours/radians, ESD: pc/kpc/Mpc)
Rmin = 1. # Minimum radius (in selected unit). Note: MICE lensing is only accurate down to ~1 arcmin.
Rmax = 100. # Minimum radius (in selected unit)
Nbins = 20 # Number of radial bins
"""
# Profile selection
Runit = 'kpc' # Select distance unit (shear: arcsec/arcmin/degrees/hours/radians, ESD: pc/kpc/Mpc)
Rmin = 20 # Minimum radius (in selected unit). Note: MICE lensing is only accurate down to ~0.2 Mpc (at z=0.2).
Rmax = 2000 # Maximum radius (in selected unit)
Nbins = 10 # Number of radial bins

plot = True
Rlog = True

# Lens selection
paramnames = np.array(['lmstellar'])
maskvals = np.array([ [8.5,11.] ])
#maskvals = np.array([ [-inf, inf] ])

srcZmin, srcZmax = [0.1, 0.9]

path_output = '/data2/brouwer/shearprofile/Lensing_results/EG_results_Oct18/%s'%(cat)


## Pipeline

# Import lens catalog
fields, path_lenscat, lenscatname, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h)

# Define radial bins
Rbins, Rcenters, Rarcmin, Rarcmax, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins, Rlog)

"""
## Weighted profiles

weightfile = '%s/%s'%(path_lenscat, weightcatname)
print('Weighted by:', weightfile, theta)

weightcat = pyfits.open(weightfile, memmap=True)[1].data
lensweights = weightcat['Wtheta%g'%theta]
weightPthetas = weightcat['Ptheta%g'%theta]

"""
lensweights = np.ones(len(lensZ))

#for f in range(len(fields)):

# Boundaries of the field
#fieldRAs, fieldDECs = [[i*20.+5.,(i+1)*20.+5.], [j*20.+5.,(j+1)*20.+5.]]
fieldRAs, fieldDECs = [[0., 10.], [0., 10.]]

# Selecting the galaxies lying within this field
fieldmask_lens = (fieldRAs[0] < lensRA)&(lensRA < fieldRAs[1]) & (fieldDECs[0] < lensDEC)&(lensDEC < fieldDECs[1])


# Importing the sources
path_srccat = '/data2/brouwer/KidsCatalogues'
if 'mice' in cat:
    #srccatname = 'mice_source_catalog_dc.fits'
    srccatname = 'mice2_source_catalog_100deg2.fits'
    srcRA, srcDEC, srcZ, srcDc, rmag_src, rmag_abs_src, e1, e2, logmstar_src =\
    utils.import_micecat(path_srccat, srccatname, h)
else:
    srccatname = 'KiDS-450_mask_%s.fits'%fields[f]
    srcRA, srcDEC, srcZ, rmag, e1, e2, weight =\
    utils.import_srccat(path_srccat, srccatname)
    
print(srcRA, srcDEC, srcDc, e1, e2)
print(lensRA, lensDEC, lensDc)

# Creating the source mask
srcmask = (srcZmin < srcZ) & (srcZ < srcZmax) & (rmag_src > 20.) & (rmag_abs_src > -19.3)
fieldmask_src = (fieldRAs[0]-1. < srcRA)&(srcRA < fieldRAs[1]+1.) & (fieldDECs[0]-1. < srcDEC)&(srcDEC < fieldDECs[1]+1.)
srcmask = srcmask * fieldmask_src

# Masking the sources
srcRA, srcDEC, srcZ, srcDc, e1, e2  = \
srcRA[srcmask], srcDEC[srcmask], srcZ[srcmask], srcDc[srcmask], e1[srcmask], e2[srcmask]

# Creating the lens mask
lensmask, filename_var = utils.define_lensmask(paramnames, maskvals, path_lenscat, lenscatname)
lensmask = lensmask*fieldmask_lens

# Masking the lenses
lensRA, lensDEC, lensZ, lensDc, lensweights = \
lensRA[lensmask], lensDEC[lensmask], lensZ[lensmask], lensDc[lensmask], lensweights[lensmask]
lensDa = lensDc/(1+lensZ)

if ('pc' not in Runit) and ('acc' not in Runit):

    ### Calculate shear profile
    config = {'min_sep': Rarcmin, 'max_sep': Rarcmax, 'nbins': Nbins, 'sep_units': Runit, 'verbose': 2}
    ng = treecorr.NGCorrelation(config)
    
    # Defining the source sample
    srccat = treecorr.Catalog(ra=srcRA, dec=srcDEC, ra_units='deg', dec_units='deg', g1=e1, g2=e2)
    
    # Defining the lens sample
    lenscat = treecorr.Catalog(ra=lensRA, dec=lensDEC, ra_units='deg', dec_units='deg')#, w=lensweights)

    # Compute the cross-correlation
    ng.process(lenscat,srccat)
    
    # The resulting shear profile
    Rnom, gamma_t, gamma_x, gamma_error, Nsrc = \
    [ng.rnom, ng.xi, ng.xi_im, np.sqrt(ng.varxi), ng.npairs]
    
    print('Nsources:', len(srcRA))

"""
#if 'acc' in Runit:
if 'pc' in Runit:

    ### Calculate the ESD profile
    config = {'min_sep': Rarcmin, 'max_sep': Rarcmax, 'nbins': Nbins,\
        'metric': 'Rlens', 'min_rpar': 0, 'verbose': 0}
#   config = {'min_sep': Rarcmin, 'max_sep': Rarcmax, 'nbins': Nbins, 'sep_units': 'arcmin', 'verbose': 0}

    ng = treecorr.NGCorrelation(config)
    print('Rbins (pc):', Rarcmin, Rarcmax, Nbins)
    
    # Define the lens redshift bins
    nZbins = 20
    
    Zlims = np.linspace(np.amin(lensZ), np.amax(lensZ), nZbins+1)
    Zbins = Zlims[0:-1] + np.diff(Zlims)/2.
    Dcbins = (cosmo.comoving_distance(Zbins).to('pc')).value
    print('Zbins:', Zlims, 'dZ:', (np.amax(Zlims)-np.amin(Zlims))/nZbins)
    
    Zmasks = [((Zlims[b] < lensZ) & (lensZ <= Zlims[b+1])) for b in np.arange(nZbins)]
    Ngals = [np.sum(Zmasks[b]) for b in np.arange(nZbins)]
    
    print('Ngalaxies:', Ngals, np.sum(Ngals), len(srcZ))
    
    # For every source redshift bin...
    for b in np.arange(nZbins):
    
        # Select lenses in the redshift bin
        binZ, binDc, binDa = Zbins[b], Dcbins[b], Dcbins[b]/(1+Zbins[b])
        Zmask = Zmasks[b]
        
        print('Zbin %i: %g - %g'%(b+1, Zlims[b], Zlims[b+1]))
        print('Number of lenses:', np.sum(Zmask))
        
        if np.sum(Zmask) > 0:
            
            # Calculate Sigma_crit for every source
            DlsoDs = (srcDc - binDc)/srcDc
            Sigma_crit = (c.value**2)/(4*np.pi*G.value) * 1/(binDa*DlsoDs)
            #Sigma_crit = np.ones(len(srcZ))
            
            # Define the lens sample
            lenscat = treecorr.Catalog(ra=lensRA[Zmask], dec=lensDEC[Zmask], ra_units='deg', dec_units='deg',\
            r=lensDc[Zmask])
            
            # Define the source sample
            srccat = treecorr.Catalog(ra=srcRA, dec=srcDEC, ra_units='deg', dec_units='deg', g1=Sigma_crit*e1, g2=Sigma_crit*e2,\
            r=srcDc, w=1/Sigma_crit**2.)
            
            # Compute the cross-correlation for this source redshift bin
            ng.process_cross(lenscat,srccat)
    
    varg = treecorr.catalog.calculateVarG(srccat)
    
    print('VarG:', varg)
    
    # Finalize by applying the total weight
    ng.finalize(varg)

    # The resulting ESD profile
    Rnom, gamma_t, gamma_x, gamma_error, Nsrc = \
    [ng.rnom, ng.xi, ng.xi_im, np.sqrt(ng.varxi), ng.npairs]

"""
if 'pc' in Runit:

    ### Calculate the ESD profile

    config = {'min_sep': Rarcmin, 'max_sep': Rarcmax, 'nbins': Nbins,\
        'metric': 'Rlens', 'min_rpar': 0, 'verbose': 0}

    kg = treecorr.KGCorrelation(config)
    print('Rbins (pc):', Rarcmin, Rarcmax, Nbins)
    
    # Define the source redshift bins
    nZbins = 20
    
    Zlims = np.linspace(srcZmin, srcZmax, nZbins+1)
    Zbins = Zlims[0:-1] + np.diff(Zlims)/2.
    Dcbins = (cosmo.comoving_distance(Zbins).to('pc')).value
    print('Zbins:', Zlims, 'dZ:', (np.amax(Zlims)-np.amin(Zlims))/nZbins)
    
    Zmasks = [(Zlims[b] < srcZ) & (srcZ <= Zlims[b+1]) for b in np.arange(nZbins)]
    Ngals = [np.sum(Zmasks[b]) for b in np.arange(nZbins)]
    
    print('Ngalaxies:', Ngals, np.sum(Ngals), len(srcZ))
    
    lenscat_list = []
    srccat_list = []

    # For every source redshift bin...
    for b in np.arange(nZbins):

        # Select sources in the redshift bin
        binZ, binDc, binDa = Zbins[b], Dcbins[b], Dcbins[b]/(1+Zbins[b])
        Zmask = Zmasks[b]

        print('Zbin %i: %g - %g'%(b+1, Zlims[b], Zlims[b+1]))
        print('Number of sources:', np.sum(Zmask))
        
        if np.sum(Zmask) > 0:
            
            # Calculate Sigma_crit for every lens
            DlsoDs = (binDc - lensDc)/binDc
            Sigma_crit = (c.value**2)/(4*np.pi*G.value) * 1/(lensDa*DlsoDs)
            #Sigma_crit = np.ones(len(lensZ))
            
            # Define the lens sample
            lenscat = treecorr.Catalog(ra=lensRA, dec=lensDEC, ra_units='deg', dec_units='deg', \
            k=Sigma_crit, w=1/Sigma_crit**2., r=lensDc)
            
            # Define the source sample
            srccat = treecorr.Catalog(ra=srcRA[Zmask], dec=srcDEC[Zmask], ra_units='deg', dec_units='deg', \
            g1=e1[Zmask], g2=e2[Zmask], r=srcDc[Zmask])
            
            srccat_list.append(srccat)
            lenscat_list.append(lenscat)
            
            # Compute the cross-correlation for this source redshift bin
            kg.process_cross(lenscat,srccat)

    vark = treecorr.catalog.calculateVarK(lenscat)
    varg = treecorr.catalog.calculateVarG(srccat_list)

    print('VarK:', vark)
    print('VarG:', varg)

    # Finalize by applying the total weight
    kg.finalize(vark, varg)

    # The resulting ESD profile
    Rnom, gamma_t, gamma_x, gamma_error, Nsrc = \
    [kg.rnom, kg.xi, kg.xi_im, np.sqrt(kg.varxi), kg.npairs]
#"""

print('Lens-source pairs:', np.sum(Nsrc))
print()

# Write the result to a file
#filename_output = '%s/lenssel-%s_Rbins-%i-%g-%g-%s_Zbins-%g_lenssplit'%(path_output, filename_var, Nbins, Rmin, Rmax, Runit, nZbins)

filename_output = '%s/%s/zcgal_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins%i_%g_%g_%s/shearcovariance/No_bins_A'\
                %(path_output, filename_var, Nbins, Rmin, Rmax, Runit)

path_output = filename_output.rsplit('/', 1)[0]

if not os.path.isdir(path_output):
    os.makedirs(path_output)
    print 'Creating new folder:', path_output

bias = np.ones(len(gamma_t))
#gamma_error = np.zeros(len(gamma_t))

utils.write_stack(filename_output+'.txt', Rcenters, Runit, gamma_t, gamma_x, \
gamma_error, bias, h, Nsrc)

# Plot the resulting shear profile
utils.write_plot(Rcenters, gamma_t, gamma_x, gamma_error, None, filename_output, Runit, Rlog, plot, h)
