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
   
h, O_matter, O_lambda = [0.7, 0.25, 0.75]
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
Runit = 'pc' # Select distance unit (arcmin/Xpc/acc)
Rmin = 1. # Minimum radius (in selected unit). Note: MICE lensing is only accurate down to ~0.2 Mpc (at z=0.2).
Rmax = 100. # Maximum radius (in selected unit)
Nbins = 20 # Number of radial bins
#"""

plot = False
Rlog = True

# Lens selection
paramnames = np.array(['abs_mag_r'])
maskvals = np.array([ [-20., -19.5] ])
#maskvals = np.array([ [-inf, inf] ])

srcZmin, srcZmax = [0.1, 0.9]

path_output = '/data2/brouwer/shearprofile/EG_results_Sep18/%s'%(cat)


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
fieldRAs, fieldDECs = [[5., 25.], [5., 25.]]

# Selecting the galaxies lying within this field
fieldmask_lens = (fieldRAs[0] < lensRA)&(lensRA < fieldRAs[1]) & (fieldDECs[0] < lensDEC)&(lensDEC < fieldDECs[1])


# Importing the sources
path_srccat = '/data2/brouwer/KidsCatalogues'
if 'mice' in cat:
    srccatname = 'mice_source_catalog_dc.fits'
    srcRA, srcDEC, srcZ, srcDc, rmag_src, rmag_abs_src, e1, e2, logmstar_src =\
    utils.import_micecat(path_srccat, srccatname, h)
else:
    srccatname = 'KiDS-450_mask_%s.fits'%fields[f]
    srcRA, srcDEC, srcZ, rmag, e1, e2, weight =\
    utils.import_srccat(path_srccat, srccatname)
   
# Creating the source mask
srcmask = (srcZmin < srcZ) & (srcZ < srcZmax) & (rmag_src > 20.) & (rmag_abs_src > -19.3)
fieldmask_src = (fieldRAs[0]-5. < srcRA)&(srcRA < fieldRAs[1]+5.) & (fieldDECs[0]-5. < srcDEC)&(srcDEC < fieldDECs[1]+5.)
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


if 'pc' not in Runit:

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
    
else:
    
    """
    
    ### Calculate the ESD profile
    config = {'min_sep': Rarcmin, 'max_sep': Rarcmax, 'nbins': Nbins,\
        'metric': 'Rlens', 'min_rpar': 0, 'verbose': 2, 'log_file': 'treecorr_log.txt'}
    ng = treecorr.NGCorrelation(config)

    # Define the lens redshift bins
    lensZlims = np.linspace(0., 0.5, 100.)
    lensZbins = lensZlims[0:-1] + np.diff(lensZlims)/2.
    lensDcbins = (cosmo.comoving_distance(lensZbins).to('pc')).value
    lensDabins = lensDcbins/(1.+lensZbins)
    
    gamma_t = np.zeros(Nbins)
    gamma_x = np.zeros(Nbins)
    gamma_error = np.zeros(Nbins)
    Nsrc = np.zeros(Nbins)
    weight = np.zeros(Nbins)
    
    # For every source redshift bin...
    for b in np.arange(len(lensZbins)):
#    for b in [14]:


        # Select lenses in the redshift bin
        binZ, binDc, binDa = lensZbins[b], lensDcbins[b], lensDabins[b]
        lensZmask = (lensZlims[b] < lensZ) & (lensZ < lensZlims[b+1])
        
        print('Zbin:', b+1, binZ)
        print('Number of lenses:', np.sum(lensZmask))
        if np.sum(lensZmask) > 0:
            
            # Calculate Sigma_crit for every lens
            DlsoDs = (srcDc - binDc)/srcDc
            Sigma_crit = (c.value**2)/(4*np.pi*G.value) * 1/(binDa*DlsoDs)

            # Define the lens sample
            lenscat = treecorr.Catalog(ra=lensRA[lensZmask], dec=lensDEC[lensZmask], r=lensDc[lensZmask], ra_units='deg', dec_units='deg')
            
            # Define the source sample
            srccat = treecorr.Catalog(ra=srcRA, dec=srcDEC, r=srcDc, ra_units='deg', dec_units='deg', \
            g1=e1*Sigma_crit, g2=e2*Sigma_crit, w=1/Sigma_crit**2.)
            
            # Compute the cross-correlation for this source redshift bin
            ng.process(lenscat,srccat)
            
            gamma_t += ng.xi/ng.weight
            gamma_x += ng.xi_im/ng.weight
            gamma_error += np.sqrt(ng.varxi)/ng.weight
            Nsrc += ng.npairs
            weight += ng.weight
            
            print(gamma_t)
    
    # The resulting ESD profile
    gamma_t, gamma_x, gamma_error, Nsrc = \
    [gamma_t/weight, gamma_x/weight,\
    gamma_error/weight, Nsrc]
    
    
    """
    
    ### Calculate the ESD profile

#    config = {'min_sep': Rarcmin, 'max_sep': Rarcmax, 'nbins': Nbins,\
#        'metric': 'Rlens', 'min_rpar': 0, 'verbose': 2, 'log_file': 'treecorr_log.txt'}
    config = {'min_sep': Rarcmin, 'max_sep': Rarcmax, 'nbins': Nbins, 'sep_units': 'arcmin', 'verbose': 0}
    ng = treecorr.NGCorrelation(config)
    print(Rarcmin, Rarcmax, Nbins)

    # Define the source redshift bins
    #srcZbins = np.arange(0.025,3.5,0.05)
    #srcZlims = np.arange(0.,3.55,0.05)
    #Nsrcbins = 10
    #srcZlims = np.linspace(srcZmin, srcZmax, Nsrcbins+1)
    #srcZbins = srcZlims[0:-1] + np.diff(srcZlims)/2.
    #srcDcbins = (cosmo.comoving_distance(srcZbins).to('pc')).value
    #print(srcZlims)
   
   
    #lenscat_list = []
    #srccat_list = []
    #gammat_list = []
    Nsources = 0.
    
    #srcZmasks = [(srcZlims[b] < srcZ) & (srcZ <= srcZlims[b+1]) for b in np.arange(len(srcZbins))]
    
    """
    lens_cats = [ treecorr.Catalog(ra=lensRA, dec=lensDEC, ra_units='deg', dec_units='deg') for srcZmask in srcZmasks]
    
    source_cats = [ treecorr.Catalog(ra=srcRA[srcZmask], dec=srcDEC[srcZmask], ra_units='deg', dec_units='deg', \
            g1=e1[srcZmask], g2=e2[srcZmask]) for srcZmask in srcZmasks]
    """
    
    #xlim = int(3490000)
    #xlim = int(1e2)
    #xlim = int(1)
    #xlim = int(len(srcRA)/2)
    xmin = 0
    xlim = int(1e4)
    xmax = int(2e4)
    #xmax = 3490727
    
    # Split calculation
    
    srccat1 = treecorr.Catalog(ra=srcRA[xmin:xlim], dec=srcDEC[xmin:xlim], ra_units='deg', dec_units='deg', \
            g1=e1[xmin:xlim], g2=e2[xmin:xlim])
   
    srccat2 = treecorr.Catalog(ra=srcRA[xlim:xmax], dec=srcDEC[xlim:xmax], ra_units='deg', dec_units='deg', \
        g1=e1[xlim:xmax], g2=e2[xlim:xmax])

    source_cats = [srccat1, srccat2]

    lens_cats = [treecorr.Catalog(ra=lensRA, dec=lensDEC, ra_units='deg', dec_units='deg')]*len(source_cats)

    for c in np.arange(len(source_cats)):
        ng.process_cross(lens_cats[c], source_cats[c])
        
        print('Sources:', [xmin, xlim, xmax][c:c+2])
        print('Lens-source pairs:', np.sum(ng.npairs))
        print()

    varg = treecorr.calculateVarG(source_cats)
    ng.finalize(varg)
    pairs_split = ng.npairs
    print(pairs_split)
    
    # Combined calculation
    ng = treecorr.NGCorrelation(config)
    
    source_cat = treecorr.Catalog(ra=srcRA[xmin:xmax], dec=srcDEC[xmin:xmax], ra_units='deg', dec_units='deg', \
        g1=e1[xmin:xmax], g2=e2[xmin:xmax])
    
    lens_cat = treecorr.Catalog(ra=lensRA, dec=lensDEC, ra_units='deg', dec_units='deg')
    
    ng.process(lens_cat, source_cat)
    pairs_combined = ng.npairs
    print(pairs_combined)
    
    print('Sources:', [xmin, xmax])
    print('Lens-source pairs:', np.sum(ng.npairs))
    print()
    
    pairs_fraction = 2.*(pairs_split - pairs_combined)/(pairs_split + pairs_combined)*100.
    print(pairs_fraction)
    plt.plot(pairs_fraction)
    plt.show()
    
    #treecorr.Catalog(ra=srcRA[srcZmask], dec=srcDEC[srcZmask], ra_units='deg', dec_units='deg', \
    #        g1=e1[srcZmask], g2=e2[srcZmask]) for srcZmask in srcZmasks
    

        
    #for c1, c2 in zip(lens_cats, source_cats):
    #    ng.process_cross(c1,c2)
    
    

    """    
    # For every source redshift bin...
    for b in np.arange(len(srcZbins)):

        # Select sources in the redshift bin
        #binZ, binDc = srcZbins[b], srcDcbins[b]
        srcZmask = (srcZlims[b] < srcZ) & (srcZ <= srcZlims[b+1])
        
        print('Zbin: %i: %g - %g'%(b+1, srcZlims[b], srcZlims[b+1]))
        print('Number of sources:', np.sum(srcZmask))
        Nsources += np.sum(srcZmask)

        if np.sum(srcZmask) > 0:
            
            # Calculate Sigma_crit for every lens
            #DlsoDs = (binDc - lensDc)/binDc
            #Sigma_crit = (c.value**2)/(4*np.pi*G.value) * 1/(lensDa*DlsoDs)
            #Sigma_crit = np.ones(len(lensZ))
            
            # Define the lens sample
            lenscat = treecorr.Catalog(ra=lensRA, dec=lensDEC, ra_units='deg', dec_units='deg')# r=lensDc, w=1/Sigma_crit**2.)
            
            # Define the source sample
            srccat = treecorr.Catalog(ra=srcRA[srcZmask], dec=srcDEC[srcZmask], ra_units='deg', dec_units='deg', \
            g1=e1[srcZmask], g2=e2[srcZmask])# r=srcDc[srcZmask])
            srccat_list.append(srccat)
            
            # Compute the cross-correlation for this source redshift bin
            ng.process_cross(lenscat,srccat)
            
            # The resulting shear profile
            #Rnom, gamma_t, gamma_x, gamma_error, Nsrc = \
            #[ng.rnom, ng.xi, ng.xi_im, np.sqrt(ng.varxi), ng.npairs]
            
            #print(gamma_t)
            #gammat_list.append(gamma_t)
    
    #vark = treecorr.catalog.calculateVarK(lenscat_list)
    varg = treecorr.catalog.calculateVarG(srccat_list)
    
    print('Nsources:', Nsources)
    
    # Finalize by applying the total weight
    ng.finalize(varg)

    """
        
    # The resulting ESD profile
    Rnom, gamma_t, gamma_x, gamma_error, Nsrc = \
    [ng.rnom, ng.xi, ng.xi_im, np.sqrt(ng.varxi), ng.npairs]
    
    ng.write('ng_test.txt')
    
print('Total:', np.sum(Nsrc))
print()

# Write the result to a file
filename_output = '%s/lenssel-%s_Rbins-%i-%g-%g-%s'%(path_output, filename_var, Nbins, Rmin, Rmax, Runit)

if not os.path.isdir(path_output):
    os.makedirs(path_output)
    print 'Creating new folder:', path_output

bias = np.ones(len(gamma_t))
#gamma_error = np.zeros(len(gamma_t))

utils.write_stack(filename_output+'.txt', Rcenters, Runit, gamma_t, gamma_x, \
gamma_error, bias, h, Nsrc)

# Plot the resulting shear profile
utils.write_plot(Rcenters, gamma_t, gamma_x, gamma_error, filename_output, Runit, Rlog, plot, h)
