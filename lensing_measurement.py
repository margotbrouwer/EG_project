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

# Profile selection
Runit = 'arcmin' # Select distance unit (arcmin/Xpc/acc)
Rmin = 2 # Minimum radius (in selected unit)
Rmax = 100 # Minimum radius (in selected unit)
Nbins = 20 # Number of radial bins

plot = True
Rlog = True

# Lens selection
paramnames = np.array(['abs_mag_r'])
maskvals = np.array([ [-20, np.inf] ])

path_output = '/data2/brouwer/shearprofile/EG_results_Sep18/%s'%(cat)


## Pipeline

# Import lens catalog
fields, path_lenscat, lenscatname, lensRA, lensDEC, lensZ, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h)

# Define radial bins
Rbins, Rcenters, Rarcmin, Rarcmax = utils.define_Rbins(Runit, Rmin, Rmax, Nbins, Rlog)


"""
## Weighted profiles

weightfile = '%s/%s'%(path_lenscat, weightcatname)
print('Weighted by:', weightfile, theta)

weightcat = pyfits.open(weightfile, memmap=True)[1].data
lensweights = weightcat['Wtheta%g'%theta]
weightPthetas = weightcat['Ptheta%g'%theta]

"""
lensweights = np.ones(len(lensZ))

for f in range(len(fields)):
    
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
    srcmask = (0.1 < srcZ) & (srcZ < 0.9) & (rmag_src > 20.) & (rmag_abs_src > -19.3)
    fieldmask_src = (fieldRAs[0]-5. < srcRA)&(srcRA < fieldRAs[1]+5.) & (fieldDECs[0]-5. < srcDEC)&(srcDEC < fieldDECs[1]+5.)
    srcmask = srcmask * fieldmask_src
    
    # Masking the sources
    srcRA, srcDEC, srcZ, srcDc, e1, e2  = \
    srcRA[srcmask], srcDEC[srcmask], srcZ[srcmask], srcDc[srcmask], e1[srcmask], e2[srcmask]
    srcDc = srcDc*1e6 # Convert source distances from Mpc to pc
    
    print('Number of sources:', len(srcZ))
    
    # Creating the lens mask
    lensmask, filename_var = utils.define_lensmask(paramnames, maskvals, path_lenscat, lenscatname)
    lensmask = lensmask*fieldmask_lens
   
    #srcZbins = np.arange(0.025,3.5,0.05)
    #nsrcZ = np.hist(srcZ,srcZbins)
    
    if 'pc' in Runit:
        lensZhist, lensZlims = np.hist(lensZ, bins=10)
        lensZbins = lensZlims[0:-1] + np.diff(lensZlims)/2.
        print(lensZhist, lensZbins)
    else:
        lensZbins = [0.]
    
    # For every lens redshift bin...
    for l in np.arange(len(lensZbins)):
        
        if 'pc' in Runit:
            
            # Select lenses in the redshift bin            
            lensZmask = (lensZlims[l] < lensZ) & (lensZ < lensZlims[l+1])
            lensmask = lensmask * lensZmask
            
            # Only use sources behind the lens
            Zbin = lensZbins[l]
            srcZmask = (srcZ > Zbin)
            srcmask = srcmask * srcZmask
            
            # Calculate the lens distance
            lensDc = (cosmo.comoving_distance(Zbin).to('pc')).value
            lensDa = lensDc / (1+Zbin)
            
            # Calculating Sigma_crit for every source
            DlsoDs = (srcDc - lensDc)/srcDc
            Sigma_crit = (c.value**2)/(4*np.pi*G.value) * 1/(lensDa*DlsoDs)
        else:
            Sigma_crit = np.ones(len(srcDc))

        # Masking the lenses
        RA, DEC, Z, weights = lensRA[lensmask], lensDEC[lensmask], lensZ[lensmask], lensweights[lensmask]
        
        """
        ### Calculate shear profile
        
        # Defining the source sample
        srccat = treecorr.Catalog(ra=srcRA, dec=srcDEC, ra_units='deg', dec_units='deg', g1=e1*Sigma_crit, g2=e2*Sigma_crit, w=1/Sigma_crit**2.)
        
        # Defining the lens sample
        lenscat = treecorr.Catalog(ra=RA, dec=DEC, ra_units='deg', dec_units='deg')#, w=weights)
        
        config = {'min_sep': Rarcmin, 'max_sep': Rarcmax, 'nbins': Nbins, 'sep_units': 'arcmin', 'verbose': 2}
        ng = treecorr.NGCorrelation(config)
        ng.process(lenscat,srccat) # Compute the cross-correlation.
        
        output_temp = 'temp_treecor.txt'
        ng.write(output_temp) # Write out to a file.
        shearfile = np.loadtxt(output_temp).T
        """
        
        ### Calculate ESD profile
        
        # Defining the source sample
        srccat = treecorr.Catalog(ra=srcRA, dec=srcDEC, ra_units='deg', dec_units='deg', g1=e1*Sigma_crit, g2=e2*Sigma_crit, w=1/Sigma_crit**2.)
        
        # Defining the lens sample
        lenscat = treecorr.Catalog(ra=RA, dec=DEC, ra_units='deg', dec_units='deg')#, w=weights)
        
        config = {'min_sep': Rarcmin, 'max_sep': Rarcmax, 'nbins': Nbins, 'sep_units': 'arcmin', 'verbose': 2}
        ng = treecorr.NGCorrelation(config)
        ng.process(lenscat,srccat) # Compute the cross-correlation.
        
        output_temp = 'temp_treecor.txt'
        ng.write(output_temp) # Write out to a file.
        shearfile = np.loadtxt(output_temp).T

        
        Rbins, gamma_t, gamma_x, gamma_error, Nsrc = \
        [shearfile[0], shearfile[3], shearfile[4], np.sqrt(shearfile[5]), shearfile[7]]

    filename_output = '%s/%s'%(path_output, filename_var)

    if not os.path.isdir(path_output):
        os.makedirs(path_output)
        print 'Creating new folder:', path_output

    bias = np.ones(len(gamma_t))
    gamma_error = np.zeros(len(gamma_t))
    utils.write_stack(filename_output+'.txt', Rcenters, Runit, gamma_t, gamma_x, \
        gamma_error, bias, h, Nsrc)

# Plot the resulting shear profile

utils.write_plot(Rcenters, gamma_t, filename_output, Runit, Rlog, plot)
