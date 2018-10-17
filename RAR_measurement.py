#!/usr/bin/python

"Module to compute the shear/ESD profiles of a lens selection."

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
Runit = 'Mpc' # Select distance unit (shear: arcsec/arcmin/degrees/hours/radians, ESD: pc/kpc/Mpc)
Rmin = 0.2 # Minimum radius (in selected unit). Note: MICE lensing is only accurate down to ~0.2 Mpc (at z=0.2).
Rmax = 20 # Maximum radius (in selected unit)
Nbins = 20 # Number of radial bins
"""
# Profile selection
Runit = 'mps2' # Select distance unit (shear: arcsec/arcmin/degrees/hours/radians, ESD: pc/kpc/Mpc)
Rmin = 1e-15 # Minimum radius (in selected unit). Note: MICE lensing is only accurate down to ~0.2 Mpc (at z=0.2).
Rmax = 1e-12 # Maximum radius (in selected unit)
Nbins = 20 # Number of radial bins
"""

plot = True
Rlog = True

# Lens selection
paramnames = np.array(['lmstellar'])
maskvals = np.array([ [10.9,11.1] ])
#maskvals = np.array([ [-inf, inf] ])

srcZmin, srcZmax = [0.1, 0.9]

path_output = '/data2/brouwer/shearprofile/EG_results_Sep18/%s'%(cat)


## Pipeline

# Import lens catalog
fields, path_lenscat, lenscatname, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h)

# Define radial bins
Rbins, Rcenters, Rmin, Rmax, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins, Rlog)

#for f in range(len(fields)):

# Boundaries of the field
#fieldRAs, fieldDECs = [[i*20.+5.,(i+1)*20.+5.], [j*20.+5.,(j+1)*20.+5.]]
fieldRAs, fieldDECs = [[5., 25.], [5., 25.]]

# Selecting the galaxies lying within this field
fieldmask_lens = (fieldRAs[0] < lensRA)&(lensRA < fieldRAs[1]) & (fieldDECs[0] < lensDEC)&(lensDEC < fieldDECs[1])


# Importing the sources
path_srccat = '/data2/brouwer/KidsCatalogues'
if 'mice' in cat:
    #srccatname = 'mice_source_catalog_dc.fits'
    srccatname = 'mice2_source_catalog_big.fits'
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


srcR, incosphi, insinphi = calc_shear(lensDa, lensRA, lensDEC, srcRA, srcDEC, e1, e2, Rmin, Rmax, logMb)


# Write the result to a file
filename_output = '%s/lenssel-%s_Rbins-%i-%g-%g-%s_Zbins-%g_lenssplit'%(path_output, filename_var, Nbins, Rmin, Rmax, Runit, nZbins)

if not os.path.isdir(path_output):
    os.makedirs(path_output)
    print 'Creating new folder:', path_output

bias = np.ones(len(gamma_t))
#gamma_error = np.zeros(len(gamma_t))

utils.write_stack(filename_output+'.txt', Rcenters, Runit, gamma_t, gamma_x, \
gamma_error, bias, h, Nsrc)

# Plot the resulting shear profile
utils.write_plot(Rcenters, gamma_t, gamma_x, gamma_error, None, filename_output, Runit, Rlog, plot, h)

