#!/usr/bin/python

import gc
import numpy as np
import os
import sys
import time

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
from astropy.io import ascii, fits as pyfits

O_matter, O_lambda, h = [0.315, 0.685, 0.7]
cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

#path_lenscat = '/data2/brouwer/KidsCatalogues/KV450_Lephare_Masked_trim_ZBlt0p6.fits'
path_lenscat = '/data2/brouwer/MergedCatalogues/GAMACatalogue_2.0.fits'

lenscat = pyfits.open(path_lenscat, ignore_missing_end=True)[1].data
galZlist = np.array(lenscat['Z'])

galZbins = np.sort(np.unique(galZlist)) # Find and sort the unique redshift values
Dclbins = np.array((cosmo.comoving_distance(galZbins).to('pc')).value) # Calculate the corresponding distances
Dcllist = Dclbins[np.digitize(galZlist, galZbins)-1] # Assign the appropriate Dcl to all lens redshifts

print('Zbins:', galZbins)
print('Dclbins:', Dclbins)

try:
    # Testing whether new-old distances gives zero
    Dcllist_old = np.array((cosmo.comoving_distance(galZlist).to('pc')).value)
    print('Null test:', np.sum(Dcllist-Dcllist_old))
except:
    'No null test possible'
