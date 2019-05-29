#!/usr/bin/python

# Import the necessary libraries
import sys
import numpy as np
import pyfits
import os

import random
from astropy.cosmology import LambdaCDM
import modules_EG as utils

# These lists will contain all the random tiles
RAlist = np.array([])
DEClist = np.array([])

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685
cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)
cat = 'kids' # Select the lens catalogue (kids/gama/mice)

# Path to the basic random tiles
path_randoms = '/data/users/brouwer/MaskCatalogues/Randoms_K1000/'
tilenames = np.array(os.listdir(path_randoms))
print(tilenames)
Ntiles = len(tilenames)

for t in range(Ntiles):
#for t in range(1):

    print(t)

    # Full directory & name of the random tile
    randomfile = '%s/%s'%(path_randoms, tilenames[t])
    randomcat = pyfits.open(randomfile, memmap=True)[1].data
    
    # List of the randoms in the tile
    tileRA = randomcat['ALPHA_J2000']
    tileDEC = randomcat['DELTA_J2000']
    
    # Apply the mask (28668)
    # 9-band no AW-r-band masks (used to create the WL mosaic catalogues):
    # 4,8,16,32,64,128,256,512,1024,2048,8192,16384 = 28668
    Mask = randomcat['MASK']
    
    Masked_index = (Mask & 28668) > 0. # 27676 the sum of the masks that you are interested in
    Mask[Masked_index]=1. # If you want to change it to a binary mask
    Mask[Mask>1.]=0.
    tilemask = (Mask==0.)
    
    # Add the tile to the list
    RAlist = np.append(RAlist, tileRA[tilemask])
    DEClist = np.append(DEClist, tileDEC[tilemask])

# The total number of random galaxies
Nrandoms = len(RAlist)
IDlist = np.arange(Nrandoms)
print('Number of random galaxies:', Nrandoms)

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

# Create redshifts for the randoms
#Nmult = int(np.ceil(Nrandoms/len(lensZ))) # Multiplier to have enough redshift values
#Zlist = np.array(random.sample(list(lensZ)*Nmult, Nrandoms))
#print(Nrandoms/len(lensZ))
#print(Nmult)

# Write everything to a catalogue
outputnames = np.array(['ID', 'RA', 'DEC'])#, 'Z'])
formats = np.array(['D']*len(outputnames))
output = np.array([IDlist, RAlist, DEClist])#, Zlist])
print(outputnames, formats, output)

filename = '/data/users/brouwer/LensCatalogues/%s_basic_randoms'%cat
utils.write_catalog('%s.fits'%filename, outputnames, formats, output)
