#!/usr/bin/python

# Import the necessary libraries
import sys
import numpy as np
import pyfits
import os

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
Nrandoms = 10 # Number of randoms to be created

# Path to the basic random tiles
path_randoms = '/data/users/brouwer/MaskCatalogues/Randoms_K1000/'
tilenames = np.array(os.listdir(path_randoms))
Ntiles = len(tilenames)
print('Number of tiles:', Ntiles)


for t in range(Ntiles):
#for t in range(100):

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
    tilemask = np.logical_not((Mask & 28668) > 0.) # 27676 the sum of the masks that you are interested in
   
    # Add the tile to the list
    RAlist = np.append(RAlist, tileRA[tilemask])
    DEClist = np.append(DEClist, tileDEC[tilemask])

    print('Tile %i: %i galaxies (%i in total)'%(t+1, np.sum(tilemask), len(RAlist)))

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

# Mask the redshifts
Zmask = (0.<lensZ)&(lensZ<0.5)
lensZ = lensZ[Zmask]

# Create the redshift histogram
Zhist, Zbin_edges = np.histogram(lensZ, 100)
Zbins = Zbin_edges[0:-1]+np.diff(Zbin_edges)

# The total number of random galaxies
Ngals = len(lensZ)
IDcat = np.arange(Ngals)

# Shuffle random galaxies for the catalogues
shuffled = np.arange(len(RAlist))
np.random.shuffle(shuffled)
RAlist = RAlist[shuffled]
DEClist = DEClist[shuffled]

for r in range(Nrandoms):
    
    # Add the right number of galaxies to the catalogue
    RAcat = RAlist[r*Ngals:(r+1)*Ngals]
    DECcat = DEClist[r*Ngals:(r+1)*Ngals]
    
    # Add redshifts to the catalogue
        
    # Compute the random redshifts
    Zcat = np.array([])
    for z in range(len(Zbins)):
        Zbin = np.random.uniform(Zbin_edges[z], Zbin_edges[z+1], Zhist[z])
        Zcat = np.append(Zcat, Zbin)
        
    # Write everything to a catalogue
    outputnames = np.array(['ID', 'RA', 'DEC', 'Z'])
    formats = np.array(['D']*len(outputnames))
    output = np.array([IDcat, RAcat, DECcat, Zcat])

    filename = '/data/users/brouwer/LensCatalogues/basic_randoms_%s_%i_randomZ'%(cat, r+1)
    utils.write_catalog('%s.fits'%filename, outputnames, formats, output)
