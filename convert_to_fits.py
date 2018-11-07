#!/usr/bin/python

"Module to convert text files into fits tables."

# Import the necessary libraries
import astropy.io.fits as pyfits
import gc
import numpy as np
import sys
import os
from glob import glob

from astropy import constants as const, units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

import modules_EG as utils


input_filename = '/data2/brouwer/KidsCatalogues/KV450_Lephare_Masked_trim_ZBlt0p6'
ext = 'csv'

#data = np.genfromtxt('%s.%s'%(input_filename, ext))
data = np.loadtxt('%s.%s'%(input_filename, ext), dtype='str').T
print(data)

# The names and values of the output data
ID, field = [np.array(data[0], dtype=np.int32), np.array(data[1], dtype='str')]
RA, DEC, Z_B, logmstar, mask = [np.array(data[x], dtype=np.float32) for x in np.arange(5)+2]
print(ID, field)

datamask = (mask==0)#&(Z_B<0.5)
ID, field, RA, DEC, Z_B, logmstar = [ID[datamask], field[datamask], RA[datamask], DEC[datamask], Z_B[datamask], logmstar[datamask]]

#galIDs = np.array([i.rsplit('_', 2) for i in ID]).T
#field = np.array(galIDs[0], dtype='str')
#galIDlist = np.array(galIDs[-1], dtype=np.float32)

outputnames = ['ID', 'thelifield', 'RA', 'DEC', 'Z', 'logmstar']
formats = ['D', '20A', 'D', 'D', 'D', 'D',]
output = [ID, field, RA, DEC, Z_B, logmstar]

# The IDs of the galaxies
#galIDlist = np.arange(len(RA))+1

# Name of the output file
#output_filename = '%s.fits'%input_filename
output_filename = '/data2/brouwer/KidsCatalogues/KV450_Lephare_Masked_trim_ZBlt0p6.fits'

# Write to the output fits catalogue
utils.write_catalog(output_filename, outputnames, formats, output)

