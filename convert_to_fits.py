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


input_filename = '/data/users/brouwer/LensCatalogues/KV450_Lephare_Masked_trim_ZBlt0p6'
ext = 'csv'

#data = np.genfromtxt('%s.%s'%(input_filename, ext))
data = np.loadtxt('%s.%s'%(input_filename, ext), dtype='str').T
print(data)

# The names and values of the output data
field, ID = [np.array(data[0], dtype='str'), np.array(data[1], dtype=np.int32)]
RA, DEC, Z_B, logmstar, mask = [np.array(data[x], dtype=np.float32) for x in np.arange(5)+2]

# Masking the data
datamask = (mask==0)#&(Z_B<0.5)
ID, field, RA, DEC, Z_B, logmstar = [ID[datamask], field[datamask], RA[datamask], DEC[datamask], Z_B[datamask], logmstar[datamask]]

print('Masked: %g of %g galaxies (%g percent)'%(np.sum(datamask), len(datamask),\
				np.float(np.sum(datamask))/np.float(len(datamask))*100.)

outputnames = ['ID', 'thelifield', 'RA', 'DEC', 'Z', 'logmstar']
formats = ['J', '20A', 'D', 'D', 'D', 'D',]
output = [ID, field, RA, DEC, Z_B, logmstar]

# Name of the output file
output_filename = '%s.fits'%input_filename

# Write to the output fits catalogue
utils.write_catalog(output_filename, outputnames, formats, output)

