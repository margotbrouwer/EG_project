#!/usr/bin/python

"Module to compute the optimal lens binning."

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

# The IDs of the galaxies
galIDlist = data[0]#np.arange(len(data[0]))+1

# The names and values of the output data
outputnames = ['RA', 'DEC', 'Z_B', 'logmstar', 'mask']
output = np.array(data[1:6], dtype=np.float32)

# Name of the output file
output_filename = '%s.fits'%input_filename

# Write to the output fits catalogue
utils.write_catalog(output_filename, galIDlist, outputnames, output)

