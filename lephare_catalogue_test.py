#!/usr/bin/python

import astropy.io.fits as pyfits
import gc
import numpy as np
import sys
import os
import time
from glob import glob

import modules_EG as utils

# Path to the KiDS catalogue
path_kidscat = '/data/users/brouwer/LensCatalogues'

#kidscatname = 'KIDS_allcats.mask.ZBlt0p6'
kidscatname = 'KIDS_allcats.LPoutput.ZBlt0p6'

# Full directory & name of the corresponding KiDS catalogue
kidscatfile = '%s/%s.fits'%(path_kidscat, kidscatname)
names = pyfits.open(kidscatfile, memmap=True)[1].columns.names
formats = pyfits.open(kidscatfile, memmap=True)[1].columns.formats
kidscat = pyfits.open(kidscatfile, memmap=True)[1].data

# Creating the ID
pointing = kidscat['Pointing']
seqnr = kidscat['SeqNr']
ID = np.array(['%s_%s'%(pointing[i], seqnr[i]) for i in range(len(seqnr))])

# List of the observables of all sources in the KiDS catalogue
output = [ID]
for name in names:
    output.append(kidscat[name])

# Writing to new catalogue
filename = '%s/%s_IDs.fits'%(path_kidscat, kidscatname)
outputnames = np.append('ID', names)
formats = np.append('30A', formats)

utils.write_catalog(filename, outputnames, formats, output)
print('Created ID file: "%s"'%filename)
