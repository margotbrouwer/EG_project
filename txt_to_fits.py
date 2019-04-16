#!/usr/bin/python

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import LambdaCDM
import astropy.io.fits as pyfits
from matplotlib import pyplot as plt
import modules_EG as utils


## Define the location and shape of the text file

#"""
# KiDS1000 ANNZ redshift catalogue
filename = '/data/users/brouwer/LensCatalogues/photozs.DR4_GAMAequ_ugri_beta_100ANNs'
outputnames = ['ID','MAG_AUTO','MAGERR_AUTO','RAJ2000','DECJ2000','A_WORLD','B_WORLD',\
    'Flag','CLASS_STAR','EXTINCTION_u','EXTINCTION_g','EXTINCTION_r','EXTINCTION_i','MAG_ISO',\
    'MAGERR_ISO','NIMAFLAGS_ISO','IMAFLAGS_ISO','SG2DPHOT','THELI_NAME','KIDS_TILE','EXTINCTION_Z',\
    'EXTINCTION_Y','EXTINCTION_J','EXTINCTION_H','EXTINCTION_Ks','MAG_GAAP_u','MAGERR_GAAP_u',\
    'MAG_GAAP_g','MAGERR_GAAP_g','MAG_GAAP_r','MAGERR_GAAP_r','MAG_GAAP_i','MAGERR_GAAP_i',\
    'MAG_GAAP_Z','MAGERR_GAAP_Z','MAG_GAAP_Y','MAGERR_GAAP_Y','MAG_GAAP_J','MAGERR_GAAP_J',\
    'MAG_GAAP_H','MAGERR_GAAP_H','MAG_GAAP_Ks','MAGERR_GAAP_Ks','Z_B','T_B','SG_FLAG','MASK',\
    'COLOUR_GAAP_u_g','COLOUR_GAAP_g_r','COLOUR_GAAP_r_i','COLOUR_GAAP_i_Z','COLOUR_GAAP_Z_Y',\
    'COLOUR_GAAP_Y_J','COLOUR_GAAP_J_H','COLOUR_GAAP_H_Ks','zANNz2ugri']
formats = np.array(['D']*len(outputnames))

"""
# KiDS1000 mass catalogue
filename = '/data/users/brouwer/LensCatalogues/photozs.DR4_GAMAequ_ugri_beta_100ANNs_LPoutput'
outputnames = ['IDENT', 'K_COR_u', 'K_COR_g', 'K_COR_r','K_COR_i','K_COR_Z','K_COR_Y','K_COR_J',\
    'K_COR_H','K_COR_Ks','MAG_ABS_u','MAG_ABS_g','MAG_ABS_r','MAG_ABS_i','MAG_ABS_Z','MAG_ABS_Y',\
    'MAG_ABS_J','MAG_ABS_H','MAG_ABS_Ks','MABS_FILTu','MABS_FILTg','MABS_FILTr','MABS_FILTi','MABS_FILTZ',\
    'MABS_FILTY','MABS_FILTJ','MABS_FILTH','MABS_FILTKs','CONTEXT','ZSPEC','MASS_MED','MASS_INF','MASS_SUP',\
    'MASS_BEST','SFR_MED','SFR_INF','SFR_SUP','SFR_BEST']
formats = np.array(['D']*len(outputnames))
"""

## Import the data and write it to a fits catalogue
data = np.genfromtxt('%s.csv'%filename, delimiter=',').T
data[0] = np.arange(len(data[0]))+1. # Use an array of integers as IDs
utils.write_catalog('%s.fits'%filename, outputnames, formats, data)
