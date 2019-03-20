#!/usr/bin/python

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import LambdaCDM
import astropy.io.fits as pyfits
from matplotlib import pyplot as plt
import modules_EG as utils

filename = '/data/users/brouwer/LensCatalogues/photozs.DR4_GAMAequ_ugri_beta_100ANNs'
data = np.genfromtxt('%s.csv'%filename, delimiter=',').T
data[0] = np.arange(len(data[0]))
print(data)

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

utils.write_catalog('%s.fits'%filename, outputnames, formats, data)
