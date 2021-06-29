#!/usr/bin/python

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import LambdaCDM
import astropy.io.fits as pyfits

import modules_EG as utils


## Configuration

# Data selection
cat = 'mice-offsetZM' # Select the lens catalogue (kids/gama/mice/matched)
#cat = 'mice'
#cat = 'gama-offsetZ'
#cat = 'matched'

# Constants
if 'mice' in cat:
    h = 0.7
    O_matter = 0.25
    O_lambda = 0.75
else:
    h = 0.7
    O_matter = 0.2793
    O_lambda = 0.7207
cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

print('Lens catalogue:', lenscatname)

# Create normally distributed offsets for the redshifts
if 'offset' in cat:
    
    Sigma_Z = 0.02*(1+lensZ)
    Sigma_M = [0.29]*len(logmstar)
    
    if 'Z' in cat:
        dZlist = np.random.normal(loc=0., scale=Sigma_Z, size=len(Sigma_Z))
        lensZ = lensZ+dZlist
        print('Added offset to lens redshifts')
        print(dZlist)

    if 'M' in cat:
        dMlist = np.random.normal(loc=0., scale=Sigma_M, size=len(Sigma_M))
        logmstar = logmstar+dMlist
        print('Added offset to lens masses')
        print(dMlist)
    
    lensDc = utils.calc_Dc(lensZ, cosmo)
lensDa = lensDc/(1.+lensZ)

# Remove all galaxies with logmstar=NAN or Z<0
logmstarcat = logmstar
nanmask = np.isfinite(logmstar) & (0.<lensZ)&(lensZ<0.5)
lensRA, lensDEC, lensDa, logmstar = [lensRA[nanmask], lensDEC[nanmask], lensDa[nanmask], logmstar[nanmask]]

# Import the faint MICE catalogue
if 'faint' in cat:
    faintID, faintRA, faintDEC, faintZ, faintDc, faint_rmag, faint_rmag_abs, faint_e1, faint_e2, faint_logmstar =\
    utils.import_micecat('/data/users/brouwer/LensCatalogues', 'mice2_faint_catalog_1000deg2.fits', h)
    
    faintmask = (faint_rmag<22.5)
    faintRA, faintDEC, faintZ, faintDc, faint_logmstar = \
        [faintRA[faintmask], faintDEC[faintmask], faintZ[faintmask], faintDc[faintmask], faint_logmstar[faintmask]]
    faintDa = faintDc/(1.+faintZ)

# Creating lens and satellite coordinates
lenscoords = SkyCoord(ra=lensRA*u.deg, dec=lensDEC*u.deg, distance=lensDa)
if 'faint' in cat:
    satcoords = SkyCoord(ra=faintRA*u.deg, dec=faintDEC*u.deg, distance=faintDa)
    logmstar_sat = faint_logmstar
else:
    satcoords = lenscoords
    logmstar_sat = logmstar


# Define the lens mass bins
logm_min = 7.
logm_max = 13.
Nlogmbins = int(1e3)
logmlims = np.linspace(logm_min, logm_max, Nlogmbins+1)
dlogm = np.diff(logmlims)[0]
logmbins = logmlims[0:-1] + dlogm


# The nearby galaxies should not be heavier than X times the galaxy
rationame = 'perc'
#massratios = [0.3, 0.25, 0.2, 0.15, 0.1]
massratios = [0.1]
rationames = [('%s'%d).replace('.', 'p') for d in massratios]
"""
rationame = 'dex'
dexvalues = [0.5, 0.6, 0.7, 0.8]
rationames = [('%s'%d).replace('.', 'p') for d in dexvalues]
massratios = [1./10**d for d in dexvalues] # ... where X=0.5 dex
"""

# This array will contain all satellite distances
satdistcat = np.zeros([len(massratios), len(logmstarcat)]) * u.Mpc

# For every mass ratio...
for d in range(len(massratios)):
    print('Satellites lighter then: %g %s * Mlens'%(massratios[d], rationame))
    
    # This list will contain the satellite distances of the lenses
    satdistlist = np.zeros(len(lenscoords)) * u.Mpc
    
    # For every stellar mass bin...
    for m in np.arange(Nlogmbins):
    
        print('Mass bin %g/%g: %g percent'%(m, Nlogmbins, m/Nlogmbins*100.))
        
        # Masking the lenses according to the stellar mass bin
        massmask_lens = (logmlims[m] < logmstar) & (logmstar <= logmlims[m+1])
        massmax_lens = np.mean(10**logmstar[massmask_lens])
        #print('Lens mass: %g'%massmax_lens)
        
        # Masking the satelites according to the lens mass
        massmax_sat = np.log10(massmax_lens * massratios[d]) # Satellites with X times the mean lens mass
        massmask_sat = (logmstar_sat > massmax_sat) # ... are "too heavy" for these lenses
        
        lenscoords_bin, satcoords_bin = [lenscoords[massmask_lens], satcoords[massmask_sat]]
        
        #print('Lenses:', len(lenscoords_bin), '/', len(lenscoords))
        #print('Satellites:', len(satcoords_bin), '/', len(satcoords))
        #print('%g < logmstar < %g: %i galaxies'%(logmlims[m], logmlims[m+1], len(lenscoords_bin)))
        #print('Max(logm) satellite: %g'%massmax_sat)
        #print('%g percent of satellites selected'%(np.sum(massmask_sat)/float(len(massmask_sat))*100.))
        #print(len(lenscoords_bin), len(satcoords_bin))
        
        # There should be galaxies in the bin
        if (len(lenscoords_bin)>0) & (len(satcoords_bin)>0):
            
            # Calculate the distance to the nearest satellite that is too heavy
            idx, sep2d, satdist3d = lenscoords_bin.match_to_catalog_3d(satcoords_bin, nthneighbor=2)
            satdistlist[massmask_lens] = satdist3d

    # Add the result to the catalogue
    (satdistcat[d])[nanmask] = satdistlist


print('logmbins:', logmlims)
print('dlogm:', dlogm)

# Write the results to a fits table
filename = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_perc_h%i'%(cat, h*100.)

if '-' in cat:
    name = cat.split('-')[-1]
    outputnames = np.append(['ID', 'logmstar_%s'%name], ['dist%s%s_%s'%(n,rationame,name) for n in rationames])
else:
    outputnames = np.append(['ID', 'logmstar'], ['dist%s%s'%(n,rationame) for n in rationames])

formats = np.append(['D']*2, ['D']*len(massratios))
output = np.append([lensID, logmstarcat], satdistcat, axis=0)
print(outputnames, formats, output)

utils.write_catalog('%s.fits'%filename, outputnames, formats, output)

