#!/usr/bin/python

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import LambdaCDM
import astropy.io.fits as pyfits

import modules_EG as utils

# Constants
h = 0.7
O_matter = 0.2793
O_lambda = 0.7207

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

## Configuration

# Data selection
cat = 'kids' # Select the lens catalogue (kids/gama/mice)

# Redshift distances: the satellite should not be closer in redshift than X
sigmas = [0.001, 0.002, 0.003, 0.004]
sigmanames = [('%s'%d).replace('.', 'p') for d in sigmas]


# Import lens catalog
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

# Create normally distributed offsets for the redshifts
if 'offset' in cat:
    #Sigma = [0.026]*len(lensZ)
    Sigma_Z = 0.018*(1+lensZ)
    #Sigma_M = [0.25]*len(logmstar)
    
    dZlist = np.random.normal(loc=0., scale=Sigma_Z, size=len(Sigma_Z))
    #dMlist = np.random.normal(loc=0., scale=Sigma_Z, size=len(Sigma_M))
    print('Added offset to lens redshifts')# and masses')
    print(dZlist)
    #print(dMlist)
    
    #logmstar = logmstar+dMlist
    lensZ = lensZ+dZlist
    Dclist = utils.calc_Dc(lensZ, cosmo)
lensDa = lensDc/(1.+lensZ)

# Remove all galaxies with logmstar=NAN
lensIDcat = lensID
logmstarcat = logmstar

nanmask = np.isfinite(logmstar)
lensID, lensRA, lensDEC, lensZ, lensDa, logmstar = \
    [lensID[nanmask], lensRA[nanmask], lensDEC[nanmask], lensZ[nanmask], lensDa[nanmask], logmstar[nanmask]]
print('Masked:', np.sum(nanmask)-len(nanmask))

# Import the faint MICE catalogue
if 'faint' in cat:
    faintID, faintRA, faintDEC, faintZ, faintDc, faint_rmag, faint_rmag_abs, faint_e1, faint_e2, faint_logmstar =\
    utils.import_micecat('/data/users/brouwer/LensCatalogues', 'mice2_faint_catalog_400deg2.fits', h)
    
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

# Mass ratio
massratio = 0.1

# Define the lens mass bins
zmin = 0.
zmax = 0.5
Nzbins = int(1e3)
zlims = np.linspace(zmin, zmax, Nzbins+1)
dz = np.diff(zlims)[0]
zbins = zlims[0:-1] + dz
Dabins = utils.calc_Dc(zbins, cosmo)/(1.+zbins)
Rbins = (3*u.Mpc / Dabins.to('Mpc')).value * u.rad

print('logzbins:', zlims)
print('dlogz:', dz)

# This array will contain all max(fsat) values
fsat_cat = np.ones([len(sigmas), len(logmstarcat)])*99.

# For every mass ratio...
for d in range(len(sigmas)):
    print('Satellites lighter then: %g * Mlens'%massratio)
    
    # This list will contain the max(fsat) values of the lenses for this sigma
    fsat_list = np.zeros(len(lenscoords))
    
    # For every redshift bin...
    for z in np.arange(Nzbins):
    
        print('Sigma %i/%i, Z-bin %g/%g: %g percent'%(d+1, len(sigmas), z, Nzbins, z/Nzbins*100.))
    
        # Masking the lenses according to the redshift bin
        zmask_lens = (zlims[z] < lensZ) & (lensZ <= zlims[z+1])
        sigma_z = sigmas[d] * (1. + zbins[z])
        zmask_sat = (zbins[z]-sigma_z < lensZ) & (lensZ <= zbins[z]+sigma_z)
        lenscoords_bin, lensmstar_bin, lensIDs_bin = \
            [lenscoords[zmask_lens], logmstar[zmask_lens], lensID[zmask_lens]]
        satcoords_bin, satmstar_bin, satIDs_bin = \
            [satcoords[zmask_sat], logmstar[zmask_sat], lensID[zmask_sat]]
        
        #print('%g < lensZ < %g: %i galaxies'%(zlims[z], zlims[z+1], len(lenscoords_bin)))
        #print('Number of satellites: %i'%len(satcoords_bin))
        
        # There should be galaxies in the bin
        if (len(lenscoords_bin)>0):
            idxsat, idxlens, sep2d, sep3d = lenscoords_bin.search_around_sky(satcoords_bin, Rbins[z])
            
            fsat_bin = [] # This list will contain the values of fsat for this redshift bin
            for l in np.arange(len(lenscoords_bin)):
                sat_index = idxsat[idxlens==l] # Find all galaxies around the lens
                if len(sat_index) > 1:
                    sat_index = sat_index[satIDs_bin[sat_index] != lensIDs_bin[l]] # Remove the lens itself
                    satmstar_max = np.amax(satmstar_bin[sat_index]) # Calculate the maximum satellite mass
                    #print(satmstar_max)
                    fsat_bin.append(10.**satmstar_max / 10.**lensmstar_bin[l]) # Add max(fsat) to the zbin list
                else:
                    fsat_bin.append(0.)
            #print(fsat_bin)
            fsat_list[zmask_lens] = fsat_bin # Add fsat to the list for this sigma
        #print()
        
    # Add the result to the catalogue
    (fsat_cat[d])[nanmask] = fsat_list # Add the fsat list to the total list for the catalogue

# Write the results to a fits table
filename = '/data/users/brouwer/LensCatalogues/%s_isolated_galaxies_2D_test_h%i'%(cat, h*100.)

if '-' in cat:
    name = cat.split('-')[-1]
    outputnames = np.append(['ID', 'logmstar_%s'%name], ['fsat_sigma%s_%s'%(n,name) for n in sigmanames])
else:
    outputnames = np.append(['ID', 'logmstar'], ['fsat_sigma%s'%n for n in sigmanames])

print(np.shape(fsat_cat), len(logmstarcat))

formats = np.append(['D']*2, ['D']*len(sigmas))
output = np.append([lensIDcat, logmstarcat], fsat_cat, axis=0)
print(outputnames, formats, output)

utils.write_catalog('%s.fits'%filename, outputnames, formats, output)

