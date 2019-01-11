#!/usr/bin/python

# Import the necessary libraries
import sys

import numpy as np
import pyfits
import os

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
import scipy.optimize as optimization
import modules_EG as utils

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

from matplotlib import gridspec

## Import Bahamas file

catnum = 1039
path_cat = '/data/users/brouwer/Simulations/Bahamas'
catname = np.array(['BAHAMAS_nu0_L400N1024_WMAP9/z_0.250/MAPS/cluster_%i.fits'%i \
    for i in np.arange(catnum)+1])

# Bahamas simulation variables
Zlens = 0.25
Npix = 2000
dpix = 0.015#/(1+Zlens) # comoving Mpc
Lpix = Npix * dpix


## Define projected distance bins R

# Creating the Rbins
Runit, Rmin, Rmax, Nbins = ['Mpc', 0.03, 3., 10]
Rbins, Rcenters, Rmin_pc, Rmax_pc, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins, True)
print('Rbins: %i bins between %g and %g %s'%(Nbins, Rmin, Rmax, Runit))

## Calculating the lens distance for every pixel
pixlist = np.arange(0., Lpix, dpix) + 0.5*dpix - 0.5*Lpix # Defining the pixel coordinates on the axes
pixgrid = np.meshgrid(pixlist, pixlist) # Making a pixel grid for the density map

pixdist = np.sqrt(pixgrid[0]**2. + pixgrid[1]**2.) # Calculating the lens distances
pixdist = np.array(np.ndarray.flatten(pixdist)) # Flatten the grid into a list
Rmask = (pixdist<Rmax) # Defining the Rmax mask
pixdist = pixdist[Rmask] # Remove pixel distances outside Rmax
#print('pixdist:', pixdist)

## Calculating DeltaSigma for every pixel

# Create fine distance bins
distbins = np.append( [0.], np.sort(np.unique(pixdist)) ) # Find and sort the unique distance values
#distbins = np.linspace(0., Rmax, 101)
Ndistbins = len(distbins)-1

#print('Distbins:', distbins)
#print('Ndistbins:', Ndistbins)

# Dividing the pixels into Rbins and distbins
binindex = np.digitize(pixdist, Rbins, right=False)
binindex_dist = np.digitize(pixdist, distbins[1::], right=False)

## Computing the ESD DeltaSigma
ESD_list = np.zeros([catnum, Nbins]) # This list wil contain the ESD profile for ever lens

print('Computing the ESD profile of:')
for c in range(catnum):
#for c in range(10):
    
    print('Cluster %i/%i'%(c+1, catnum))
    
    # Full directory & name of the Bahamas catalogue
    catfile = '%s/%s'%(path_cat, catname[c])
    
    # Import the surface density Sigma
    cat = pyfits.open(catfile, memmap=True)[0].data / (1.e6)**2. #*(1+Zlens)**2. # in Msun/(comoving pc/h)^2
    Sigmalist = np.ndarray.flatten(cat)[Rmask] # Remove pixels outside Rmax
    
    ## Calculate the ESD (SigmaCrit) for each pixel
    
    N_dist, foo = np.histogram(pixdist, distbins) # The number of pixels in each Rbin
    #print('N_dist:', N_dist)
    Sigmatot_dist = np.array([ np.sum(Sigmalist[binindex_dist==r]) \
        for r in np.arange(Ndistbins) ]) # The total density in each Rbin
    #print('Sigmatot_dist:', Sigmatot_dist)
    avSigma_dist = np.cumsum(Sigmatot_dist)/np.cumsum(N_dist) # Average Sigma within each distance circle
    #print('avgSigma_dist:', avSigma_dist)
    
    avSigmalist = avSigma_dist[np.digitize(pixdist, distbins[1::])-1] # Assign the appropriate avSigma to all pixels
    DeltaSigmalist = avSigmalist - Sigmalist # Calculate SigmaCrit for each pixel
    #print('avSigmalist:', avSigmalist)
    
    # Averaging the SigmaCrit in each Rbin
    N_bin = np.array([ np.sum([binindex==r]) for r in np.arange(Nbins)+1]) # The number of pixels in each Rbin
    Sigmatot_bin = np.array([ np.sum(DeltaSigmalist[binindex==r]) for r in np.arange(Nbins)+1 ]) # The total density in each Rbin
    DeltaSigma_bin = Sigmatot_bin/N_bin # The average density in each Rbin

    ESD_list[c] = DeltaSigma_bin

"""    
## Ploting the result

plt.plot(Rcenters, DeltaSigma_bin, marker='.', ls=':')

# Define the labels for the plot
xlabel = r'Radius R [$Mpc/h$]'
ylabel = r'Excess Surface Density $\Delta\Sigma$ [$M_\odot/(pc/h)^2$]'
plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)

plt.xscale('log')
plt.yscale('log')

plt.show()
"""

# Writing the result to a Fits file
filename = '%s/ESD_profiles_Rbins-%i_%g-%g%s.fits'%(path_cat, Nbins, Rmin, Rmax, Runit)
outputnames = ['cluster', 'ESD']
formats = ['I', '%iD'%Nbins]
output = [np.arange(catnum)+1, ESD_list]

utils.write_catalog(filename, outputnames, formats, output)
