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

# Import constants
G = const.G.to('m3/Msun s2')
c = const.c.to('m/s')
inf = np.inf
h=0.7
pc_to_meter = 3.08567758e16 # meters

## Import Bahamas file
catnum = 402 #1039
lenslist = np.arange(catnum)
lenslist = np.delete(lenslist, [322,326,648,758,867])
catnum = len(lenslist)

path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'

catname = '%s/catalog.dat'%path_cat
catalog = np.loadtxt(catname).T[:,lenslist]
M200list = 10.**catalog[3] # M200 of each galaxy
r200list = catalog[4] # r200 of each galaxy
logmstarlist = catalog[5] # Stellar mass of each lens galaxy

# Bahamas simulation variables
Zlens = 0.25 # The redshift of the galaxies
Npix = 2000 # The number of pixels on each side
dpix = 1.5e4 / (1.+Zlens)/h # The pixel size of the grid (in physical pc/h70)
Lpix = Npix * dpix # The length of the grid (in physical pc/h70)

## Define projected distance bins R

# Creating the Rbins
Runit, Nbins, Rmin, Rmax = ['Mpc', 15, 0.03, 3.] # Fixed Rbins
#Runit, Nbins, Rmin, Rmax = ['Mpc', 16, -999, 999] # Same R-bins as PROFILES
#Runit, Nbins, Rmin, Rmax = ['mps2', 20, 1e-15, 5e-12] # gbar-bins

Rbins, Rcenters, Rmin_pc, Rmax_pc, xvalue = utils.define_Rbins(Runit, Rmin, Rmax, Nbins, True)
print('Rbins: %i bins between %g and %g %s'%(Nbins, Rmin, Rmax, Runit))

## Calculating the lens distance for every pixel
pixlist = 0.5*dpix + np.arange(0., Lpix, dpix) - 0.5*Lpix # Defining the pixel coordinates on the axes
pixgrid = np.meshgrid(pixlist, pixlist) # Making a pixel grid for the density map

pixdist = np.sqrt(pixgrid[0]**2. + pixgrid[1]**2.) # Calculating the lens distances
pixdist = np.array(np.ndarray.flatten(pixdist))/xvalue # Flatten the grid into a list, and translate to Runit

## Computing the ESD DeltaSigma
Rbins_list = np.zeros([catnum, Nbins+1]) # This list wil contain the radial bins for every lens
Sigma_list = np.zeros([catnum, Nbins]) # This list wil contain the projected density profile for every lens
ESD_list = np.zeros([catnum, Nbins]) # This list wil contain the ESD profile for every lens

print('Computing the ESD profile of:')
for c in range(catnum): # For every cluster/lens...

    print('Cluster %i (%i/%i)'%(lenslist[c], c+1, catnum))
    
    ## Define the radial distance bins
    
    if Runit == 'mps2': # Translate pixel distances to gbar
        mstar = 10.**logmstarlist[c]
        print('    log10[Mstar/Msun]: %g'%np.log10(mstar))
        
        Rdist_c = (G.value * mstar)/(pixdist*xvalue*pc_to_meter)**2 # the distance to the pixels (in m/s^2)
        Rmask = (Rdist_c>Rmin) # Defining the Rmax mask for the pixels
        Rbins_list[c] = np.sqrt((G.value * mstar)/Rbins) / pc_to_meter # in pc
        
    else: # Keep pixel distances in Xpc
        Rdist_c = pixdist # The distance to the pixels (in Xpc)
    
        if Rmax == 999: # Use the same radial bins as the true BAHAMAS profiles
            
            # Import the BAHAMAS profiles
            profname = '%s/PROFILES/cluster_%i_Menclosed_profile.dat'%(path_cat, lenslist[c])
            profile = np.loadtxt(profname).T
            profiles_centers = profile[0][0:Nbins] * r200list[c] # in Mpc
            
            # Calculate bin radii from logarithmic bin centers
            logprofiles_centers = np.log10(profiles_centers)
            logprofiles_diff = np.diff(logprofiles_centers)
            logprofiles_radius = np.append([logprofiles_centers[0] - logprofiles_diff[0]/2.], \
                                logprofiles_centers[0:Nbins-1] + logprofiles_diff/2.)
            logprofiles_radius = np.append(logprofiles_radius, \
                                [logprofiles_centers[Nbins-1] + logprofiles_diff[Nbins-2]/2.])
            profiles_radius = 10.**logprofiles_radius
            Rbins = profiles_radius
            
            Rbins_list[c] = Rbins # Save the radial bins to a list
            
        else: # Use the radial bins given
            profiles_centers = Rcenters
            Rbins_list[c] = Rbins # Save the radial bins to a list
            pass

    # Remove pixel distances outside Rmax
    Rmask = (Rdist_c<np.amax(Rbins)) # Defining the Rmax mask for the pixels
    Rdist_c, pixdist_c = [Rdist_c[Rmask], pixdist[Rmask]] # Apply the mask to the grid
    
    
    ## Import the BAHAMAS density maps
    
    # Full directory & name of the Bahamas catalogue
    mapfile = '%s/MAPS/cluster_%i.fits'%(path_cat, lenslist[c])
    
    # Import the surface density Sigma
    cat = pyfits.open(mapfile, memmap=True)[0].data * (1.+Zlens)**2.*h**2. / (1.e6)**2. # in Msun/(pc/h70)^2
    Sigmalist = np.ndarray.flatten(cat)[Rmask] # Remove pixels outside Rmax
    
    ## (As a check) calculate the average Sigma in each Rbin

    # Averaging the densities in each Rbin
    binindex_R = np.digitize(Rdist_c, Rbins)
    N_bin, foo = np.histogram(Rdist_c, Rbins) # The number of pixels in each Rbin
    
    Sigmatot_bin = np.array([ np.sum(Sigmalist[binindex_R==r]) for r in np.arange(Nbins)+1 ]) # The total Sigma in each Rbin
    Sigma_bin = Sigmatot_bin/N_bin # The average Sigma in each Rbin
    Sigma_list[c] = Sigma_bin
    
    ## Calculate the ESD (SigmaCrit) for each pixel
        
    # Create fine pixel distance bins
    distbins = np.append( [0.], np.sort(np.unique(pixdist_c)) ) # Find and sort the unique distance values
    Ndistbins = len(distbins)-1
    #print('Distbins:', distbins)
    #print('Ndistbins:', Ndistbins)

    # Dividing the pixel values into the fine distbins
    binindex_dist = np.digitize(pixdist_c, distbins[1::])
    N_dist, foo = np.histogram(pixdist_c, distbins) # The number of pixels in each distbin
    
    Sigmatot_dist = np.array([ np.sum(Sigmalist[binindex_dist==r]) \
        for r in np.arange(Ndistbins) ]) # The total density in each distbin
    avSigma_dist = np.cumsum(Sigmatot_dist)/np.cumsum(N_dist) # Average Sigma within each distance circle
    avSigmalist = avSigma_dist[np.digitize(pixdist_c, distbins[1::])-1] # Assign the appropriate avSigma to all pixels
    DeltaSigmalist = avSigmalist - Sigmalist # Calculate DeltaSigma (ESD) for each pixel
    #print('N_dist:', N_dist)
    #print('Sigmatot_dist:', Sigmatot_dist)
    #print('avgSigma_dist:', avSigma_dist)
    #print('avSigmalist:', avSigmalist)
   
    # Calculate the ESD in each Rbin
    DeltaSigmatot_bin = np.array([ np.sum(DeltaSigmalist[binindex_R==r]) for r in np.arange(Nbins)+1 ]) # The total ESD in each Rbin
    DeltaSigma_bin = DeltaSigmatot_bin/N_bin # The average density in each Rbin
    ESD_list[c] = DeltaSigma_bin


## Ploting the result

for i in np.arange(catnum):
    
    Rbins_centers = (Rbins_list[i])[0:-1] + np.diff(Rbins_list[i])/2.
    
    plt.plot(Rbins_centers, Sigma_list[i], marker='.', ls=':')
    plt.plot(profiles_centers[0:Nbins], Sigma_list[i], marker='.', ls=':')
    
    print(Rbins_centers)
    print(Rbins_list[i])
    
# Define the labels for the plot
xlabel = r'Radius R [$Mpc/h$]'
ylabel = r'Surface Density $\Delta\Sigma$ [$M_\odot/(pc/h)^2$]'
plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)

plt.xscale('log')
plt.yscale('log')

plt.show()
plt.clf


# Writing the result to a Fits file
filename = '%s/ESD/ESD_profiles_Rbins-%i_%g-%g%s_with_Sigma.fits'%(path_cat, Nbins, Rmin, Rmax, Runit)
outputnames = ['cluster', 'Rbins', 'Sigma', 'ESD']
formats = ['I', '%iD'%(Nbins+1), '%iD'%Nbins, '%iD'%Nbins]
output = [lenslist, Rbins_list, Sigma_list, ESD_list]

utils.write_catalog(filename, outputnames, formats, output)

