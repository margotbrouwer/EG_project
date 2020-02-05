#!/usr/bin/python

# Import the necessary libraries
import sys
import numpy as np
import pyfits
import os

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
import scipy.optimize as optimization
import scipy.stats as stats
import modules_EG as utils

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams
from matplotlib import gridspec

def bins_to_name(binlims):
    binname = str(binlims).replace(', ', '_')
    binname = str(binname).replace('[', '')
    binname = str(binname).replace(']', '')
    binname = str(binname).replace('.', 'p')
    return(binname)

# Constants
h = 0.7
O_matter = 0.2793
O_lambda = 0.7207

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Make use of TeX
rc('text',usetex=True)
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})

# Colours
# Blue, green, turquoise, cyan
blues = ['#332288', '#44AA99', '#117733', '#88CCEE']

# Light red, Red, light pink, pink
reds = ['#CC6677', '#882255', '#CC99BB', '#AA4499']
#colors = np.array([reds,blues])

blacks = ['black', '#0571b0']

#colors = ['#0571b0', '#92c5de', '#d7191c']*2#, '#fdae61']
colors = ['#0571b0', '#92c5de', '#d7191c', '#fdae61']*2


# Defining the paths to the data
blind = 'A'
logplot = False
massbias = True
miceoffset = True

# Import constants
pi = np.pi
G = const.G.to('pc3 / (M_sun s2)').value
c = const.c.to('m/s').value
H0 = h * 100 * (u.km/u.s)/u.Mpc
H0 = H0.to('s-1').value
pc_to_m = 3.08567758e16

def gobs_mond(gbar, g0=1.2e-10):
    gobs = gbar / (1 - np.exp( -np.sqrt(gbar/g0) ))
    return gobs

def gobs_verlinde(gbar):
    gobs = gbar + np.sqrt((c*H0)/6) * np.sqrt(gbar)
    return gobs

# rho = const
def calc_gobs_0(gbar):
    gD_0 = np.sqrt((c*H0)/6) * np.sqrt(4*gbar)
    gobs_0 = gbar + gD_0
    return gD_0, gobs_0

# rho = r^{-2}?
def calc_gobs_2(gbar):
    gD_2 = np.sqrt((c*H0)/6) * np.sqrt(2*gbar)
    gobs_2 = gbar + gD_2
    return gD_2, gobs_2

# Conversion from baryonic to apparent DM distribution 
def calc_Md(Mb_r, r):
    H = H0 #* np.sqrt(O_matter*(1+z)**3 + O_lambda)
    Cd = (c * H0) / (G * 6)
    Md_r = (Cd * np.gradient(Mb_r * r, r))**0.5 * r
    gd_r = G * Md_r / r**2. * pc_to_m # in m/s^2
    return gd_r

# Define guiding lines
gbar_mond = np.logspace(-12, -8, 50)
gbar_ext = np.logspace(-15, -12, 30)
gbar_uni = np.logspace(-15, -8, 50)
gbar_log = gbar_uni

gobs_mond_M16 = gobs_mond(gbar_mond)
gobs_mond_ext = gobs_mond(gbar_ext)
if not logplot:
    gobs_mond_M16 = np.log10(gobs_mond_M16)
    gobs_mond_ext = np.log10(gobs_mond_ext)
    
    gbar_mond = np.log10(gbar_mond)
    gbar_ext = np.log10(gbar_ext)
    gbar_uni = np.log10(gbar_uni)

## Import shear and random profiles

# Fiducial plotting parameters
Runit = 'pc'
datatitles = []
Nrows = 1

path_sheardata = '/data/users/brouwer/Lensing_results/EG_results_Jan20'

## Input lens selections

"""
"""

# KiDS + GAMA + Verlinde (isolated)

param1 = ['']
param2 = ['KiDS-1000 data (isolated galaxies)', r'GAMA data (isolated galaxies)']

N1 = 1
N2 = len(param2)
Nrows = 1

path_lenssel = np.array([['No_bins/dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0_0p5_lw-logmbar_GL', 'No_bins_gama/Z_0_0p5-dist0p1perc_3_inf-logmstarGL_0_11_lw-logmbar_GL']]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_filename = np.array([['shearcovariance/No_bins_A']*N2]*N1)

path_mocksel =  np.array([['No_bins/dist0p1perc_3_inf-logmstar_0_11-zcgal_0_0p5_lw-logmbar']*N2]*N1)
path_mockcosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_mockfilename = np.array([['shearcovariance/No_bins_A']*N2]*N1)
mocklabels = np.array(['GL-MICE mocks (isolated galaxies)'])

bahlabels = np.array(['BAHAMAS mocks (isolated galaxies)'])
Nmocks = [1, 1]

#masses_navarro = ['4.4E10'] # No mass bins (in Msun)

datalabels = param2
plotfilename = '%s/Plots/RAR_KiDS+GAMA+Verlinde_Nobins_isolated_zoomout'%path_sheardata
"""

# KiDS + GAMA + MICE (isolated)

param1 = ['']
param2 = [r'GAMA data (isolated galaxies)', 'KiDS-1000 data (isolated galaxies)']

N1 = 1
N2 = len(param2)
Nrows = 1

path_lenssel = np.array([['No_bins_gama/Z_0_0p5-dist0p1perc_3_inf-logmstarGL_0_11_lw-logmbar_GL', 'No_bins/dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0_0p5_lw-logmbar_GL']]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_filename = np.array([['shearcovariance/No_bins_A']*N2]*N1)

path_mocksel =  np.array([['No_bins/dist0p1perc_3_inf-logmstar_0_11-zcgal_0_0p5_lw-logmbar']*N2]*N1)
path_mockcosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_mockfilename = np.array([['shearcovariance/No_bins_A']*N2]*N1)
mocklabels = np.array(['GL-MICE mocks (isolated galaxies)'])

bahlabels = np.array(['BAHAMAS mocks (isolated galaxies)'])
Nmocks = [1, 1]

masses_navarro = ['4.4E10'] # No mass bins (in Msun)

datalabels = param2
plotfilename = '%s/Plots/RAR_KiDS+GAMA+Navarro_Nobins_isolated'%path_sheardata



# KiDS + Verlinde / MICE (isolated, 4 stellar mass bins)

param1 = [8.5,10.3,10.6,10.8,11.]
binname = bins_to_name(param1)

param2 = [r'GL-KiDS data (isolated galaxies)']#: $r_{\rm sat}(f_{\rm M_*}>0.1)>3$ Mpc/$h_{%g}$)'%(h*100)]

N1 = len(param1)-1
N2 = len(param2)
Nrows = 2

path_lenssel = np.array([['logmstar_GL_%s/dist0p1perc_3_inf-zANNZKV_0_0p5_lw-logmbar_GL'%(binname)]*N2]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_filename = np.array([['shearcovariance/shearcovariance_bin_%i_A'%p1]*N2 for p1 in np.arange(N1)+1])

path_mocksel =  np.array([['logmstar_%s/dist0p1perc_3_inf-zcgal_0_0p5_lw-logmbar'%(binname)]*N2]*N1)
path_mockcosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_mockfilename = np.array([['shearcovariance/shearcovariance_bin_%i_A'%p1]*N2 for p1 in np.arange(N1)+1])
mocklabels = np.array(['GL-MICE mocks (isolated galaxies)'])

#masses_navarro = ['1.3E10', '3.7E10', '6.0E10', '9.0E10'] # Mass bins (in Msun)

bahlabels = np.array(['BAHAMAS mocks (isolated galaxies)'])
datalabels = param2

#plotfilename = '%s/Plots/RAR_KiDS+Verlinde_4-massbins_isolated'%path_sheardata
plotfilename = '%s/Plots/RAR_KiDS+Verlinde_4-massbins_isolated'%path_sheardata # Substitute: Verlinde / MICE



# KiDS + Verlinde / MICE (isolated, 4 Central Surface Brightness bins)

param1 = [0.,10.,14.,17.,22.5]
binname = bins_to_name(param1)
massbias = False

param2 = [r'GL-KiDS data (isolated galaxies)']#: $r_{\rm sat}(f_{\rm M_*}>0.1)>3$ Mpc/$h_{%g}$)'%(h*100)]

N1 = len(param1)-1
N2 = len(param2)
Nrows = 2

path_lenssel = np.array([['mu0_2dphot_%s/dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0_0p5_lw-logmbar_GL'%(binname)]*N2]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_filename = np.array([['shearcovariance/shearcovariance_bin_%i_A'%p1]*N2 for p1 in np.arange(N1)+1])

path_mocksel =  np.array([['logmstar_%s/dist0p1perc_3_inf-zcgal_0_0p5_lw-logmbar'%(binname)]*N2]*N1)
path_mockcosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_mockfilename = np.array([['shearcovariance/shearcovariance_bin_%i_A'%p1]*N2 for p1 in np.arange(N1)+1])
mocklabels = np.array(['GL-MICE mocks (isolated galaxies)'])

#masses_navarro = ['1.3E10', '3.7E10', '6.0E10', '9.0E10'] # Mass bins (in Msun)

bahlabels = np.array(['BAHAMAS mocks (isolated galaxies)'])
datalabels = param2

#plotfilename = '%s/Plots/RAR_KiDS+Verlinde_4-massbins_isolated'%path_sheardata
plotfilename = '%s/Plots/RAR_KiDS+Verlinde_4-mu0bins_isolated'%path_sheardata # Substitute: Verlinde / MICE


# KiDS + BAHAMAS + MICE (isolated)

param1 = ['']
param2 = ['KiDS-1000 data (isolated galaxies)']

N1 = 1
N2 = len(param2)
Nrows = 1

path_lenssel = np.array([['No_bins/dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0_0p5_lw-logmbar_GL']*N2]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_filename = np.array([['shearcovariance/No_bins_A']*N2]*N1)

path_mocksel =  np.array([['No_bins/dist0p1perc_3_inf-logmstar_0_11-zcgal_0_0p5_lw-logmbar']*N2]*N1)
path_mockcosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_mockfilename = np.array([['shearcovariance/No_bins_A']*N2]*N1)
mocklabels = np.array(['GL-MICE mocks (isolated galaxies)'])

bahlabels = np.array(['BAHAMAS mocks (isolated galaxies)'])
Nmocks = [1, 1]

#masses_navarro = ['4.4E10'] # No mass bins (in Msun)

datalabels = param2
plotfilename = '%s/Plots/RAR_KiDS+MICE+Bahamas_No_Nobins_isolated_zoomout'%path_sheardata


# KiDS + MICE (all, 4 stellar mass bins)

param1 = [8.5,10.3,10.6,10.8,11.]
binname = bins_to_name(param1)

param2 = [r'GL-KiDS data (isolated galaxies)']#: $r_{\rm sat}(f_{\rm M_*}>0.1)>3$ Mpc/$h_{%g}$)'%(h*100)]

N1 = len(param1)-1
N2 = len(param2)
Nrows = 2

path_lenssel = np.array([['logmstar_GL_%s/zANNZKV_0_0p5_lw-logmbar_GL'%(binname)]*N2]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_filename = np.array([['shearcovariance/shearcovariance_bin_%i_A'%p1]*N2 for p1 in np.arange(N1)+1])

path_mocksel =  np.array([['logmstar_%s/zcgal_0_0p5_lw-logmbar'%(binname)]*N2]*N1)
path_mockcosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_mockfilename = np.array([['shearcovariance/shearcovariance_bin_%i_A'%p1]*N2 for p1 in np.arange(N1)+1])
mocklabels = np.array(['GL-MICE mocks (all galaxies)'])

massbias = True
miceoffset = False

datalabels = param2
#datatitles = [r'$%g < \log(M_*) < %g \, {\rm M_\odot}/h_{%g}^{2}$'%(param1[p1], param1[p1+1], h*100) for p1 in range(N1)]
plotfilename = '%s/Plots/RAR_KiDS+MICE_4-massbins_all'%path_sheardata


"""

## Measured ESD
esdfiles = np.array([['%s/%s/%s/%s.txt'%\
	(path_sheardata, path_lenssel[i,j], path_cosmo[i,j], path_filename[i,j]) \
	for j in np.arange(np.shape(path_lenssel)[1])] for i in np.arange(np.shape(path_lenssel)[0]) ])

Nbins = np.shape(esdfiles)
Nsize = np.size(esdfiles)
esdfiles = np.reshape(esdfiles, [Nsize])

print('Plots, profiles:', Nbins)

# Importing the shearprofiles and lens IDs
data_x_log, R_src, data_y_log, error_h, error_l, N_src = utils.read_esdfiles(esdfiles)

# Importing the upper and lower limits due to the stellar mass bias
if massbias and ('KiDS' in plotfilename):
    esdfiles_min = [f.replace('GL', 'min') for f in esdfiles]
    esdfiles_max = [f.replace('GL', 'max') for f in esdfiles]
    foo, foo, data_y_min, error_h_min, foo, foo = utils.read_esdfiles(esdfiles_min)
    foo, foo, data_y_max, error_h_max, foo, foo = utils.read_esdfiles(esdfiles_max)

## Random subtraction
path_randoms = '%s/Randoms/combined_randoms.txt'%path_sheardata

try:
    print()
    print('Import random signal:')
    random_esdfiles = [path_randoms]
    random_data_x, random_src_R, random_data_y, random_error_h, random_error_l, random_N_src = utils.read_esdfiles(random_esdfiles)

    # Subtract random signal
    data_y_log = data_y_log-random_data_y
    error_h = np.sqrt(error_h**2. + random_error_h**2)
    error_l = np.sqrt(error_l**2. + random_error_l**2)
    
    if massbias and ('KiDS' in plotfilename):
        data_y_min, data_y_max = [data_y_min-random_data_y, data_y_max-random_data_y]
    
    print('Random percentage:', np.mean(abs(random_data_y)/data_y_log*100.,1))

except:
    print('No randoms subtracted!')
    print()
    pass

# Convert ESD (Msun/pc^2) to acceleration (m/s^2)
data_y_log, error_h, error_l = 4. * G * pc_to_m * np.array([data_y_log, error_h, error_l])

# Adding 0.1 dex to the error bars to account for the translation into the RAR
error_h, error_l = np.array([error_h, error_l]) # * 10.**0.1

# Keep this error for the chi2 analysis
error_log = error_h

if massbias and ('KiDS' in plotfilename):
    data_y_min, data_y_max, error_h_min, error_h_max = \
        4. * G * pc_to_m * np.array([data_y_min, data_y_max, error_h_min, error_h_max])
    data_y_min_log = data_y_min
    data_y_max_log = data_y_max
    
## Convert the signals into log-signals
if not logplot:
    error_h = 1./np.log(10.) * error_h/abs(data_y_log)
    error_l = 1./np.log(10.) * error_l/abs(data_y_log)
    
    floor = 1e-15
    error_h[data_y_log<0.] = data_y_log[data_y_log<0.] + error_h[data_y_log<0.] - floor
    
    data_y = np.zeros(np.shape(data_y_log))
    data_y[data_y_log<0.] = floor
    data_y[data_y_log>0.] = np.log10(data_y_log[data_y_log>0.])
    data_x = np.log10(data_x_log)
    
    if massbias and ('KiDS' in plotfilename):
        data_y_min = np.log10(data_y_min)
        data_y_max = np.log10(data_y_max)
else:
    data_x = data_x_log
    data_y = data_y_log


## Find the mean galaxy masses
cat = 'kids'

# Import the Lens catalogue
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

IDfiles = np.array([m.replace('A.txt', 'lensIDs.txt') for m in esdfiles])
lensIDs_selected = np.array([np.loadtxt(m) for m in IDfiles])#+1
N_selected = [len(m) for m in lensIDs_selected]

# Import the mass catalogue
path_masscat = '/data/users/brouwer/LensCatalogues/baryonic_mass_catalog_%s.fits'%cat
masscat = pyfits.open(path_masscat, memmap=True)[1].data

logmstar = masscat['logmstar_GL']
logmbar = masscat['logmbar_GL']

# Calculate the galaxy masses
mean_mstar, median_mstar, mean_mbar, median_mbar = \
    [np.zeros(len(esdfiles)), np.zeros(len(esdfiles)), np.zeros(len(esdfiles)), np.zeros(len(esdfiles))]
for m in range(len(esdfiles)):
    IDmask = np.in1d(lensID, lensIDs_selected[m])
    
    print(np.amax(logmstar[IDmask*np.isfinite(logmstar)]))
    
    mean_mstar[m] = np.log10(np.mean(10.**logmstar[IDmask*np.isfinite(logmstar)]))
    median_mstar[m] = np.median(logmstar[IDmask*np.isfinite(logmstar)])

    mean_mbar[m] = np.log10(np.mean(10.**logmbar[IDmask*np.isfinite(logmbar)]))
    median_mbar[m] = np.median(logmbar[IDmask*np.isfinite(logmbar)])

print()
print('Number of galaxies:', N_selected) 
print()
print('mean logmstar:', mean_mstar)
print('median logmstar:', median_mstar)
print()
print('mean logmbar:', mean_mbar)
print('median logmbar:', median_mbar)
print()

# Print reliability limit of isolated KiDS-1000 galaxies
if ('KiDS' in plotfilename):# and ('isolated' in plotfilename):
    isoR = 0.3 # in Mpc
    isolim_log = (G * 10.**mean_mbar)/(isoR * 1e6)**2. * pc_to_m
    print('Isolated KiDS signal reliability limit:', isolim_log)
    
    if not logplot:
        isolim = np.log10(isolim_log)
    else:
        isolim = isolim_log

# Create titles of the bins
datatitles = [r'$\log\langle M_{\rm g}/h_{%g}^{-2} {\rm M_\odot} \rangle = %.4g$'%(h*100, mean_mbar[p1]) for p1 in range(N1)]

# When binning by Central Surface Brightness
if 'mu0' in path_lenssel[0,0]:
    
    # Import the CSB catalogue
    path_structcat = '/data/users/brouwer/LensCatalogues/photozs.DR4_trained-on-GAMAequ_ugri+KV_version0.9_struct.fits'
    structcat = pyfits.open(path_structcat, memmap=True)[1].data
    mu0 = structcat['mu0_2dphot']

    # Calculate the mean CSB per bin
    mean_mu0 = np.zeros(len(esdfiles))
    for m in range(len(esdfiles)):
        IDmask = np.in1d(lensID, lensIDs_selected[m])
        mean_mu0[m] = np.mean(mu0[IDmask])
    
    datatitles = [r'$\langle \mu_0 \rangle = %.4g \, {\rm mag/arcsec^2}$'%(mean_mu0[p1]) for p1 in range(N1)]



"""
# Import Crescenzo's RAR from Early Type Galaxies
cres = np.loadtxt('RAR_profiles/crescenzo_RAR.txt').T
gbar_cres = 10**cres[0]
gobs_cres = 10**cres[1]
errorl_cres = 10**cres[1] - 10**(cres[1]+cres[2])
errorh_cres = 10**(cres[1]+cres[3]) - 10**cres[1]
"""

if 'Navarro' in plotfilename:
    # Import Kyle's RAR (based on Navarro et al. 2017)
    gbar_navarro = []
    gobs_navarro = []
    for m in range(len(masses_navarro)):
        data_navarro = np.loadtxt('RAR_profiles/RAR_Mbar%s.txt'%masses_navarro[m]).T
        
        if logplot:
            gbar_navarro.append(data_navarro[1])
            gobs_navarro.append(data_navarro[1] + data_navarro[0])
        else:
            gbar_navarro.append(np.log10(data_navarro[1]))
            gobs_navarro.append(np.log10(data_navarro[1] + data_navarro[0]))

"""
if ('massbins' in plotfilename) and ('all' in plotfilename):
    # Import the MICE satellite distributions
    purpose = 'mstarbins'
    offset = ''
    
    # Full directory & name of the satellite distribution file
    path_satcat = '%s/Plots/satellite_histogram_%s_%s%s.fits'%(path_sheardata, 'mice', purpose, offset)
    satcat = pyfits.open(path_satcat, memmap=True)[1].data
    
    rcenters = satcat['rcenters'] * 1e6 # In pc
    Mb_all = np.array([satcat['Mtot_bin%i'%m] for m in np.arange(N1)+1])
    
    gb_all = np.array([ (G * Mb_all[m]/rcenters**2. * pc_to_m) for m in range(N1)] )
    gb_point = np.array([ (G * Mb_all[m,0]/rcenters**2. * pc_to_m) for m in range(N1)] )
    
    gd_all = np.array([calc_Md(Mb_all[m], rcenters) for m in range(N1)] )
    gtot_all = (gb_all + gd_all)

    print(gtot_all)
    print(np.log10(gobs_verlinde(gbar_log)))
    
    if not logplot:
        gb_all, gb_point, gtot_all = [np.log10(gb_all), np.log10(gb_point), np.log10(gtot_all)]
"""


# Import McGaugh data
data_mcgaugh = np.loadtxt('RAR_profiles/mcgaugh2016_RAR.txt').T
gbar_mcgaugh, gbar_mcgaugh_error, gobs_mcgaugh, gobs_mcgaugh_error = np.array([data_mcgaugh[d] for d in range(4)])

data_mcgaugh_binned = np.loadtxt('RAR_profiles/mcgaugh2016_RAR_binned.txt').T
gbar_mcgaugh_binned, gobs_mcgaugh_binned, sd_mcgaugh, N_mcgaugh = np.array([data_mcgaugh_binned[d] for d in range(4)])

if logplot:
    gbar_mcgaugh, gbar_mcgaugh_error, gobs_mcgaugh, gobs_mcgaugh_error, \
    gbar_mcgaugh_binned, gobs_mcgaugh_binned = \
    [10.**gbar_mcgaugh, 10.**gbar_mcgaugh_error, 10.**gobs_mcgaugh, \
    10.**gobs_mcgaugh_error, 10.**gbar_mcgaugh_binned, 10.**gobs_mcgaugh_binned]

"""
# Import Lelli+2017 dSph data
data_dsph = np.genfromtxt('RAR_profiles/Lelli2017_dSph_data.txt', delimiter=',').T
loggbar_dsph, loggbar_dsph_ehigh, loggbar_dsph_elow, loggobs_dsph, loggobs_dsph_ehigh, loggobs_dsph_elow \
    = np.array([data_dsph[d] for d in np.arange(15,21)])

# Apply the High-Quality mask
hqsample = data_dsph[1]
hqmask = (hqsample==1)
"""

## Mocks

# MICE mocks
if 'MICE' in plotfilename:

    print()
    print('Import mock signal:')

    # Defining the mock profiles
    esdfiles_mock = np.array([['%s/%s/%s/%s.txt'%\
        (path_sheardata, path_mocksel[i,j], path_mockcosmo[i,j], path_mockfilename[i,j]) \
        for j in np.arange(np.shape(path_lenssel)[1])] for i in np.arange(np.shape(path_lenssel)[0]) ])

    Nmocks = np.shape(esdfiles_mock)
    esdfiles_mock = np.reshape(esdfiles_mock, [Nsize])

    if miceoffset and ('KiDS' in plotfilename):
        esdfiles_mock_offset = [f.replace('perc', 'percoffsetZM') for f in esdfiles_mock]
        if 'No_bins' in esdfiles_mock[0]:
            esdfiles_mock_offset = [f.replace('logmstar', 'logmstaroffsetZM') for f in esdfiles_mock_offset]        
        else:        
            esdfiles_mock_offset = [f.replace('logmstar', 'logmstar_offsetZM') for f in esdfiles_mock_offset]
        foo, foo, data_y_mock_offset, foo, foo, foo = utils.read_esdfiles(esdfiles_mock_offset)
        data_y_mock_offset = 4. * G * pc_to_m * data_y_mock_offset

    if Nmocks[1] > 5:
        valpha = 0.3
    else:
        valpha = 0.6

    # Importing the mock shearprofiles
    esdfiles_mock = np.reshape(esdfiles_mock, [Nsize])

    data_x_mock, R_src_mock, data_y_mock, error_h_mock, error_l_mock, N_src_mock = utils.read_esdfiles(esdfiles_mock)
    data_y_mock, error_h_mock, error_l_mock = 4. * G * pc_to_m *\
        np.array([data_y_mock, error_h_mock, error_l_mock]) # Convert ESD (Msun/pc^2) to acceleration (m/s^2)

    if not logplot:
        error_h_mock = 1./np.log(10.) * error_h_mock/data_y_mock
        error_l_mock = 1./np.log(10.) * error_l_mock/data_y_mock
        data_x_mock, data_y_mock = np.log10(np.array([data_x_mock, data_y_mock]))
        
        if miceoffset and ('KiDS' in plotfilename):
            data_y_mock_offset = np.log10(data_y_mock_offset)

    IDfiles_mock = np.array([m.replace('A.txt', 'lensIDs.txt') for m in esdfiles_mock])
    lensIDs_selected_mock = np.array([np.loadtxt(m) for m in IDfiles_mock])

    # Import mock lens catalog
    fields, path_mockcat, mockcatname, lensID_mock, lensRA_mock, lensDEC_mock, \
        lensZ_mock, lensDc_mock, rmag_mock, rmag_abs_mock, logmstar_mock =\
        utils.import_lenscat('mice', h, cosmo)
    lensDc_mock = lensDc_mock.to('pc').value
    lensDa_mock = lensDc_mock/(1.+lensZ_mock)

    max_gbar = np.zeros(len(esdfiles_mock))
    for m in range(len(esdfiles_mock)):
        IDmask = np.in1d(lensID_mock, lensIDs_selected_mock[m])
        Da_max = np.amax(lensDa_mock[IDmask*np.isfinite(lensDa_mock)])
        mstar_mean_mock = np.mean(10.**logmstar_mock[IDmask*np.isfinite(logmstar_mock)])
            
        pixelsize = 2. * 0.43 / 60. * pi/180. # arcmin to radian
        min_R = pixelsize * Da_max
        max_gbar[m] = (G * pc_to_m * mstar_mean_mock)/min_R**2.
    if not logplot:
        max_gbar = np.log10(max_gbar)
            
    #    print('Da_max:', Da_max)
    #    print('mstar_mean_mock:', mstar_mean_mock)
    #    print('min_R:', min_R)
    print('max_gbar:', max_gbar)

else:
    print('No mock signal imported!')
    pass

# BAHAMAS mocks

data_x_bah = []
data_y_bah = []
bah_y_std = []

if 'Bahamas' in plotfilename:

    ## Import true enclosed mass profiles
    profbins = 40
    catnum = 515
    lenslist = np.arange(catnum)
    
    path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_isolated_new/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
    catname = '%s/catalog.dat'%path_cat
    catalog = np.loadtxt(catname).T[:,lenslist]

    # Import galaxy observables (M200, r200, logmstar)
    M200list = 10.**catalog[3] # M200 of each galaxy
    r200list = catalog[4] * 1e6 # r200 of each galaxy (in Xpc)
    logmstar_bah = catalog[5] # Stellar mass of each lens galaxy

    profiles_gbar = []
    profiles_gobs = []
    profiles_min = []
    profiles_max = []
       
    # Calculate baryonic galaxy mass
    fcold_bah = 10.**(-0.69*logmstar_bah + 6.63)
    mstar_bah = 10.**logmstar_bah
    mbar_bah = mstar_bah * (1 + fcold_bah)
    mstar_bah = np.reshape(mstar_bah[0:catnum], [catnum,1])
    mbar_bah = np.reshape(mbar_bah[0:catnum], [catnum,1])
    
    profiles_radius = np.zeros([catnum, profbins])
    profiles_Menclosed = np.zeros([catnum, profbins])
    for x in range(catnum):
        profname = '%s/PROFILES/cluster_%i_Menclosed_profile_types.dat'%(path_cat, lenslist[x])
        profile_c = np.loadtxt(profname).T
        profiles_radius[x] = profile_c[0] * 1e6 # * r200list[x] # in pc # profile_c[0,0:Nbins]
        profiles_Menclosed[x] = profile_c[1]# * M200list[x] # in Msun # profile_c[1,0:Nbins]

    # Calculate true gbar and gobs from enclosed mass profiles
    profiles_gbar = (G * mbar_bah) / (profiles_radius)**2. * pc_to_m # in m/s^2
    profiles_gobs = (G * profiles_Menclosed) / (profiles_radius)**2. * pc_to_m # in m/s^2

    # For every stellar mass bin:
    for m in range(Nbins[0]):
      
        # Select galaxies within this stellar mass bin
        if Nbins[0] > 1:
            logmstarmask = (param1[m] < logmstar_bah) & (logmstar_bah <= param1[m+1])        
            print('Mstarbin:', m+1, param1[m], param1[m+1])
        else:
            logmstarmask = logmstar_bah < 11.
        
        profiles_gbar_bin, profiles_gobs_bin = [profiles_gbar[logmstarmask], profiles_gobs[logmstarmask]]
        
        data_x_bin, data_y_bin, bin_y_std = utils.mean_profile(profiles_gbar_bin, profiles_gobs_bin, \
            0, 0, profbins, True)
        
        data_x_bah.append(data_x_bin)
        data_y_bah.append(data_y_bin)
        bah_y_std.append(bin_y_std)
    
    data_x_bah = np.array(data_x_bah)
    data_y_bah = np.array(data_y_bah)
    bah_y_std = np.array(bah_y_std)
    
    if not logplot:
        bah_y_std = 1./np.log(10.) * bah_y_std/data_y_bah # This must be calculated first
        data_x_bah = np.log10(data_x_bah)
        data_y_bah = np.log10(data_y_bah)


    print('data_x_bah', data_x_bah)
    print('data_y_bah', data_y_bah)
    print('bah_y_std', bah_y_std)


## Perform chi2 analysis

print()
print('Chi2 analysis:')
print()

for Nc in range(N2):
    
    print('** %s **'%param2[Nc])
    
    # Import covariance matrix
    if 'No_bins' in esdfiles[0]:
        covfile, covfile_min, covfile_max = [ f.replace('bins_A', 'matrix_A') \
            for f in [esdfiles[Nc], esdfiles_min[Nc], esdfiles_max[Nc]] ]
    else:
        covfile, covfile_min, covfile_max = [ f.replace('bin_1_A', 'matrix_A') \
            for f in [esdfiles[0], esdfiles_min[0], esdfiles_max[0]] ]

    covariance, covariance_min, covariance_max = [ np.loadtxt(f).T \
        for f in [covfile, covfile_min, covfile_max] ]
    
    # Translating the ESD covariance into gobs covariance, adding 0.1 dex to the error
    covariance[4] = (4. * G * pc_to_m)**2. * covariance[4]
    covariance_min[4] = (4. * G * pc_to_m)**2. * covariance_min[4]
    covariance_max[4] = (4. * G * pc_to_m)**2. * covariance_max[4]
    
    # Import data
    if N2>1: # When multiple datasets are used
        data_x_chi2, data_y_chi2 = [np.array([data_x_log[Nc]]), np.array([data_y_log[Nc]])]
        data_y_min_chi2, data_y_max_chi2 = [np.array([data_y_min_log[Nc]]), np.array([data_y_max_log[Nc]])]
    else: # When one datasets is used (with multiple bins)
        data_x_chi2, data_y_chi2 = [data_x_log, data_y_log]
        data_y_min_chi2, data_y_max_chi2 = [data_y_min_log, data_y_max_log]
        
    nRbins = len(data_y[0])
    dof = N1 * nRbins
    print('dof:', dof)
    print()
    
    if 'KiDS' in plotfilename:
        
        # Create a mask for data beyond the isolation limit
        isomask = [np.ravel( np.argwhere(data_x_log[y] < isolim_log[y]) ) \
            + y * nRbins for y in range(N1)]
        isomask = np.concatenate(isomask).ravel()
        
    # Testing Emergent Gravity and the MOND relation
    if 'Verlinde' in plotfilename:
        chi2_mond = utils.calc_chi2(data_y_chi2, gobs_mond(data_x_chi2), covariance)        
        chi2_EG = utils.calc_chi2(data_y_chi2, gobs_verlinde(data_x_chi2), covariance)
        
        print( 'MOND: %g (%g)'%(chi2_mond/dof, chi2_mond) )
        print( 'Verlinde: %g (%g)'%(chi2_EG/dof, chi2_EG) )
        print()
        
        if 'KiDS' in param2[Nc]:
            # Apply mask beyond the isolation limit
            chi2_mond = utils.calc_chi2(data_y_chi2, gobs_mond(data_x_chi2), covariance, masked=isomask)        
            chi2_EG = utils.calc_chi2(data_y_chi2, gobs_verlinde(data_x_chi2), covariance, masked=isomask)
            
            print( 'MOND (isolation mask): %g (%g)'%(chi2_mond/dof, chi2_mond) )
            print( 'Verlinde (isolation mask): %g (%g)'%(chi2_EG/dof, chi2_EG) )
            print()
            
            chi2_mond_min = utils.calc_chi2(data_y_min_chi2, gobs_mond(data_x_chi2), covariance_min, masked=isomask)        
            chi2_mond_max = utils.calc_chi2(data_y_max_chi2, gobs_mond(data_x_chi2), covariance_max, masked=isomask)
            
            print( 'MOND (mask + mass_min): %g (%g)'%(chi2_mond_min/dof, chi2_mond_min) )
            print( 'MOND (mask + mass_max): %g (%g)'%(chi2_mond_max/dof, chi2_mond_max) )
            print()
print()


## Create the plot

Ncolumns = int(Nbins[0]/Nrows)

if 'zoomout' in plotfilename:
    # Zoomed out
    xlims = [1e-15, 2e-9]
    ylims = [1e-13, 2e-9]
else:
    # Zoomed in
    xlims = [1e-15, 1e-11]
    ylims = [1e-13, 1e-10]


if logplot:
    plt.xscale('log')
    plt.yscale('log')
else:
    xlims, ylims = [np.log10(xlims), np.log10(ylims)]

# Plotting the ueber matrix
if Nbins[0] > 1:
    fig = plt.figure(figsize=(Ncolumns*5.,Nrows*4))
else:
    fig = plt.figure(figsize=(8,6))

gs_full = gridspec.GridSpec(1,1)
gs = gridspec.GridSpecFromSubplotSpec(Nrows, Ncolumns, wspace=0, hspace=0, subplot_spec=gs_full[0,0])

ax = fig.add_subplot(gs_full[0,0])

for N1 in range(Nrows):
    for N2 in range(Ncolumns):
    
        ax_sub = fig.add_subplot(gs[N1, N2])
        
        N = np.int(N1*Ncolumns + N2)
        
        # Plot guiding lines
        ax_sub.plot(gbar_uni, gbar_uni, label = r'Unity (No dark matter: $g_{\rm obs} = g_{\rm bar}$)', \
            color='grey', ls=':', marker='', zorder=5)

        if 'Verlinde' in plotfilename:
            # Plot Verlinde slopes
            #gD_0, gobs_0 = calc_gobs_0(gbar_uni)
            #gD_2, gobs_2 = calc_gobs_2(gbar_uni)
            #ax_sub.plot(np.log10(gbar_log), np.log10(gobs_0), ls='--', marker='', color=colors[1], label=r'Verlinde (Flat density distribution: $\rho(r)$ = const.)', zorder=4)
            #ax_sub.plot(np.log10(gbar_log), np.log10(gobs_2), ls='--', marker='', color=colors[2], label=r'Verlinde (Singular Isothermal Sphere: $\rho(r)\sim 1/r^2$)', zorder=4)
            ax_sub.plot(gbar_uni, np.log10(gobs_verlinde(gbar_log)), ls = '--', marker='', color=colors[2], label = r'Verlinde+16 theory (point mass)', zorder=4) #: ${\rm M_b}(r)$=const.)'
            
        # McGaugh fitting function
        ax_sub.plot(gbar_mond, gobs_mond_M16, color='grey', ls='-', marker='', zorder=5)
        ax_sub.plot(gbar_ext, gobs_mond_ext, label = r'McGaugh+16 fitting function (extrapolated)', \
            color='grey', ls='-', marker='', zorder=5)
        
        """
        # Plot Verlinde prediction including full satellite distribution
        if ('massbins' in plotfilename) and ('all' in plotfilename):
            ax_sub.plot(gb_point[N], gtot_all[N], ls='-', marker='', color=colors[3], \
            label=r'Verlinde (full mass distribution)',  zorder=4)
        """    
            
        for Nplot in range(Nbins[1]):
            #print('Nplot:', Nplot)
            
            Ndata = Nplot + N*(Nbins[1])
            
            """
            print('Ndata=',Ndata)
            print('N=',N)
            print('Nplot=',Nplot)
            print('Nbins[0]=',Nbins[0])
            print()
            """
            
            if Nbins[1] > 1:
                shift = 0.004
                data_x_plot = data_x[Ndata] * (1.-shift/2. + shift*Nplot)
            else:
                data_x_plot = data_x[Ndata]
                
            # Plot data
            if Nsize==Nbins:
                ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                color=blacks[Nplot], ls='', marker='.', zorder=8)
            else:
                ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                color=blacks[Nplot], ls='', marker='.', label=datalabels[Nplot], zorder=8)
            
        # Plot KiDS-1000 isolation limit
        if 'KiDS' in param2[N2]:
            ax_sub.axvline(x=isolim[N], ls=':', color='black', label=r'KiDS isolation criterion limit ($R < %g$ Mpc/h)'%isoR)
        
        # Navarro analytical model
        if ('Navarro' in plotfilename):
            # Plot Navarro predictions
            ax_sub.plot(gbar_navarro[N], gobs_navarro[N], ls='-', marker='', color=colors[3], \
            label=r'Navarro+2017 prediction (based on $\langle M_* \rangle$)',  zorder=4)
        
        ## Plot McGaugh observations
        
        # Binned
        ax_sub.plot(gbar_mcgaugh_binned, gobs_mcgaugh_binned, label='McGaugh+16 observations (mean)', \
            ls='', marker='s', markerfacecolor='red', markeredgecolor='black', zorder=1)
        
        #if 'zoomout' not in plotfilename:
        # 2D histogram
        ax_sub.hist2d(gbar_mcgaugh, gobs_mcgaugh, bins=[gbar_mond, gbar_mond], cmin=1, cmap='Blues', zorder=0)
        """
        # Not binned
        ax_sub.errorbar(gbar_mcgaugh, gobs_mcgaugh, xerr=gbar_mcgaugh_error, \
        yerr=gobs_mcgaugh_error, ls='', marker='.', color=colors[0], alpha=0.04, \
        label='McGaugh+2016 observations', zorder=2)
        
        # Plot Lelli+2017 dwarf Spheroidals
        ax_sub.errorbar(loggbar_dsph, loggobs_dsph, xerr=[-loggbar_dsph_elow,loggbar_dsph_ehigh], \
        yerr=[-loggobs_dsph_elow,loggobs_dsph_ehigh], marker='o', color='orange', ls='', zorder=3, alpha=0.1)
        
        ax_sub.errorbar(loggbar_dsph[hqmask], loggobs_dsph[hqmask], \
        xerr=[-loggbar_dsph_elow[hqmask],loggbar_dsph_ehigh[hqmask]], yerr=[-loggobs_dsph_elow[hqmask],loggobs_dsph_ehigh[hqmask]], \
        marker='o', color='orange', ls='', label='Lelli+2017 Dwarf Spheroidals (velocity dispersions at r$_{1/2}$)', zorder=4)
        """            
#        except:
#            pass

        if 'MICE' in plotfilename:
            # Plot MICE mock shearprofiles
            for Nmock in range(Nmocks[1]):

                Ndata = Nplot + N*(Nmocks[1])
                gbar_mask = data_x[Ndata]<max_gbar[Ndata] # np.inf 
                
                #ax_sub.axvline(x=max_gbar[Ndata], ls='--', color=colors[1])#, label='MICE pixel size (0.43 arcmin)')
                
                if miceoffset and ('KiDS' in plotfilename) and ('isolated' in plotfilename):
                    ax_sub.fill_between(data_x_plot[gbar_mask], (data_y_mock[Ndata])[gbar_mask], \
                    (data_y_mock_offset[Ndata])[gbar_mask], color=colors[1], label=mocklabels[0], alpha=valpha, zorder=7)
                else:
                    ax_sub.plot(data_x_plot[gbar_mask], (data_y_mock[Ndata])[gbar_mask], \
                    marker='', ls='-', color=colors[1], label=mocklabels[0], zorder=7)
                    
        if 'Bahamas' in plotfilename:
            # Plot BAHAMAS mock shearprofiles
            for Nmock in range(Nmocks[1]):

                Ndata = Nplot + N*(Nmocks[1])
                
                ax_sub.plot(data_x_bah[Nmock], data_y_bah[Nmock], \
                marker='', ls='-', color=colors[3], label=bahlabels[0], alpha=1., zorder=6)
                
                ax_sub.fill_between(data_x_bah[Nmock], (data_y_bah[Nmock]-0.5*bah_y_std[Nmock]), \
                    (data_y_bah[Nmock]+0.5*bah_y_std[Nmock]), color=colors[3], alpha=0.5, zorder=6)
                
                print('Bin', Nmock+1)
                print(data_y_bah[Nmock]-0.5*bah_y_std[Nmock])
                print(data_y_bah[Nmock]+0.5*bah_y_std[Nmock])
                
        # Plot stellar mass limits
        if massbias and ('KiDS' in param2[N2]):
            ax_sub.fill_between(data_x_plot, data_y_min[N], data_y_max[N], \
            color='black', alpha=0.1, label=r'Systematic stellar mass bias (M$_* \pm 0.2$ dex)', zorder=4)
        
        
        # Plot Crescenzo's data
        #ax_sub.errorbar(gbar_cres, gobs_cres, yerr=[errorl_cres, errorh_cres], ls='', marker='.', label="Tortora+2017 (Early Type Galaxies)", zorder=4)
        
        ## Plot the axes and title
        
        ax_sub.xaxis.set_label_position('top')
        ax_sub.yaxis.set_label_position('right')

        ax.tick_params(labelleft='off', labelbottom='off', top='off', bottom='off', left='off', right='off')

        xticklabels = np.linspace(xlims[0], xlims[1], 5)
        yticklabels = np.linspace(ylims[0], ylims[1], 7)
        
        if (N1+1) != Nrows:
            ax_sub.tick_params(axis='x', labelbottom='off')
        else:
            ax_sub.tick_params(labelsize='14')
            if (Nbins[0]>1) and ('zoomout' not in plotfilename):
                ax_sub.set_xticklabels(['%.0f'%x for x in xticklabels[0:-1]])
                ax_sub.set_yticklabels(['%.1f'%y for y in yticklabels[0:-1]])
                
        if N2 != 0:
            ax_sub.tick_params(axis='y', labelleft='off')
            ax_sub.set_xticklabels(['%.0f'%x for x in xticklabels])
        else:
            ax_sub.tick_params(labelsize='14')
        
        #plt.autoscale(enable=False, axis='both', tight=None)
        
        ax.xaxis.set_label_coords(0.5, -0.09)
        ax.yaxis.set_label_coords(-0.09, 0.5)

        plt.xlim(xlims)
        plt.ylim(ylims)
        
        #plt.gca().invert_xaxis()
        #plt.gca().invert_yaxis()
        
        if Nbins[0]>1:
            plt.title(datatitles[N], x = 0.35, y = 0.85, fontsize=16)

# Define the labels for the plot
xlabel = r'Stellar radial acceleration log($g_*$ [$h_{%g} \, {\rm m/s^2}$])'%(h*100)
ylabel = r'Observed radial acceleration log($g_{\rm obs}$ [$h_{%g} \, {\rm m/s^2}$])'%(h*100)
ax.set_xlabel(xlabel, fontsize=16)
ax.set_ylabel(ylabel, fontsize=16)

handles, labels = ax_sub.get_legend_handles_labels()

# Plot the legend
if Nbins[0] > 1:
    lgd = ax_sub.legend(handles[::-1], labels[::-1], loc='lower right', bbox_to_anchor=(1.9, 0.75)) # side
else:
    lgd = plt.legend(handles[::-1], labels[::-1], loc='upper left')
    plt.tight_layout()

# Save plot
for ext in ['png', 'pdf']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf

