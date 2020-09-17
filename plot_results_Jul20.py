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

# Import constants
pi = np.pi
G = const.G.to('pc3 / (M_sun s2)').value
c = const.c.to('m/s').value
pc_to_m = 3.08567758e16

# Define cosmology
O_matter = 0.2793
O_lambda = 0.7207
h = 0.7
H0 = h * 100 * (u.km/u.s)/u.Mpc
H0 = H0.to('s-1').value

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

blacks = ['black', '#0571b0', 'grey', '#92c5de']

# Dark blue, light blue, red, orange
colors = ['#0571b0', '#92c5de', '#d7191c', '#fdae61']*2

# Defining default plot parameters
blind = 'C'
logplot = False
massbias = True
miceoffset = True
randomsub = True

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

path_sheardata = '/data/users/brouwer/Lensing_results/EG_results_Aug20'

## Input lens selections

"""

# KiDS + GAMA + Verlinde / Navarro (isolated)

param1 = ['']
param2 = [r'GL-KiDS isolated galaxies ($1000 \,{\rm deg}^2$)', r'GAMA isolated galaxies ($180 \, {\rm deg}^2$)']

N1 = 1
N2 = len(param2)
Nrows = 1

path_lenssel = np.array([['No_bins/dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0p1_0p5_lw-logmbar_GL', \
                          'No_bins_gama/Z_0p1_0p5-dist0p1perc_3_inf-logmstarGL_0_11_lw-logmbar_GL']]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_filename = np.array([['shearcovariance/No_bins_%s'%blind]*N2]*N1)

masses_navarro = ['4.8E10'] # (in Msun) No mass bins

datalabels = param2
#plotfilename = '%s/Plots/RAR_KiDS+GAMA+Verlinde_Nobins_isolated_zoomout'%path_sheardata
plotfilename = '%s/Plots/RAR_KiDS+GAMA+Navarro_Nobins_isolated'%path_sheardata

"""

# KiDS + BAHAMAS + MICE (isolated)

param1 = ['']
param2 = [r'GL-KiDS isolated galaxies ($1000 \,{\rm deg}^2$)']

N1 = 1
N2 = len(param2)
Nrows = 1

path_lenssel = np.array([['No_bins/dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0p1_0p5_lw-logmbar_GL']*N2]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_filename = np.array([['shearcovariance/No_bins_%s'%blind]*N2]*N1)

path_mocksel =  np.array([['No_bins/dist0p1perc_3_inf-logmstar_0_11-zcgal_0p1_0p5_lw-logmbar']*N2]*N1)
path_mockcosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_mockfilename = np.array([['shearcovariance/No_bins_%s'%blind]*N2]*N1)
mocklabels = np.array(['GL-MICE isolated mock galaxies'])

labels_bhm = np.array(['BAHAMAS isolated mock galaxies (mass profiles)', 'BAHAMAS isolated mock galaxies (density maps)'])
Nmocks = [1, 1]

datalabels = param2
plotfilename = '%s/Plots/RAR_KiDS+MICE+Bahamas+Verlinde_No_Nobins_isolated_zoomout'%path_sheardata

"""

# KiDS + Verlinde + MICE (isolated, 4 stellar mass bins)

param1 = [8.5,10.3,10.6,10.8,11.]
binname = bins_to_name(param1)

param2 = [r'GL-KiDS isolated galaxies ($1000 \,{\rm deg}^2$)']

N1 = len(param1)-1
N2 = len(param2)
Nrows = 2

path_lenssel = np.array([['logmstar_GL_%s/dist0p1perc_3_inf-zANNZKV_0p1_0p5_lw-logmbar_GL'%(binname)]*N2]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_filename = np.array([['shearcovariance/shearcovariance_bin_%i_%s'%(p1, blind)]*N2 for p1 in np.arange(N1)+1])

path_mocksel =  np.array([['logmstar_%s/dist0p1perc_3_inf-zcgal_0p1_0p5_lw-logmbar'%(binname)]*N2]*N1)
path_mockcosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_mockfilename = np.array([['shearcovariance/shearcovariance_bin_%i_%s'%(p1, blind)]*N2 for p1 in np.arange(N1)+1])
mocklabels = np.array(['GL-MICE isolated mock galaxies'])

#masses_navarro = ['1.3E10', '3.7E10', '6.0E10', '9.0E10'] # Mass bins (in Msun)

datalabels = param2

plotfilename = '%s/Plots/RAR_KiDS+MICE+Verlinde_4-massbins_isolated'%path_sheardata # Substitute: Verlinde / MICE



# KiDS + MICE (all, 4 stellar mass bins)

param1 = [8.5,10.3,10.6,10.8,11.]
binname = bins_to_name(param1)

param2 = [r'All GL-KiDS galaxies ($1000 \,{\rm deg}^2$)']

N1 = len(param1)-1
N2 = len(param2)
Nrows = 2

path_lenssel = np.array([['logmstar_GL_%s/zANNZKV_0p1_0p5_lw-logmbar_GL'%(binname)]*N2]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_filename = np.array([['shearcovariance/shearcovariance_bin_%i_%s'%(p1, blind)]*N2 for p1 in np.arange(N1)+1])

path_mocksel =  np.array([['logmstar_%s/zcgal_0p1_0p5_lw-logmbar'%(binname)]*N2]*N1)
path_mockcosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N2]*N1)
path_mockfilename = np.array([['shearcovariance/shearcovariance_bin_%i_%s'%(p1, blind)]*N2 for p1 in np.arange(N1)+1])
mocklabels = np.array(['All GL-MICE mock galaxies'])

massbias = True
miceoffset = False

datalabels = param2
#datatitles = [r'$%g < \log(M_*) < %g \, {\rm M_\odot}/h_{%g}^{2}$'%(param1[p1], param1[p1+1], h*100) for p1 in range(N1)]
plotfilename = '%s/Plots/RAR_KiDS+MICE+Verlinde_4-massbins_all'%path_sheardata



# Binning by galaxy type with same mass range

param1 = [r'GL-KiDS isolated elipticals', r'GL-KiDS isolated spirals']
param2 = [r'Sersic index $n$ (split at 2)', r'$u-r$ colour (split at 2.5 mag)']

N1 = len(param1)
N2 = len(param2)
Nrows = 1

path_lenssel = np.array([ ['n_2dphot_0p0_2p0_inf/selected_1-zANNZKV_0p1_0p5_lw-logmbar_GL']*N1, \
                          ['MAG_GAAP_u-r_0p0_2p5_inf/selected_1-zANNZKV_0p1_0p5_lw-logmbar_GL']*N1 ])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_1em15_5em12_mps2']*N1]*N1)
path_filename = np.array([['shearcovariance/shearcovariance_bin_2_%s'%blind, \
                            'shearcovariance/shearcovariance_bin_1_%s'%blind]]*N1)

massbias = False
miceoffset = False

datalabels = param1
datatitles = param2
plotfilename = '%s/Plots/RAR_KiDS_galtypes_isolated_samemass'%path_sheardata


#Dwarf galaxies (Edwin)

param1 = ['']
param2 = [r'Light dwarfs: isolated galaxies with log$(M_*/$M$_\odot) < 9.3$',  \
    r'Dwarfs: isolated galaxies with log$(M_*/$M$_\odot) < 10$', r'GL-KiDS isolated lens galaxies ($1000 \,{\rm deg}^2$)']

N1 = 1
N2 = len(param2)
Nrows = 1

path_lenssel = np.array([['No_bins/dist0p1perc_3_inf-logmstarGL_0_9p3-zANNZKV_0p1_0p5_lw-logmbar_GL', \
                          'No_bins/dist0p1perc_3_inf-logmstarGL_0_10-zANNZKV_0p1_0p5_lw-logmbar_GL', \
                          'No_bins/dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0p1_0p5_lw-logmbar_GL']]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins5_1em15_5em12_mps2']*N2]*N1)
path_filename = np.array([['shearcovariance/No_bins_%s'%blind]*N2]*N1)

datalabels = param2
plotfilename = '%s/Plots/RAR_KiDS+dwarfs_Nobins_isolated_zoomout'%path_sheardata
#plotfilename = '%s/Plots/RAR_KiDS+GAMA+Navarro_Nobins_isolated'%path_sheardata

massbias = False
randomsub = False

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

if randomsub:
    ## Random subtraction
    path_randoms = '%s/Randoms/combined_randoms.txt'%path_sheardata

    #try:
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
else:
    print('No randoms subtracted!')
    print()
    pass

# Convert ESD (Msun/pc^2) to acceleration (m/s^2)
data_y_log, error_h, error_l = 4. * G * pc_to_m * np.array([data_y_log, error_h, error_l])

# Adding 0.1 dex to the error bars to account for the translation into the RAR
error_h, error_l = np.array([error_h, error_l]) * 10.**0.1

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

IDfiles = np.array([m.replace('%s.txt'%blind, 'lensIDs.txt') for m in esdfiles])
lensIDs_selected = np.array([np.loadtxt(m) for m in IDfiles])#+1 # Sometimes +1 is needed to line up the ID's
N_selected = [len(m) for m in lensIDs_selected]

# Import the mass catalogue
path_masscat = '/data/users/brouwer/LensCatalogues/baryonic_mass_catalog_%s.fits'%cat
masscat = pyfits.open(path_masscat, memmap=True)[1].data

logmstar = masscat['logmstar_GL']
logmbar = masscat['logmbar_GL']

# Calculate the galaxy masses
mean_mstar, median_mstar, mean_mbar, median_mbar = \
    [np.zeros(len(esdfiles)), np.zeros(len(esdfiles)), np.zeros(len(esdfiles)), np.zeros(len(esdfiles))]
m = []
for m in range(len(esdfiles)):
    IDmask = np.in1d(lensID, lensIDs_selected[m])
       
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

# Reliability limit of isolated KiDS-1000 galaxies
if ('KiDS' in plotfilename):# and ('isolated' in plotfilename):
    isoR = 0.3 # in Mpc
    isolim_log = (G * 10.**mean_mbar)/(isoR * 1e6)**2. * pc_to_m
    print('Isolated KiDS signal reliability limit:', isolim_log)
    
    if not logplot:
        isolim = np.log10(isolim_log)
    else:
        isolim = isolim_log

# Create titles of the bins
if 'massbins' in plotfilename:
    datatitles = [r'$\log\langle M_{\rm gal}/h_{%g}^{-2} {\rm M_\odot} \rangle = %.4g$'%(h*100, mean_mbar[p1]) for p1 in range(N1)]
"""
if 'galtypes' in plotfilename:
    datatitles[0] = r'n$<2$, %s'%datatitles[0]
    datatitles[1] = r'n$<2$, %s'%datatitles[1]
    datatitles[2] = r'n$>2$, $\mu_0>10$, %s'%datatitles[2]
    datatitles[3] = r'n$>2$, $\mu_0>10$, %s'%datatitles[3]
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

    IDfiles_mock = np.array([m.replace('%s.txt'%blind, 'lensIDs.txt') for m in esdfiles_mock])
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
        Da_mean = np.mean(lensDa_mock[IDmask*np.isfinite(lensDa_mock)])
        mstar_mean_mock = np.mean(10.**logmstar_mock[IDmask*np.isfinite(logmstar_mock)])
            
        pixelsize = 3. * 0.43 / 60. * pi/180. # arcmin to radian
        min_R = pixelsize * Da_mean
        print('min_R mock (Mpc):', min_R/1e6)
        max_gbar[m] = (G * pc_to_m * mstar_mean_mock)/min_R**2.
    if not logplot:
        max_gbar = np.log10(max_gbar)
            
    #    print('Da_max:', Da_max)
    #    print('mstar_mean_mock:', mstar_mean_mock)
    #    print('min_R:', min_R)
    print('max_gbar:', 10.**max_gbar)

else:
    print('No mock signal imported!')
    pass


## BAHAMAS mocks

# These lists will contain the Bahamas results in each observable bin
data_x_profiles = []
data_y_profiles = []
std_y_profiles = []

data_x_maps = []
data_y_maps = []
std_y_maps = []

if 'Bahamas' in plotfilename:

    ## Import Bahamas data
    catnum = 515
    lenslist = np.arange(catnum)
    
    # Import galaxy observables from the general catalogue
    path_cat = '/data/users/brouwer/Simulations/Bahamas/BAHAMAS_isolated_new/BAHAMAS_nu0_L400N1024_WMAP9/z_0.250'
    catname = '%s/catalog.dat'%path_cat
    catalog = np.loadtxt(catname).T[:,lenslist]
    
    logM200_bhm = 10.**catalog[3] # M200 of each galaxy
    r200_bhm = catalog[4] * 1e6 # r200 of each galaxy (in Xpc)
    logmstar_bhm = catalog[5] # Stellar mass of each lens galaxy
    
    # Calculate baryonic galaxy mass
    fcold_bhm = 10.**(-0.69*logmstar_bhm + 6.63)
    mstar_bhm = 10.**logmstar_bhm
    mbar_bhm = mstar_bhm * (1 + fcold_bhm)
    mstar_bhm = np.reshape(mstar_bhm[0:catnum], [catnum,1])
    mbar_bhm = np.reshape(mbar_bhm[0:catnum], [catnum,1])
    
    ## Import true enclosed mass profiles
    profbins = 40
    profiles_radius = np.zeros([catnum, profbins])
    profiles_Menclosed = np.zeros([catnum, profbins])
    for x in range(catnum):
        profname = '%s/PROFILES/cluster_%i_Menclosed_profile_types.dat'%(path_cat, lenslist[x])
        profile_c = np.loadtxt(profname).T
        profiles_radius[x] = profile_c[0] * 1e6 # * r200list[x] # in pc # profile_c[0,0:Nbins]
        profiles_Menclosed[x] = profile_c[1]# * M200list[x] # in Msun # profile_c[1,0:Nbins]

    # Calculate true gbar and gobs from enclosed mass profiles
    profiles_gbar = (G * mbar_bhm) / (profiles_radius)**2. * pc_to_m # in m/s^2
    profiles_gobs = (G * profiles_Menclosed) / (profiles_radius)**2. * pc_to_m # in m/s^2

    ## Import ESD profiles from density maps
    mapscatfile = '%s/ESD/Bahamas_ESD_profiles_Rbins-%i_%g-%g%s_isolated.fits'%(path_cat, 15, 1e-15, 5e-12, 'mps2')
    mapscat = pyfits.open(mapscatfile, memmap=True)[1].data
    maps_ESD = mapscat['ESD'][0:catnum]
    maps_Rbins = mapscat['Rbins'][0:catnum]
    print('Imported:', mapscatfile)
    
    # Calculate gobs from the ESD using the SIS assumption
    maps_gobs = maps_ESD * 4.*G * pc_to_m # Convert ESD (Msun/pc^2) to acceleration (m/s^2)

    # Calculate gbar from R
    maps_gbar_bins = (G * mbar_bhm) / (maps_Rbins)**2. * pc_to_m # in m/s^2
    maps_gbar = np.array([(maps_gbar_bins[i])[0:-1] + np.diff(maps_gbar_bins[i])/2. for i in range(catnum)])
    
    ## Select Bahamas lenses with the right masses
    
    # For every stellar mass bin:
    for m in range(Nbins[0]):
      
        # Select galaxies within this stellar mass bin
        if Nbins[0] > 1:
            logmstarmask = (param1[m] < logmstar_bhm) & (logmstar_bhm <= param1[m+1])        
            print('Mstarbin:', m+1, param1[m], param1[m+1])
        else:
            logmstarmask = logmstar_bhm < 11.
            print('Mstar < 11')
        
        # Mask the profiles and density map results
        profiles_gbar_bin, profiles_gobs_bin = [profiles_gbar[logmstarmask], profiles_gobs[logmstarmask]]
        maps_gbar_bin, maps_gobs_bin = [maps_gbar[logmstarmask], maps_gobs[logmstarmask]]
        
        # Calculate the means and standard deviations
        profile_gbar_bin, profile_gobs_bin, profile_gobs_std = utils.mean_profile(profiles_gbar_bin, profiles_gobs_bin, \
            0, 0, profbins, True)
        map_gbar_bin = np.nanmean(maps_gbar, 0)
        map_gobs_bin = np.nanmean(maps_gobs, 0)
        map_gobs_std = np.nanstd(maps_gobs, 0)
        
        # Append result to the list of all bins
        data_x_profiles.append(profile_gbar_bin)
        data_y_profiles.append(profile_gobs_bin)
        std_y_profiles.append(profile_gobs_std)

        data_x_maps.append(map_gbar_bin)
        data_y_maps.append(map_gobs_bin)
        std_y_maps.append(map_gobs_std)
    
    # Make the lists into numpy arrays    
    data_x_profiles = np.array(data_x_profiles)
    data_y_profiles = np.array(data_y_profiles)
    std_y_profiles = np.array(std_y_profiles)
    
    data_x_maps = np.array(data_x_maps)
    data_y_maps = np.array(data_y_maps)
    std_y_maps = np.array(std_y_maps)
    
    if not logplot:
        std_y_profiles = 1./np.log(10.) * std_y_profiles/data_y_profiles # This must be calculated first
        data_x_profiles = np.log10(data_x_profiles)
        data_y_profiles = np.log10(data_y_profiles)
        
        std_y_maps = 1./np.log(10.) * std_y_maps/data_y_maps # This must be calculated first
        data_x_maps = np.log10(data_x_maps)
        data_y_maps = np.log10(data_y_maps)

## Perform chi2 analysis

print()
print('Chi2 analysis:')

for Nc in range(N2):
    print()
    print('** %s **'%param2[Nc])
    
    # Import covariance matrix
    if 'No_bins' in esdfiles[0]:
        covfile = esdfiles[Nc].replace('bins_%s'%blind, 'matrix_%s'%blind)
        if massbias:
            covfile_min, covfile_max = [ f.replace('bins_%s'%blind, 'matrix_%s'%blind) \
            for f in [esdfiles_min[Nc], esdfiles_max[Nc]] ]
    else:
        covfile = esdfiles[0].replace('bin_1_%s'%blind, 'matrix_%s'%blind)
        if 'matrix' not in covfile: # In case bin 2 is plotted first
            covfile = esdfiles[0].replace('bin_2_%s'%blind, 'matrix_%s'%blind)
        if massbias:
            covfile_min, covfile_max = [ f.replace('bin_1_%s'%blind, 'matrix_%s'%blind) \
            for f in [esdfiles_min[0], esdfiles_max[0]] ]
    
    # Translating the ESD covariance into gobs covariance, and adding 0.1 dex to the error
    covariance = np.loadtxt(covfile).T
    covariance[4] = (4. * G * pc_to_m)**2. * covariance[4] * (10.**0.1)**2.
    
    if massbias:
        covariance_min, covariance_max = [ np.loadtxt(f).T \
            for f in [covfile_min, covfile_max] ]
        covariance_min[4] = (4. * G * pc_to_m)**2. * covariance_min[4] * (10.**0.1)**2.
        covariance_max[4] = (4. * G * pc_to_m)**2. * covariance_max[4] * (10.**0.1)**2.
    
    # Import data
    if N2>1: # When multiple datasets are used
        data_x_chi2, data_y_chi2 = [np.array([data_x_log[Nc]]), np.array([data_y_log[Nc]])]
        if massbias:
            data_y_min_chi2, data_y_max_chi2 = [np.array([data_y_min_log[Nc]]), np.array([data_y_max_log[Nc]])]
    else: # When one datasets is used (with multiple bins)
        data_x_chi2, data_y_chi2 = [data_x_log, data_y_log]
        if massbias:
            data_y_min_chi2, data_y_max_chi2 = [data_y_min_log, data_y_max_log]
        
    nRbins = len(data_y[0])
    dof = N1 * nRbins
    print('DOF:', dof)
    
    if 'KiDS' in plotfilename:
        
        # Create a mask for data beyond the isolation limit
        isomask = [np.ravel( np.argwhere(data_x_log[y] < isolim_log[y]) ) \
            + y * nRbins for y in range(N1)]
        isomask = np.concatenate(isomask).ravel()
        
    
    if 'Navarro' in plotfilename:
        
        if 'KiDS' in param2[Nc]:
            dof_iso = dof - len(np.ndarray.flatten(isomask))
            chi2_navarro = utils.calc_chi2(data_y_chi2, 10.**gobs_navarro[0], covariance, masked=isomask)
            pvalue_navarro = stats.chi2.cdf(chi2_navarro, dof_iso)
            sigma_navarro = np.sqrt(stats.chi2.ppf(pvalue_navarro, 1.))
            print('DOF (isomask): %i'%dof_iso)
            print( 'Navarro: chi2=%g / %i = %g, sigma=%g'%(chi2_navarro, dof_iso, chi2_navarro/dof_iso, sigma_navarro) )
        else:
            chi2_navarro = utils.calc_chi2(data_y_chi2, 10.**gobs_navarro[0], covariance)
            pvalue_navarro = stats.chi2.cdf(chi2_navarro, dof)
            sigma_navarro = np.sqrt(stats.chi2.ppf(pvalue_navarro, 1.))
            print( 'Navarro: chi2=%g / %i = %g, sigma=%g'%(chi2_navarro, dof, chi2_navarro/dof, sigma_navarro) )
    
        
    # Testing Emergent Gravity and the MOND relation
    if 'Verlinde' in plotfilename:
                
        print('Covariance:', np.shape(covariance))
        
        chi2_mond = utils.calc_chi2(data_y_chi2, gobs_mond(data_x_chi2), covariance)
        chi2_EG = utils.calc_chi2(data_y_chi2, gobs_verlinde(data_x_chi2), covariance)
        
        # Calculating the sigma- from the p-value
        pvalue_mond = stats.chi2.cdf(chi2_mond, dof)
        pvalue_EG = stats.chi2.cdf(chi2_EG, dof)
        
        sigma_mond = np.sqrt(stats.chi2.ppf(pvalue_mond, 1.))
        sigma_EG = np.sqrt(stats.chi2.ppf(pvalue_EG, 1.))
        
        print( 'MOND: chi2=%g / %i = %g, sigma=%g'%(chi2_mond, dof, chi2_mond/dof, sigma_mond) )
        print( 'Verlinde: chi2=%g / %i = %g, sigma=%g'%(chi2_EG, dof, chi2_EG/dof, sigma_EG) )
        print()
        
        if 'KiDS' in param2[Nc]:
            # Apply mask beyond the isolation limit
            chi2_mond = utils.calc_chi2(data_y_chi2, gobs_mond(data_x_chi2), covariance, masked=isomask)        
            chi2_EG = utils.calc_chi2(data_y_chi2, gobs_verlinde(data_x_chi2), covariance, masked=isomask)

            print(np.ndarray.flatten(isomask))
            dof_iso = dof - len(np.ndarray.flatten(isomask))
            print('DOF (isomask): %i'%dof_iso)
            
            # Calculating the sigma- from the p-value
            pvalue_mond = stats.chi2.cdf(chi2_mond, dof_iso)
            pvalue_EG = stats.chi2.cdf(chi2_EG, dof_iso)
        
            sigma_mond = np.sqrt(stats.chi2.ppf(pvalue_mond, 1.))
            sigma_EG = np.sqrt(stats.chi2.ppf(pvalue_EG, 1.))
            
            print( 'MOND (isolation mask): chi2=%g / %i = %g, sigma=%g'%(chi2_mond, dof_iso, chi2_mond/dof_iso, sigma_mond) )
            print( 'Verlinde (isolation mask): chi2=%g / %i = %g, sigma=%g'%(chi2_EG, dof_iso, chi2_EG/dof_iso, sigma_EG) )
            print()
            
            if massbias:
                chi2_min = utils.calc_chi2(data_y_min_chi2, gobs_mond(data_x_chi2), covariance_min, masked=isomask)
                chi2_max = utils.calc_chi2(data_y_max_chi2, gobs_mond(data_x_chi2), covariance_max, masked=isomask)
                    
                pvalue_min = stats.chi2.cdf(chi2_min, dof_iso)
                pvalue_max = stats.chi2.cdf(chi2_max, dof_iso)
            
                sigma_min = np.sqrt(stats.chi2.ppf(pvalue_min, 1.))
                sigma_max = np.sqrt(stats.chi2.ppf(pvalue_max, 1.))
                
                print( 'MOND (mask + mass_min): chi2=%g / %i = %g, sigma=%g'%(chi2_min, dof_iso, chi2_min/dof_iso, sigma_min) )
                print( 'MOND (mask + mass_max): chi2=%g / %i = %g, sigma=%g'%(chi2_max, dof_iso, chi2_max/dof_iso, sigma_max) )
                print()

    
    # Testing the MICE simulation
    if 'MICE' in plotfilename:
        
        # Create a mask for data beyond the MICE resolution limit
        micemask = [np.ravel( np.argwhere(10.**max_gbar[y] < data_x_log[y]) ) \
            + y * nRbins for y in range(N1)]
        micemask = np.concatenate(micemask).ravel()
        
        dof_mice = dof - len(np.ndarray.flatten(micemask))
        print('DOF (micemask): %i'%dof_mice)
        
        # Calculate chi2 for the truly isolated MICE galaxies
        chi2_mock = utils.calc_chi2(data_y_chi2, 10.**(data_y_mock), covariance, masked=micemask)
        pvalue_mock = stats.chi2.cdf(chi2_mock, dof_mice)
        sigma_mock = np.sqrt(stats.chi2.ppf(pvalue_mock, 1.))
        print('Mock: chi2=%g / %i = %g, sigma=%g'%(chi2_mock, dof_mice, chi2_mock/dof_mice, sigma_mock) )
        
        if miceoffset:
            # Calculate chi2 for the offset MICE galaxies
            chi2_mock_offset = utils.calc_chi2(data_y_chi2, 10.**(data_y_mock_offset), covariance, masked=micemask)
            pvalue_mock_offset = stats.chi2.cdf(chi2_mock_offset, dof_mice)
            sigma_mock_offset = np.sqrt(stats.chi2.ppf(pvalue_mock_offset, 1.))
            print('Mock (offset): chi2=%g / %i = %g, sigma=%g'%(chi2_mock_offset, dof_mice, chi2_mock_offset/dof_mice, sigma_mock_offset) )
        
    # Testing the difference between ellipticals and spirals
    if 'galtypes' in plotfilename:

        ## Difference: fractional and dex
        #diff_galtype = np.mean(data_y_log[Nc*N2]/data_y_log[Nc*N2+1])
        
        diff_galtype = np.mean(data_y_log[Nc*N2] / data_y_log[Nc*N2+1])
            #(np.mean(data_y_log[Nc*N2]+data_y_log[Nc*N2+1])/2)
        
        print('Fractional difference: %g (%g dex)'%(diff_galtype, np.log10(diff_galtype)) )
        
        # With isolation limit
        isomask_galtype = (isolim_log[Nc*N2] < data_x_log[Nc*N2])
        diff_galtype_iso = np.mean((data_y_log[Nc*N2])[isomask_galtype] / (data_y_log[Nc*N2+1])[isomask_galtype] )
        
        print('... with isolation limit: %g (%g dex)'%(diff_galtype_iso, np.log10(diff_galtype_iso)) )
        
        ## Difference: Chi-squared
        
        # Use only the covariance matrix and DOF of one galaxy type
        dof = nRbins
        covariance = np.delete(covariance, np.arange(nRbins**2., 4.*nRbins**2.), 1)
        
        # Calculate the chi-squared and sigma values
        chi2_galtype = utils.calc_chi2([data_y_log[Nc*N2]], [data_y_log[Nc*N2+1]], covariance)#, masked=isomask)
        pvalue_galtype = stats.chi2.cdf(chi2_galtype, nRbins)
        sigma_galtype = np.sqrt(stats.chi2.ppf(pvalue_galtype, 1.))
        
        print( 'Difference: chi2=%g / %i = %g, sigma=%g'%(chi2_galtype, dof, chi2_galtype/dof, sigma_galtype) )
        
            
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
    if 'Navarro' in plotfilename:
        xlims = [1e-15, 1e-10]
        ylims = [1e-13, 4e-10]

if logplot:
    plt.xscale('log')
    plt.yscale('log')
else:
    xlims, ylims = [np.log10(xlims), np.log10(ylims)]

# Define the colors of the ESD profiles
if len(param2) > 2:
    plotcolors = colors
else:
    plotcolors = blacks

# Plotting the ueber matrix
if Nbins[0] > 1:
    fig = plt.figure(figsize=(Ncolumns*5.,Nrows*4))
else:
    fig = plt.figure(figsize=(8,6))

gs_full = gridspec.GridSpec(1,1)
gs = gridspec.GridSpecFromSubplotSpec(Nrows, Ncolumns, wspace=0, hspace=0, subplot_spec=gs_full[0,0])

ax = fig.add_subplot(gs_full[0,0])

# Create the plot panels
for NR in range(Nrows):
    for NC in range(Ncolumns):
        
        N = np.int(NR*Ncolumns + NC)
        print('Plotting: row %i, column %i (panel %i)'%(NR, NC, N))
        
        ax_sub = fig.add_subplot(gs[NR, NC])
                
        # Plot guiding lines
        if 'Bahamas' in plotfilename:
            ax_sub.plot(gbar_uni, gbar_uni, color='grey', ls=':', marker='', zorder=5)
        else:
            ax_sub.plot(gbar_uni, gbar_uni, color='grey', ls=':', marker='', zorder=5, \
                label = r'Unity (No dark matter: $g_{\rm obs} = g_{\rm bar}$)')

        if ('Navarro' not in plotfilename):
            # Verlinde's Emergent Gravity
            ax_sub.plot(gbar_uni, np.log10(gobs_verlinde(gbar_log)), ls = '--', marker='', color=colors[2], label = r'Verlinde+16 Emergent Gravity (point mass)', zorder=5) #: ${\rm M_b}(r)$=const.)'
            
            # Verlinde slopes
            #gD_0, gobs_0 = calc_gobs_0(gbar_uni)
            #gD_2, gobs_2 = calc_gobs_2(gbar_uni)
            #ax_sub.plot(np.log10(gbar_log), np.log10(gobs_0), ls='--', marker='', color=colors[1], label=r'Verlinde (Flat density distribution: $\rho(r)$ = const.)', zorder=4)
            #ax_sub.plot(np.log10(gbar_log), np.log10(gobs_2), ls='--', marker='', color=colors[2], label=r'Verlinde (Singular Isothermal Sphere: $\rho(r)\sim 1/r^2$)', zorder=4)

            # McGaugh fitting function        
            ax_sub.plot(gbar_mond, gobs_mond_M16, color='grey', ls='-', marker='', zorder=4)
            ax_sub.plot(gbar_ext, gobs_mond_ext, label = r'McGaugh+16 fitting function (extrapolated)', \
                color='grey', ls='-', marker='', zorder=4)
        
        
        # Plot the data in this panel
        for Nplot in range(Nbins[1]):
            Ndata = Nplot + N*(Nbins[1])
            
            print('    Nplot %i (Ndata %i): %s'%(Nplot, Ndata, param2[Nplot]))
                       
            if Nbins[1] > 1:
                shift = 0.004
                data_x_plot = data_x[Ndata] * (1.-shift/2. + shift*Nplot)
            else:
                data_x_plot = data_x[Ndata]
                
            # Plot data
            if Nsize==Nbins:
                ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                color=plotcolors[Nplot], ls='', marker='.', zorder=8)
            else:
                ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                color=plotcolors[Nplot], ls='', marker='.', label=datalabels[Nplot], zorder=8)
        
        # Plot the mocks in this panel
        if 'MICE' in plotfilename:
            # Plot MICE mock shearprofiles
            for Nmock in range(Nmocks[1]):

                Ndata = Nplot + N*(Nmocks[1])
                gbar_mask = data_x[Ndata]<max_gbar[Ndata] # np.inf 
                
                #ax_sub.axvline(x=max_gbar[Ndata], ls='--', color=colors[1])#, label='MICE pixel size (0.43 arcmin)')
                
                if miceoffset and ('KiDS' in plotfilename) and ('isolated' in plotfilename):
                    ax_sub.fill_between(data_x_plot[gbar_mask], (data_y_mock[Ndata])[gbar_mask], \
                    (data_y_mock_offset[Ndata])[gbar_mask], color=colors[2], label=mocklabels[0], alpha=valpha, zorder=7)
                else:
                    ax_sub.plot(data_x_plot[gbar_mask], (data_y_mock[Ndata])[gbar_mask], \
                    marker='', ls='-', color=colors[2], label=mocklabels[0], zorder=7)
                    
        if 'Bahamas' in plotfilename:
            # Plot BAHAMAS mock shearprofiles
            for Nmock in range(Nmocks[1]):

                Ndata = Nplot + N*(Nmocks[1])

                # From density maps
                ax_sub.plot(data_x_maps[Nmock], data_y_maps[Nmock], \
                marker='', ls='-', color=colors[4], label=labels_bhm[1], alpha=1., zorder=6)
                
                ax_sub.fill_between(data_x_maps[Nmock], (data_y_maps[Nmock]-0.5*std_y_maps[Nmock]), \
                    (data_y_maps[Nmock]+0.5*std_y_maps[Nmock]), color=colors[4], alpha=0.5, zorder=6)
                
                # From true mass profiles
                ax_sub.plot(data_x_profiles[Nmock], data_y_profiles[Nmock], \
                marker='', ls='-', color=colors[3], label=labels_bhm[0], alpha=1., zorder=6)
                
                ax_sub.fill_between(data_x_profiles[Nmock], (data_y_profiles[Nmock]-0.5*std_y_profiles[Nmock]), \
                    (data_y_profiles[Nmock]+0.5*std_y_profiles[Nmock]), color=colors[3], alpha=0.5, zorder=6)
                
            
        # Extras for KiDS data (isolation limit and/or stellar mass bias)
        if 'KiDS' in plotfilename:
            if 'iso' in plotfilename:
                # Plot KiDS-1000 isolation limit
                #ax_sub.axvline(x=isolim[N], ls=':', color=blacks[1], \
                #    label=r'KiDS isolation criterion limit ($R > %g \, {\rm Mpc/h_{70}}$)'%isoR)
                ax_sub.axvspan(isolim[N], -16., color=blacks[1], alpha=0.1, \
                    label=r'KiDS isolation criterion limit ($R > %g \, {\rm Mpc/h_{70}}$)'%isoR)
                
            if massbias:
            # Plot stellar mass limits
                print('Plotting stellar mass bias')
                ax_sub.fill_between(data_x_plot, data_y_min[N], data_y_max[N], \
                color=blacks[0], alpha=0.15, label=r'Systematic stellar mass bias (${\rm M_*} \pm 0.2$ dex)', zorder=4)
        print()
        
        # Navarro analytical model
        if ('Navarro' in plotfilename):
            # Plot Navarro predictions
            ax_sub.plot(gbar_navarro[N], gobs_navarro[N], ls='-', marker='', color=colors[3], \
            label=r'Navarro+2017 prediction (based on $\langle M_{\rm gal} \rangle$)',  zorder=4)
        
        ## Plot McGaugh+16 observations
        
        # McGaugh - Binned
        ax_sub.plot(gbar_mcgaugh_binned, gobs_mcgaugh_binned, label='SPARC rotation curves (mean)', \
            ls='', marker='s', markerfacecolor='red', markeredgecolor='black', zorder=1)
        
        # McGaugh - 2D histogram
        ax_sub.hist2d(gbar_mcgaugh, gobs_mcgaugh, bins=[gbar_mond, gbar_mond], cmin=1, cmap='Blues', zorder=0)

        
        ## Plot the axes and title
        
        ax_sub.xaxis.set_label_position('top')
        ax_sub.yaxis.set_label_position('right')

        ax.tick_params(labelleft='off', labelbottom='off', top='off', bottom='off', left='off', right='off')

        xticklabels = np.linspace(xlims[0], xlims[1], 5)
        yticklabels = np.linspace(ylims[0], ylims[1], 7)
        
        if (NR+1) != Nrows:
            ax_sub.tick_params(axis='x', labelbottom='off')
        else:
            ax_sub.tick_params(labelsize='14')
            if (Nbins[0]>1) and ('zoomout' not in plotfilename):
                ax_sub.set_xticklabels(['%.0f'%x for x in xticklabels[0:-1]])
                ax_sub.set_yticklabels(['%.1f'%y for y in yticklabels[0:-1]])
                
        if NC != 0:
            ax_sub.tick_params(axis='y', labelleft='off')
            ax_sub.set_xticklabels(['%.0f'%x for x in xticklabels])
        else:
            ax_sub.tick_params(labelsize='14')
        
        #plt.autoscale(enable=False, axis='both', tight=None)
        
        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.yaxis.set_label_coords(-0.1, 0.5)

        plt.xlim(xlims)
        plt.ylim(ylims)
        
        #plt.gca().invert_xaxis()
        #plt.gca().invert_yaxis()
        
        if Nbins[0]>1:
            plt.title(datatitles[N], x = 0.50, y = 0.85, fontsize=16)

# Define the labels for the plot
xlabel = r'Baryonic (stars+cold gas) radial acceleration log($g_{\rm bar}$ [$h_{%g} \, {\rm m/s^2}$])'%(h*100)
ylabel = r'Observed radial acceleration log($g_{\rm obs}$ [$h_{%g} \, {\rm m/s^2}$])'%(h*100)
ax.set_xlabel(xlabel, fontsize=16)
ax.set_ylabel(ylabel, fontsize=16)

handles, labels = ax_sub.get_legend_handles_labels()

# Plot the legend
if Nbins[0] > 1:
    lgd = ax_sub.legend(handles[::-1], labels[::-1], loc='lower right', \
    bbox_to_anchor=(1.0, Nrows*1.0), ncol=2) # top
    #bbox_to_anchor=(2.0, 0.75)) # side
else:
    lgd = plt.legend(handles[::-1], labels[::-1], loc='upper left')
    plt.tight_layout()

# Save plot
for ext in ['pdf']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf

