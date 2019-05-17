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

# Constants
h = 0.7
O_matter = 0.315
O_lambda = 0.685

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})

# Colours
# Blue, green, turquoise, cyan
blues = ['#332288', '#44AA99', '#117733', '#88CCEE']

# Light red, Red, light pink, pink
reds = ['#CC6677', '#882255', '#CC99BB', '#AA4499']
#colors = np.array([reds,blues])

#colors = ['#0571b0', '#92c5de', '#d7191c']*2#, '#fdae61']
colors = ['#0571b0', '#92c5de', '#d7191c', '#fdae61']*2


# Defining the paths to the data
blind = 'A'
logplot = False

#Import constants
pi = np.pi
G = const.G.to('pc3/Msun s2').value
c = const.c.to('m/s').value
H0 = 2.269e-18 # 70 km/s/Mpc in 1/s
h=0.7

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


# Define guiding lines
gbar_mond = np.logspace(-12, -8, 50)
gbar_ext = np.logspace(-15, -12, 30)
gbar_uni = np.logspace(-15, -8, 50)

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

#paramlims = [8.5, 9.8, 10.4, 10.9, 12.0]
#dexvalues = ['0p6', '0p7', '0p8']
#percvalues = ['0p25', '0p2', '0p1']
#distvalues = ['3','4','4p5']


#path_sheardata = 'data2/brouwer/shearprofile/EG_results'
path_sheardata = '/data/users/brouwer/Lensing_results/EG_results_Mar19'

## Input lens selections

"""

# Mass bins (4) with equal S/N (GAMA vs MICE)
paramlims = [8.5, 10.5, 10.8, 11.1, 12.0]
N = len(paramlims)-1
Nrows = 2
path_lenssel = np.array([['logmstar_8p5_10p5_10p8_11p1_12p0/dist0p1perc_4p5_inf-nQ_3_inf_lw-logmbar']]*N) # dist0p1perc_4p5_inf-
path_cosmo = np.array([['ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']]*N)
path_filename = np.array([['shearcovariance_bin_%i_A'%i] for i in np.arange(N)+1])

datalabels = [r'KiDS+GAMA (isolated)']
mocklabels = [r'MICE (isolated: ${\rm D_c}(> 0.1{\rm M_*})>4.5$Mpc)']
datatitles = [r'($%g < \rm{M_*} < %g \rm{M_\odot}$)'%(paramlims[i], paramlims[i+1]) for i in range(N)]

# Mocks
path_mocksel = np.array([['lmstellar_8p5_10p5_10p8_11p1_12p0/dist0p1perc_4p5_inf-lmstellar_8p5_12_lw-lmstellar']]*N) # dist0p1perc_4p5_inf-
path_mockcosmo = np.array([['zcgal_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']]*N)
path_mockfilename = np.array([['shearcovariance_bin_%i_A'%i] for i in np.arange(N)+1])

plotfilename = '%s/Plots/RAR_GAMA+Navarro_4-massbins_isolated_strong'%path_sheardata

"""

# Mass bins (4) with equal S/N (KiDS-only vs MICE)
paramlims = np.array([8.5, 10.3, 10.6, 10.8, 11.0])
N = len(paramlims)-1
Nrows = 1

path_lenssel = np.array([['logmbar_8p5_10p3_10p6_10p8_11p0/dist0p1perc_4p5_inf-zANNz2ugri_0p1_0p5_lw-logmbar']*N])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']*N])
path_filename = np.array([['shearcovariance/shearcovariance_bin_%i_A'%i for i in np.arange(N)+1]])

datatitles = [r'KiDS-1000 (isolated)']
datalabels = [r'{\rm $%g < M_{bar} < %g M_\odot$}'%(paramlims[i], paramlims[i+1]) for i in range(N)]
plotfilename = '%s/Plots/RAR_KiDS_4-massbins_equal_SN_oneplot'%path_sheardata

"""

# Compare isolation criteria (New vs. Old)
path_lenssel = np.array([['No_bins/dist0p1perc_4p5_inf-nQ_3_inf_lw-logmbar', \
    'No_bins/dist0p2perc_4_inf-nQ_3_inf_lw-logmbar', 'No_bins/RankBCG_m999_2-isocen3_1-nQ_3_inf_lw-logmbar']])
path_cosmo = np.array([['ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']*N])

datalabels= [r'New (strong): ${\rm D_c}(> 0.1{\rm M_*})>4.5$Mpc', r'New (weak): ${\rm D_c}(> 0.2{\rm M_*})>4$Mpc', \
    'Old: Brouwer et al. (2017)']


# Compare isolation criteria (massperc and distmin)
path_lenssel = np.array([['No_bins/dist%sperc_4_inf-nQ_3_inf_lw-logmbar'%d for d in percvalues], \
    ['No_bins/dist0p2perc_%s_inf-nQ_3_inf_lw-logmbar'%d for d in distvalues]])
path_cosmo = np.array([['ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']*N, \
    ['ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']*N])
path_filename = np.array([['No_bins_A']*N, ['No_bins_A']*N])

datalabels = ['Weaker (%sperc/%sMpc)'%(percvalues[0],distvalues[0]), 'Fiducial (%sperc/%sMpc)'%(percvalues[1],distvalues[1]), \
    'Stronger (%sperc/%sMpc)'%(percvalues[2],distvalues[2])]
datatitles = [r'Maximum satellite mass (Rmin=4Mpc)', r'Minimum satellite distance (Mmax=0p2perc)']



# GAMA+KiDS (isolated)
path_lenssel = np.array([['No_bins/dist0p1perc_4p5_inf-nQ_3_inf_lw-logmbar', \
                        'No_bins/dist0p1perc_4p5_inf-zANNz2ugri_0p1_0p5_lw-logmbar']])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2', \
                        'ZB_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']])
path_filename = np.array([['shearcovariance/No_bins_A']*2])

datalabels = [r'K1000 + GAMA-II (isolated: D(M$>$0.1)$>$4.5 Mpc)', \
                r'KiDS-1000 foreground galaxies (isolated)']

plotfilename = '%s/Plots/RAR_GAMA_KiDS_isolated_lin'%path_sheardata



# GAMA+MICE (isolated)
path_lenssel = np.array([['No_bins/dist0p1perc_4p5_inf-nQ_3_inf_lw-logmbar']])#, 'No_bins/dist0p2perc_4_inf_lw-lmstellar']])
path_cosmo = np.array([['ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']])#, 'zcgal_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']])
path_filename = np.array([['No_bins_A']])

datalabels = [r'KiDS+GAMA, isolated $({\rm log}_{10}({\rm \bar{M}}_*)=11.1 {\rm M}_\odot)$']


path_mocksel = np.array([['No_bins/dist0p1perc_4p5_inf_lw-lmstellar']])
path_mockcosmo = np.array([['zcgal_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']])
path_mockfilename = np.array([['No_bins_A']])
mocklabels = [r'MICE, isolated: ${\rm D_c}(> 0.1{\rm M_*})>4.5$Mpc']

plotfilename = '%s/Plots/RAR_GAMA+MICE+Navarro_isolated_strong'%path_sheardata
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
data_x, data_y, error_h, error_l = utils.read_esdfiles(esdfiles)

# Convert ESD (Msun/pc^2) to acceleration (m/s^2)
data_y, error_h, error_l = 4. * G * 3.08567758e16 *\
    np.array([data_y, error_h, error_l])

print('data_y:', data_y)
print('error_h:', error_h)

if not logplot:
    # Convert the errors into log-errors
    error_h = 1./np.log(10.) * error_h/abs(data_y)
    error_l = 1./np.log(10.) * error_l/abs(data_y)

    floor = 1e-15
    error_h[data_y<0.] = data_y[data_y<0.] + error_h[data_y<0.] - floor
    data_y[data_y<0.] = floor
    data_y[data_y>0.] = np.log10(data_y[data_y>0.])
    data_x = np.log10(data_x)

# Find the mean galaxy mass
IDfiles = np.array([m.replace('A.txt', 'lensIDs.txt') for m in esdfiles])
lensIDs_selected = np.array([np.loadtxt(m) for m in IDfiles])
N_selected = [len(m) for m in lensIDs_selected]

fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat('gama', h, cosmo)

mean_mass = np.zeros(len(esdfiles))
median_mass = np.zeros(len(esdfiles))
for m in range(len(esdfiles)):
    IDmask = np.in1d(lensID, lensIDs_selected[m])
    mean_mass[m] = np.log10(np.mean(10.**logmstar[IDmask*np.isfinite(logmstar)]))
    median_mass[m] = np.median(logmstar[IDmask*np.isfinite(logmstar)])

print('Number of galaxies:', N_selected) 
print('mean logmstar:', mean_mass)
print('median logmstar:', median_mass)

"""
# Import Crescenzo's RAR from Early Type Galaxies
cres = np.loadtxt('RAR_profiles/crescenzo_RAR.txt').T
gbar_cres = 10**cres[0]
gobs_cres = 10**cres[1]
errorl_cres = 10**cres[1] - 10**(cres[1]+cres[2])
errorh_cres = 10**(cres[1]+cres[3]) - 10**cres[1]
"""

# Import Kyle's RAR (based on Navarro et al. 2017)
masses_navarro = ['1.5E10', '3.2E10', '4.6E10', '8.9E10'] # (in Msun)
#masses_navarro = ['1.5E10', '4.6E10', '8.9E10', '1.8E11'] # Median bin mass (in Msun)
#masses_navarro = ['6.3E10'] # Median of all isolated galaxies

gbar_navarro = []
gobs_navarro = []
for m in range(len(masses_navarro)):
    data_navarro = np.loadtxt('RAR_profiles/RAR_Mstar%s.txt'%masses_navarro[m]).T
    
    if logplot:
        gbar_navarro.append(data_navarro[1])
        gobs_navarro.append(data_navarro[1] + data_navarro[0])
    else:
        gbar_navarro.append(np.log10(data_navarro[1]))
        gobs_navarro.append(np.log10(data_navarro[1] + data_navarro[0]))

#gbar_navarro = np.array([data_navarro[m][1] for m in range(len(masses_navarro))])
#gobs_navarro = np.array([data_navarro[m][1] + data_navarro[m][0] for m in range(len(masses_navarro))])
#gobs_kyle = np.array([kyle9[1]+kyle9[0], kyle10[1]+kyle10[0], kyle11[1]+kyle11[0]])
#gbar_navarro, gobs_navarro = [np.log10(gbar_navarro), np.log10(gobs_navarro)]

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
gbar_mcgaugh_error_low = gbar_mcgaugh - 10.**(loggbar_mcgaugh-loggbar_mcgaugh_error)
gbar_mcgaugh_error_high = 10.**(loggbar_mcgaugh+loggbar_mcgaugh_error) - gbar_mcgaugh
gobs_mcgaugh_error_low = gobs_mcgaugh - 10.**(loggobs_mcgaugh-loggobs_mcgaugh_error)
gobs_mcgaugh_error_high = 10.**(loggobs_mcgaugh+loggobs_mcgaugh_error) - gobs_mcgaugh

#gbar_mcgaugh_error = np.log(10.) * gbar_mcgaugh * loggbar_mcgaugh_error
#gobs_mcgaugh_error = np.log(10.) * gobs_mcgaugh * loggobs_mcgaugh_error


# Bin the McGaugh data
Nbins_gbar = 14
gbar_lims = np.linspace(-11.5, -8.8, Nbins_gbar)
inds = np.digitize(gbar_mcgaugh, gbar_lims)

gbar_mcgaugh_mean = np.array([np.mean(gbar_mcgaugh[inds==b]) for b in range(Nbins_gbar)])
gobs_mcgaugh_mean = np.array([np.mean(gobs_mcgaugh[inds==b]) for b in range(Nbins_gbar)])
gbar_mcgaugh_number = np.array([len(gbar_mcgaugh[inds==b]) for b in range(Nbins_gbar)])

gbar_mcgaugh_std = np.array([np.std(gbar_mcgaugh[inds==b]) for b in range(Nbins_gbar)])
gobs_mcgaugh_std = np.array([np.std(gobs_mcgaugh[inds==b]) for b in range(Nbins_gbar)])

gbar_mcgaugh_meanerror = np.sqrt(np.array([np.sum((gbar_mcgaugh_error[inds==b])**2.) for b in range(Nbins_gbar)]))/gbar_mcgaugh_number
gobs_mcgaugh_meanerror = np.sqrt(np.array([np.sum((gobs_mcgaugh_error[inds==b])**2.) for b in range(Nbins_gbar)]))/gbar_mcgaugh_number

gobs_mcgaugh_meanerror_low = np.sqrt(np.array([np.sum((gobs_mcgaugh_error_low[inds==b])**2.) for b in range(Nbins_gbar)]))
gobs_mcgaugh_meanerror_high = np.sqrt(np.array([np.sum((gobs_mcgaugh_error_high[inds==b])**2.) for b in range(Nbins_gbar)]))
gobs_mcgaugh_meanerror_low, gobs_mcgaugh_meanerror_high = \
[gobs_mcgaugh_meanerror_low, gobs_mcgaugh_meanerror_high]/gbar_mcgaugh_number

gbar_mcgaugh_meanerror_low = np.sqrt(np.array([np.sum((gbar_mcgaugh_error_low[inds==b])**2.) for b in range(Nbins_gbar)]))
gbar_mcgaugh_meanerror_high = np.sqrt(np.array([np.sum((gbar_mcgaugh_error_high[inds==b])**2.) for b in range(Nbins_gbar)]))
gbar_mcgaugh_meanerror_low, gbar_mcgaugh_meanerror_high = \
[gbar_mcgaugh_meanerror_low, gbar_mcgaugh_meanerror_high]/gbar_mcgaugh_number

gobs_mcgaugh_stderror = gobs_mcgaugh_std#/np.sqrt(gbar_mcgaugh_number)
gbar_mcgaugh_stderror = gbar_mcgaugh_std#/np.sqrt(gbar_mcgaugh_number)
"""

# Import Lelli+2017 dSph data
data_dsph = np.genfromtxt('RAR_profiles/Lelli2017_dSph_data.txt', delimiter=',').T
loggbar_dsph, loggbar_dsph_ehigh, loggbar_dsph_elow, loggobs_dsph, loggobs_dsph_ehigh, loggobs_dsph_elow \
    = np.array([data_dsph[d] for d in np.arange(15,21)])

# Apply the High-Quality mask
hqsample = data_dsph[1]
hqmask = (hqsample==1)

## Mocks
print()
print('Import mock signal:')
try:
    # Defining the mock profiles
    esdfiles_mock = np.array([['%s/%s/%s/shearcovariance/%s.txt'%\
        (path_sheardata, path_mocksel[i,j], path_mockcosmo[i,j], path_mockfilename[i,j]) \
        for j in np.arange(np.shape(path_lenssel)[1])] for i in np.arange(np.shape(path_lenssel)[0]) ])

    Nmocks = np.shape(esdfiles_mock)

    if Nmocks[1] > 5:
        valpha = 0.3
    else:
        valpha = 1.

    # Importing the mock shearprofiles
    esdfiles_mock = np.reshape(esdfiles_mock, [Nsize])

    data_x_mock, data_y_mock, error_h_mock, error_l_mock = utils.read_esdfiles(esdfiles_mock)
    data_y_mock, error_h_mock, error_l_mock = 4. * G * 3.08567758e16 *\
        np.array([data_y_mock, error_h_mock, error_l_mock]) # Convert ESD (Msun/pc^2) to acceleration (m/s^2)
    
    if not logplot:
        error_h_mock = 1./np.log(10.) * error_h_mock/data_y_mock
        error_l_mock = 1./np.log(10.) * error_l_mock/data_y_mock
        data_x_mock, data_y_mock = np.log10(np.array([data_x_mock, data_y_mock]))
    
    IDfiles_mock = np.array([m.replace('A.txt', 'lensIDs.txt') for m in esdfiles_mock])
    lensIDs_selected_mock = np.array([np.loadtxt(m) for m in IDfiles_mock])

    # Import lens catalog
    fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
    utils.import_lenscat('mice', h, cosmo)
    lensDc = lensDc.to('pc').value

    max_gbar = np.zeros(len(esdfiles_mock))
    for m in range(len(esdfiles_mock)):
        IDmask = np.in1d(lensID, lensIDs_selected_mock[m])
        Dc_max = np.amax(lensDc[IDmask*np.isfinite(lensDc)])
        mstar_mean = np.mean(10.**logmstar[IDmask*np.isfinite(logmstar)])
            
        pixelsize = 0.43 / 60. * pi/180. # arcmin to radian
        min_R = pixelsize * Dc_max
        max_gbar[m] = np.log10((G*3.08567758e16 * mstar_mean)/min_R**2.)

    #    print('Dc_max:', Dc_max)
    #    print('mstar_mean:', mstar_mean)
    #    print('min_R:', min_R)
    print('max_gbar:', max_gbar)
except:
    print('No mock signal imported!')
    pass

## Randoms
path_randoms = np.array([ [''] ])

try:
    print()
    print('Import random signal:')

    path_randoms = np.reshape(path_randoms, [Nsize])
    random_esdfile = np.array(['/%s/%s/%s/%s'%(path_sheardata, path_random, path_cosmo, path_filename) for path_random in path_randoms])
    random_data_x, random_data_y, random_error_h, random_error_l = utils.read_esdfiles(random_esdfile)
    
    # Subtract random signal
    data_y = data_y-random_data_y
    error_h = np.sqrt(error_h**2. + random_error_h**2)
    error_l = np.sqrt(error_l**2. + random_error_l**2)

except:
    print('No randoms subtracted!')
    print()
    pass
    

## Create the plot

Ncolumns = int(Nbins[0]/Nrows)

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
        ax_sub.plot(gbar_mond, gobs_mond_M16, label = r'McGaugh+2016 fitting function (extrapolated)',\
            color='grey', ls='-', marker='', zorder=1)
        ax_sub.plot(gbar_ext, gobs_mond_ext, color='grey', ls='--', marker='', zorder=1)
        ax_sub.plot(gbar_uni, gbar_uni, label = r'Unity (No dark matter: $g_{\rm tot} = g_{\rm bar}$)', \
        color='grey', ls=':', marker='', zorder=1)
                
        for Nplot in range(Nbins[1]):
            
            Ndata = Nplot + N*(Nbins[1])
            """
            print('Ndata=',Ndata)
            print('N=',N)
            print('Nplot=',Nplot)
            print('Nbins[0]=',Nbins[0])
            print()
            """
            
            if Nbins[1] > 2:
                data_x_plot = data_x[Ndata] * (1.+0.001*Nplot)
            else:
                data_x_plot = data_x[Ndata]
            
            # Plot data
            if Nsize==Nbins:
                ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                color=colors[Nplot], ls='', marker='.', zorder=4)
            else:
                ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                color=colors[Nplot], ls='', marker='.', label=datalabels[Nplot], zorder=4)
            """
            # Plot Navarro predictions
            ax_sub.plot(gbar_navarro[Ndata], gobs_navarro[Ndata], ls='--', marker='', color=colors[Ndata], \
            label='Navarro+2017 ($M_*=%s} M_\odot$)'%masses_navarro[Ndata].replace('E','\cdot10^{'), zorder=5)
            """
            
        ## Plot McGaugh observations
        
        # Binned
        ax_sub.plot(gbar_mcgaugh_binned, gobs_mcgaugh_binned, label='McGaugh+2016 observations (mean)', \
            ls='', marker='s', markerfacecolor='red', markeredgecolor='black', zorder=2)
        """
        # 2D histogram
        ax_sub.hist2d(gbar_mcgaugh, gobs_mcgaugh, bins=[gbar_mond, gbar_mond], cmin=1, cmap='Blues')
        
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
        
        try:
            # Plot mock shearprofiles
            for Nmock in range(Nmocks[1]):

                Ndata = Nplot + N*(Nmocks[1])
                gbar_mask = data_x[Ndata]<max_gbar[Ndata] # np.inf 
                
                if Ndata==0:
                    ax_sub.plot(data_x_plot[gbar_mask], (data_y_mock[Ndata])[gbar_mask], \
                    marker='', ls='-', color=colors[Ndata], label=mocklabels[0], alpha=valpha, zorder=1)
                else:
                    ax_sub.plot(data_x_plot[gbar_mask], (data_y_mock[Ndata])[gbar_mask], \
                    marker='', ls='-', color=colors[Ndata], label=mocklabels[0], alpha=valpha, zorder=1)
                
                ax_sub.axvline(x=max_gbar[Ndata], ls='--', color=colors[Ndata], label='MICE pixel size (0.43 arcmin)')
                
        except:
            pass
        
        
        # Plot the axes and title
        
        ax_sub.xaxis.set_label_position('top')
        ax_sub.yaxis.set_label_position('right')

        ax.tick_params(labelleft='off', labelbottom='off', top='off', bottom='off', left='off', right='off')

        if (N1+1) != Nrows:
            ax_sub.tick_params(axis='x', labelbottom='off')
        else:
            ax_sub.tick_params(labelsize='14')
        if N2 != 0:
            ax_sub.tick_params(axis='y', labelleft='off')
        else:
            ax_sub.tick_params(labelsize='14')
        
        #plt.autoscale(enable=False, axis='both', tight=None)
        
        ax.xaxis.set_label_coords(0.5, -0.15)
        ax.yaxis.set_label_coords(-0.15/Ncolumns, 0.5)
            
        # Plot Crescenzo's data
        #ax_sub.errorbar(gbar_cres, gobs_cres, yerr=[errorl_cres, errorh_cres], ls='', marker='.', label="Tortora+2017 (Early Type Galaxies)", zorder=4)

        # Plot Verlinde slopes
        #gD_0, gobs_0 = calc_gobs_0(gbar_log)
        #gD_2, gobs_2 = calc_gobs_2(gbar_log)
        #ax_sub.plot(np.log10(gbar_log), np.log10(gobs_0), ls='--', marker='', color=colors[1], label=r'Verlinde (Flat density distribution: $\rho(r)$ = const.)', zorder=6)
        #ax_sub.plot(np.log10(gbar_log), np.log10(gobs_2), ls='--', marker='', color=colors[2], label=r'Verlinde (Singular Isothermal Sphere: $\rho(r)\sim 1/r^2$)', zorder=6)
        #ax_sub.plot(np.log10(gbar_log), np.log10(gobs_verlinde(gbar_log)), ls = '--', marker='', color=colors[0], label = r'Verlinde (Point mass: ${\rm M_b}(r)$=const.)', zorder=6)
        
        #"""
        # Zoomed in
        xlims = [1e-15, 1e-11]
        ylims = [1e-13, 1e-10]
        """
        # Zoomed out
        xlims = [1e-15, 2e-9]
        ylims = [2e-13, 2e-9]
        """
        
        if logplot:
            plt.xscale('log')
            plt.yscale('log')
        else:
            xlims, ylims = [np.log10(xlims), np.log10(ylims)]
        

        plt.xlim(xlims)
        plt.ylim(ylims)
        
        #plt.gca().invert_xaxis()
        #plt.gca().invert_yaxis()
        
        if Nbins[0]>1:
            plt.title(datatitles[N], x = 0.35, y = 0.85, fontsize=16)

# Define the labels for the plot
xlabel = r'Baryonic radial acceleration log($g_{\rm bar}$ [${\rm h_{%g} \, m/s^2}$])'%(h*100)
ylabel = r'Total radial acceleration log($g_{\rm tot}$ [${\rm h_{%g} \, m/s^2}$])'%(h*100)
ax.set_xlabel(xlabel, fontsize=16)
ax.set_ylabel(ylabel, fontsize=16)

handles, labels = ax_sub.get_legend_handles_labels()

# Plot the legend
#plt.legend()

if Nbins[0] > 1:
    plt.legend(loc='lower right')
#    lgd = ax_sub.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.5*Ncolumns, 0.7*Nrows)) # side
#    plt.legend(handles[::-1], labels[::-1], loc='lower right')
else:
    plt.legend()#loc='lower right')
#    plt.legend(handles[::-1], labels[::-1], loc='best')
#    lgd = ax_sub.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.85, 1.55)) # top



plt.tight_layout()

# Save plot
for ext in ['pdf', 'png']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf


