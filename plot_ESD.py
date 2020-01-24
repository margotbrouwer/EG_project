#!/usr/bin/python

# Import the necessary libraries
import sys

import numpy as np
import pyfits
import os

from astropy import constants as const, units as u
from astropy.cosmology import LambdaCDM
import scipy.optimize as optimization
from scipy import stats
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
colors = ['#d7191c', '#0571b0', '#92c5de', '#fdae61']*2


## Define paths of ESD profiles

# Fiducial plotting parameters
Runit = 'Mpc'
datatitles = []
Nrows = 1

path_sheardata = '/data/users/brouwer/Lensing_results/EG_results_Jan20'

"""
# KiDS vs. GAMA comparison
params = ['Z', 'zANNz2ugri']
N = len(params)
Nrows = 1

path_lenssel = np.array([['No_bins/%s_0_0p5'%p for p in params]])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N])
path_filename = np.array([['shearcovariance/No_bins_A']*N])

datalabels = [r'GAMA', 'KiDS']
datatitles = [r'']

plotfilename = '%s/Plots/ESD_GAMA_KiDS_all'%path_sheardata


# Lens redshift comparison
params1 = ['KiDS', 'GAMA']
params2 = np.arange(4)+1
N1 = len(params1)
N2 = len(params2)
Nrows = 2

path_lenssel = np.array([['zANNz2ugri_0p1_0p2_0p3_0p4_0p5/zANNz2ugri_0_0p5',
                            'Z_0p1_0p2_0p3_0p4_0p5_GAMA/Z_0_0p5']]*N2)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N1]*N2)
path_filename = np.array([['shearcatalog/shearcatalog_bin_%i_A'%p2]*N1 for p2 in params2])

datalabels = params1
datatitles = [r'$0.1<Z<0.2$',r'$0.2<Z<0.3$',r'$0.3<Z<0.4$',r'$0.4<Z<0.5$']

plotfilename = '%s/Plots/ESD_KiDS_GAMA_Zbins'%path_sheardata


# Isolation test: f_iso (KiDS)

params1 = ['KiDS']
params2 = ['0p0', '0p1', '0p2']
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['No_bins/dist%sperc_3_inf-logmstar_10p4_10p6-zANNz2ugri_0_0p5'%p for p in params2]])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcatalog/No_bins_A']*N2]*N1)

datalabels = [r'$f_{\rm iso}=0.0$', r'$f_{\rm iso}=0.1$', r'$f_{\rm iso}=0.2$', r'$f_{\rm iso}=0.3$']
datatitles = params1

plotfilename = '%s/Plots/ESD_KiDS_isotest_fiso'%path_sheardata


# Isolation test: f_iso (MICE)

params1 = ['KiDS']
params2 = ['0p0', '0p1', '0p2']
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['No_bins_400deg2/dist%sperc_3_inf-logmstar_10p4_10p6-zcgal_0_0p5'%p for p in params2]])
path_cosmo = np.array([['zcgal_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcatalog/No_bins_A']*N2]*N1)

datalabels = [r'$f_{\rm iso}=0.0$', r'$f_{\rm iso}=0.1$', r'$f_{\rm iso}=0.2$']#, r'$f_{\rm iso}=0.3$']
datatitles = params1

plotfilename = '%s/Plots/ESD_MICE_isotest_fiso'%path_sheardata


# Isolation test: r_iso (KiDS)

params1 = ['0p1']
params2 = ['3', '4', '5', '6']
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['No_bins/dist%sperc_%s_inf-logmstarGL_0_11-zANNz2ugri_0_0p5'%(p1,p2) \
                                                            for p2 in params2] for p1 in params1])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcatalog/No_bins_A']*N2]*N1)

datatitles = ['']
datalabels = [r'$r_{\rm sat} = %s$ Mpc/h'%p for p in params2]

plotfilename = '%s/Plots/ESD_KiDS_isotest_riso'%path_sheardata


# Isolation test: r_iso (MICE)

params1 = ['0p1']
params2 = ['3', '4p5', '6']
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['No_bins_400deg2/dist%sperc_%s_inf-logmstar_10p9_11p1-zcgal_0_0p5'%(p1,p2) \
                                                                for p2 in params2] for p1 in params1])
path_cosmo = np.array([['zcgal_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcatalog/No_bins_A']*N2]*N1)

datatitles = ['']
datalabels = ['3 Mpc', '4.5 Mpc', '6 Mpc']

plotfilename = '%s/Plots/ESD_MICE_isotest_riso_fsat0p1'%path_sheardata


# Isolated vs. not isolated (KiDS)

params1 = ['KiDS']
params2 = [r'All galaxies', r'Isolated ($r_{\rm iso}=3$ Mpc)', r'Isolated, bright (m$_{\rm r}<17.5$)']
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['No_bins/zANNz2ugri_0_0p5', 'No_bins/dist0p1perc_3_inf-zANNz2ugri_0_0p5', \
                            'No_bins/MAGAUTO_0_17p5-dist0p1perc_3_inf-zANNz2ugri_0_0p5']])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcatalog/No_bins_A']*N2]*N1)

datatitles = params1
datalabels = params2

plotfilename = '%s/Plots/ESD_KiDS_isotest'%path_sheardata


# Isolated vs. not isolated (KiDS, log(M_*)< 11.)

params1 = ['KiDS']
params2 = [r'All galaxies', r'Isolated ($r_{\rm iso}=3$ Mpc)']
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['No_bins/logmstarGL_0_11-zANNZKV_0_0p5', 'No_bins/dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0_0p5']])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcatalog/No_bins_A']*N2]*N1)

datatitles = params1
datalabels = params2

plotfilename = '%s/Plots/ESD_KiDS_isotest'%path_sheardata


# Isolated vs. not isolated (GAMA)

params1 = ['GAMA']
params2 = [r'All galaxies', r'Isolated: $r_{\rm iso}=3$ Mpc', r'Isolated, bright: mag$<17.3$']
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['No_bins_GAMAII/nQ_3_inf', 'No_bins_GAMAII/dist0p1perc_3_inf-nQ_3_inf', \
                            'No_bins_GAMAII/Rpetro_0_17p3-dist0p1perc_4p5_inf-nQ_3_inf']])
path_cosmo = np.array([['ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcatalog/No_bins_A']*N2]*N1)

datatitles = params1
datalabels = params2

plotfilename = '%s/Plots/ESD_GAMA_isotest'%path_sheardata


# Isolated vs. not isolated (MICE)

params1 = ['MICE']
params2 = [r'All galaxies', r'Isolated ($r_{\rm iso}=3$ Mpc)', r'Isolated, bright (m$_{\rm r}<17.5$)']
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['No_bins_400deg2/zcgal_0_0p5', 'No_bins_400deg2/dist0p1perc_3_inf-logmstar_10p4_10p6-zcgal_0_0p5', \
                            'No_bins_400deg2/dist0p1percfaint_3_inf-logmstar_10p4_10p6-zcgal_0_0p5']])
path_cosmo = np.array([['zcgal_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcatalog/No_bins_A']*N2]*N1)

datatitles = params1
datalabels = params2

plotfilename = '%s/Plots/ESD_MICE_isotest'%path_sheardata

"""

# True vs. offset redshifts (MICE)

params1 = ['MICE']
params2 = ['All galaxies', r'Isolated: $r_{\rm sat}(f_{\rm M_*}>0.1)>$3 Mpc/$h_{70}$, log$(M_*)<11\,{\rm M_\odot}/h_{70}^2$', \
            r'Isolated, offset: $\sigma_{\rm z}/(1+z)=0.018$, $\sigma_{\rm M_*}=0.21$ dex']
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['No_bins/zcgal_0_0p5', 'No_bins/dist0p1perc_3_inf-logmstar_0_11-zcgal_0_0p5', \
                            'No_bins/dist0p1percoffsetZM_3_inf-logmstaroffsetZM_0_11-zcgal_0_0p5']])
path_cosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcatalog/No_bins_A']*N2]*N1)

datatitles = params1
datalabels = params2

plotfilename = '%s/Plots/ESD_MICE_isotest_offset'%path_sheardata

"""


# 2D isolation test (MICE): True vs. offset redshifts 
dz = [0.001, 0.005, 0.01, 0.018]

params1 = ['', 'offset']
params2 = [str(s).replace('.', 'p') for s in dz]
print(params2)

N1 = len(params1)
N2 = len(params2)
Nrows = 2

path_lenssel = np.array([['No_bins_mice/fsatsigma%s%s_0_0p1-logmstar_0_11-zcgal_0_0p5'%(p2,p1) \
                            for p1 in params1] for p2 in params2])
path_cosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N1]*N2)
path_filename = np.array([['shearcatalog/No_bins_A']*N1]*N2)

datatitles = ['Isolated, $\delta z=%s$'%s for s in dz]
datalabels = ['MICE', 'MICE-offset']

plotfilename = '%s/Plots/ESD_MICE_isotest-sigma_offset'%path_sheardata


# 2D isolation test (KiDS)
#dz = [0.001, 0.005, 0.01, 0.018]
dz = [0.001, 0.002, 0.003, 0.004, 0.005]

params1 = ['KiDS-1000']
params2 = [str(s).replace('.', 'p') for s in dz]
print(params2)

N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['No_bins/fsatsigma%s_0_0p1-logmstarGL_0_11-zANNz2ugri_0_0p5'%(p2) for p2 in params2]]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcatalog/No_bins_A']*N2]*N1)

datatitles = params1
datalabels = ['Isolated, $\delta z=%s$'%s for s in dz]

plotfilename = '%s/Plots/ESD_KiDS_isotest_dz-0p001-0p005'%path_sheardata


# Stellar mass bins (KiDS)

massbins = [8.5,10.3,10.6,10.8,11.]

params1 = ['Isolated']
params2 = ['$%g <$ log($M_*$) $< %g M_\odot$'%(massbins[m], massbins[m+1]) for m in range(len(massbins)-1)]
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['logmstar_GL_8p5_10p3_10p6_10p8_11p0/dist0p1perc_3_inf-zANNZKV_0_0p5']*N2]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcovariance/shearcovariance_bin_%i_A'%p2 for p2 in np.arange(N2)+1]*N1])

datatitles = params1
datalabels = params2

plotfilename = '%s/Plots/ESD_Mstarbins-4_eqSNall_isolated'%path_sheardata


# Lens photoz errors versus no photoz errors

params1 = ['GAMA']
params2 = ['Without lens photo-z errors', 'With lens photo-z errors']
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['No_bins_gamatest/Z_0p1_0p2', 'No_bins_gamatest_withsigma/Z_0p1_0p2']]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcovariance/No_bins_A']*N2]*N1)

datatitles = params1
datalabels = params2

plotfilename = '%s/Plots/ESD_photoz-test'%path_sheardata


# Stellar mass bins (KiDS)

#massbins = [8.5,10.5,10.7,10.9,11.]
#massbins = [9.5,10.,10.5,10.75,11.]
massbins = [8.5,10.3,10.6,10.8,11.]
binname = bins_to_name(massbins)

params1 = ['']#, 'dist0p1perc_3_inf-']
params2 = [r'$%g <$ log($M_*$) $< %g \, {\rm M_\odot}$'%(massbins[m], massbins[m+1]) for m in range(len(massbins)-1)]
N1 = len(params1)
N2 = len(params2)
Nrows = 1

path_lenssel = np.array([['logmstar_GL_%s/%szANNz2ugri_0_0p5'%(binname, p1)]*N2 for p1 in params1])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcovariance/shearcovariance_bin_%i_A'%p2 for p2 in np.arange(N2)+1]]*N1)

datatitles = ['All galaxies', r'Isolated: $r_{\rm sat}(f_{\rm M_*}>0.1)>$3 Mpc/$h_{70}$']
datalabels = params2

plotfilename = '%s/Plots/ESD_Mstarbins-%s_iso'%(path_sheardata, binname)


#/data/users/brouwer/Lensing_results/EG_results_Jun19/No_bins_#/zcgal_0_0p5/zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_0p03_3_Mpc/shearcatalog/No_bins_A.txt
"""

## Import measured ESD
cat = 'mice'
h = 0.7

if 'mice' in cat:
    O_matter = 0.25
    O_lambda = 0.75
else:
    O_matter = 0.2793
    O_lambda = 0.7207

cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

# Import the Lens catalogue
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

esdfiles = np.array([['%s/%s/%s/%s.txt'%\
	(path_sheardata, path_lenssel[i,j], path_cosmo[i,j], path_filename[i,j]) \
	for j in np.arange(np.shape(path_lenssel)[1])] for i in np.arange(np.shape(path_lenssel)[0]) ])

Nbins = np.shape(esdfiles)
Nsize = np.size(esdfiles)
esdfiles = np.reshape(esdfiles, [Nsize])

print('Plots, profiles:', Nbins)

# Importing the shearprofiles and lens IDs
data_x, R_src, data_y, error_h, error_l, N_src = utils.read_esdfiles(esdfiles)
#print('mean error ratio:', np.mean(error_h[0]/error_h[1]))


## Find the mean galaxy mass
IDfiles = np.array([m.replace('A.txt', 'lensIDs.txt') for m in esdfiles])
lensIDs_selected = np.array([np.loadtxt(m) for m in IDfiles])#+1
N_selected = [len(m) for m in lensIDs_selected]

# Import the mass catalogue
path_masscat = '/data/users/brouwer/LensCatalogues/baryonic_mass_catalog_%s.fits'%cat
masscat = pyfits.open(path_masscat, memmap=True)[1].data

"""
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
print('Bin limits:', massbins)
print('Number of galaxies:', N_selected) 
print()
print('mean logmstar:', mean_mstar)
print('median logmstar:', median_mstar)
print()
print('mean logmbar:', mean_mbar)
print('median logmbar:', median_mbar)
print()
"""


# Calculate the difference between subsequent bins
for n in range(len(data_y)-1):
    chisquared = np.sum((data_y[n] - data_y[n+1])**2. / ((error_h[n]+error_h[n+1])/2.))
    dof = len(data_y[0])
    prob = 1. - stats.chi2.cdf(chisquared, dof)

    diff = (data_y[n+1] - data_y[n])/((data_y[n+1] + data_y[n])/2.)
    diff_error = (error_h[n+1] - error_h[n])/((error_h[n+1] + error_h[n])/2.)
    
    print()
    print('Difference:', np.mean(diff[8:-1]))
    print('Difference, error:', np.mean(diff_error))
    #print(params2[n], 'vs.', params2[n+1])
    print('Chi2:', chisquared)
    print('DoF:', dof)
    print('P-value:', prob)
print()


if 'mice' in cat:
    z_mice = np.mean(lensZ)
    Da_mice = utils.calc_Dc(z_mice, cosmo)/(1.+z_mice)
    Da_mice = Da_mice.to(Runit).value
    micelim = 0.43/60. * np.pi/180. * Da_mice
    print('MICE limit:', micelim, Runit)
    
## Create the plot

Ncolumns = int(Nbins[0]/Nrows)

# Plotting the ueber matrix
if Nbins[0] > 1:
    fig = plt.figure(figsize=(Ncolumns*5.,Nrows*4))
else:
    fig = plt.figure(figsize=(7,5))

gs_full = gridspec.GridSpec(1,1)
gs = gridspec.GridSpecFromSubplotSpec(Nrows, Ncolumns, wspace=0, hspace=0, subplot_spec=gs_full[0,0])

ax = fig.add_subplot(gs_full[0,0])


for N1 in range(Nrows):
    for N2 in range(Ncolumns):
    
        ax_sub = fig.add_subplot(gs[N1, N2])
        
        N = np.int(N1*Ncolumns + N2)

        for Nplot in range(Nbins[1]):
            
            Ndata = Nplot + N*(Nbins[1])
            """
            print('Ndata=',Ndata)
            print('N=',N)
            print('Nplot=',Nplot)
            print('Nbins[0]=',Nbins[0])
            print()
            """
            
            if Nbins[1] > 1:
                dx = 0.04
                data_x_plot = data_x[Ndata] * (1.-dx/2.+dx*Nplot)
            else:
                data_x_plot = data_x[Ndata]

            # Plot data
            if 'mice' not in cat:
                if Nsize==Nbins:
                    ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                    color=colors[Nplot], ls='', marker='.', zorder=4)
                else:
                    ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                    color=colors[Nplot], ls='', marker='.', label=datalabels[Nplot], zorder=4)
            else:
                if Nsize==Nbins:
                    ax_sub.plot(data_x_plot, data_y[Ndata], \
                    color=colors[Nplot], ls='-', marker='.', zorder=4)
                else:
                    ax_sub.plot(data_x_plot, data_y[Ndata], \
                    color=colors[Nplot], ls='-', marker='.', label=datalabels[Nplot], zorder=4)
        
        if 'mice' in cat:
            ax_sub.axvline(x=micelim, ls='--', color='grey')
        
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
        
        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.yaxis.set_label_coords(-0.1/Ncolumns, 0.5)
        
        if Nbins[0]>1:
            plt.title(datatitles[N], x = 0.5, y = 0.9, fontsize=16)

        plt.xlim([0.03, 3])
        plt.ylim([1e-1, 1e2])

        plt.xscale('log')
        plt.yscale('log')
        
# Define the labels for the plot
xlabel = r'Radius R (${\rm %s} / h_{%g}$)'%(Runit, h*100)
ylabel = r'Excess Surface Density $\Delta\Sigma$ ($h_{%g} {\rm M_{\odot} / {\rm pc^2}}$)'%(h*100)
ax.set_xlabel(xlabel, fontsize=16)
ax.set_ylabel(ylabel, fontsize=16)

handles, labels = ax_sub.get_legend_handles_labels()

# Plot the legend
#plt.legend()

if Nbins[0] > 1:
    plt.legend(loc='lower left', fontsize=12)
#    lgd = ax_sub.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.5*Ncolumns, 0.7*Nrows)) # side
#    plt.legend(handles[::-1], labels[::-1], loc='lower right')
else:
    plt.legend(loc='best', fontsize=12)#loc='lower right')
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
