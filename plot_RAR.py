#!/usr/bin/python

# Import the necessary libraries
import sys

import numpy as np
import pyfits
import os

from astropy import constants as const, units as u
import scipy.optimize as optimization
import modules_EG as utils

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

from matplotlib import gridspec

#import matplotlib.style
#matplotlib.style.use('classic')

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

colors = ['#0571b0', '#92c5de', '#d7191c', '#fdae61']

# Defining the paths to the data
blind = 'A'

#Import constants
G = const.G.to('pc3/Msun s2').value
c = const.c.to('pc/s')
h=0.7

def gobs_mcgaugh(gbar, g0=1.2e-10):
    gobs = gbar / (1 - np.exp( -np.sqrt(gbar/g0) ))
    return gobs

gbar_mcgaugh = np.logspace(-12, -11, 50)
gbar_ext = np.logspace(-15, -12, 30)
gbar_uni = np.logspace(-14, -11, 50)

## Import shear and random profiles

# Fiducial troughs:sizes

Runit = 'pc'
plotfilename = '/data/users/brouwer/Lensing_results/EG_results_Nov18/Plots/RAR_isolation_test'
Nrows = 1

#paramlims = [8.5, 9.8, 10.4, 10.9, 12.0]
#paramlims = [8.5, 10.5, 10.8, 11.1, 12.0]
#N = len(paramlims)-1


#path_sheardata = 'data2/brouwer/shearprofile/EG_results'
path_sheardata = '/data/users/brouwer/Lensing_results/EG_results_Nov18'

# Input lens selections

path_lenssel = np.array([['No_bins/dist0p5dex_3_inf-nQ_3_inf_lw-logmbar', 'No_bins/RankBCG_m999_2-isocen3_1-nQ_3_inf_lw-logmbar']])
#path_lenssel = np.array(['No_bins/logmstar_8p5_11_lw-logmbar', 'No_bins/lmstellar_8p5_11_lw-lmstellar'])
#path_lenssel = np.array(['No_bins/logmstar_8p5_11_lw-logmbar', 'No_bins/logmstar_8p5_11_lw-logmbar'])
#path_lenssel = np.array([['logmstar_8p5_10p5_10p8_11p1_12p0/nQ_3_inf_lw-logmbar']*N])
#path_lenssel = np.array([['logmstar_8p5_10p5_10p8_11p1_12p0/nQ_3_inf_lw-logmbar']*Nbins, \
#                            ['lmstellar_8p5_10p5_10p8_11p1_12p0/lmstellar_8p5_12_lw-lmstellar']*Nbins]).T

path_cosmo = np.array([['ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2', 'ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']])
#path_cosmo = np.array([['ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']*N])
#path_cosmo = np.array([['ZB_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']*Nbins, \
#                            ['zcgal_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']*Nbins]).T

path_filename = np.array([['No_bins_A', 'No_bins_A']])
#path_filename = np.array(['shearcovariance_bin_%i_%s.txt'%(i, blind) for i in np.arange(2)+1])
#path_filename = np.array([['shearcovariance_bin_%i_A'%i for i in np.arange(N)+1]])
#path_filename = np.array([['shearcovariance_bin_%i_A'%i for i in np.arange(Nbins)+1], \
#                                ['shearcovariance_bin_%i_A'%j for j in np.arange(Nbins)+1]]).T

datalabels= [r'New (dist[$M_{\rm gal}>0.5$ dex]$>3$Mpc)', 'Old (Brouwer et al. 2017)']
#datalabels = [r'($%g < M* < %g M_\odot$)'%(paramlims[i], paramlims[i+1]) for i in range(N)]
#datalabels = [r'KiDS+GAMA  (isolated centrals)', r'KiDS+GAMA (isolated centrals, $M_* \leq 10^{11} M_\odot$)']
#datalabels = [r'KiDS+GAMA (isolated centrals, $M_* \leq 10^{11} M_\odot$)']
#datalabels = [r'KiDS+GAMA (isolated centrals, $8.5 \leq$ log($M_*$) $\leq 10.8 M_\odot$)', r'KiDS+GAMA (isolated centrals, $10.8 \leq$ log($M_*$) $\leq 11 M_\odot$)']
#datalabels = [r'KiDS+GAMA, isolated spirals ($n_r<2.5$)', r'KiDS+GAMA, isolated ellipticals ($n_r>2.5$)']
#datalabels = [r'KiDS+GAMA, isolated (small: $r_e<4$kpc)', r'KiDS+GAMA, isolated (large: $r_e>4$kpc)']
#datalabels = [r'KiDS-GAMA ($M_*<10^{11} M_{\odot}$)', r'MICE ($M_*<10^{11} M_{\odot}$)']
#datalabels = [r'KiDS-GAMA (New)', r'KiDS-GAMA (Old)']
#datalabels = [r'($%g < M* < %g M_\odot$)'%(paramlims[i], paramlims[i+1]) for i in range(Nbins)]
datatitles = [r'KiDS+GAMA']

#esdfiles = np.array(['%s/%s_%s.txt'%(path_sheardata, path_lenssel, path_cosmo[i]) for i in np.arange(len(path_cosmo))])
#esdfiles = np.array([['%s/%s/%s/shearcovariance/%s.txt'%\
#    (path_sheardata, path_lenssel[i], path_cosmo[i], path_filename[i]) for i in np.arange(len(path_lenssel))]])
esdfiles = np.array([['%s/%s/%s/shearcovariance/%s.txt'%\
	(path_sheardata, path_lenssel[i,j], path_cosmo[i,j], path_filename[i,j]) \
	for j in np.arange(np.shape(path_lenssel)[1])] for i in np.arange(np.shape(path_lenssel)[0]) ])

Nbins = np.shape(esdfiles)
Nsize = np.size(esdfiles)
esdfiles = np.reshape(esdfiles, [Nsize])

print('Plots, profiles:', Nbins)

# Importing the shearprofiles and lens IDs
data_x, data_y, error_h, error_l = utils.read_esdfiles(esdfiles)
data_y, error_h, error_l = 4. * G * 3.08567758e16 *\
    np.array([data_y, error_h, error_l]) # Convert ESD (Msun/pc^2) to acceleration (m/s^2)
print(data_x, data_y)
print('Average S/N:', np.mean(data_y/error_h, 1))

# Import Crescenzo's RAR from Early Type Galaxies
cres = np.loadtxt('RAR_profiles/crescenzo_RAR.txt').T
gbar_cres = 10**cres[0]
gtot_cres = 10**cres[1]
errorl_cres = 10**cres[1] - 10**(cres[1]+cres[2])
errorh_cres = 10**(cres[1]+cres[3]) - 10**cres[1]

# Import Kyle's RAR from LCDM
kyle9 = np.loadtxt('RAR_profiles/RAR_Mstar1.0E9.txt').T
kyle10 = np.loadtxt('RAR_profiles/RAR_Mstar2.1E10.txt').T
kyle11 = np.loadtxt('RAR_profiles/RAR_Mstar1.0E11.txt').T
gbar_kyle = np.array([kyle9[1], kyle10[1], kyle11[1]])
gtot_kyle = np.array([kyle9[1]+kyle9[0], kyle10[1]+kyle10[0], kyle11[1]+kyle11[0]])


## Mocks
"""
path_mocksel = np.array([['lmstellar_8p5_10p5_10p8_11p1_12p0/lmstellar_8p5_12_lw-lmstellar']*N])
path_mockcosmo = np.array([['zcgal_0p1_0p9-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins10_1em15_5em12_mps2']*N])
mocklabels = [r'MICE simulation predictions']
"""
try:
# Defining the mock profiles

    esdfiles_mock = np.array([['%s/%s/%s/shearcovariance/%s.txt'%\
        (path_sheardata, path_mocksel[i,j], path_mockcosmo[i,j], path_filename[i,j]) \
        for j in np.arange(np.shape(path_mocksel)[1])] for i in np.arange(np.shape(path_mocksel)[0]) ])

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

except:
    pass


## Randoms

path_randoms = np.array([ [''] ])

try:
    print('Import random signal:')

    path_randoms = np.reshape(path_randoms, [Nsize])
    random_esdfile = np.array(['/%s/%s/%s/%s'%(path_sheardata, path_random, path_cosmo, path_filename) for path_random in path_randoms])
    random_data_x, random_data_y, random_error_h, random_error_l = utils.read_esdfiles(random_esdfile)
    
    # Subtract random signal
    data_y = data_y-random_data_y
    error_h = np.sqrt(error_h**2. + random_error_h**2)
    error_l = np.sqrt(error_l**2. + random_error_l**2)

except:
    print()
    print('No randoms subtracted!')
    print()
    pass
    

# Create the plot

Ncolumns = int(Nbins[0]/Nrows)

# Plotting the ueber matrix
print(Nbins)
if Nbins[0] > 1:
    fig = plt.figure(figsize=(Ncolumns*3.,Nrows*3.))
else:
    fig = plt.figure(figsize=(6,5))

gs_full = gridspec.GridSpec(1,1)
gs = gridspec.GridSpecFromSubplotSpec(Nrows, Ncolumns, wspace=0, hspace=0, subplot_spec=gs_full[0,0])

ax = fig.add_subplot(gs_full[0,0])

print(Nrows)
print(Ncolumns)

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
            
            if Nbins[1] > 2:
                data_x_plot = data_x[Ndata] * (1.+0.3*Nplot)
            else:
                data_x_plot = data_x[Ndata]
            

            if Nsize==Nbins:
                ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                color=colors[Ndata], ls='', marker='.', zorder=4)
            else:
                ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                color=colors[Ndata], ls='', marker='.', label=datalabels[Nplot], zorder=4)
            
        try:
            # Plot mock shearprofiles
            for Nmock in range(Nmocks[1]):

                Ndata = Nplot + N*(Nmocks[1])
                
                if Ndata==0:
                    ax_sub.plot(data_x_plot, data_y_mock[Ndata], marker='', ls='-', \
                    color=colors[Ndata], label=mocklabels[0], alpha=valpha, zorder=1)
                else:
                    ax_sub.plot(data_x_plot, data_y_mock[Ndata], marker='', ls='-', \
                    color=colors[Ndata], alpha=valpha, zorder=1)
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
        
        # Define the labels for the plot
      
        xlabel = r'Baryonic radial acceleration $g_{\rm bar}$ [${\rm h_{%g} \, m/s^2}$]'%(h*100)
        ylabel = r'Total radial acceleration $g_{\rm tot}$ [${\rm h_{%g} \, m/s^2}$]'%(h*100)
        
        ax.xaxis.set_label_coords(0.5, -0.12)
        ax.yaxis.set_label_coords(-0.12, 0.5)
    
        # Plot guiding lines
        ax_sub.plot(gbar_mcgaugh, gobs_mcgaugh(gbar_mcgaugh),\
            color='black', ls='-', marker='', zorder=3)
        ax_sub.plot(gbar_ext, gobs_mcgaugh(gbar_ext), label = 'McGaugh+2016 (extrapolation)',\
            color='black', ls=':', marker='', zorder=2)
        #ax_sub.plot(gbar_uni, gbar_uni, label = 'Unity (no dark matter)', ls=':', marker='', zorder=1)

        # Plot Crescenzo's data
        #ax_sub.errorbar(gbar_cres, gtot_cres, yerr=[errorl_cres, errorh_cres], ls='', marker='.', label="Tortora+2017 (Early Type Galaxies)", zorder=4)

        # Plot Kyle's prediction
        #ax_sub.plot(gbar_kyle[0], gtot_kyle[0], ls='-', marker='', label="Navarro+2017 ($M_*=10^{9} M_\odot$)", zorder=5)
        #ax_sub.plot(gbar_kyle[1], gtot_kyle[1], ls='-', marker='', label="Navarro+2017 ($M_*=2.1*10^{10} M_\odot$)", zorder=6)
        #ax_sub.plot(gbar_kyle[2], gtot_kyle[2], ls='-', marker='', label="Navarro+2017 ($M_*=10^{11} M_\odot$)", zorder=6)
        
        plt.xscale('log')
        plt.yscale('log')
        #plt.gca().invert_xaxis()
        #plt.gca().invert_yaxis()
        
        if Nbins[0]>1:
            plt.title(datatitles[N], x = 0.73, y = 0.86, fontsize=16)

# Define the labels for the plot
ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)

handles, labels = ax_sub.get_legend_handles_labels()

# Plot the legend
#plt.legend()

if Nbins[0] > 1:
    lgd = ax_sub.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.5*Ncolumns, 0.7*Nrows)) # side
else:
    #plt.legend(handles[::-1], labels[::-1], loc='best')
    plt.legend()
#    lgd = ax_sub.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.85, 1.55)) # top



plt.tight_layout()

# Save plot
for ext in ['pdf']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf


