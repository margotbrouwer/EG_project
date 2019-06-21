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
colors = ['#d7191c', '#0571b0', '#92c5de', '#fdae61']*2


## Define paths of ESD profiles

# Fiducial plotting parameters
Runit = 'Mpc'
datatitles = []
Nrows = 1

path_sheardata = '/data/users/brouwer/Lensing_results/EG_results_Mar19'

## Import the randoms (KiDS)

params1 = []
params2 = [r'Randoms', r'All galaxies']

Nrows = 1
"""
# Randoms only
path_lenssel = np.array([['No_bins_randoms/Z_0_0p5']])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']])
path_filename = np.array([['shearcovariance/No_bins_A']])
"""
# Randoms + All galaxies
path_lenssel = np.array([['No_bins_randoms/Z_0_0p5', 'No_bins/zANNz2ugri_0_0p5']])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p315-Ol_0p685-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*2])
path_filename = np.array([['shearcovariance/No_bins_A', 'shearcatalog/No_bins_A']])
#"""

datatitles = params1
datalabels = params2

plotfilename = '%s/Plots/ESD_KiDS_randoms'%path_sheardata

## Import measured ESD
esdfiles = np.array([['%s/%s/%s/%s.txt'%\
	(path_sheardata, path_lenssel[i,j], path_cosmo[i,j], path_filename[i,j]) \
	for j in np.arange(np.shape(path_lenssel)[1])] for i in np.arange(np.shape(path_lenssel)[0]) ])
print(esdfiles)

Nbins = np.shape(esdfiles)
Nsize = np.size(esdfiles)
esdfiles = np.reshape(esdfiles, [Nsize])

print('Plots, profiles:', Nbins)

# Importing the shearprofiles and lens IDs
data_x, data_y, error_h, error_l = utils.read_esdfiles(esdfiles)
#print('mean error ratio:', np.mean(error_h[0]/error_h[1]))
data_y, error_h, error_l = [data_y*data_x, error_h*data_x, error_l*data_x]

print('Ratio:', np.mean(abs(data_y[0]/data_y[1])))

zeroline = np.zeros(len(data_x[0]))
chisquared = np.sum((data_y[0] - zeroline)**2. / (error_h[0]))
dof = len(data_y[0])
prob = 1. - stats.chi2.cdf(chisquared, dof)

print('Chi2:', chisquared)
print('DoF:', dof)
print('P-value:', prob)

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
            if Nsize==Nbins:
                ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                color=colors[Nplot], ls='', marker='.', zorder=4)
            else:
                ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                color=colors[Nplot], ls='', marker='.', label=datalabels[Nplot], zorder=4)
     
        
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

        plt.xscale('log')
        #plt.yscale('log')

        ax_sub.errorbar(data_x_plot, zeroline, color='black', ls='--', marker='', zorder=1)

# Define the labels for the plot
xlabel = r'Radius R (${\rm %s} / {\rm h_{%g}}$)'%(Runit, h*100)
ylabel = r'Excess Surface Density $\Delta\Sigma$ (${\rm h_{%g} M_{\odot} / {\rm pc^2}}$)'%(h*100)
ax.set_xlabel(xlabel, fontsize=16)
ax.set_ylabel(ylabel, fontsize=16)

handles, labels = ax_sub.get_legend_handles_labels()


plt.xlim([0.03, 3])
#plt.ylim([2e-1, 2e2])

# Plot the legend
#plt.legend()

if Nbins[0] > 1:
    plt.legend(loc='best')
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
