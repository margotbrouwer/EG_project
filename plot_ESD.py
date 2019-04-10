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


## Define paths of ESD profiles

# Fiducial plotting parameters
Runit = 'Mpc'
datatitles = []
Nrows = 1

path_sheardata = '/data/users/brouwer/Lensing_results/EG_results_Mar19'

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

"""
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

#"""

datalabels = params1
datatitles = [r'$0.1<Z<0.2$',r'$0.2<Z<0.3$',r'$0.3<Z<0.4$',r'$0.4<Z<0.5$']

plotfilename = '%s/Plots/ESD_KiDS_GAMA_Zbins'%path_sheardata


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
print('mean error ratio:', np.mean(error_h[0]/error_h[1]))


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
                dx = 0.05
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
        plt.yscale('log')

# Define the labels for the plot
xlabel = r'Radius R (${\rm %s} / {\rm h_{%g}}$)'%(Runit, h*100)
ylabel = r'Excess Surface Density $\Delta\Sigma$ (${\rm h_{%g} M_{\odot} / {\rm pc^2}}$)'%(h*100)
ax.set_xlabel(xlabel, fontsize=16)
ax.set_ylabel(ylabel, fontsize=16)

handles, labels = ax_sub.get_legend_handles_labels()


# Zoomed in
#plt.xlim([1e-15, 1e-11])
#plt.ylim([0.5e-13, 1e-10])

# Plot the legend
#plt.legend()

if Nbins[0] > 1:
    plt.legend(loc='best')
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