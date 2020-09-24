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

# Make use of TeX
rc('text',usetex=True)

# Change all fonts to 'Computer Modern'
rc('font',**{'family':'serif','serif':['DejaVu Sans']})

# Colours
# Blue, green, turquoise, cyan
blues = ['#332288', '#44AA99', '#117733', '#88CCEE']

# Light red, Red, light pink, pink
reds = ['#CC6677', '#882255', '#CC99BB', '#AA4499']

blacks = ['black', '#0571b0']

# Dark blue, orange, red, light blue
colors = ['#0571b0', '#fdae61', '#d7191c', '#92c5de']*2


# Import constants
pi = np.pi
G = const.G.to('pc3 / (M_sun s2)').value
c = const.c.to('pc/s').value
h = 0.7
H0 = h * 100 * (u.km/u.s)/u.Mpc
H0 = H0.to('s-1').value
pc_to_m = 3.08567758e16

# Define functions
def vrot_mond(Mbar, r, g0=1.2e-10/pc_to_m):
    gbar = (G * 10.**Mbar) / r**2
    gobs = gbar / (1 - np.exp( -np.sqrt(gbar/g0) ))
    Vobs = np.sqrt(gobs * r) * pc_to_m
    return Vobs

def vrot_verlinde(Mbar, r):
    Mobs = 10.**Mbar + np.sqrt((c*H0)/(6.*G))*np.sqrt(10.**Mbar) * r
    Vobs = np.sqrt((G * Mobs) / r) * pc_to_m
    #Vobs = np.sqrt( (G*10.**Mbar)/r + np.sqrt((G*c*H0*10.**Mbar)/6.) ) * pc_to_m
    return Vobs

def bins_to_name(binlims):
    binname = str(binlims).replace(', ', '_')
    binname = str(binname).replace('[', '')
    binname = str(binname).replace(']', '')
    binname = str(binname).replace('.', 'p')
    return(binname)

## Define paths of ESD profiles

# Fiducial plotting parameters
Runit = 'Mpc'
datatitles = []
Nrows = 1
blind = 'C'
vrot = False
isolim = True
cat = 'kids'

# Define minimum and maximum radius
Rmin = 0.03 # Mpc
Rmax = 3. # Mpc

# Define path to the lensing results
path_sheardata = '/data/users/brouwer/Lensing_results/EG_results_Aug20'

"""


# Isolated vs. not isolated (KiDS)

param1 = ['KiDS']
param2 = [r'Isolated, bright (m$_{\rm r}<17.5$ mag)', \
        r'Isolated: $r_{\rm sat}>3$ Mpc/$h_{70}$, $M_*<10^{11}\,{\rm M_\odot}/h_{70}^2$', \
        r'All KiDS galaxies']
N1 = len(param1)
N2 = len(param2)
Nrows = 1
isolim = False

path_lenssel = np.array([['No_bins/MAGAUTOCALIB_0_17p5-dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0p1_0p5', \
                           'No_bins/dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0p1_0p5', 'No_bins/zANNZKV_0p1_0p5']])
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcovariance/No_bins_C']*N2]*N1)

datatitles = param1
datalabels = param2

plotfilename = '%s/Plots/ESD_KiDS_isotest'%path_sheardata

"""

# True vs. offset redshifts (MICE)

cat = 'mice'
param1 = ['MICE']
param2 = [r'Isolated, offset: $\sigma_{\rm z}/(1+z)=0.02$, $\sigma_{\rm M_*}=0.12$ dex', \
    r'Isolated: $r_{\rm sat}>3$ Mpc/$h_{70}$, $M_*<10^{11}\,{\rm M_\odot}/h_{70}^2$', 'All MICE mock galaxies']
N1 = len(param1)
N2 = len(param2)
Nrows = 1

path_lenssel = np.array([['No_bins/dist0p1percoffsetZM_3_inf-logmstaroffsetZM_0_11-zcgal_0p1_0p5', \
                        'No_bins/dist0p1perc_3_inf-logmstar_0_11-zcgal_0p1_0p5', 'No_bins/zcgal_0p1_0p5']])
path_cosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcovariance/No_bins_%s'%blind]*N2]*N1)

datatitles = param1
datalabels = param2

plotfilename = '%s/Plots/ESD_mice_isotest_offset'%path_sheardata

"""

# Isolation test (MICE)

cat = 'mice'
param1 = ['MICE']
param2 = [r'Isolated: $r_{\rm sat}(f_{\rm M_*}<0.1)>3$ Mpc/$h_{70}$', \
    r'Isolated: $r_{\rm sat}(f_{\rm M_*}<0.01)>3$ Mpc/$h_{70}$', 'All MICE mock galaxies']
N1 = len(param1)
N2 = len(param2)
Nrows = 1

path_lenssel = np.array([['No_bins/dist0p1perc_3_inf-logmstar_0_11-zcgal_0p1_0p5', \
                        'No_bins/dist0p01perc_3_inf-logmstar_0_11-zcgal_0p1_0p5', 'No_bins/zcgal_0p1_0p5']])
path_cosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcovariance/No_bins_%s'%blind]*N2]*N1)

datatitles = param1
datalabels = param2

plotfilename = '%s/Plots/ESD_mice_isotest_offset'%path_sheardata



# Rotation curve KiDS - 4 stellar mass bins (KiDS)

massbins = [8.5,10.3,10.6,10.8,11.]
binname = bins_to_name(massbins)

param1 = [r'$%g <$ log($M_*$) $< %g \, {\rm M_\odot}$'%(massbins[m], massbins[m+1]) for m in range(len(massbins)-1)]
param2 = [r'GL-KiDS isolated lens galaxies (SIS assumption)']
N1 = len(param1)
N2 = len(param2)
Nrows = 2

path_lenssel = np.array([['logmstar_GL_%s/dist0p1perc_3_inf-zANNZKV_0p1_0p5'%(binname)]*N2]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcovariance/shearcovariance_bin_%i_%s'%(p1,blind)]*N2 for p1 in np.arange(N1)+1])

path_mocksel =  np.array([['logmstar_%s/dist0p1perc_3_inf-zcgal_0p1_0p5'%(binname)]*N2]*N1)
path_mockcosmo = np.array([['zcgal_0p1_1p2-Om_0p25-Ol_0p75-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_mockfilename =  np.array([['shearcovariance/shearcovariance_bin_%i_%s'%(p1,blind)]*N2 for p1 in np.arange(N1)+1])
mocklabels = np.array(['GL-MICE mocks (isolated galaxies)'])

vrot = True
miceoffset = True

datalabels = param2

plotfilename = '%s/Plots/ESD_KiDS_massbins-%s_iso'%(path_sheardata, binname)



# Lensing rotation curve KiDS + GAMA

vrot = True

param1 = ['']
param2 = [r'Isolated GAMA lenses (180 deg$^2$)', 'Isolated KiDS lenses (1000 deg$^2$)']

N1 = 1
N2 = len(param2)
Nrows = 1

path_lenssel = np.array([['No_bins_gama/Z_0_0p5-dist0p1perc_3_inf-logmstarGL_0_11', 'No_bins/dist0p1perc_3_inf-logmstarGL_0_11-zANNZKV_0_0p5']]*N1)
path_cosmo = np.array([['ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc']*N2]*N1)
path_filename = np.array([['shearcatalog/No_bins_A']*N2]*N1)

path_mocksel =  np.array([['']*N2]*N1)
path_mockcosmo = np.array([['']*N2]*N1)
path_mockfilename = np.array([['shearcovariance/No_bins_A']*N2]*N1)
mocklabels = np.array(['GL-MICE mocks (isolated galaxies)'])

bahlabels = np.array(['BAHAMAS mocks (isolated galaxies)'])
Nmocks = [1, 1]

#masses_navarro = ['4.4E10'] # No mass bins (in Msun)

datalabels = param2
plotfilename = '%s/Plots/RAR_KiDS+GAMA+Verlinde_Nobins_isolated_zoomout'%path_sheardata

"""

# Define cosmological parameters
if 'mice' in cat:
    O_matter = 0.25
    O_lambda = 0.75
else:
    O_matter = 0.2793
    O_lambda = 0.7207
cosmo = LambdaCDM(H0=h*100., Om0=O_matter, Ode0=O_lambda)

## Import measured ESD

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

IDfiles = np.array([m.replace('%s.txt'%blind, 'lensIDs.txt') for m in esdfiles])
lensIDs_selected = np.array([np.loadtxt(m) for m in IDfiles])#+1 # Sometimes +1 is needed to line up the ID's
N_selected = [len(m) for m in lensIDs_selected]


# Calculate the difference between subsequent bins
for n in range(len(data_y)-1):
    chisquared = np.sum((data_y[n] - data_y[n+1])**2. / ((error_h[n]+error_h[n+1])/2.))
    dof = len(data_y[0])
    prob = 1. - stats.chi2.cdf(chisquared, dof)

    diff = (data_y[n+1] - data_y[n])/((data_y[n+1] + data_y[n])/2.)
    diff_error = (error_h[n+1] - error_h[n])/((error_h[n+1] + error_h[n])/2.)
    
    print('Difference:', np.mean(diff[8:-1]))
    print('Difference, error:', np.mean(diff_error))
    #print(param2[n], 'vs.', param2[n+1])
    print('Chi2:', chisquared)
    print('DoF:', dof)
    print('P-value:', prob)
print()


if 'mice' in cat:
    # Calculate MICE resolution limit
    lensDa = lensDc.to('Mpc').value/(1.+lensZ)
    
    IDmask = np.in1d(lensID, lensIDs_selected[1])
    Da_mean = np.mean(lensDa[IDmask*np.isfinite(lensDa)])
    
    pixelsize = 3. * 0.43 / 60. * pi/180. # arcmin to radian
    micelim = pixelsize * Da_mean
    print('MICE limit:', micelim, Runit)

if 'MICE' in plotfilename:

    print()
    print('Import mock signal:')

    # Defining the mock profiles
    esdfiles_mock = np.array([['%s/%s/%s/%s.txt'%\
        (path_sheardata, path_mocksel[i,j], path_mockcosmo[i,j], path_mockfilename[i,j]) \
        for j in np.arange(np.shape(path_lenssel)[1])] for i in np.arange(np.shape(path_lenssel)[0]) ])

    Nmocks = np.shape(esdfiles_mock)
    esdfiles_mock = np.reshape(esdfiles_mock, [Nsize])

    if Nmocks[1] > 5:
        valpha = 0.3
    else:
        valpha = 0.6

    # Importing the mock shearprofiles
    esdfiles_mock = np.reshape(esdfiles_mock, [Nsize])

    data_x_mock, R_src_mock, data_y_mock, error_h_mock, error_l_mock, N_src_mock = utils.read_esdfiles(esdfiles_mock)
    
    if vrot:
        data_y_mock = np.sqrt(4 * G * data_x_mock*1e6 * data_y_mock) * pc_to_m/1e3 # Convert ESD (Msun/pc^2) to acceleration (km/s^2)

    IDfiles_mock = np.array([m.replace('A.txt', 'lensIDs.txt') for m in esdfiles_mock])
    lensIDs_selected_mock = np.array([np.loadtxt(m) for m in IDfiles_mock])

    if miceoffset and ('KiDS' in plotfilename):
        esdfiles_mock_offset = [f.replace('perc', 'percoffsetZM') for f in esdfiles_mock]
        if 'No_bins' in esdfiles_mock[0]:
            esdfiles_mock_offset = [f.replace('logmstar', 'logmstaroffsetZM') for f in esdfiles_mock_offset]        
        else:        
            esdfiles_mock_offset = [f.replace('logmstar', 'logmstar_offsetZM') for f in esdfiles_mock_offset]
        foo, foo, data_y_mock_offset, foo, foo, foo = utils.read_esdfiles(esdfiles_mock_offset)
        
        if vrot:
            data_y_mock_offset = np.sqrt(4 * G * data_x_mock*1e6 * data_y_mock_offset) * pc_to_m/1e3 # Convert ESD (Msun/pc^2) to acceleration (km/s^2)

    # Import mock lens catalog
    fields, path_mockcat, mockcatname, lensID_mock, lensRA_mock, lensDEC_mock, \
        lensZ_mock, lensDc_mock, rmag_mock, rmag_abs_mock, logmstar_mock =\
        utils.import_lenscat('mice', h, cosmo)
    lensDc_mock = lensDc_mock.to('pc').value
    lensDa_mock = lensDc_mock/(1.+lensZ_mock)

    Rres = np.zeros(len(esdfiles_mock))
    for m in range(len(esdfiles_mock)):
        IDmask = np.in1d(lensID_mock, lensIDs_selected_mock[m])
        Da_max = np.amax(lensDa_mock[IDmask*np.isfinite(lensDa_mock)])
        #mstar_mean_mock = np.mean(10.**logmstar_mock[IDmask*np.isfinite(logmstar_mock)])
        pixelsize = 3. * 0.43 / 60. * pi/180. # arcmin to radian
        Rres[m] = pixelsize * Da_max / 1e6 # Minimum R due to MICE resolution (in Mpc)
    print('MICE resolution Rres=', Rres)

else:
    print('No mock signal imported!')
    pass


## Find the mean galaxy masses
#cat = 'kids'

# Import the Lens catalogue
fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar =\
utils.import_lenscat(cat, h, cosmo)

IDfiles = np.array([m.replace('%s.txt'%blind, 'lensIDs.txt') for m in esdfiles])
lensIDs_selected = np.array([np.loadtxt(m) for m in IDfiles])#+1 # Sometimes +1 is needed to line up the ID's
N_selected = [len(m) for m in lensIDs_selected]

# Import the mass catalogue
path_masscat = '/data/users/brouwer/LensCatalogues/baryonic_mass_catalog_%s.fits'%cat
masscat = pyfits.open(path_masscat, memmap=True)[1].data

print(path_masscat)

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

if vrot:
    ## Convert the ESD into the rotation velocity
    data_y_vrot = np.sqrt(4 * G * data_x*1e6 * data_y) * pc_to_m/1e3
    
    error_l, error_h = [ data_y_vrot * 0.5 * (d / data_y) for d in [error_l, error_h] ]
    data_y = data_y_vrot
    
    
    ## Import Lelli's rotation curve data
    data_lelli = np.loadtxt('Lelli_rotation_curves.txt', usecols=[2,3]).T
    name_data = np.array(list(np.genfromtxt('Lelli_rotation_curves.txt', dtype=None, usecols=[0])))

    R_lelli = data_lelli[0] / 1e3 # R in Mpc.
    Vobs_lelli = data_lelli[1] # Vobs in km/s
    
    info_lelli = np.loadtxt('Lelli_galaxy_info.txt', usecols=[7]).T
    name_info = np.array(list(np.genfromtxt('Lelli_galaxy_info.txt', dtype=None, usecols=[0])))
    
    Mstar_lelli = info_lelli * 1e9 * 0.5
    
    # Define minimum and maximum radius
    Rmin = 3.e-3 # Mpc
    Rmax = 3. # Mpc
    Rcenters = np.logspace(np.log10(Rmin), np.log10(Rmax), 100)
    
    
    ## Import Kyle's rotation curves
    filenames_kyle = ['RAR_profiles/gobs_isolated_massbin_%i.txt'%i for i in range(N1)]
    data_kyle = np.array([np.loadtxt(f).T for f in filenames_kyle])
    
    Rcenters_kyle, gobs_kyle, gmin_kyle, gmax_kyle = [data_kyle[:,d] for d in range(4)]
    Vrot_kyle, Vmin_kyle, Vmax_kyle = [np.sqrt(g_kyle * Rcenters_kyle*pc_to_m) \
        for g_kyle in [gobs_kyle, gmin_kyle, gmax_kyle] ] # in km/s
    
    # Calculate the difference between the two conversion methods
    print('Difference Vrot (SIS/PPL): %g dex'%np.log10(np.mean(data_y[n]/Vrot_kyle[n])))
    print('Difference in gobs (SIS/PPL): %g dex'%np.log10(np.mean(data_y[n]**2./Vrot_kyle[n]**2.)))
    print()
    
if 'NFW' in plotfilename:
    ## NWF rotation curves
    
    C200 = 12.
    M200 = 1e12 # Msun
    
    R = np.logspace(1, 7, 40) # in pc
    Rs = (1 / C200) * ((G * M200) / (100 * H0**2.))**(1./3.)

    x = R / Rs
    
    print('Rs', Rs)
    print('x', x)
    
    M_NFW = ( M200 / (np.log(1+C200) - C200 / (1+C200) )) * ( np.log(1+x) - x/(1+x) )
    Vrot_NFW = np.sqrt( (G * M_NFW) / R ) * pc_to_m / 1e3
    
    #V200 = np.exp( (1./3.)*(np.log(M200) + np.log(10.*G*H0)) )
    
    #Vrot_NFW = V200 * np.sqrt( (C200 / x) * \
    #    (( np.log(1+x) - x/(1+x) ) / (np.log(1+C200) - C200 / (1+C200) )) ) * pc_to_m
    
    print(Vrot_NFW)   

    
# Plot titles
if 'massbins' in plotfilename:
    datatitles = [r'$\log\langle M_{\rm gal}/h_{%g}^{-2} {\rm M_\odot} \rangle = %.4g$'%(h*100, mean_mbar[p1]) for p1 in range(N1)]

# Reliability limit of isolated KiDS-1000 galaxies
if 'iso' in plotfilename:
    isoR = 0.3 # in Mpc
    print('Isolated KiDS signal reliability limit:', isoR)


## Create the plot

Ncolumns = int(Nbins[0]/Nrows)

# Plotting the ueber matrix
if Nbins[0] > 1:
    fig = plt.figure(figsize=(Ncolumns*6.,Nrows*4))
else:
    fig = plt.figure(figsize=(7.,5.))

gs_full = gridspec.GridSpec(1,1)
gs = gridspec.GridSpecFromSubplotSpec(Nrows, Ncolumns, wspace=0, hspace=0, subplot_spec=gs_full[0,0])

ax = fig.add_subplot(gs_full[0,0])

if len(param2) > 2:
    plotcolors = colors
else:
    plotcolors = blacks

for NR in range(Nrows):
    for NC in range(Ncolumns):
    
        ax_sub = fig.add_subplot(gs[NR, NC])

        N = np.int(NR*Ncolumns + NC)
        print('Plotting: row %i, column %i (panel %i)'%(NR, NC, N))

        for Nplot in range(Nbins[1]):
            
            Ndata = Nplot + N*(Nbins[1])
            print('Ndata =', Ndata)
            print('Nplot =', Nplot)
            print()
            
            dx = 0.15
            if (Nbins[1] > 1) and ('mice' not in cat):
                data_x_plot = data_x[Ndata] * (1.-dx/2.+dx*Nplot)
            else:
                data_x_plot = data_x[Ndata]

            # Plot data
            if 'mice' not in cat:
                if Nsize==Nbins:
                    # Margot's rotation curves
                    ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                    color=plotcolors[Nplot], ls='', marker='.', zorder=8)

                    # Kyle's rotation curves
                    ax_sub.errorbar(Rcenters_kyle[N] * (1.-dx/2.+dx), Vrot_kyle[N], \
                        yerr=[Vrot_kyle[N]-Vmin_kyle[N], Vmax_kyle[N]-Vrot_kyle[N]], \
                        color=plotcolors[1], ls='', marker='.', zorder=8)

                else:
                    # Margot's rotation curves
                    ax_sub.errorbar(data_x_plot, data_y[Ndata], yerr=[error_l[Ndata], error_h[Ndata]], \
                    color=plotcolors[Nplot], ls='', marker='.', label=datalabels[Nplot], zorder=8)
            
                    # Kyle's rotation curves
                    ax_sub.errorbar(Rcenters_kyle[N] * (1.-dx/2.+dx), Vrot_kyle[N], yerr=[Vrot_kyle[N]-Vmin_kyle[N], Vmax_kyle[N]-Vrot_kyle[N]], \
                        color=plotcolors[1], ls='', marker='.', label=r'GL-KiDS isolated lens galaxies (PPL method)', zorder=8)
                    
            else:
                if Nsize==Nbins:
                    ax_sub.plot(data_x_plot, data_y[Ndata], \
                    color=plotcolors[Nplot], ls='-', marker='.', zorder=8)
                else:
                    ax_sub.plot(data_x_plot, data_y[Ndata], \
                    color=plotcolors[Nplot], ls='-', marker='.', label=datalabels[Nplot], zorder=8)
        
            if 'MICE' in plotfilename:
                
                Rmin_mask = data_x_mock[Ndata]>Rmin[Ndata]
                
                if miceoffset and ('KiDS' in plotfilename) and ('iso' in plotfilename):
                    ax_sub.fill_between((data_x_mock[Ndata])[Rmin_mask], (data_y_mock[Ndata])[Rmin_mask], \
                        (data_y_mock_offset[Ndata])[Rmin_mask], color=plotcolors[2], label=mocklabels[Nplot], alpha=valpha, zorder=6)
                else:
                    ax_sub.plot((data_x_mock[Ndata])[Rmin_mask], (data_y_mock[Ndata])[Rmin_mask], \
                        color=plotcolors[2], ls='-', marker='.', label=mocklabels[Nplot], zorder=6)
                
                #ax_sub.axvline(x = Rmin[Ndata], color=colors[2], ls=':', label='MICE resolution limit')
                
        if vrot:
            
            # Plot Lelli rotation curve data
            name_mask = (10.**massbins[Ndata] < Mstar_lelli) & (Mstar_lelli < 10.**massbins[Ndata+1])
            info_mask = np.in1d(name_data, name_info[name_mask])
                        
            data_lelli_x_mean, data_lelli_y_mean, data_lelli_y_std = \
                utils.mean_profile(R_lelli[info_mask], Vobs_lelli[info_mask], 3e-3, 6e-2, 11, True)
            
            ax_sub.plot(data_lelli_x_mean, data_lelli_y_mean, \
                ls='', marker='s', markerfacecolor='red', markeredgecolor='black', label='SPARC rotation curves (mean)')
            #ax_sub.fill_between(data_lelli_x_mean, data_lelli_y_mean+0.5*data_lelli_y_std,
            #    data_lelli_y_mean-0.5*data_lelli_y_std, alpha=0.3, color=blues[3])
            #ax_sub.scatter(R_lelli[info_mask], Vobs_lelli[info_mask], alpha=0.05, color=blues[1])
            ax_sub.hist2d(R_lelli[info_mask], Vobs_lelli[info_mask], \
                bins=[np.logspace(np.log10(3e-3), np.log10(3), 40), np.linspace(0., 330., 40)], cmin=1, cmap='Blues', zorder=0)
            
            # Kyle's rotation curves
            #ax_sub.plot(Rcenters_kyle[N], Vrot_kyle[N], color=colors[1])
            #ax_sub.fill_between(Rcenters_kyle[N], Vmin_kyle[N], \
            #            Vmax_kyle[N], color=colors[1], alpha=0.3)
            
            """
            # NFW profile
            #ax_sub.plot(R/1e6, Vrot_NFW, label=r'NFW profile ($M_{200}=%g, R_s = %.3g$ Mpc/h'%(M200, Rs/1e6))
            
            # McGaugh rotation curve
            #Vobs_mond = vrot_mond(mean_mstar[N], data_x[N]*1e6)
            ax_sub.plot(data_x[N], vrot_mond(mean_mstar[N], data_x[N]*1e6)/1e3, \
                color='grey', ls='-', marker='', label=r'McGaugh+16 fitting function (extrapolated)')
            
            # Verlinde rotation curve
            Vobs_verlinde = vrot_verlinde(mean_mstar[N], data_x[N]*1e6)
            ax_sub.plot(data_x[N], Vobs_verlinde/1e3, \
                ls = '--', marker='', color=colors[2], label = r'Verlinde+16 Emergent Gravity (point mass)')
            """
        
        
        if 'mice' in plotfilename:
            ax_sub.axvline(x=micelim, ls='--', color='grey', label=r'MICE resolution limit ($3\times0.43$ arcmin)')
        
        # Plot the axes and title
        
        ax_sub.xaxis.set_label_position('top')
        ax_sub.yaxis.set_label_position('right')

        ax.tick_params(labelleft='off', labelbottom='off', top='off', bottom='off', left='off', right='off')

        if (NR+1) == Nrows:
            ax_sub.tick_params(labelsize='14')
        else:
            ax_sub.tick_params(axis='x', labelbottom='off')

        if NC == 0:
            ax_sub.tick_params(labelsize='14')
        else:
            ax_sub.tick_params(axis='y', labelleft='off')

        
        #plt.autoscale(enable=False, axis='both', tight=None)
        
        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.yaxis.set_label_coords(-0.1/Ncolumns, 0.5)
        
        if Nbins[0]>1:
            plt.title(datatitles[N], x = 0.5, y = 0.87, fontsize=16)
        
        if vrot:
            plt.xlim([Rmin, Rmax])
            plt.ylim([0., 330.])
        else:
            plt.xlim([Rmin, Rmax])
            plt.ylim([2e-1, 2e2])
            plt.yscale('log')
        
        plt.xscale('log')
        
        # Extras for KiDS data (isolation limit)
        if isolim:
            # Plot KiDS-1000 isolation limit
            ax_sub.axvspan(isoR, Rmax, color=blacks[1], alpha=0.1, \
                label=r'KiDS isolation criterion limit ($R > %g \, {\rm Mpc}/h_{70}$)'%isoR)
        
# Define the labels for the plot
xlabel = r'Radius R (${\rm %s} / h_{%g}$)'%(Runit, h*100)
ylabel = r'Excess Surface Density $\Delta\Sigma$ ($h_{%g} {\rm M_{\odot} / {\rm pc^2}}$)'%(h*100)
if vrot:
    ylabel = r'Rotational velocity [$h_{%g} \, {\rm km/s}$])'%(h*100)
    
ax.set_xlabel(xlabel, fontsize=16)
ax.set_ylabel(ylabel, fontsize=16)

#handles, labels = ax_sub.get_legend_handles_labels()

# Plot the legend

handles, labels = ax_sub.get_legend_handles_labels()

if Nbins[0] > 1:
    lgd = plt.legend(handles[::-1], labels[::-1], loc='lower left', fontsize=12)
else:
    plt.legend(handles[::-1], labels[::-1], loc='best', fontsize=12)#loc='lower right')
#    plt.legend(handles[::-1], labels[::-1], loc='best')
#    lgd = ax_sub.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.85, 1.55)) # top

plt.tight_layout()

# Save plot
for ext in ['pdf']:
    plotname = '%s.%s'%(plotfilename, ext)
    plt.savefig(plotname, format=ext, bbox_inches='tight')
    
print('Written: ESD profile plot:', plotname)

plt.show()
plt.clf
