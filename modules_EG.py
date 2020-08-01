#!/usr/bin/python
"""
# This contains all the modules for the EG project.
"""
import astropy.io.fits as pyfits
import gc
import numpy as np
import sys
import os
import time
from glob import glob

from astropy import constants as const, units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

# Import constants
G = const.G.to('pc3/Msun s2')
c = const.c.to('pc/s')
inf = np.inf

# Import and mask all used data from the sources in this KiDS field
def import_kidscat(path_kidscat, kidscatname, h):
    
    # Full directory & name of the corresponding KiDS catalogue
    kidscatfile = '%s/%s'%(path_kidscat, kidscatname)
    kidscat = pyfits.open(kidscatfile, memmap=True)[1].data
    
    # List of the observables of all sources in the KiDS catalogue
    galID =  kidscat['IDENT']
    galRA = kidscat['RAJ2000']
    galDEC = kidscat['DECJ2000']
    galZ = kidscat['Z_ANNZ_KV']
    #galZ = kidscat['zANNz2ugri']
    
    rmag = kidscat['MAG_AUTO_CALIB']
    rmag_gaap = kidscat['MAG_GAAP_r']
    rmag_abs = kidscat['MAG_ABS_r']
    
    logmstar = kidscat['MASS_BEST']
    logmstar = logmstar + (rmag_gaap-rmag)/2.5 - 2.*np.log10(h/0.7)
    
    #gmag = kidscat['MAG_GAAP_g']
    #imag = kidscat['MAG_GAAP_i']
    #logML = -0.68 + 0.70*(gmag - imag)

    return galID, galRA, galDEC, galZ, rmag, rmag_abs, logmstar

def import_gamacat(path_gamacat, gamacatname, h):
    
    # Full directory & name of the corresponding KiDS catalogue
    gamacatfile = '%s/%s'%(path_gamacat, gamacatname)
    gamacat = pyfits.open(gamacatfile, memmap=True)[1].data
    
    # List of the observables of all lenses in the GAMA catalogue
    galID = gamacat['ID']
    galRA = gamacat['RA']
    galDEC = gamacat['DEC']
    galZ = gamacat['Z']
    
    rmag = gamacat['Rpetro']
    rmag_abs = gamacat['absmag_r']
    logmstar = gamacat['logmstar']
    
    # Fluxscale, needed for absolute magnitude and stellar mass correction
    fluxscale = gamacat['fluxscale']
    rmag_abs = rmag_abs - 2.5*np.log10(fluxscale) + 5*np.log10(h/0.7)
    logmstar = logmstar + np.log10(fluxscale) - 2.*np.log10(h/0.7)
    
    """
    nQ = gamacat['nQ']
    gamamask = (nQ>=3)
    
    galRA, galDEC, galZ, rmag, rmag_abs, logmstar = \
    galRA[gamamask], galDEC[gamamask], galZ[gamamask], rmag[gamamask], rmag_abs[gamamask], logmstar[gamamask]
    """
    
    return galID, galRA, galDEC, galZ, rmag, rmag_abs, logmstar


def import_micecat(path_micecat, micecatname, h):
    
    # Full directory & name of the corresponding KiDS catalogue
    micecatfile = '%s/%s'%(path_micecat, micecatname)
    micecat = pyfits.open(micecatfile, memmap=True)[1].data
    
    # List of the observables of all galaxies in the mice catalogue
    galID =  micecat['unique_gal_id']
    galRA = micecat['ra_gal']
    galDEC = micecat['dec_gal']
    
    galZ = micecat['z_cgal']
    galDc = micecat['cgal']*(h/0.7)*u.Mpc # Convert source distances from Mpc to pc/h
    
    rmag = micecat['sdss_r_true']
    rmag_abs = micecat['sdss_r_abs_mag']
    
    # For the lenses
    try:
        logmstar = micecat['lmstellar']
    except:
        logmstar = np.ones(len(galZ))
    
    # For the sources
    try:
        e1 = -micecat['gamma1']
        e2 = micecat['gamma2']
    except:
        e1 = np.zeros(len(galZ))
        e2 = np.zeros(len(galZ))
    
    return galID, galRA, galDEC, galZ, galDc, rmag, rmag_abs, e1, e2, logmstar


def import_lepharecat(path_lepharecat, lepharecatname, h):
    
    # Full directory & name of the corresponding KiDS catalogue
    lepharecatfile = '%s/%s'%(path_lepharecat, lepharecatname)
    lepharecat = pyfits.open(lepharecatfile, memmap=True)[1].data
    
    # List of the observables of all galaxies in the mice catalogue
    galID =  lepharecat['IDENT']
    galRA = lepharecat['ALPHA_J2000']
    galDEC = lepharecat['DELTA_J2000']
    
    galZ = lepharecat['SPECZ']
    
    rmag = lepharecat['MAG_AUTO']
    rmag_abs = lepharecat['MAG_ABS_r']
    logmstar = lepharecat['MASS_BEST']
    
    return galID, galRA, galDEC, galZ, rmag, rmag_abs, logmstar

# Import lens catalogue
def import_lenscat(cat, h, cosmo):

    path_lenscat = '/data/users/brouwer/LensCatalogues'
    
    if 'gama' in cat:
        fields = ['G9', 'G12', 'G15']

        lenscatname = 'GAMACatalogue_2.0.fits'
        lensID, lensRA, lensDEC, lensZ, rmag, rmag_abs, logmstar =\
        import_gamacat(path_lenscat, lenscatname, h)
    
    if 'kids' in cat:
        fields = ['K1000']
        #lenscatname = 'photozs.DR4_GAMAequ_ugri_beta_100ANNs_masses.fits'
        #lenscatname = 'photozs.DR4_trained-on-GAMAequ_ugri+KV_version0.9_struct.fits'
        lenscatname = 'photozs.DR4.1_bright_ugri+KV_struct.fits'
        lensID, lensRA, lensDEC, lensZ, rmag, rmag_abs, logmstar =\
        import_kidscat(path_lenscat, lenscatname, h)

    if 'lephare' in cat:
        fields = ['G9', 'G12', 'G15', 'G23', 'GS']
        lenscatname = 'KIDS_allcats.LPoutput.gamalike.fits'
        lensID, lensRA, lensDEC, lensZ, rmag, rmag_abs, logmstar =\
        import_lepharecat(path_lenscat, lenscatname, h)
        
    if 'mice' in cat:
        fields = ['M1']
        lenscatname = 'mice2_gama_catalog_1000deg2.fits'
        lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, e1, e2, logmstar =\
        import_micecat(path_lenscat, lenscatname, h)
    
    # Calculating galaxy distances (in Mpc)
    if 'mice' not in cat:
        lensDc = calc_Dc(lensZ, cosmo)
        #print(lensDc)
    
    return fields, path_lenscat, lenscatname, lensID, lensRA, lensDEC, lensZ, lensDc, rmag, rmag_abs, logmstar

# Import source catalogue
def import_srccat(path_srccat, srccatname):
    
    # Full directory & name of the corresponding KiDS catalogue
    srccatfile = '%s/%s'%(path_srccat, srccatname)
    srccat = pyfits.open(srccatfile, memmap=True)[2].data
    
    # List of the observables of all sources in the KiDS catalogue
    galRA = srccat['ALPHA_J2000']
    galDEC = srccat['DELTA_J2000']
    galZ = srccat['Z_B']
    rmag = srccat['MAG_GAAP_r_CALIB']
    
    e1 = srccat['e1_A']
    e2 = srccat['e2_A']
    weight = srccat['weight_A']
    
    return srcRA, srcDEC, srcZ, rmag, e1, e2, weight


# Define radial bins
def define_Rbins(Runit, Rmin, Rmax, Nbins, Rlog):

    if Rlog:
        Rbins = np.logspace(np.log10(Rmin), np.log10(Rmax), Nbins+1)
    else:
        Rbins = np.linspace(Rmin, Rmax, Nbins+1)
    Rcenters = Rbins[0:-1] + np.diff(Rbins)/2.

    if 'pc' in Runit:
        # Define the value of X in Xpc
        if 'M' in Runit:
            xvalue = 10.**6
        if 'k' in Runit:
            xvalue = 10.**3
            
        # Translate radial distances from Xpc to pc
        Rmin = Rmin*xvalue
        Rmax = Rmax*xvalue
    else:
        xvalue = 1.
    
    print('Translating Rbins from %s to pc: multiplying by %g'%(Runit, xvalue))

    # Translate radial distances from Xpc to arcmin
    #Rarcmin = np.degrees(Rmin/(lensDa/xvalue))*60.
    #Rarcmax = np.degrees(Rmax/(lensDa/xvalue))*60.
    
    return Rbins, Rcenters, Rmin, Rmax, xvalue


def define_lensmask(paramnames, maskvals, path_lenscat, lenscatnames, h):
    
    paramlist = []
    
    for p in range(len(paramnames)):
                
        # Import lens parameters
        lenscatfile = '%s/%s'%(path_lenscat, lenscatnames[p])
        lenscat = pyfits.open(lenscatfile, memmap=True)[1].data
        paramname = paramnames[p]
        paramvals = lenscat[paramname]

        if (paramname == 'logmstar') or (paramname == 'absmag_r'):
            # Fluxscale, needed for absolute magnitude and stellar mass correction
            fluxscale = lenscat['fluxscale']
            
            if paramname == 'logmstar':
                paramvals = paramvals + np.log10(fluxscale) - 2.*np.log10(h/0.7)
            
            elif paramname == 'absmag_r':
                paramvals = paramvals - 2.5*np.log10(fluxscale) + 5*np.log10(h/0.7)
            
        paramlist.append(paramvals)
        
    # Selecting the lenses
    masklist = np.ones(len(paramlist[0]))
    filename_var = ''
    
    # Applying the mask for each parameter
    for m in range(len(paramnames)):
        lensmask = (maskvals[m,0] <= paramlist[m]) & (paramlist[m] < maskvals[m,1])
        masklist[np.logical_not(lensmask)] = 0
        
        filename_var = '%s~%s~%g~%g'%(filename_var, paramnames[m], maskvals[m,0], maskvals[m,1])
        
    lensmask = (masklist == 1)
    
    # Define filename
    filename_var = filename_var.replace('.', 'p')
    filename_var = filename_var.replace('-', 'm')
    filename_var = filename_var.replace('~', '_')
    filename_var = filename_var.split('_', 1)[1]
    
    print
    print('Selection:', filename_var)
    print('Selected: %i of %i lenses (%g percent)'\
    %(np.sum(masklist), len(masklist), np.sum(masklist)/len(masklist)*100.))
    print
    
    return lensmask, filename_var


# Calculating the absolute magnitude  
def calc_absmag(rmag, galZ, gmag, imag, h, O_matter, O_lambda):
    
    # Calculating the distance modulus
    cosmo = LambdaCDM(H0=h*100, Om0=O_matter, Ode0=O_lambda)
    galDl = (cosmo.luminosity_distance(galZ).to('pc')).value
    DM = 5.*np.log10(galDl/10.)
    
    # Calculating the K-corrections
    kcorfile = np.loadtxt('kcorrection_list.txt').T
    zbins = kcorfile[5]
    kparams = kcorfile[6:9]

    # Calculating the K-correction per redshift bin
    galKcor = np.zeros(len(galZ))
    for k in range(len(zbins)):
        
        zmask = (zbins[k]-0.005 <= galZ) & (galZ < zbins[k]+0.005)
        
        a, b, c = kparams[:,k]
        Kcor = a*(gmag-imag)**2 + b*(gmag-imag) + c
        
        galKcor[zmask] = Kcor[zmask]
    
    # Calculating the absolute magnitude    
    rmag_abs = rmag - DM + galKcor
    
    return rmag_abs

# Compute distances for a list of redshifts
def calc_Dc(Z, cosmo):
    
    # Find and sort the unique redshift values
    Zbins = np.sort(np.unique(Z))
    
    # Calculate the corresponding distances
    Dcbins = cosmo.comoving_distance(Zbins)

    # Assign the appropriate distances to all lens redshifts
    Dc = Dcbins[np.digitize(Z, Zbins)-1]
    
    return Dc

# Define grid points for trough selection
def define_gridpoints(fieldRAs, fieldDECs, gridspace, gridtype):
    
    ## Creating a grid to measure the galaxy density
    
    # Creating DEC
    if 'random' in gridtype:
        gridDEC = np.arange(fieldDECs[0], fieldDECs[1], gridspace)
    else:
        gridDEC = np.arange(fieldDECs[0]+gridspace/2., fieldDECs[1], gridspace)
    
    # At these DECs, calculate the correction for the equatorial coordinate system
    eqcor = np.cos(np.radians(np.abs(gridDEC)))
    
    if 'corrected' in gridtype: # Calculate RA, with correction the grid for the equatorial coordinate system
        
        cormin, cormax = [1./np.amax(eqcor), 1./np.amin(eqcor)]
        print('Applying correction (min,max):', cormin, cormax)
        
        gridRA = [ np.arange(fieldRAs[0]+gridspace/(2.*eqcor[DEC]), fieldRAs[1], \
        gridspace/eqcor[DEC]) for DEC in range(len(gridDEC)) ] # Corrected for equatorial frame
        lenRA = len(gridRA[0])
    
        gridlist = [ [ [gridRA[DEC][RA], gridDEC[DEC]] for RA in range(len(gridRA[DEC]))] for DEC in range(len(gridDEC)) ]
        gridlist = np.vstack(gridlist)
        
        gridRAlist, gridDEClist = gridlist[:,0], gridlist[:,1]
    
    else: # Calculate RA, without correction the grid for the equatorial coordinate system
        
        print('Not correcting for equatorial coordinates!')
        
        gridRA = np.arange(fieldRAs[0]+gridspace/2., fieldRAs[1], gridspace)
        lenRA = len(gridRA)
        
        gridRAlist, gridDEClist = np.meshgrid(gridRA, gridDEC)
        gridRAlist = np.reshape(gridRAlist, np.size(gridRAlist))
        gridDEClist = np.reshape(gridDEClist, np.size(gridDEClist))
        #gridlist = [ [ [RA, DEC] for RA in gridRA ] for DEC in gridDEC ]

    gridcoords = SkyCoord(ra=gridRAlist*u.deg, dec=gridDEClist*u.deg) # All grid coordinates    
    
    """
    ## Masking grid points
    inmask = (fieldRAs[0] < gridRAlist) & (gridRAlist < fieldRAs[1])
    gridRAlist, gridDEClist = gridRAlist[inmask], gridDEClist[inmask]
    
    print( 'Number of grid coordinates:', len(gridRA), 'x', len(gridDEC), '=', len(gridRA)*len(gridDEC), '/', np.sum(inmask) )
    
    # Distances of grid points to the nearest source
    idx, d2dsrc, d3d = gridcoords.match_to_catalog_sky(srccoords)

    # Find grid points that are outside the field
    inmask = (d2dsrc > 1*u.deg) # Points that lie outside the source field
    incoords = gridcoords[inmask]
    
    if len(outcoords) > 0:
        # Remove points that lie close to the edge of the galaxy field
        idx, d2dout, d3d = gridcoords.match_to_catalog_sky(outcoords)
        gridmask = (d2dout > 0.5*u.deg)
        
        # Define new grid coordinates
        gridRA, gridDEC, gridcoords = gridRA[gridmask], gridDEC[gridmask], gridcoords[gridmask]
    """
    print('Gridcoords:', lenRA, 'x', len(gridDEC), '=', len(gridcoords))
    
    
    return gridRAlist, gridDEClist, gridcoords

# Write the results to a fits catalogue
def write_catalog(filename, outputnames, formats, output):
    
    fitscols = []
    
    # Adding the output
    [fitscols.append(pyfits.Column(name = outputnames[c], format = formats[c], array = output[c])) \
        for c in range(len(outputnames))]
    
    cols = pyfits.ColDefs(fitscols)
    tbhdu = pyfits.BinTableHDU.from_columns(cols)

    #   print
    if os.path.isfile(filename):
        os.remove(filename)
        print('Old catalog overwritten:', filename)
    else:
        print('New catalog written:', filename)
    print

    tbhdu.writeto(filename)


# Importing the ESD profiles
def read_esdfiles(esdfiles):
    
    data = np.loadtxt(esdfiles[0]).T
    data_x = data[0]

    data_x = []
    data_y = []
    error_h = []
    error_l = []
    R_src = []
    N_src = []
    
    print('Imported ESD profiles: %i'%len(esdfiles))
    print(esdfiles)
    
    for f in range(len(esdfiles)):
        # Load the text file containing the stacked profile
        data = np.loadtxt(esdfiles[f]).T
    
        bias = data[4]
        bias[bias==-999] = 1
        
        datax = data[0]
       
        datay = data[1]/bias
        datay[datay==-999] = np.nan
    
        errorh = (data[3])/bias # covariance error
        errorl = (data[3])/bias # covariance error
        errorh[errorh==-999] = np.nan
        errorl[errorl==-999] = np.nan
        
        Rsrc = np.ones(len(bias))
        Nsrc = data[8]
        
        data_x.append(datax)     
        data_y.append(datay) 
        error_h.append(errorh) 
        error_l.append(errorl)
        R_src.append(Rsrc)
        N_src.append(Nsrc)
    
    data_x, R_src, data_y, error_h, error_l, N_src = \
    np.array(data_x), np.array(R_src), np.array(data_y), np.array(error_h), np.array(error_l), np.array(N_src)
    
    return data_x, R_src, data_y, error_h, error_l, N_src


# Printing stacked ESD profile to a text file
def write_stack(filename, Rcenters, Runit, ESDt_tot, ESDx_tot, \
    error_tot, bias_tot, wk2_tot, w2k2_tot, Nsrc, variance, h):
    
    if ('pc' in Runit) or ('mps2' in Runit):
        filehead = '# Radius({0})	ESD_t(h{1:g}*M_sun/pc^2)' \
                   '   ESD_x(h{1:g}*M_sun/pc^2)' \
                   '    error(h{1:g}*M_sun/pc^2)^2	bias(1+K)' \
                   '    variance(e_s)     wk2     w2k2' \
                   '     Nsources'.format(Runit, h*100)
    else:
        filehead = '# Radius({0})    gamma_t    gamma_x    error' \
                   '    bias(1+K)    variance(e_s)    wk2    w2k2' \
                   '    Nsources'.format(Runit)
    
    index = np.where(np.logical_not((0.0 < error_tot) & (error_tot < inf)))
    ESDt_tot.setflags(write=True)
    ESDx_tot.setflags(write=True)
    error_tot.setflags(write=True)
    bias_tot.setflags(write=True)
    wk2_tot.setflags(write=True)
    w2k2_tot.setflags(write=True)
    Nsrc.setflags(write=True)
    
    ESDt_tot[index] = int(-999)
    ESDx_tot[index] = int(-999)
    error_tot[index] = int(-999)
    bias_tot[index] = int(-999)
    wk2_tot[index] = int(-999)
    w2k2_tot[index] = int(-999)
    Nsrc[index] = int(-999)

    data_out = np.vstack((Rcenters.T, ESDt_tot.T, ESDx_tot.T, error_tot.T, \
                          bias_tot.T, variance*np.ones(bias_tot.shape).T, \
                          wk2_tot.T, w2k2_tot.T, Nsrc.T)).T
    
    np.savetxt(filename, data_out, delimiter='\t', header=filehead)
    
    print('Written: ESD profile data:', filename)
    
    return

def write_plot(Rcenters, gamma_t, gamma_x, gamma_error, labels, filename_output, Runit, Rlog, plot, h):
    
    shape = np.shape(gamma_t)
    
    if len(shape) == 1:
        plt.errorbar(Rcenters, gamma_t, gamma_error, ls='', marker='.')
        try:
            plt.errorbar(Rcenters, gamma_x, gamma_error, ls='', marker='.')
        except:
            pass

        
    else:
        for i in np.arange(len(gamma_t)):
            plt.errorbar(Rcenters[i], gamma_t[i], gamma_error[i], label=labels[i], ls='', marker='.')
            try:
                plt.errorbar(Rcenters[i], gamma_x[i], gamma_error[i], ls='', marker='.')
            except:
                pass
        plt.legend(loc='best')        
    
    if 'mps2' in Runit:
        xlabel = r'Expected baryonic acceleration $g_{\rm bar}$ $(m/s^2)$'
        #ylabel = r'Observed acceleration $g_{\rm obs}$ $(m/s^2)$'
        
        ylabel = r'ESD $\langle\Delta\Sigma\rangle$ [h$_{%g}$ M$_{\odot}$/pc$^2$]'%(h*100)
    else:
        if 'pc' in Runit:
            xlabel = r'Radius $R$ (%s/h$_{%g}$)'%(Runit, h*100)
            ylabel = r'ESD $\langle\Delta\Sigma\rangle$ [h$_{%g}$ M$_{\odot}$/pc$^2$]'%(h*100)
        else:
            xlabel = r'Angular separation $\theta$ (arcmin)'
            ylabel = r'Shear $\gamma$'
    
    plt.axhline(y=0., ls=':', color='black')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if Rlog:
        plt.xscale('log')
        plt.yscale('log')
        
    #plt.axis([Rmin,Rmax,ymin,ymax])
    ymin, ymax = [np.amin(gamma_t)*0.4, np.amax(gamma_t)*2.]
    plt.ylim(ymin, ymax)
    plt.tight_layout()

    # Save plot
    for ext in ['pdf']:
        plotname = '%s.%s'%(filename_output, ext)
        plt.savefig(plotname, format=ext, bbox_inches='tight')
        
    print('Written: ESD profile plot:', plotname)

    if plot:
        plt.show()

    plt.clf


def calc_chi2(data, model, covariance, masked=[]):

    Nbins = len(data)
    Rbins = len(data[0])
    
    # Reshaping the data and the model
    data, model = [np.reshape(x, [Nbins*Rbins, 1]) for x in [data, model]]
    
    # Sorting the covariance [Rbin1, Obsbin1, Rbin2, Obsbin2] and turning it into a matrix
    ind = np.lexsort((covariance[3,:], covariance[1,:], covariance[2,:], covariance[0,:]))
    covariance = np.reshape(covariance[4][ind], [Nbins*Rbins, Nbins*Rbins])

    # Applying the mask to data, model and covariance matrix
    if len(masked) > 0:
        data, model = [np.delete(y, masked, 0) for y in [data, model]]
        covariance = np.delete( np.delete(covariance, masked, 0) , masked, 1)
    
    # Turning the data, model and covariance into matrices    
    data, model = np.matrix(data), np.matrix(model)
    covariance = np.matrix(covariance)
    
    # Calculating chi2 from the matrices
    chi2_cov = np.dot((model-data).T, np.linalg.inv(covariance))
    chi2_tot = np.dot(chi2_cov, (model-data))[0,0]
    
    return chi2_tot

# Calculating the mean of profiles with different binning in x
def mean_profile(data_x, data_y, binmin, binmax, Nbins, log):
    
    # Flattening the x- and y-data
    data_x = np.ndarray.flatten(data_x)
    data_y = np.ndarray.flatten(data_y)
    
    if (binmin==0) and (binmax==0):
        binmin, binmax = [np.amin(data_x), np.amax(data_x)] # Minimum and maximum bin values
    else:
        pass
    
    # Defining the bins in the x-axis
    if log:
        binedges = np.logspace(np.log10(binmin), np.log10(binmax), Nbins+1)
    else:
        binedges = np.linspace(binmin, binmax, Nbins+1)
    
    inds = np.digitize(data_x, binedges) # Indices indicated the bin of each values
    
    # Calculating the mean (x and y) and standard deviation (y) in each x-bin
    data_x_mean = np.array([np.median(data_x[inds==x]) for x in range(Nbins)])
    data_y_mean = np.array([np.median(data_y[inds==x]) for x in range(Nbins)])
    data_y_std = np.array([np.std(data_y[inds==x]) for x in range(Nbins)])
        
    return data_x_mean, data_y_mean, data_y_std

