import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import multiprocessing as mp
import os
from astropy.table import Table

###########################################################
# directories

Dir = '/home/heh15/research/FIRE/'
picDir = Dir+'pictures/'
logDir = Dir+'logs/'

###########################################################
# basic information

incls = ['v0', 'v1', 'v2', 'v3']


###########################################################
# functions

def count_time(stop, start):
    '''
    Convert the time difference into human readable form.
    '''
    dure=stop-start
    m,s=divmod(dure,60)
    h,m=divmod(m,60)
    print("%d:%02d:%02d" %(h, m, s))

    return

def make_mom0(dataCubes, pixsize=100, value=1e10):
    '''
    ------
    Parameters
    dataCubes: 3d numpy array
        3D numpy array to make moment maps
    pixsize: float
        Size of each pixel in pc
    value: float
        pixel unit in solar mass
    '''
    mom0 = np.nansum(dataCubes, axis=2) * value / pixsize**2

    return mom0

def make_mom1(dataCubes, vel_1d):
    '''
    ------
    Parameters
    dataCubes: 3d numpy array
        3D numpy array to make moment maps
    vel_1d: 1d numpy array
        1D array of velocity values corresponding to the 3rd
        axis of the dataCubes.
    '''
    vel_3d = np.full(np.shape(dataCubes),fill_value=np.nan)
    vel_3d[:] = vel_1d
    mom1 = np.nansum(vel_3d*dataCubes,axis=2) / np.nansum(dataCubes,axis=2)

    return mom1

def make_mom2(dataCubes, vel_1d):
    '''
    ------
    Parameters
    dataCubes: 3d numpy array
        3D numpy array to make moment maps
    vel_1d: 1d numpy array
        1D array of velocity values corresponding to the 3rd
        axis of the dataCubes.
    '''
    vel_3d = np.full(np.shape(dataCubes),fill_value=np.nan)
    vel_3d[:] = vel_1d
    mom1 = np.nansum(vel_3d*dataCubes,axis=2) / np.nansum(dataCubes,axis=2)

    mom1_3d = np.repeat(mom1[:,:,np.newaxis], len(vel_1d), axis=2)
    vel_diff = vel_3d - mom1_3d
    mom2 = np.sqrt(np.nansum(vel_diff**2*dataCubes,axis=2) / np.nansum(dataCubes,axis=2))

    return mom2

def make_Tpeak(dataCubes, pixsize=100, value=1e10, alphaCO=4.3, ratio=0.7, deltaV=2):
    '''
    ------
    Parameters
    dataCubes: 3d numpy array
        3D numpy array to make moment maps
    pixsize: float
        Size of each pixel in pc
    value: float
        pixel unit in solarmass
    alphaCO: float
        CO-H2 conversion factor
    ratio: float
        CO 2-1/1-0 ratio
    deltaV : float
        Velocity resolution per channel
    '''
    mom8 = np.nanmax(dataCubes, axis=2)*value/pixsize**2
    Tpeak = mom8 / alphaCO / deltaV * ratio
    Tpeak[np.where(np.isnan(Tpeak))] = 0

    return Tpeak

def round_sig(x, sig=2):
    '''
    round the x to certain significant figures
    '''
    return round(x, sig-int(floor(log10(abs(x))))-1)


def density_contour(
        x, y, weights=None, xlim=None, ylim=None,
        overscan=(0.1, 0.1), logbin=(0.02, 0.02), smooth_nbin=(3, 3),
        levels=(0.393, 0.865, 0.989), alphas=(0.75, 0.50, 0.25),
        color='k', contour_type='contourf', ax=None, **contourkw):
    """
    Generate data density contours (in log-log space).
    Parameters
    ----------
    x, y : array_like
        x & y coordinates of the data points
    weights : array_like, optional
        Statistical weight on each data point.
        If None (default), uniform weight is applied.
        If not None, this should be an array of weights,
        with its shape matching `x` and `y`.
    xlim, ylim : array_like, optional
        Range to calculate and generate contour.
        Default is to use a range wider than the data range
        by a factor of F on both sides, where F is specified by
        the keyword 'overscan'.
    overscan : array_like (length=2), optional
        Factor by which 'xlim' and 'ylim' are wider than
        the data range on both sides. Default is 0.1 dex wider,
        meaning that xlim = (Min(x) / 10**0.1, Max(x) * 10**0.1),
        and the same case for ylim.
    logbin : array_like (length=2), optional
        Bin widths (in dex) used for generating the 2D histogram.
        Usually the default value (0.02 dex) is enough, but it
        might need to be higher for complex distribution shape.
    smooth_nbin : array_like (length=2), optional
        Number of bins to smooth over along x & y direction.
        To be passed to `~scipy.ndimage.gaussian_filter`
    levels : array_like, optional
        Contour levels to be plotted, specified as levels in CDF.
        By default levels=(0.393, 0.865, 0.989), which corresponds
        to the integral of a 2D normal distribution within 1-sigma,
        2-sigma, and 3-sigma range (i.e., Mahalanobis distance).
        Note that for an N-level contour plot, 'levels' must have
        length=N+1, and its leading element must be 0.
    alphas : array_like, optional
        Transparancy of the contours. Default: (0.75, 0.50, 0.25)
    color : mpl color, optional
        Base color of the contours. Default: 'k'
    contour_type : {'contour', 'contourf'}, optional
        Contour drawing function to call
    ax : `~matplotlib.axes.Axes` object, optional
        The Axes object to plot contours in.
    **contourkw
        Keywords to be passed to the contour drawing function
        (see keyword "contour_type")
    Returns
    -------
    ax : `~matplotlib.axes.Axes` object
        The Axes object in which contours are plotted.
    """
    
    if xlim is None:
        xlim = (10**(np.nanmin(np.log10(x))-overscan[0]),
                10**(np.nanmax(np.log10(x))+overscan[0]))
    if ylim is None:
        ylim = (10**(np.nanmin(np.log10(y))-overscan[1]),
                10**(np.nanmax(np.log10(y))+overscan[1]))

    if ax is None:
        ax = plt.subplot(111)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # force to change to log-log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    # generate 2D histogram
    lxedges = np.arange(
        np.log10(xlim)[0], np.log10(xlim)[1]+logbin[0], logbin[0])
    lyedges = np.arange(
        np.log10(ylim)[0], np.log10(ylim)[1]+logbin[1], logbin[1])
    if weights is None:
        hist, lxedges, lyedges = np.histogram2d(
            np.log10(x), np.log10(y),
            bins=[lxedges, lyedges])
    else:
        hist, lxedges, lyedges = np.histogram2d(
            np.log10(x), np.log10(y), weights=weights,
            bins=[lxedges, lyedges])
    xmids = 10**(lxedges[:-1] + 0.5*logbin[0])
    ymids = 10**(lyedges[:-1] + 0.5*logbin[1])
    
    # smooth 2D histogram
    pdf = gaussian_filter(hist, smooth_nbin).T
    
    # calculate cumulative density distribution (CDF)
    cdf = np.zeros_like(pdf).ravel()
    for i, density in enumerate(pdf.ravel()):
        cdf[i] = pdf[pdf >= density].sum()
    cdf = (cdf/cdf.max()).reshape(pdf.shape)

    # plot contourf
    if contour_type == 'contour':
        contourfunc = ax.contour
        contourlevels = levels
    elif contour_type == 'contourf':
        contourfunc = ax.contourf
        contourlevels = np.hstack([[0], levels])
    else:
        raise ValueError(
            "'contour_type' should be either 'contour' or 'contourf'")
    contourfunc(
        xmids, ymids, cdf, contourlevels,
        colors=[mpl.colors.to_rgba(color, a) for a in alphas],
        **contourkw)
    
    return ax

###########################################################
# main program

### import literature data
# import the literature data from Sun+2020
filename= logDir+'Sun_2020_GMC.txt'
skiprows=28
table=pd.read_csv(filename,header=None,sep=r"\s+",skiprows=skiprows,engine='python')
table_90pc = table.loc[(table[1]==90)]
table_center = table.loc[(table[3]==1)]
table_offcenter = table.loc[(table[3]==0)]
# read the galaxy type
filename = logDir+'Sun_2020_galaxies.txt'
skiprows = 47
table_galaxy = pd.read_fwf(filename, colspecs='infer', header=None, skiprows=skiprows,engine='python')
galaxy_barred = np.array(table_galaxy[0].loc[(table_galaxy[2]=='Y')])
galaxy_unbarred = np.array(table_galaxy[0].loc[(table_galaxy[2]=='N')])
# classify center into bared and unbarred. 
table_center_barred = table_center.loc[table_center[0].isin(galaxy_barred)]
table_center_unbarred = table_center.loc[table_center[0].isin(galaxy_unbarred)]

# import the literature data from Sun+2018 for Antennae, M31 and M33.
filename = logDir+'Sun_2018_GMCs.txt'
table2 = pd.read_csv(filename,header=None,sep=r"\s+",skiprows=skiprows,engine='python')
table_120pc = table2.loc[(table2[1]==120)]
table_M31 = table_120pc.loc[(table_120pc[0]=='M31')]
table_M33 = table_120pc.loc[(table_120pc[0]=='M33')]
table_M3 = pd.concat([table_M31,table_M33])

table_main=table_120pc.loc[(table_120pc[0]!='Antennae')]
table_main = table_main.loc[(table_main[0]!='M33')]
table_main = table_main.loc[(table_main[0]!='M31')]

# Import pixel analyses on Antennae and NGC 3256
filename = logDir+'Antennae_Brunetti_pix.ecsv'
table_antennae = Table.read(filename)
table_ant = table_antennae.to_pandas()
table_ant_90pc = table_ant.loc[table_ant['beam_fwhm']==90]
# For NGC 3256
filename = logDir+'NGC3256_Brunetti_pix.ecsv'
table_3256 = (Table.read(filename)).to_pandas()
table_3256_80pc = table_3256.loc[table_3256['beam_fwhm']==80]

### Draw the contour plot
orbit = 'e1'
incls = ['v0', 'v1', 'v2', 'v3']
incls = ['v0']
# ids = np.arange(520, 712, 2)
ids = np.arange(250, 412, 2)

levels = [0.2, 0.5, 0.8]

def draw_contour(orbit, incl, idNo):
    '''
    Main body function to draw the contour
    '''
    # read the fits file
    fitsfile = Dir+orbit+'/losvd_FIRE2_'+orbit+'_'+str(idNo)+'_gas_'+str(incl)+'__32.fits'
    data = fits.open(fitsfile)[0].data
    hdr = fits.open(fitsfile)[0].header
    data[np.where(data==0)] = np.nan
    
    # make moment maps
    mom0 = make_mom0(data)
    vel = np.arange(-500,500,2)
    mom1 = make_mom1(data, vel)
    mom2 = make_mom2(data, vel)
    Tpeak = make_Tpeak(data)
    
    counts = np.count_nonzero(~np.isnan(data),axis=2)
    mom0[np.where(mom0<1)] = np.nan
    mom2[np.where(counts<2)] = np.nan  
    mask = np.ma.mask_or(np.isnan(mom0.flatten()), np.isnan(mom2.flatten()))

    # flag the nan values
    m1 = mom0.flatten()[np.where(~mask)]
    m2 = mom2.flatten()[np.where(~mask)]

    # make the scatter plot
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('$\Sigma_{\mathrm{mol}}$ (M$_{\odot}$ pc$^{-2}$)',fontsize=15)
    plt.ylabel(r'$\sigma_v$ (km$^2$ s$^{-2}$)',fontsize=15)

    # Draw the scatter and contour plot for simulation 
#     plt.scatter(mom0.flatten(), mom2.flatten(), alpha=0.05, color='gray')
    density_contour(m1, m2, weights=m1, levels=levels, ax=ax, color='black', 
                    contour_type='contour',alphas=(1,1,1))
    plt.ylim(bottom=0.1)
    plt.ylim(top=250)

    # Draw the contour plot for PHANGS data
    density_contour(table_90pc[7], table_90pc[8], weights=table_90pc[7], levels=levels, 
                    ax=ax, color='tab:blue')
    # Draw the virial parameter
    ax.set_xlim(); ax.set_ylim()
    Sigmol_theory = 10**np.linspace(-1,5,5)
    vdep_theory = np.sqrt(Sigmol_theory * 3.1 / 5.77 * (45/40))
    plt.plot(Sigmol_theory, vdep_theory, linestyle='--',linewidth=2,color='red')

    # Draw the contour plot for Antennae
    density_contour(table_ant_90pc['surface_density'], table_ant_90pc['velocity_dispersion'],
                    weights=table_ant_90pc['surface_density'], type='contour', levels=levels, ax=ax, 
                    color='tomato')
#    density_contour(table_ant_90pc['surface_density']/4, table_ant_90pc['velocity_dispersion'],
#                     weights=table_ant_90pc['surface_density']/4, contour_type='contour',
#                     levels=levels, ax=ax, color='orangered', linestyles='solid', alphas=(1,1,1))
    # Draw the contour plot for NGC3256
    density_contour(table_3256_80pc['surface_density'], table_3256_80pc['velocity_dispersion'],
                    weights=table_3256_80pc['surface_density'], contour_type='contour', 
                    levels=levels, ax=ax, color='blue', alphas=(1,1,1))
    # Draw the contour plot for M31 and M33
    density_contour(table_M31[3], table_M31[4], weights=table_M31[3], levels=levels, 
                    ax=ax, contour_type='contour', alphas=(1,1,1), color='green') 

    # label the legend
    line_simul = mlines.Line2D([],[],color='black', linestyle='solid', label='FIRE G2&G3')
    patch = mpatches.Patch(color='tab:blue', label='PHANGS galaxy')
    patch2 = mpatches.Patch(color='tomato', label=r'Antennae, $\alpha_{\mathrm{CO}}=4.3$')
#    line=mlines.Line2D([], [], color='tomato', linestyle='solid', label=r'Antennae, $\alpha_{\mathrm{CO}}=4.3$')
    line2=mlines.Line2D([], [], color='blue', linestyle='solid', label=r'NGC 3256, $\alpha_{\mathrm{CO}}=1.1$')
    line3=mlines.Line2D([], [], color='green', linestyle='solid', label='M31')
    legend = plt.legend(handles=[line_simul],loc='upper left')
    legend1 = plt.legend(handles=[patch, patch2, line2, line3], loc='lower right')
    plt.gca().add_artist(legend)
    
    # save the picture. 
    plt.savefig(picDir+'Sigmol_vdep_G2G3_'+orbit+'_'+str(idNo)+'_'+str(incl)+'.png', bbox_inches='tight', pad_inches=0.2, dpi=300)

    return 

start = time.time()
for incl in incls:
    for idNo in ids:
        draw_contour(orbit, incl, idNo)

stop = time.time()
count_time(stop, start)
