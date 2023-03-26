# import packages
from astropy.utils import data
from radio_beam import Beam
from astropy import units as u
import time
import numpy as np
import matplotlib.pyplot as plt
import math
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
from scipy.stats.stats import pearsonr
from matplotlib import rcParams
import seaborn as sns
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from math import log10, floor
import pickle
from astropy.table import Table
from statsmodels.stats.weightstats import DescrStatsW
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mtick
from astropy.coordinates import SkyCoord

# import functions
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

def mpl_setup(
        figtype='paper-1/2', aspect=0.8, lw=None, ms=None, fs=None,
        style='seaborn-paper', rcParams={}):
    """
    Configure matplotlib rcParams.
    Parameters
    ----------
    figtype : {'paper-1/1', 'paper-1/2', 'paper-1/3', 'talk', float}
        This parameter specifies the type and width of the figure(s):
            * 'paper-1/1': figure width = full textwidth (7.1 inch)
                           (normal lw, ms, fs)
            * 'paper-1/2': figure width = 1/2 textwidth
                           (normal lw, ms, fs)
            * 'paper-1/3': figure width = 1/3 textwidth
                           (normal lw, ms, fs)
            * 'talk': figure height = 4.0 inches
                      (larger lw, ms, fs)
            * float: specifying figure width (in inches)
                     (normal lw, ms, fs)
    aspect : float, optional
        Aspect ratio (height/width) of a figure.
        Default is 0.8.
    lw : float, optional
        Line width (in points). Default is 1 (1.5 for figtype='talk').
    ms : float, optional
        Marker size (in points). Default is 1 (1.5 for figtype='talk').
    fs : float, optional
        Font size (in points). Default is 11 (18 for figtype='talk').
    style : string, optional
        Style name to be passed to `~matplotlib.pyplot.style.use`.
        Default is 'seaborn-paper'.
    rcParams : dict, optional
        Other `~matplotlib.rcParams`
    """

    # style
    plt.style.use(style)

    # figure
    textwidth = 7.1  # inches
    slideheight = 4.0  # inches
    if figtype == 'paper-1/1':
        fw = 0.98 * textwidth
    elif figtype == 'paper-1/2':
        fw = 0.48 * textwidth
    elif figtype == 'paper-1/3':
        fw = 0.31 * textwidth
    elif figtype == 'talk':
        fw = slideheight / aspect
    else:
        fw = figtype
    plt.rcParams['figure.figsize'] = (fw, fw*aspect)
    plt.rcParams['figure.dpi'] = 200

    # default sizes
    if figtype == 'talk':
        if lw is None:
            lw = 1.5
        if ms is None:
            ms = 1.5
        if fs is None:
            fs = 18.0
    else:
        if lw is None:
            lw = 1.0
        if ms is None:
            ms = 1.0
        if fs is None:
            fs = 11.0

    # font
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [
        'Times', 'Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['font.sans-serif'] = [
        'Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.monospace'] = [
        'Terminal', 'monospace']
    plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    
    # font weight 
    if figtype == 'talk':
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['figure.titleweight'] = 'bold'

    # fontsize
    plt.rcParams['font.size'] = fs
    plt.rcParams['xtick.labelsize'] = 'small'
    plt.rcParams['ytick.labelsize'] = 'small'
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['axes.labelsize'] = 'large'
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['figure.titlesize'] = 'large'

    # linewidth
    for key in [
            'lines.linewidth', 'axes.linewidth',
            'patch.linewidth', 'hatch.linewidth',
            'grid.linewidth', 'lines.markeredgewidth',
            'xtick.major.width', 'xtick.minor.width',
            'ytick.major.width', 'ytick.minor.width']:
        plt.rcParams[key] = lw

    # errorbar cap size
    plt.rcParams['errorbar.capsize'] = 2 * lw

    # markersize
    plt.rcParams['lines.markersize'] = ms

    # axes
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 4 * lw
    plt.rcParams['xtick.minor.size'] = 2 * lw
    plt.rcParams['xtick.major.pad'] = 0.3 * fs
    plt.rcParams['xtick.minor.pad'] = 0.2 * fs
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.size'] = 4 * lw
    plt.rcParams['ytick.minor.size'] = 2 * lw
    plt.rcParams['ytick.major.pad'] = 0.2 * fs
    plt.rcParams['ytick.minor.pad'] = 0.1 * fs

    # legend
    plt.rcParams['legend.handlelength'] = 1.0
    plt.rcParams['legend.handletextpad'] = 0.5
    plt.rcParams['legend.framealpha'] = 0.2
    plt.rcParams['legend.edgecolor'] = u'0.0'

    # hatch
    plt.rcParams['hatch.color'] = 'gray'

    # image
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'none'

    plt.rcParams.update(rcParams)

    return

###########################################################
# directories

npzDir = '../FIRE_GalSep/'
fitsDir = '../e2/'
logDir = '../logs/'
picDir = '../pictures/'
figDir = '../test/'

tex_textwidth = 7.1
# plt.style.use('seaborn-paper')
# plt.rcParams.update({'font.size': 12})
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = [
#     'Times', 'Times New Roman', 'DejaVu Serif', 'serif']
# plt.rcParams['font.sans-serif'] = [
#     'Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
# plt.rcParams['font.monospace'] = [
#     'Terminal', 'monospace']
# plt.rcParams.update(rcParams)

###########################################################
# main 

# Load the literature data

# import the literature data from Sun+2020
filename= logDir+'Sun_2020_GMC.txt'
skiprows=28
table=pd.read_csv(filename,header=None,sep=r"\s+",skiprows=skiprows,engine='python')
table_90pc=table.loc[(table[1]==90)]
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
# select Antennae overlap regions
table_antennae = table_120pc.loc[(table_120pc[0]=='Antennae')]
# select M31 and M33. 
table_M31 = table_120pc.loc[(table_120pc[0]=='M31')]
table_M33 = table_120pc.loc[(table_120pc[0]=='M33')]
table_M3 = pd.concat([table_M31,table_M33])

# M31 properties (Nieten et al.2006; Sick et al. 2014)
M31_Mmol = 3.6e8
M31_Mgas = 5.2e9
M31_Mstar = 8.8e10

# Import pixel analyses on Antennae and NGC 3256
filename = logDir+'Antennae_Brunetti_pix.ecsv'
table_antennae = Table.read(filename)
table_ant = table_antennae.to_pandas()
table_ant_90pc = table_ant.loc[table_ant['beam_fwhm']==90]
# For NGC 3256
filename = logDir+'NGC3256_Brunetti_pix.ecsv'
table_3256 = (Table.read(filename)).to_pandas()
table_3256_80pc = table_3256.loc[table_3256['beam_fwhm']==80]

## plot the figure

# make the moment maps
def make_moments(fitsfile, vel):
    data = fits.open(fitsfile)[0].data
    data[np.where(data==0)] = np.nan
    mom0 = make_mom0(data)
    mom1 = make_mom1(data, vel)
    mom2 = make_mom2(data, vel)
    return data, mom0, mom1, mom2

def set_threshold(data, mom0, mom2):
    
    counts = np.count_nonzero(~np.isnan(data), axis=2)
    mom0_filter = np.copy(mom0)
    mom2_filter = np.copy(mom2)
    mom0_filter[np.where(mom0<1)] = np.nan
    mom2_filter[np.where(counts<2)] = np.nan
    mask = np.ma.mask_or(np.isnan(mom0_filter.flatten()), np.isnan(mom2_filter.flatten()))
    m0 = mom0.flatten()[np.where(~mask)]
    m2 = mom2.flatten()[np.where(~mask)]
    return m0, m2

vel = np.arange(-500, 500, 2)

# load data and information for G2 galaxy
fitsfile_i2 = '../losvd_FIRE2_i2_025_gas_pa30__32.fits'
data_i2, mom0_i2, mom1_i2, mom2_i2 = make_moments(fitsfile_i2, vel)
m0_i2, m2_i2 = set_threshold(data_i2, mom0_i2, mom2_i2)

# Loat data and information for G3 galaxy
fitsfile_i3 = '../losvd_FIRE2_i3_025_gas_pa30__32.fits'
data_i3, mom0_i3, mom1_i3, mom2_i3 = make_moments(fitsfile_i3, vel)
m0_i3, m2_i3 = set_threshold(data_i3, mom0_i3, mom2_i3)

# load the data and information for G3 galaxy with inclination of 60 degrees
fitsfile_i3_60 = '../losvd_FIRE2_i3_025_gas_pa60__32.fits'
data_i3_60, mom0_i3_60, mom1_i3_60, mom2_i3_60 = make_moments(fitsfile_i3_60, vel)
m0_i3_60, m2_i3_60 = set_threshold(data_i3_60, mom0_i3_60, mom2_i3_60)

# load the data and information for G3 galaxy with inclination of 80 degrees
fitsfile_i3_80 = '../losvd_FIRE2_i3_025_gas_pa80__32.fits'
data_i3_80, mom0_i3_80, mom1_i3_80, mom2_i3_80 = make_moments(fitsfile_i3_80, vel)
m0_i3_80, m2_i3_80 = set_threshold(data_i3_80, mom0_i3_80, mom2_i3_80)

# calculate the G3 mass weighted surface density for this snapshot
wq = DescrStatsW(data=m0_i3, weights=m0_i3)
surf_median = wq.quantile(probs=[0.5], return_pandas=False)
print('median surface density: '+str(round(surf_median[0],2)))
wq = DescrStatsW(data=m2_i3, weights=m0_i3)
vdep_median = wq.quantile(probs=[0.5], return_pandas=False)
print('median velocity dispersion: '+str(round(vdep_median[0],2)))
aVir = 5.77 * m2_i3**2 * m0_i3**(-1) * (50/40)**(-1)
wq = DescrStatsW(data=aVir, weights=m0_i3)
aVir_median = wq.quantile(probs=[0.5], return_pandas=False)
print('median virial parameter: '+str(round(aVir_median[0],2)))

# plot the figure

levels = [0.2,0.5,0.8]

mpl_setup()
fig = plt.figure() # paper figsize=(0.5*tex_textwidth,0.5*0.8*tex_textwidth)
# fig = plt.figure(figsize=(7, 4)) # abstract
ax = plt.subplot(111)
# plt.xscale('log')
# plt.yscale('log')
# 
# plt.xlabel('$\Sigma_{\mathrm{mol}}$ (M$_{\odot}$ pc$^{-2}$)',fontsize=15)
# plt.ylabel(r'$\sigma_v$ (km s$^{-1}$)',fontsize=15)
# 
# # Draw the scatter and contour plot for simulation 
# density_contour(m0_i2, m2_i2, weights=m0_i2, levels=levels, ax=ax, color='black',
#                 contour_type='contour',alphas=(1,1,1))
# density_contour(m0_i3, m2_i3, weights=m0_i3, levels=levels, ax=ax, color='brown',
#                 contour_type='contour',alphas=(1,1,1))
# plt.ylim(bottom=0.1, top=150)
# 
# # Draw the contour plot for Sun+2020 data
# density_contour(table_offcenter[7], table_offcenter[8], weights=table_offcenter[7],levels=levels,
#                 ax=ax, color='tab:blue')
# density_contour(table_center_barred[7], table_center_barred[8], weights=table_center_barred[7], levels=levels,
#                 ax=ax, color='salmon')
# density_contour(table_center_unbarred[7], table_center_unbarred[8], weights=table_center_unbarred[7],
#                 levels=levels,
#                 ax=ax, color='brown',
#                 contour_type='contour', linestyles='dashed',alphas=(1,1,1))
# # Draw the contour plot for Sun+2018 data
# # density_contour(table_antennae[3], table_antennae[4], weights=table_antennae[3], levels=levels,
# #                     ax=ax, contour_type='contour', alphas=(1,1,1), color='red')
# density_contour(table_M31[3], table_M31[4], weights=table_M31[3], levels=levels,
#                 ax=ax, contour_type='contour', alphas=(1,1,1), color='green')
# 
# # Draw the virial parameter
# ax.set_xlim(); ax.set_ylim()
# Sigmol_theory = 10**np.linspace(-1,5,5)
# vdep_theory = np.sqrt(Sigmol_theory * 3.1 / 5.77 * (45/40))
# plt.plot(Sigmol_theory, vdep_theory, linestyle='--',linewidth=2,color='red')
# 
# # label the legend
# line_simul2 = mlines.Line2D([],[],color='black', linestyle='solid', label='FIRE G2')
# line_simul3 = mlines.Line2D([],[],color='brown', linestyle='solid', label='FIRE G3')
# patch1 = mpatches.Patch(color='tab:blue', label='PHANGS disks')
# patch2 = mpatches.Patch(color='salmon', label='PHANGS barred galaxy centers')
# line=mlines.Line2D([], [], color='brown', linestyle='dashed', label='PHANGS unbarred galaxy centers')
# # line2=mlines.Line2D([], [], color='red', linestyle='solid', label='Antennae overlap region')
# line3=mlines.Line2D([], [], color='green', linestyle='solid', label='M31')
# legend = plt.legend(handles=[line_simul2, line_simul3], loc='upper left',fontsize=10) # fontsize=10
# legend1 = plt.legend(handles=[patch1, patch2, line, line3], loc='lower right',fontsize=10) # fontsize=10
# plt.gca().add_artist(legend)
# fig.tight_layout()
plt.savefig('G2G3_iso_Sigmol_vdep.png', bbox_inches='tight')
