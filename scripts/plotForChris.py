import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
# import seaborn as sns
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
# functions

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

# Import pixel analyses on Antennae and NGC 3256
filename = logDir+'Antennae_Brunetti_pix.ecsv'
table_antennae = Table.read(filename)
table_ant = table_antennae.to_pandas()
table_ant_90pc = table_ant.loc[table_ant['beam_fwhm']==90]
# For NGC 3256
filename = logDir+'NGC3256_Brunetti_pix.ecsv'
table_3256 = (Table.read(filename)).to_pandas()
table_3256_80pc = table_3256.loc[table_3256['beam_fwhm']==80]

# make the scatter plot
fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\Sigma_{\mathrm{mol}}$ (M$_{\odot}$ pc$^{-2}$)',fontsize=15)
plt.ylabel(r'$\sigma$ (km$^2$ s$^{-2}$)',fontsize=15)
levels=(0.4, 0.85, 0.99)
# Draw the contour plot for PHANGS data
density_contour(table_90pc[7], table_90pc[8], weights=table_90pc[7], levels=levels,
                ax=ax, color='tab:blue')
# Draw the contour plot for Antennae
density_contour(table_ant_90pc['surface_density'], table_ant_90pc['velocity_dispersion'],
                weights=table_ant_90pc['surface_density'], levels=levels, ax=ax,
                color='tomato')
# Draw the contour plot for NGC3256
density_contour(table_3256_80pc['surface_density'], table_3256_80pc['velocity_dispersion'],
                weights=table_3256_80pc['surface_density'], contour_type='contour',
                levels=levels, ax=ax, color='blue', alphas=(1,1,1))
# Draw the virial parameter
ax.set_xlim(); ax.set_ylim()
Sigmol_theory = 10**np.linspace(-1,5,5)
vdep_theory = np.sqrt(Sigmol_theory * 3.1 / 5.77 * (45/40))
plt.plot(Sigmol_theory, vdep_theory, linestyle='--',linewidth=2,color='red')

# set the limit
ax.set_xlim(right=10**4.5)
ax.set_ylim(bottom=0.5)
# label the legend
patch = mpatches.Patch(color='tab:blue', label='PHANGS')
patch2 = mpatches.Patch(color='tomato', label=r'Antennae')
# line=mlines.Line2D([], [], color='orangered', linestyle='solid', label=r'Antennae, $\alpha_{\mathrm{CO}}=1.1$')
line2=mlines.Line2D([], [], color='blue', linestyle='solid', label='NGC 3256')
plt.legend(handles=[patch, patch2,line2], loc='lower right')
plt.savefig('PHANGS_NGC3256_Antennae.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
