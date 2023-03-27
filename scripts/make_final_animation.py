from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pandas as pd
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython.display import HTML

###########################################################
# directories

Dir = '/home/heh15/research/FIRE/'
picDir = Dir + 'snapshots/'
npzDir = Dir + 'FIRE_GalSep/'
logDir = Dir + 'logs/'
figDir = Dir + 'figures/'

orbit = 'e2'
orbit = 'e1'
fitsDir = Dir + orbit + '/'

###########################################################
# basic settings

incl = 'v0'
ids = np.arange(520,712,2)
ids = np.arange(278,412,2)

t_cols = [2.616, 2.831]
t_cols = [1.41, 1.53]

frames = len(ids)
print(frames)

# picture style settings
my_dpi = 96

###########################################################
# functions

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

def draw_contour(ax, orbit, incl, idNo):
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
    mom2 = make_mom2(data, vel)

    counts = np.count_nonzero(~np.isnan(data),axis=2)
    mom0[np.where(mom0<1)] = np.nan
    mom2[np.where(counts<2)] = np.nan
    mask = np.ma.mask_or(np.isnan(mom0.flatten()), np.isnan(mom2.flatten()))

    # flag the nan values
    m1 = mom0.flatten()[np.where(~mask)]
    m2 = mom2.flatten()[np.where(~mask)]
    # make the scatter plot
    ax.set_xscale('log')
    ax.set_yscale('log')

#     plt.xlabel('$\Sigma_{\mathrm{mol}}$ (M$_{\odot}$ pc$^{-2}$)',fontsize=15)
    ax.set_ylabel(r'$\sigma_v$ (km s$^{-1}$)',fontsize=10)

    # Draw the scatter and contour plot for simulation 
    levels = [0.2, 0.5, 0.8]
#     plt.scatter(mom0.flatten(), mom2.flatten(), alpha=0.05, color='gray')
    density_contour(m1, m2, weights=m1, levels=levels, ax=ax, color='black',
                    contour_type='contour',alphas=(1,1,1))
    ax.set_ylim(0.1, 250)
    ax.set_xlim(0.8, 8e3)

    # Draw the contour plot for PHANGS data
    density_contour(table_90pc[7], table_90pc[8], weights=table_90pc[7], levels=levels,
                    ax=ax, color='tab:blue')
    # Draw the virial parameter
    ax.set_xlim(); ax.set_ylim()
    Sigmol_theory = 10**np.linspace(-1,5,5)
    vdep_theory = np.sqrt(Sigmol_theory * 3.1 / 5.77 * (45/40))
    ax.plot(Sigmol_theory, vdep_theory, linestyle='--',linewidth=2,color='red')

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
    patch = mpatches.Patch(color='tab:blue', label='PHANGS galaxies')
    patch2 = mpatches.Patch(color='tomato', label=r'Antennae, $\alpha_{\mathrm{CO}}=4.3$')
#    line=mlines.Line2D([], [], color='tomato', linestyle='solid', label=r'Antennae, $\alpha_{\mathrm{CO}}=4.3$')
    line2=mlines.Line2D([], [], color='blue', linestyle='solid', label=r'NGC 3256, $\alpha_{\mathrm{CO}}=1.1$')
    line3=mlines.Line2D([], [], color='green', linestyle='solid', label='M31')
    legend = ax.legend(handles=[line_simul],loc='upper left', fontsize=8)
    legend1 = ax.legend(handles=[patch, patch2, line2, line3], loc='lower right', fontsize=8, framealpha=0.5)
#     plt.gca().add_artist(legend)
    ax.add_artist(legend)
   
    return

def make_moments(ax, orbit, incl, idNo):
    '''
    Make the moment maps for different snapshots
    Return colorbar axis
    '''
    # read the fits file
    fitsfile = Dir+orbit+'/losvd_FIRE2_'+orbit+'_'+str(idNo)+'_gas_'+str(incl)+'__32.fits'
    data = fits.open(fitsfile)[0].data
    hdr = fits.open(fitsfile)[0].header
    data[np.where(data==0)] = np.nan
   
    extent = [-12.5,12.5,-12.5,12.5] 
#     # zoom into the central region
#     FOV = np.array([10,10]) # in kpc
#     data = data[int(124-FOV[0]/2*10):int(124+FOV[0]/2*10), int(124-FOV[1]/2*10):int(124+FOV[1]/2*10), :]
#     extent=[-FOV[0]/2, FOV[0]/2, -FOV[1]/2, FOV[1]/2]

#     # set the color bar
#     colors1 = plt.cm.binary(np.linspace(0., 1, 128))
#     colors2 = plt.cm.gist_heat(np.linspace(0, 1, 128))
#     colors_combined = np.vstack((colors1, colors2))
#     mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)

    # make moment maps
    mom0 = make_mom0(data)
    counts = np.count_nonzero(~np.isnan(data),axis=2)
    mask1 = mom0<1; mask2 = counts<2
    mask = np.ma.mask_or(mask1, mask2, copy=True)
    mom0[mask] = np.nan
#     vel = np.arange(-500,500,2)
#     mom1 = make_mom1(data, vel)
#     mom2 = make_mom2(data, vel)
#     Tpeak = make_Tpeak(data)

    # plot the moment maps
    vmax = np.nanpercentile(mom0.flatten(), 99.5)
#     vmin = np.nanpercentile(mom0.flatten(), 0.5)
    vmin = vmax/100
#     ax.set_title('Moment 0 Map', fontsize=15)
    ax.tick_params(direction='in')
    im = ax.imshow(mom0, origin='lower', cmap='gist_heat_r',
                   norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                   extent=extent)
#     ax.set_xlabel('kpc', fontsize=15)
#     ax.set_ylabel('kpc', fontsize=15)

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('top', size='5%', pad=0.05)
#     cax = fig.add_axes([ax.get_position().x0, ax.get_position().y1, 0.8*(ax.get_position().x1-ax.get_position().x0), 0.01])
    cbar = plt.colorbar(im, orientation='horizontal', cax=cax)
#     cax.set_xlabel(r'M$_{\odot}$ pc$^{-2}$', fontsize=15,loc='right', labelpad=-15)
    cax.get_xaxis().set_ticks_position('top')
#     cax.xaxis.set_label_position('top')

    # annotate with the plots
    ax.annotate('$\Sigma_{\mathrm{mol}}$\n(M$_{\odot}$ pc$^{-2}$)', (0.08,0.85), xycoords='axes fraction', fontsize=10,
               ha='left', va='center') 

    # draw the red aperture around the center with diameter of 1 kpc
#     if draw_aperture == True:
#     circle = plt.Circle((0, 0), 1.0, color='r', fill=False, linewidth=2)
#     ax.add_patch(circle)

    
    return cax


###########################################################
# main program

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

# draw snapshots
fig = plt.figure(figsize=(600/my_dpi, 500/my_dpi))
gc = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, height_ratios=[1, 3],
                        hspace=0.1)
ax0 = fig.add_subplot(gc[0]) # plot SFR history
gc1 = gridspec.GridSpecFromSubplotSpec(ncols=2, nrows=1, subplot_spec=gc[1], width_ratios=[6,5], wspace=0.2)
ax1 = fig.add_subplot(gc1[0]) # plot vdep vs Sigmol
ax2 = fig.add_subplot(gc1[1]) # plot surface density map
divider = make_axes_locatable(ax2)
cax = divider.append_axes('top', size='5%', pad=0.05)

def animate(i):
    ax0.clear(); ax1.clear(); ax2.clear(); cax.clear()
# for i, idNo in enumerate(ids):
    idNo = ids[i]
    # plot the SFR history for 'e2' orbit
    filename=npzDir+'galsep_G2G3_'+orbit[0]+'_orbit_'+orbit[1:]+'.npz'
    data = np.load(filename)
    SFRs = data['sfr']
    Times = data['time']
    Snapshots = data['isnap']
    ax0.set_yscale('log')
    ax0.plot(Times, SFRs)
    ax0.set_xlabel('Time (Gyr)', fontsize=10)
    ax0.set_ylabel('SFR (M$_{\odot}$ yr$^{-1}$', fontsize=10)
    ax0.axvline(t_cols[0], linestyle='dashed', color='black')
    ax0.axvline(t_cols[1], linestyle='dashed', color='black')
    ax0.axvline(Times[idNo], linestyle='solid', color='black')

    # plot the scatter plot
    draw_contour(ax1, orbit, incl, idNo)
    ax1.annotate('$t$: '+str(round(Times[idNo],2))+' Gyr', (0.05, 0.78), 
         xycoords='axes fraction', fontsize=10)
    ax1.set_aspect('equal')
#     ax.annotate('SFR: '+str(round(SFRs[idNo],2))+' M$_{\odot}$yr$^{-1}$', (0.01, 0.68), 
#          xycoords='axes fraction', fontsize=10)
    ax1.set_xlabel('$\Sigma_{\mathrm{mol}}$ (M$_{\odot}$ pc$^{-2}$)',fontsize=10)

    # plot surface density maps
    make_moments(ax2, orbit, incl, idNo)
    ax2.set_xlabel('kpc',fontsize=10)
    ax2.set_ylabel('kpc', fontsize=10, labelpad=-8)
#    plt.savefig(picDir+'G2G3_'+orbit+'_'+incl+'_'+str(idNo)+'.png',bbox_inches='tight',dpi=my_dpi)

anim = FuncAnimation(fig, animate, frames=frames, interval=100, repeat=False)
anim_html = HTML(anim.to_jshtml())
with open(figDir+'G2G3_'+orbit+'_'+incl+'_final.html', 'w') as f:
    f.write(anim_html.data)

