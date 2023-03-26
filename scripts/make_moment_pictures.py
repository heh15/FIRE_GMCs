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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

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

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

###########################################################
# main program 

orbit = 'e2'
incls = ['v0', 'v1', 'v2', 'v3']
ids = np.arange(520, 712, 2)

# incls = ['v0','v1']
# ids = np.arange(520, 712, 2)
# ids = np.arange(586, 588, 2)
orbit = 'e1'
incls = ['v0']
ids = np.arange(250, 412, 2)

def make_moments(orbit, incl, idNo):
    '''
    Make the moment maps for different 
    '''
    filename = picDir+'G2G3_'+orbit+'_mom0_'+str(idNo)+'_'+str(incl)+'.png'
    # read the fits file
    fitsfile = Dir+orbit+'/losvd_FIRE2_'+orbit+'_'+str(idNo)+'_gas_'+str(incl)+'__32.fits'
    data = fits.open(fitsfile)[0].data
    hdr = fits.open(fitsfile)[0].header
    data[np.where(data==0)] = np.nan

    # set the color bar
    colors1 = plt.cm.binary(np.linspace(0., 1, 128))
    colors2 = plt.cm.gist_heat(np.linspace(0, 1, 128))
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined) 
   
    # make moment maps
    mom0 = make_mom0(data)
    vel = np.arange(-500,500,2)
    mom1 = make_mom1(data, vel)
    mom2 = make_mom2(data, vel)
    Tpeak = make_Tpeak(data)

    # plot the moment maps
#     vmax = np.nanpercentile(mom0.flatten(), 99.5)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_title('Moment 0 Map', fontsize=15)
    ax.tick_params(direction='in')
    im = ax.imshow(mom0, origin='lower', cmap=mymap,
                   norm=colors.LogNorm(vmin=1, vmax=np.nanmax(mom0)),
                   extent=[-12.5, 12.5, -12.5, 12.5])
    ax.set_xlabel('kpc', fontsize=15)
    ax.set_ylabel('kpc', fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    
    cbar = plt.colorbar(im, cax=cax)
    cax.set_ylabel(r'M$_{\odot}$ pc$^{-2}$', fontsize=15)

    # draw the red aperture around the center with diameter of 1 kpc
    if draw_aperture == True:
        circle = plt.Circle((0, 0), 1.0, color='r', fill=False, linewidth=2)
        ax.add_patch(circle)
        filename = filename.replace('.png', '_aperture.png')

    set_size(5,5)
#     fig.set_size_inches(6,6)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.2, dpi=300) 
 
start = time.time()
draw_aperture = False
for incl in incls:
    for idNo in ids:
        make_moments(orbit, incl, idNo)

stop = time.time()
count_time(stop, start)

