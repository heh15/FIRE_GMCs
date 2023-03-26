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
import weightedstats as ws
from statsmodels.stats.weightstats import DescrStatsW
import pickle

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

###########################################################
# main program

# load the global history for merger evolution
# filename=logDir+'galsep_G2G3_e_orbit_2.npz'
# galsep = np.load(filename)
# SFRs = galsep['sfr']
# Times = galsep['time']
# Snapshots = galsep['isnap']
# orbit = 'e2'
# incls = ['v0', 'v1', 'v2', 'v3']
# ids = np.arange(520, 712, 2)
# incls = ['v0']

# Try e1 orbit
filename = logDir+'galsep_G2G3_e_orbit_1.npz'
galsep = np.load(filename)
SFRs = galsep['sfr']
Times = galsep['time']
Snapshots = galsep['isnap']
orbit = 'e1'
ids = np.arange(250, 412, 2)
incls = ['v0']


G2G3_globals = {}
G2G3_globals['Sigmol_med'] = np.array([])
G2G3_globals['vdep_med'] = np.array([])
G2G3_globals['alphavir_med'] = np.array([])
G2G3_globals['sfr'] = SFRs[np.isin(Snapshots, ids)]
G2G3_globals['time'] = Times[np.isin(Snapshots, ids)] 

def calc_GMCs_median(orbit, incl, idNo):
    '''
    Calculate the median virial parameter for different snapshots 
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

    # mask values
    counts = np.count_nonzero(~np.isnan(data),axis=2)
    mask1 = mom0<1; mask2 = counts<2
    mask = np.ma.mask_or(mask1, mask2, copy=True)
    m0 = mom0[~mask].flatten()
    m2 = mom2[~mask].flatten()


    # calculate the mass weighted median surface densities
    wq =  DescrStatsW(data=m0, weights=m0)
    Sigmol_meds = wq.quantile(probs=[0.16,0.5,0.84], return_pandas=False)
    G2G3_globals['Sigmol_med'] = np.append(G2G3_globals['Sigmol_med'], Sigmol_meds)
    G2G3_globals['Sigmol_med'] = np.reshape(G2G3_globals['Sigmol_med'], (-1, 3)) 

    # Calculate the mass weighted median velocity dispersions
    wq = DescrStatsW(data=m2, weights=m0)
    vdep_meds = wq.quantile(probs=[0.16,0.5,0.84], return_pandas=False)
    G2G3_globals['vdep_med'] = np.append(G2G3_globals['vdep_med'], vdep_meds)
    G2G3_globals['vdep_med'] = np.reshape(G2G3_globals['vdep_med'], (-1,3))

    # calculate the alphavir (Sun+2018)
    alphaVir = 5.77 * mom2**2 * mom0**(-1) * (50/40)**(-1)
    aVir = alphaVir[~mask].flatten()
    # calculate the median of the alphavir (mask and no mask has large difference)
    wq = DescrStatsW(data=aVir, weights=m0)
    alphaVir_meds = wq.quantile(probs=[0.16, 0.5, 0.84], return_pandas=False)
    G2G3_globals['alphavir_med'] = np.append(G2G3_globals['alphavir_med'], 
                                    alphaVir_meds)
    G2G3_globals['alphavir_med'] = np.reshape(G2G3_globals['alphavir_med'], (-1,3))    

    return 

start = time.time()
for incl in incls:
    for idNo in ids:
        calc_GMCs_median(orbit, incl, idNo)
    # write it into the pkl file
    outputfile = 'G2G3_'+orbit+'_'+incl+'_coalesce.pkl'
    with open(logDir+outputfile, 'wb') as handle:
        pickle.dump(G2G3_globals, handle)

stop = time.time()
count_time(stop, start)

