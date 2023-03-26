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
from photutils.aperture import SkyCircularAperture
import astropy.units as u
from astropy.coordinates import SkyCoord
from photutils.aperture import aperture_photometry
from astropy.wcs import WCS


###########################################################
# directories

Dir = '/home/heh15/research/FIRE/'
picDir = Dir+'pictures/'
logDir = Dir+'logs/'
fitsDir = Dir+'e2/'

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

def fits_import(fitsimage, item=0):
    hdr = fits.open(fitsimage)[item].header
    wcs = WCS(hdr).celestial
    data=fits.open(fitsimage)[item].data
    data=np.squeeze(data)

    return wcs, data

def Apmask_convert(aperture,data_cut):
    apmask=aperture.to_mask(method='center')
    shape=data_cut.shape
    mask=apmask.to_image(shape=((shape[0],shape[1])))
    ap_mask=mask==0
    ap_masked=np.ma.masked_where(ap_mask,data_cut)

    return ap_masked

###########################################################
# main program

# load the global history for merger evolution
filename=logDir+'galsep_G2G3_e_orbit_2.npz'
galsep = np.load(filename)
SFRs = galsep['sfr']
Times = galsep['time']
Snapshots = galsep['isnap']

orbit = 'e2'
incls = ['v0', 'v1', 'v2', 'v3']
ids = np.arange(520, 712, 2)

incls = ['v0']

G2G3_center = {}
G2G3_center['Sigmol_med'] = np.array([])
G2G3_center['vdep_med'] = np.array([])
G2G3_center['alphavir_med'] = np.array([])
G2G3_center['tff_mean'] = np.array([])
G2G3_center['sfr'] = SFRs[np.isin(Snapshots, ids)]
G2G3_center['time'] = Times[np.isin(Snapshots, ids)] 

def calc_GMCs_median(mom0file, mom2file):
    '''
    Calculate the median virial parameter for different snapshots 
    '''
    # read the moment fits files
    wcs, mom0 = fits_import(mom0file)
    wcs, mom2 = fits_import(mom2file)

    # mask the region outside 1 kpc circle
    position = SkyCoord(ra=0, dec=0, unit='deg')
    central_sky = SkyCircularAperture(position, r=1*u.arcsec)
    central_pix = central_sky.to_pixel(wcs)
    mom0_center = Apmask_convert(central_pix, mom0)
    mom2_center = Apmask_convert(central_pix, mom2)

    # mask values
    mom2_center[np.where(mom2_center==0)] = np.nan
    mask = np.isnan(mom2_center.flatten())
    m0 = mom0_center.flatten()[np.where(~mask)]
    m2 = mom2_center.flatten()[np.where(~mask)]
    # flag the masked non-central region                                         
    m0 = m0.data[np.where(~m0.mask)]
    m2 = m2.data[np.where(~m2.mask)] 

    # calculate the mass weighted median surface densities
    wq =  DescrStatsW(data=m0, weights=m0)
    Sigmol_meds = wq.quantile(probs=[0.16,0.5,0.84], return_pandas=False)
    G2G3_center['Sigmol_med'] = np.append(G2G3_center['Sigmol_med'], Sigmol_meds)
    G2G3_center['Sigmol_med'] = np.reshape(G2G3_center['Sigmol_med'], (-1, 3)) 

    # Calculate the mass weighted median velocity dispersions
    wq = DescrStatsW(data=m2*10, weights=m0) # low values give 
    vdep_meds = wq.quantile(probs=[0.16,0.5,0.84], return_pandas=False) / 10
    G2G3_center['vdep_med'] = np.append(G2G3_center['vdep_med'], vdep_meds)
    G2G3_center['vdep_med'] = np.reshape(G2G3_center['vdep_med'], (-1,3))

    # calculate the alphavir (Sun+2018)
    aVir = 5.77 * m2**2 * m0**(-1) * (50/40)**(-1)
    # calculate the median of the alphavir (mask and no mask has large difference)
    wq = DescrStatsW(data=aVir, weights=m0)
    alphaVir_meds = wq.quantile(probs=[0.16, 0.5, 0.84], return_pandas=False)
    G2G3_center['alphavir_med'] = np.append(G2G3_center['alphavir_med'], 
                                    alphaVir_meds)
    G2G3_center['alphavir_med'] = np.reshape(G2G3_center['alphavir_med'], (-1,3))    

    # calculate the free-fall time
    G = 6.67e-11      
    tff = np.sqrt(3)/(4*G)*(m2*1000)/(m0*2e30/3.1e16**2)/(3600*24*365)
    tff = 1 / np.sqrt(8*G*(m0*2e30/3.1e16**2)/3.14/(50*3.1e16))/(3600*24*365)
    tff_mean = 1 / np.average(1/tff, weights=m0)
    G2G3_center['tff_mean'] = np.append(G2G3_center['tff_mean'], tff_mean)

    return 

start = time.time()
for incl in incls:
    for idNo in ids:
        mom0file = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32_ccords_mom0.fits'
        mom2file = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32_ccords_mom2.fits'
        calc_GMCs_median(mom0file, mom2file)
    # write it into the pkl file
    outputfile = 'G2G3_'+orbit+'_'+incl+'_coalesce_center.pkl'
    with open(logDir+outputfile, 'wb') as handle:
        pickle.dump(G2G3_center, handle)

stop = time.time()
count_time(stop, start)

