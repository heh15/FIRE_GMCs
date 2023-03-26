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

def fits_import(fitsimage, item=0):
    hdr = fits.open(fitsimage)[item].header
    wcs = WCS(hdr).celestial
    data=fits.open(fitsimage)[item].data
    data=np.squeeze(data)

    return wcs, data

def count_time(stop, start):
    '''
    Convert the time difference into human readable form.
    '''
    dure=stop-start
    m,s=divmod(dure,60)
    h,m=divmod(m,60)
    print("%d:%02d:%02d" %(h, m, s))

    return

###########################################################
# main program

incls = ['v0', 'v1', 'v2', 'v3']
ids = np.arange(520, 712, 2)

incls = ['v0', 'v1']
# ids = np.arange(520, 526, 2)


def calc_fcenter(fitsfile, G2G3_globals):
    '''
    add the celestial coordinates to the file
    '''
    # read the fits file
    wcs, data = fits_import(fitsfile)

    # calculate the total gas mass
    Mmol = np.nansum(data) * 100**2
    G2G3_globals['Mmol'] = np.append(G2G3_globals['Mmol'], Mmol)

    # create circlular aperture in the center
    position = SkyCoord(ra=0, dec=0, unit='deg')
    central_sky = SkyCircularAperture(position, r=1*u.arcsec) 
    central_pix = central_sky.to_pixel(wcs)

    # measure the total gas mass in the center
    phot_table = aperture_photometry(data, central_pix)
    Mmol_central = phot_table['aperture_sum'] * 100**2
    G2G3_globals['Mmol_central'] = np.append(G2G3_globals['Mmol_central'], Mmol_central)

start = time.time()
for incl in incls:
    # read the dictionaries of GMC global properties
    filename = logDir+'G2G3_e2_'+incl+'_coalesce.pkl'
    with open(filename, 'rb') as handle:
        G2G3_globals = pickle.load(handle, encoding='latin')
    G2G3_globals['Mmol_central'] = np.array([])
    G2G3_globals['Mmol'] = np.array([])
    for idNo in ids:
        # write the central gas fraction
        fitsfile = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32_ccords_mom0.fits'
        calc_fcenter(fitsfile, G2G3_globals)
    # write it into the pkl file
    outputfile = 'G2G3_e2_'+incl+'_coalesce.pkl'
    with open(logDir+outputfile, 'wb') as handle:
        pickle.dump(G2G3_globals, handle)

stop = time.time()
count_time(stop, start)

# fitsfile = Dir+'losvd_FIRE2_i3_025_gas_pa30__32_ccords_mom0.fits'
# G3_iso_dicts = {}
# G3_iso_dicts['Mmol_central'] = np.array([])
# G3_iso_dicts['Mmol'] = np.array([])
# calc_fcenter(fitsfile, G3_iso_dicts)
# print(G3_iso_dicts)
