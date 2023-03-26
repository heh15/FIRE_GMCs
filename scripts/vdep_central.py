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
from regions import read_ds9
from spectral_cube import SpectralCube

###########################################################
# directories

Dir = '/home/heh15/research/FIRE/'
picDir = Dir+'pictures/'
logDir = Dir+'logs/'
fitsDir = Dir+'e2/'
regionDir = Dir+'regions/'

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

def Apmask_convert(aperture,data_cut):
    apmask=aperture.to_mask(method='center')
    shape=data_cut.shape
    mask=apmask.to_image(shape=((shape[0],shape[1])))
    ap_mask=mask==0
    ap_masked=np.ma.masked_where(ap_mask,data_cut)

    return ap_masked

def measure_aperture_linewidth(fitsfile):
    # read the data cube 
    cube = SpectralCube.read(fitsfile)
    cube = cube.with_spectral_unit(u.km/u.s)

    # Extract the spectrum of the central regions
#    regionfiles = regionDir+'central_1kpc.reg'
    reg_str = 'fk5; circle(0:00:00.000, 0:00:00.00, 1")'
    subcube = cube.subcube_from_ds9region(reg_str)
    spectrum = subcube.sum(axis=(1, 2))
    intensities = spectrum.array
    vels = spectrum.spectral_axis
    vel_mean = np.average(vels, weights=intensities)
    vdep_aperture = np.sqrt(np.sum((vels-vel_mean)**2*intensities)/np.sum(intensities))
    vdep_aperture = vdep_aperture.value

    return vdep_aperture 

def measure_peak_linewidth(mom0file, mom2file):
    # read the moment fits files
    wcs, mom0 = fits_import(mom0file)
    wcs, mom2 = fits_import(mom2file)

    # mask the region outside 1 kpc circle
    position = SkyCoord(ra=0, dec=0, unit='deg')
    central_sky = SkyCircularAperture(position, r=1*u.arcsec)
    central_pix = central_sky.to_pixel(wcs)
    mom0_center = Apmask_convert(central_pix, mom0)
    mom2_center = Apmask_convert(central_pix, mom2)

    # get the velocity dispersion for pixel with peak intensities
    peak_pos = np.where(mom0_center==np.ma.max(mom0_center))
    vdep_peak = mom2_center[peak_pos[0][0],peak_pos[1][0]] 

    return vdep_peak

###########################################################
# main program

incls = ['v0', 'v1']
ids = np.arange(520, 712, 2)

# start = time.time()
# for incl in incls:
#     vdeps_peak = np.array([])
#     vdeps_aperture = np.array([])
#     for idNo in ids:
#         # measure the velocity dispersion of the entire aperture
#         cubefile = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32_ccords.fits'
#         vdeps_aperture = np.append(vdeps_aperture, measure_aperture_linewidth(cubefile))
#         # measure the velocity dispersion of the pixel with peak intensity
#         mom0file = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32_ccords_mom0.fits'
#         mom2file = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32_ccords_mom2.fits'
#         vdeps_peak = np.append(vdeps_peak, measure_peak_linewidth(mom0file,mom2file))
# 
#     outfile = logDir+'G2G3_e2_'+incl+'_central_vdeps.npz'
#     np.savez(outfile, vdep_aperture=vdeps_aperture, vdep_peak=vdeps_peak)

# Extract the velocity spectrum for a given snapshot
incl = 'v0'; idNo = 586
cubefile = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32_ccords.fits'
cube = SpectralCube.read(cubefile)
cube = cube.with_spectral_unit(u.km/u.s)
# Extract the spectrum of the central regions
# regionfiles = regionDir+'central_1kpc.reg'
reg_str = 'fk5; circle(0:00:00.000, 0:00:00.00, 1")'
subcube = cube.subcube_from_ds9region(reg_str)
spectrum_reg = subcube.sum(axis=(1, 2))
spectrum_reg_binned = np.average(spectrum_reg.value.reshape(-1, 10), axis=1)
spectrum_reg.value.dump(logDir+'G2G3_e2_v0_587_central_spectrum.pkl')

# Extract the spectrum for the peak pixel
mom0file = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32_ccords_mom0.fits'
mom2file = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32_ccords_mom2.fits'
wcs, mom0 = fits_import(mom0file)
wcs, mom2 = fits_import(mom2file)
# mask the region outside 1 kpc circle
position = SkyCoord(ra=0, dec=0, unit='deg')
central_sky = SkyCircularAperture(position, r=1*u.arcsec)
central_pix = central_sky.to_pixel(wcs)
mom0_center = Apmask_convert(central_pix, mom0)
mom2_center = Apmask_convert(central_pix, mom2)
peak_pos = np.where(mom0_center==np.ma.max(mom0_center))
spectrum_pix = cube[:, peak_pos[0][0], peak_pos[1][0]]
spectrum_pix_binned = np.average(spectrum_pix.value.reshape(-1, 10), axis=1)
spectrum_pix.value.dump(logDir+'G2G3_e2_v0_587_peak_spectrum.pkl')

# # test the spectrum
# vels = np.arange(-500, 500, 20)
# fig = plt.figure()
# plt.plot(vels, spectrum_reg_binned/np.max(spectrum_reg_binned))
# plt.plot(vels, spectrum_pix_binned/np.max(spectrum_pix_binned))
# plt.show()
