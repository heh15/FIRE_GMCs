from spectral_cube import SpectralCube
from astropy.io import fits
import numpy as np
import casatasks
import os, sys
from astropy.wcs import WCS
import shutil
import time
import astropy.units as u

###########################################################
# Directories

Dir = '/home/heh15/research/FIRE/'
fitsDir = Dir+'e2/'
picDir = Dir+'pictures/'
logDir = Dir+'logs/'

###########################################################
# basic information

incls = ['v0', 'v1', 'v2', 'v3']
pixsize = 100 # in pc
resvel = 2 # in km/s

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

def make_mom0_fits(fitsfile):
    '''
    Make the moment 0 maps
    '''
    # read the fits file
    cube = SpectralCube.read(fitsfile)  
    cube = cube.with_spectral_unit(u.km/u.s)
    mom0_fits = cube.moment(order=0)

    # output moment 0 map to the fits files
    outfile = fitsfile.replace('.fits', '_mom0.fits')
    mom0_fits.write(outfile, format='fits', overwrite=True)

    return outfile

def make_mom2_fits(fitsfile):
    '''
    Make the moment 0 maps
    '''
    # read the fits file
    cube = SpectralCube.read(fitsfile)
    cube = cube.with_spectral_unit(u.km/u.s)
    mom2_fits = cube.linewidth_sigma()

    # output moment 0 map to the fits files
    outfile = fitsfile.replace('.fits', '_mom2.fits')
    mom2_fits.write(outfile, format='fits', overwrite=True)

    return outfile


###########################################################
# main program

incls = ['v0', 'v1', 'v2', 'v3']
ids = np.arange(520, 712, 2)

incls = ['v0', 'v1']

# # create moment maps for mergers during the second passage. 
# start = time.time()
# for incl in incls:
#     for idNo in ids:
#         fitsfile = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32_ccords.fits'
# #         make_mom0_fits(fitsfile)
#         make_mom2_fits(fitsfile)
# stop = time.time()
# count_time(stop, start)

# # create moment 0 maps for star formation maps
# start = time.time()
# for incl in incls:
#     for idNo in ids:
#         fitsfile = fitsDir+'losvd_FIRE2_YoungStars_e2_'+str(idNo)+'_stars_'+str(incl)+'__32_ccords.fits'
#         make_mom0_fits(fitsfile)
# stop = time.time()
# count_time(stop, start)

# create moment maps for isolated galaxies
fitsfile = Dir+'losvd_FIRE2_i3_025_gas_pa30__32_ccords.fits'
make_mom2_fits(fitsfile)
