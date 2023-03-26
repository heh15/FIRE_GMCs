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
# directories

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

###########################################################
# main program

incls = ['v0', 'v1', 'v2', 'v3']
incls = ['v0', 'v1']
ids = np.arange(520, 712, 2)

def add_ccords(fitsfile):
    '''
    add the celestial coordinates to the file
    '''
    # read the fits file
    transfile = fitsfile.replace('.fits', '_imtrans.cube')
    if os.path.exists(transfile):
        shutil.rmtree(transfile)
    casatasks.imtrans(imagename=fitsfile, outfile=transfile, order='120')

    # export the cube to be the fitsfile
    outfile = transfile.replace('_imtrans.cube', '_ccords.fits')
    casatasks.exportfits(imagename=transfile, fitsimage=outfile, overwrite=True)

    # add the celestial coordinates
    with fits.open(outfile) as hdu:
        header = hdu[0].header
        data = hdu[0].data
    # create a celestial coordinate
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'VELOCITY']
    wcs.wcs.cdelt = [0.1, 0.1, 2]
    wcs.wcs.crval = [0, 0, 0]
    wcs.wcs.crpix = [125, 125, 250]
    wcs.wcs.cunit = ['arcsec','arcsec','km/s']
    # add to existing header
    hdr = wcs.to_header()
    for key in hdr.keys():
        header[key] = hdr[key]

    # make the brightness unit to be Msol/pc^2
    data = data*1e10/pixsize**2/resvel

    # write the modified header into the fitsfile
    with fits.open(outfile) as hdu:
        hdu[0].header = header
        hdu[0].data = data
        hdu.writeto(outfile, overwrite=True)

    # remove the intermediate file
    if os.path.exists(transfile):
        shutil.rmtree(transfile)

    return outfile

# start = time.time()
# for incl in incls:
#     for idNo in ids: 
#         fitsfile = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32.fits'
#         outfile = add_ccords(fitsfile)
# stop = time.time()
# count_time(stop, start)

# # Convert isolated galaxies
# fitsfile = Dir+'losvd_FIRE2_i3_025_gas_pa30__32.fits'
# outfile = add_ccords(fitsfile)

# convert the cubes for stars younger than 10 Myr
start = time.time()
for incl in incls:
    for idNo in ids: 
        fitsfile = fitsDir+'losvd_FIRE2_YoungStars_e2_'+str(idNo)+'_stars_'+str(incl)+'__32.fits'
        outfile = add_ccords(fitsfile)
stop = time.time()
count_time(stop, start)

# # test opening it with spectralCube
# cube = SpectralCube.read(outfile)  
# cube = cube.with_spectral_unit(u.km/u.s)
# print(np.nanmax(cube.moment(order=0)))
