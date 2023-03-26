import numpy as np
from scipy import ndimage
from spectral_cube import SpectralCube

###########################################################
# directories

Dir = '/home/heh15/research/FIRE/'
fitsDir = Dir+'e2/'
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

###########################################################
# main program

incl = 'v0'
idNo = 520

fitsfile = fitsDir+'losvd_FIRE2_e2_'+str(idNo)+'_gas_'+str(incl)+'__32_ccords.fits'
cube = SpectralCube.read(fitsfile)
data_unmasked = cube.unmasked_data[:,:,:]
data_unmasked = data_unmasked.value
data_unmasked = data_unmasked * 1e10 / 100**2 # convert unit to Msol/pc^2

s = ndimage.generate_binary_structure(3,1)*1.0
labels, num_patches = ndimage.label(data_unmasked,s)

