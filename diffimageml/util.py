import os 

import astropy
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm
from astropy.table import Table,Column,Row,vstack,setdiff,join

import photutils
from photutils.datasets import make_gaussian_sources_image

import itertools


def pixtosky(self,pixel):
    """
    Given a pixel location returns the skycoord
    """
    hdu = self.sci
    hdr = hdu.header
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
    xp,yp = pixel
    sky = pixel_to_skycoord(xp,yp,wcs)
    return sky

def skytopix(self,sky):
    """
    Given a skycoord (or list of skycoords) returns the pixel locations
    """
    hdu = self.sci
    hdr = hdu.header
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
    pixel = skycoord_to_pixel(sky,wcs)
    return pixel

def cut_hdu(self,location,size,writetodisk=False,saveas=None):
    """
    cutout size lxw ~ (dy x dx) box on fits file centered at a pixel or skycoord location
    if size is scalar gives a square dy=dx 
    updates hdr wcs keeps other info from original
    """
    hdu = self.sci
    cphdu = hdu.copy()
    dat = cphdu.data
    hdr = cphdu.header
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
        
    cut = Cutout2D(dat,location,size,wcs=wcs) 
    cutwcs = cut.wcs
    cphdu.data = cut.data
    cphdu.header.update(cut.wcs.to_header())  
    
    if writetodisk:  
        cphdu.writeto(saveas,overwrite=True)
    
    return cphdu

def get_lattice_positions(self):
    """Function for constructing list of pixels in a grid over the image"""

    hdu = self.diffim.sci
    #reads number of rows/columns from header and creates a grid of locations for planting
    hdr = hdu.header
    wcs,frame=WCS(hdr),hdr['RADESYS'].lower()
    
    NX = hdr['naxis1']
    NY = hdr['naxis2']
    edge = 100 # pixels away from edge
    spacing = 100 # pixels between each location on lattice
    x = list(range(0+edge,NX-edge+1,spacing)) # +1 to make inclusive
    y = list(range(0+edge,NY-edge+1,spacing))
    pixels = list(itertools.product(x, y))
    skycoords = [] # skycoord locations corresponding to the pixels  
    for i in range(len(pixels)):
        pix = pixels[i]
        skycoord=astropy.wcs.utils.pixel_to_skycoord(pix[0],pix[1],wcs)
        skycoords.append(skycoord)

    self.has_lattice = True # if makes it through lattice update has_lattice

    return pixels,skycoords

def lco_epsf(self):
    """
    Another ePSF option besides building just use circular gaussian from lco header on the static sky search im
    At some point may want to have a class like telescope_psf where we can store pre-defined (i.e. not built) epsf
    """
    
    hdu = self.searchim.sci

    # LCO measures PSF stored in header
    # L1FWHM ~ Frame FWHM in arcsec, PIXSCALE ~ arcsec/pixel
    hdr = hdu.header
    l1fwhm = hdr['L1FWHM']
    pixscale = hdr['PIXSCALE']

    sigma = gaussian_fwhm_to_sigma*l1fwhm
    sigma *= 1/pixscale # to pixels
    
    shape = (50,50)
    constant,amplitude,xmean,ymean,xstd,ystd=0,1,shape[0]/2,shape[1]/2,sigma,sigma
    flux = 10**5 # if flux and amplitude present flux is ignored
    table = Table()
    table['constant'] = [constant]
    table['flux'] = [flux]
    table['x_mean'] = [xmean]
    table['y_mean'] = [ymean]
    table['x_stddev'] = [sigma]
    table['y_stddev'] = [sigma]
    epsf = photutils.datasets.make_gaussian_sources_image(shape, table,oversample=1)

    self.has_lco_epsf = True # update bool if makes it through this function
    self.lco_epsf = epsf

    return epsf



