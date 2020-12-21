import astropy
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm
from astropy.table import Table,Column,Row,vstack,setdiff,join

import photutils
from photutils.datasets import make_gaussian_sources_image

import numpy as np
import itertools
import copy

def plant_unit_test(self,accuracy=0.05):
    # unit test
    planthdu = self.plant_fakes_in_sci
    hdu = self.hdu

    fitsflux = np.sum(planthdu.data - hdu.data)
    epsfflux = int(planthdu.header['N_fake'])*float(planthdu.header['f_fake'])
    print(fitsflux,epsfflux)
    if np.abs(fitsflux-epsfflux)/epsfflux < accuracy:
        print("plant was successful")

def has_fakes(self):
    """Check if fake stars have been planted in the image"""
    return self.has_fakes

def plant_fakes(self,epsf_model,locations,writetodisk=False,saveas="planted.fits"):
    """Function for planting fake stars in the diff image.
    """
    # using the ePSF model embedded in the fits file, plant a grid
    # of fakes or plant fakes around galaxies with varying magnitudes
    # (fluxes), mimicking strong-lensing sources

    # write info into the fits header for each planted fake, including
    # the location and total flux

    # optional:  write out the modified image with fakes planted as
    # a new fits file record in the image db that fakes have been
    # planted in the image

    # self is the fakesnplanter object class
    # epsf_model a class with these properties
    hdu = epsf_model.hdu # the fits opened hdu 
    epsf = epsf_model.epsf # the array of data 

    # copying so can leave original data untouched
    cphdu = hdu.copy()
    cpim = cphdu.data
    cphdr = cphdu.header
    
    wcs,frame = WCS(cphdr),cphdr['RADESYS'].lower()
    
    # location should be list of pixels [(x1,y1),(x2,y2)...(xn,yn)]
    for pix in locations:
        pix = list(pix)
        revpix = copy.copy(pix)
        revpix.reverse()
        row,col=revpix
        nrows,ncols=epsf.shape
        # +2 in these to grab a couple more than needed, the correct shapes for broadcasting taken using actual psf.shapes
        rows=np.arange(int(np.round(row-nrows/2)),int(np.round(row+nrows/2))+2) 
        cols=np.arange(int(np.round(col-ncols/2)),int(np.round(col+ncols/2))+2) 
        rows = rows[:epsf.shape[0]]
        cols = cols[:epsf.shape[1]]
        cpim[rows[:, None], cols] += epsf
        np.float64(cpim)
    
    # inserting some new header values
    cphdr['fakeSN']=True 
    #cphdr['fakeSNlocs']=str(locations) this is too long to stick in hdr 
    cphdr['N_fake']=str(len(locations))
    cphdr['f_fake']=str(np.sum(epsf))
    
    if writetodisk:
        fits.writeto(saveas,cpim,cphdr,overwrite=True)
        #cphdu = fits.open(saveas)[0]
    
    self.has_fakes = True # if makes it through this plant_fakes update has_fakes

    return cphdu

def has_lattice(self):
    """Check if a grid of locations exists"""
    return self.has_lattice

def lattice(self):
    """Function for constructing list of pixels in a grid over the image"""

    self.hdu = hdu
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

def has_lco_epsf(self):
    """Check if lco epsf (circular gaussian) exists"""
    return self.has_lco_epsf

def lco_epsf(self):
    """
    Another ePSF option besides building just use circular gaussian from header
    """
    
    hdu = self.hdu

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

    return epsf