import os 
import numpy as np

import astropy
from astropy.wcs import WCS, utils as wcsutils

from astropy.nddata import Cutout2D
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm
from astropy.table import Table,Column,Row,vstack,setdiff,join

import photutils
from photutils.datasets import make_gaussian_sources_image

import itertools

def get_example_data():
    """Returns a dict with the filepath for each of the input images used
    as example data"""
    example_data = {}
    example_data['dir'] = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'test_data'))
    example_data['diffim1'] = os.path.abspath(os.path.join(
        example_data['dir'], 'diff_pydia_1.fits.fz'))
    example_data['fakediffim1'] = os.path.abspath(os.path.join(
        example_data['dir'], 'diff_pydia_1_fakegrid.fits'))
    example_data['searchim1'] = os.path.abspath(os.path.join(
        example_data['dir'], 'sky_image_1.fits.fz'))
    example_data['templateim1'] = os.path.abspath(os.path.join(
        example_data['dir'], 'template_1.fits.fz'))

    example_data['diffim2'] = os.path.abspath(os.path.join(
        example_data['dir'], 'diff_pydia_2.fits.fz'))
    example_data['fakediffim2'] = os.path.abspath(os.path.join(
        example_data['dir'], 'diff_pydia_2_fakegrid.fits'))
    example_data['searchim2'] = os.path.abspath(os.path.join(
        example_data['dir'], 'sky_image_2.fits.fz'))
    example_data['templateim2'] = os.path.abspath(os.path.join(
        example_data['dir'], 'template_2.fits.fz'))

    return example_data


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
    try:
        hdu = self.sci
    except:
        hdu = self

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

    self.postage_stamp = cphdu
    
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

    # want the epsf here to be same shape as build so can compare them
    # TODO once build_epsf is working smoothly in each class rewrite 
    try: 
        self.epsf
        epsf = self.epsf # the build_epsf
        shape = epsf.shape
        oversample = epsf.oversampling
    except:
        shape = (51,51)
        oversample = 2

    # LCO measures PSF stored in header
    # L1FWHM ~ Frame FWHM in arcsec, PIXSCALE ~ arcsec/pixel
    hdr = hdu.header
    l1fwhm = hdr['L1FWHM']
    pixscale = hdr['PIXSCALE']

    sigma = gaussian_fwhm_to_sigma*l1fwhm
    sigma *= 1/pixscale # to pixels
    
    constant,amplitude,xmean,ymean,xstd,ystd=0,1,shape[0]/2,shape[1]/2,sigma,sigma
    flux = 10**5 # if flux and amplitude present flux is ignored
    table = Table()
    table['constant'] = [constant]
    table['flux'] = [flux]
    table['x_mean'] = [xmean]
    table['y_mean'] = [ymean]
    table['x_stddev'] = [sigma]
    table['y_stddev'] = [sigma]
    epsf = photutils.datasets.make_gaussian_sources_image(shape, table,oversample=oversample)

    self.has_lco_epsf = True # update bool if makes it through this function
    self.lco_epsf = epsf

    return epsf

def _extract_psf_fitting_names(psf):
    """
    Determine the names of the x coordinate, y coordinate, and flux from
    a model.  Returns (xname, yname, fluxname)
    """

    if hasattr(psf, 'xname'):
        xname = psf.xname
    elif 'x_0' in psf.param_names:
        xname = 'x_0'
    else:
        raise ValueError('Could not determine x coordinate name for '
                         'psf_photometry.')

    if hasattr(psf, 'yname'):
        yname = psf.yname
    elif 'y_0' in psf.param_names:
        yname = 'y_0'
    else:
        raise ValueError('Could not determine y coordinate name for '
                         'psf_photometry.')

    if hasattr(psf, 'fluxname'):
        fluxname = psf.fluxname
    elif 'flux' in psf.param_names:
        fluxname = 'flux'
    else:
        raise ValueError('Could not determine flux name for psf_photometry.')

    return xname, yname, fluxname

def add_psf(self, psf, posflux, subshape=None,writetodisk=False,saveas="planted.fits"):
    """
    Add (or Subtract) PSF/PRFs from an image.

    Parameters
    ----------
    data : `~astropy.nddata.NDData` or array (must be 2D)
        Image data.
    psf : `astropy.modeling.Fittable2DModel` instance
        PSF/PRF model to be substracted from the data.
    posflux : Array-like of shape (3, N) or `~astropy.table.Table`
        Positions and fluxes for the objects to subtract.  If an array,
        it is interpreted as ``(x, y, flux)``  If a table, the columns
        'x_fit', 'y_fit', and 'flux_fit' must be present.
    subshape : length-2 or None
        The shape of the region around the center of the location to
        subtract the PSF from.  If None, subtract from the whole image.

    Returns
    -------
    subdata : same shape and type as ``data``
        The image with the PSF subtracted
    """

    # copying so can leave original data untouched
    hdu = self.sci
    cphdu = hdu.copy()
    data = cphdu.data
    cphdr = cphdu.header

    wcs,frame = WCS(cphdr),cphdr['RADESYS'].lower()

    if data.ndim != 2:
        raise ValueError(f'{data.ndim}-d array not supported. Only 2-d '
                         'arrays can be passed to subtract_psf.')

    #  translate array input into table
    if hasattr(posflux, 'colnames'):
        if 'x_fit' not in posflux.colnames:
            raise ValueError('Input table does not have x_fit')
        if 'y_fit' not in posflux.colnames:
            raise ValueError('Input table does not have y_fit')
        if 'flux_fit' not in posflux.colnames:
            raise ValueError('Input table does not have flux_fit')
    else:
        posflux = Table(names=['x_fit', 'y_fit', 'flux_fit'], data=posflux)

    # Set up contstants across the loop
    psf = psf.copy()
    xname, yname, fluxname = _extract_psf_fitting_names(psf)
    indices = np.indices(data.shape)
    subbeddata = data.copy()
    addeddata = data.copy()
    
    n = 0
    if subshape is None:
        indicies_reversed = indices[::-1]

        for row in posflux:
            getattr(psf, xname).value = row['x_fit']
            getattr(psf, yname).value = row['y_fit']
            getattr(psf, fluxname).value = row['flux_fit']

            xp,yp,flux_fit = row['x_fit'],row['y_fit'],row['flux_fit']
            sky = wcsutils.pixel_to_skycoord(xp,yp,wcs)
            idx = str(n).zfill(3) 
            cphdr['FK{}X'.format(idx)] = xp
            cphdr['FK{}Y'.format(idx)] = yp
            cphdr['FK{}RA'.format(idx)] = str(sky.ra.hms)
            cphdr['FK{}DEC'.format(idx)] = str(sky.dec.dms)
            cphdr['FK{}F'.format(idx)] = flux_fit
            # TO-DO, once have actual epsf classes will be clearer to fill the model
            cphdr['FK{}MOD'.format(idx)] = "NA"
            n += 1

            subbeddata -= psf(*indicies_reversed)
            addeddata += psf(*indicies_reversed)
    else:
        for row in posflux:
            x_0, y_0 = row['x_fit'], row['y_fit']

            # float dtype needed for fill_value=np.nan
            y = extract_array(indices[0].astype(float), subshape, (y_0, x_0))
            x = extract_array(indices[1].astype(float), subshape, (y_0, x_0))

            getattr(psf, xname).value = x_0
            getattr(psf, yname).value = y_0
            getattr(psf, fluxname).value = row['flux_fit']

            xp,yp,flux_fit = row['x_fit'],row['y_fit'],row['flux_fit']
            sky = wcsutils.pixel_to_skycoord(xp,yp,wcs)
            idx = str(n).zfill(3) 
            cphdr['FK{}X'.format(idx)] = xp
            cphdr['FK{}Y'.format(idx)] = yp
            cphdr['FK{}RA'.format(idx)] = str(sky.ra.hms)
            cphdr['FK{}DEC'.format(idx)] = str(sky.dec.dms)
            cphdr['FK{}F'.format(idx)] = flux_fit
            # TO-DO, once have actual epsf classes will be clearer to fill the model
            cphdr['FK{}MOD'.format(idx)] = "NA"
            n += 1
            
            subbeddata = add_array(subbeddata, -psf(x, y), (y_0, x_0))
            addeddata = add_array(addeddata, psf(x, y), (y_0, x_0))
    
    # the copied hdu written/returned should have data with the added psfs 
    cphdu.data = addeddata
    # inserting some new header values
    cphdr['fakeSN']=True 
    cphdr['N_fake']=str(len(posflux))
    cphdr['F_epsf']=str(psf.flux)
    
    if writetodisk:
        fits.writeto(saveas,cphdu.data,cphdr,overwrite=True)
    
    self.plants = [cphdu,posflux]
    self.has_fakes = True # if makes it through this plant_fakes update has_fakes

    return cphdu

def model2dG_build(self):
    """Function to fit a 2d-Gaussian to the built epsf
    """
    
    # TODO once build_epsf is working smoothly in each class rewrite 
    try: 
        self.epsf
        epsf = self.epsf # the build_epsf
    except:
        epsf = self

    # use photutils 2d gaussian fit on the built epsf
    # TODO give option to restrict fit params, force xmean,ymean to be the ctr, constant to be zero
    gaussian = photutils.centroids.fit_2dgaussian(epsf.data)
    print(gaussian.param_names,gaussian.parameters)
    # unpack the parameters of fit
    constant,amplitude,x_mean,y_mean,x_stddev,y_stddev,theta=gaussian.parameters
    # Theta is in degrees, rotating the sigma_x,sigma_y ccw from +x 
    
    # Put fit values into table 
    # TODO the build_epsf is oversampled with respect to the fits image class data
    # ie the x_stddev, y_stddev are scaled by the oversampling
    # I think what makes the most sense is to rescale the build_epsf array, I'm not clear on how to do that
    table = Table()
    table['constant'] = [constant]
    table['amplitude'] = [amplitude]
    #table['flux'] = [flux] # if flux and amplitude flux is ignored
    table['x_mean'] = [x_mean]
    table['y_mean'] = [y_mean]
    table['x_stddev'] = x_stddev #[x_stddev/epsf.oversampling[0]]
    table['y_stddev'] = y_stddev #[y_stddev/epsf.oversampling[1]]
    # theta is a ccw rotation from +x in deg
    # ie the 2dgaussian grabs hold of a sigma_x sigma_y and then rotated
    table['theta'] = [theta]
    
    # get epsf of the model fit     
    shape=epsf.shape
    modeled_epsf = make_gaussian_sources_image(shape, table)
    resid = modeled_epsf.data - epsf.data
    
    return gaussian,table,modeled_epsf


