import requests

import random
import sys
import glob
import os
from optparse import OptionParser
parser = OptionParser()
(options,args)=parser.parse_args()
import shutil

import copy
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle
from matplotlib.colors import BoundaryNorm
import numpy as np
import itertools
import collections 
from scipy.optimize import curve_fit

import astropy
from astropy.io import ascii,fits
from astropy.table import vstack,Table,Column,Row,setdiff,join
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.units import Quantity
from astroquery.gaia import Gaia
from astropy.convolution import Gaussian2DKernel
from astropy.visualization import ZScaleInterval,simple_norm
zscale = ZScaleInterval()
from astropy.nddata import Cutout2D,NDData
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

import photutils
from photutils import find_peaks
from photutils.psf import extract_stars
from photutils import EPSFBuilder
from photutils import detect_threshold
from photutils import detect_sources
from photutils import deblend_sources
from photutils import source_properties, EllipticalAperture
from photutils import BoundingBox
from photutils import Background2D, MedianBackground

import pandas as pd

import lco_figures

# Suppress warnings. Relevant for astroquery. Comment this out if you wish to see the warning messages
import warnings
warnings.filterwarnings('ignore')

def get_data(path):
    my_data = {} 
    """
    # shouldn't be any images directly in the folder path to field this is used for
    # should be tucked in dirs like dia_out, dia_trim, source_im
    ims=glob.glob('*fits')
    for i in range(len(ims)):
        filename = ims[i].split('/')[-1]
        my_data[filename] = fits.open(ims[i])[0]
    """
    # this source im is a dir that you should make for each field with the image you want to use
    # ie will be the one that psf measured on and planting into
    SOURCE_IM=glob.glob(os.path.join(path,'source_im/*fits'))
    for i in range(len(SOURCE_IM)):
        filename=SOURCE_IM[i].split('/')[-1]
        my_data[filename] = fits.open(SOURCE_IM[i])[0]
    DIA_OUT=glob.glob(os.path.join(path,'dia_out/*'))
    for i in range(len(DIA_OUT)):
        filename = DIA_OUT[i].split('/')[-1]
        my_data[filename] = fits.open(DIA_OUT[i])[0]
    DIA_TRIM=glob.glob(os.path.join(path,'dia_trim/*'))
    for i in range(len(DIA_TRIM)):
        filename = DIA_TRIM[i].split('/')[-1]
        my_data[filename] = fits.open(DIA_TRIM[i])[0]
    #print(my_data)
    return my_data

def gaia_results(image):
    # image wcs and frame, for conversions pixels/skycoord
    wcs,frame=WCS(image.header),image.header['RADESYS'].lower()
    # coord of strong lensing galaxy
    ra=image.header['CAT-RA']
    dec=image.header['CAT-DEC']
    coord = SkyCoord(ra,dec,unit=(u.hourangle,u.deg))

    # the pixscale is same along x&y and rotated to default (N up, E left) cdi_j ~ delta_ij
    cdi_i = image.header['CD1_1'] # deg/pixel
    naxis = image.header['NAXIS1'] # naxis1=naxis2
    radius = 3600*cdi_i*naxis/2 # approx 800 arcsec entire image
    radius*=.75 # do 3/4 of that
    # do the search
    r = Gaia.query_object_async(coordinate=coord, radius=radius*u.arcsec)
    return r,image

def stars2(results,Nbrightest=None):
    # stars are extracted from image to be ready for use in determine ePSF
    # note ref.fits doesn't have saturate and maxlin available the image should be just one of the trims

    # unpack gaia_results into the gaia catalog and image
    r,image = results
    # need to give the results a column name x and y (pixel locations on image) for extract stars fcn which am going to apply
    wcs,frame=WCS(image.header),image.header['RADESYS'].lower()
    positions,pixels=[],[]
    for i in range(len(r)):
        position=SkyCoord(ra=r[i]['ra'],dec=r[i]['dec'],unit=u.deg,frame=frame)
        positions.append(position)
        pixel=skycoord_to_pixel(position,wcs)
        pixels.append(pixel)
    x,y=[i[0] for i in pixels],[i[1] for i in pixels]
    x,y=Column(x),Column(y)
    r.add_column(x,name='x')
    r.add_column(y,name='y')
    print('there are {} stars available within fov from gaia results queried'.format(len(r)))

    # I am finding bboxes of the extractions I will do so I can remove any stars with overlaps 
    # I want to extract all the stars wo overlaps before I start to remove any using photometry constraints to get 'good' ones for psf
    bboxes = []
    for i in r:
        x = i['x']
        y = i['y']
        size = 25
        ixmin,ixmax = int(x - size/2), int(x + size/2)
        iymin, iymax = int(y - size/2), int(y + size/2)
        
        bbox = BoundingBox(ixmin=ixmin, ixmax=ixmax, iymin=iymin, iymax=iymax)
        bboxes.append(bbox)
    bboxes = Column(bboxes)
    r.add_column(bboxes,name='bbox')
    
    # using the bbox of each star from results to determine intersections, dont want confusion of multi-stars for ePSF
    intersections = []
    for i,obj1 in enumerate(bboxes):
        for j in range(i+1,len(bboxes)):
            obj2 = bboxes[j]
            if obj1.intersection(obj2):
                #print(obj1,obj2)
                # these are the ones to remove 
                intersections.append(obj1) 
                intersections.append(obj2)
    # use the intersections found to remove stars
    j=0
    rows=[]
    for i in r:
        if i['bbox'] in intersections:
            #tmp.remove(i)
            row=j
            rows.append(row)
        j+=1
    r.remove_rows(rows)
    print('{} stars, after removing intersections'.format(len(r)))

    # I am going to extract stars with strong signal in rp filter (the one lco is looking in)
    r = r[r['phot_rp_mean_flux_over_error']>100]
    print('restricting extractions to stars w rp flux/error > 100 we have {} to consider'.format(len(r)))

    data = image.data
    hdr = image.header
    # the header has L1 bkg values; should be the same as sigma clipped stats 
    L1mean,L1med,L1sigma,L1fwhm = hdr['L1MEAN'],hdr['L1MEDIAN'],hdr['L1SIGMA'],hdr['L1FWHM'] # counts, fwhm in arcsec 
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)  
    WMSSKYBR = hdr['WMSSKYBR'] # mag/arcsec^2 of sky bkg measured
    
    # AGGMAG the guide star magnitude header value would be simpler but it is given as unknown, ra/dec are provided for it though
    # grab some other useful header values now
    pixscale,saturate,maxlin = hdr['PIXSCALE'],hdr['SATURATE'],hdr['MAXLIN'] # arcsec/pixel, counts for saturation and non-linearity levels
    
    # need bkg subtracted to extract stars, want to build ePSF using just star brightness 
    data -= median_val # L1med
    nddata = NDData(data=data)
    stars = extract_stars(nddata,catalogs=r, size=25)
    tmp = [i for i in stars] # get a list of stars rather than single photutils obj with all of them 
    print('kfo len tmp {}, len r {}'.format(len(tmp),len(r)))

    """
    # using the bbox of each star from results to determine intersections, dont want confusion of multi-stars for ePSF
    # this was done with all stars not just those extracted, this is an optional sanity check but don't need it
    intersections = []
    for i,obj1 in enumerate(stars.bbox):
        for j in range(i+1,len(stars.bbox)):
            obj2 = stars.bbox[j]
            if obj1.intersection(obj2):
                #print(obj1,obj2)
                # these are the ones to remove 
                intersections.append(obj1) 
                intersections.append(obj2)
    # use the intersections found to remove stars
    
    for i in tmp:
        if i.bbox in intersections:
            tmp.remove(i)
    #print('{} stars, after removing intersections'.format(len(tmp)))
    """

    j=0
    rows=[]
    for i in range(len(r)):
        tmpi = tmp[i] # the ith photutils extract star
        ri = r[i]
        if np.max(tmpi.data) > saturate:
            tmp.remove(tmpi)
            rows.append(i)
        elif np.max(tmpi.data) > maxlin:
            tmp.remove(tmpi)
            rows.append(i)
    r.remove(rows)

    print('removed stars above saturation or non-linearity level ~ {}, {} ADU; now have {} = {}'.format(saturate,maxlin,len(tmp),len(r)))
    good_stars = photutils.psf.EPSFStars(tmp)
    
    
    """
    # you should look at the images to make sure these are good stars
    nrows = 4
    ncols = 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                            squeeze=True)
    ax = ax.ravel()
    for i in range(len(brightest_results)):
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
    """
    #return stars
    return good_stars,r,image

def stars(results,Nbrightest=None):
    # stars are extracted from image to be ready for use in determine ePSF
    # note ref.fits doesn't have saturate and maxlin available the image should be just one of the trims

    # unpack gaia_results into the gaia catalog and image
    r,image = results
    # need to give the results a column name x and y (pixel locations on image) for extract stars fcn which am going to apply
    wcs,frame=WCS(image.header),image.header['RADESYS'].lower()
    positions,pixels=[],[]
    for i in range(len(r)):
        position=SkyCoord(ra=r[i]['ra'],dec=r[i]['dec'],unit=u.deg,frame=frame)
        positions.append(position)
        pixel=skycoord_to_pixel(position,wcs)
        pixels.append(pixel)
    x,y=[i[0] for i in pixels],[i[1] for i in pixels]
    x,y=Column(x),Column(y)
    r.add_column(x,name='x')
    r.add_column(y,name='y')
    print('there are {} stars available within fov from gaia results queried'.format(len(r)))

    # I am finding bboxes of the extractions I will do so I can remove any stars with overlaps 
    # I want to extract all the stars wo overlaps before I start to remove any using photometry constraints to get 'good' ones for psf
    bboxes = []
    for i in r:
        x = i['x']
        y = i['y']
        size = 25
        ixmin,ixmax = int(x - size/2), int(x + size/2)
        iymin, iymax = int(y - size/2), int(y + size/2)
        
        bbox = BoundingBox(ixmin=ixmin, ixmax=ixmax, iymin=iymin, iymax=iymax)
        bboxes.append(bbox)
    bboxes = Column(bboxes)
    r.add_column(bboxes,name='bbox')
    # using the bbox of each star from results to determine intersections, dont want confusion of multi-stars for ePSF
    intersections = []
    for i,obj1 in enumerate(bboxes):
        for j in range(i+1,len(bboxes)):
            obj2 = bboxes[j]
            if obj1.intersection(obj2):
                #print(obj1,obj2)
                # these are the ones to remove 
                intersections.append(obj1) 
                intersections.append(obj2)
    # use the intersections found to remove stars
    j=0
    rows=[]
    for i in r:
        if i['bbox'] in intersections:
            #tmp.remove(i)
            row=j
            rows.append(row)
        j+=1
    r.remove_rows(rows)
    print('{} stars, after removing intersections'.format(len(r)))

    # I am going to extract stars with strong signal in rp filter (the one lco is looking in)
    r = r[r['phot_rp_mean_flux_over_error']>100]
    print('restricting extractions to stars w rp flux/error > 100 we have {} to consider'.format(len(r)))

    # sort by the strongest signal/noise in r' filter
    # r.sort('phot_rp_mean_flux_over_error')
    """
    # don't think it will be necessary to limit to some N stars, might as well take all that will give good data for building psf
    if Nbrightest == None:
        Nbrightest = len(r)
    brightest_results = r[:Nbrightest]
    """

    data = image.data
    hdr = image.header
    # the header has L1 bkg values; should be the same as sigma clipped stats 
    L1mean,L1med,L1sigma,L1fwhm = hdr['L1MEAN'],hdr['L1MEDIAN'],hdr['L1SIGMA'],hdr['L1FWHM'] # counts, fwhm in arcsec 
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)  
    WMSSKYBR = hdr['WMSSKYBR'] # mag/arcsec^2 of sky bkg measured
    # AGGMAG the guide star magnitude header value would be simpler but it is given as unknown, ra/dec are provided for it though
    # grab some other useful header values now
    pixscale,saturate,maxlin = hdr['PIXSCALE'],hdr['SATURATE'],hdr['MAXLIN'] # arcsec/pixel, counts for saturation and non-linearity levels
    
    # need bkg subtracted to extract stars, want to build ePSF using just star brightness 
    data -= median_val # L1med
    nddata = NDData(data=data)
    stars = extract_stars(nddata,catalogs=r, size=25)
    # using the bbox of each star from results to determine intersections, dont want confusion of multi-stars for ePSF
    # this was done with all stars not just those extracted, this is an optional sanity check but don't need it
    intersections = []
    for i,obj1 in enumerate(stars.bbox):
        for j in range(i+1,len(stars.bbox)):
            obj2 = stars.bbox[j]
            if obj1.intersection(obj2):
                #print(obj1,obj2)
                # these are the ones to remove 
                intersections.append(obj1) 
                intersections.append(obj2)
    # use the intersections found to remove stars
    tmp = [i for i in stars] # get a list of stars rather than single photutils obj with all of them 
    for i in tmp:
        if i.bbox in intersections:
            tmp.remove(i)
    #print('{} stars, after removing intersections'.format(len(tmp)))
    

    # note ref.fits doesn't have saturate and maxlin available the image should be just one of the trims
    for i in tmp:
        if np.max(i.data) > saturate:
            tmp.remove(i)
        elif np.max(i.data) > maxlin:
            tmp.remove(i)

    print('removed stars above saturation or non-linearity level ~ {}, {} ADU; now have {}'.format(saturate,maxlin,len(tmp)))
    good_stars = photutils.psf.EPSFStars(tmp)
    
    """
    # you should look at the images to make sure these are good stars
    nrows = 4
    ncols = 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                            squeeze=True)
    ax = ax.ravel()
    for i in range(len(brightest_results)):
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
    """
    #return stars
    return good_stars,image

def ePSF(stars,name='psf.fits',oversampling=2):
    # using all the available Gaia results which are below non-linearity/saturation to build an effective PSF 

    # unpack the stars results into the good_stars and image 
    good_stars, image = stars

    hdr = image.header
    L1mean,L1med,L1sigma,L1fwhm = hdr['L1MEAN'],hdr['L1MEDIAN'],hdr['L1SIGMA'],hdr['L1FWHM'] # counts, fwhm in arcsec 
    pixscale,saturate,maxlin = hdr['PIXSCALE'],hdr['SATURATE'],hdr['MAXLIN'] # arcsec/pixel, counts for saturation and non-linearity levels

    # oversampling chops pixels of each star up further to get better fit
    # this is okay since stacking multiple ...
    # however more oversampled the ePSF is, the more stars you need to get smooth result
    # LCO is already oversampling the PSFs, the fwhm ~ 2 arcsec while pixscale ~ 0.4 arcsec; should be able to get good ePSF measurement without any oversampling
    # ePSF basic x,y,sigma 3 param model should be easily obtained if consider that 3*pixscale < fwhm
    epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=10,
                                progress_bar=True)  
    epsf, fitted_stars = epsf_builder(good_stars)  
    """
    # take a look at the ePSF image 
    norm = simple_norm(epsf.data, 'log', percent=99.)
    plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
    plt.colorbar()
    """
    #fits.writeto(name,epsf.data,hdr,overwrite=True)
    #         fits.writeto(plantname,image.data,hdr,overwrite=True)

    return epsf, fitted_stars
from photutils.datasets import make_gaussian_sources_image


def stock_epsf(l1fwhm,pixscale=0.389):
    # use the available psf in header l1fwhm in pixels to have a fall back if my epsf fails, as well as to compare
    #fwhm,pixscale = img.header['L1FWHM'],img.header['pixscale'] # [arcsec] Frame FWHM in arcsec, arcsec/pixel
    sigma = gaussian_fwhm_to_sigma*l1fwhm
    sigma *= 1/pixscale # to pixels
    constant,amplitude,xmean,ymean,xstd,ystd=0,1,25,25,sigma,sigma
    flux = 100 # if flux and amplitude present flux is ignored
    table = Table()
    table['constant'] = [constant]
    #table['amplitude'] = [amplitude]
    #flux = np.max(epsf)*(2*np.pi*sigma*sigma) = amplitude*(2pi*sigma^2)
    table['flux'] = [flux]
    table['x_mean'] = [xmean]
    table['y_mean'] = [ymean]
    table['x_stddev'] = [sigma]
    table['y_stddev'] = [sigma]
    shape = (50,50) # in epsf, 25x25 box shape extracted but oversampling of 2 makes it 50x50
    epsf = photutils.datasets.make_gaussian_sources_image(shape, table,oversample=1)
    return epsf 


def gaussian2d(epsf,hdr=None):
    # use photutils 2d gaussian fit on the epsf
    gaussian = photutils.centroids.fit_2dgaussian(epsf.data)
    #print('gaussian fit to epsf (the epsf oversampled by two so these fit sigmas --> /2 will be used for the image):')
    #print(gaussian.param_names,gaussian.parameters)
    # unpack the parameters of fit
    constant,amplitude,x_mean,y_mean,x_stddev,y_stddev,theta=gaussian.parameters
    # may want to force xmean,ymean to be the ctr, constant to be zero, theta to be zero
    
    sigma = (abs(x_stddev)+abs(y_stddev))/2 # the average of x and y 
    sigma*=1/epsf.oversampling[0] # the model fit was to oversampled pixels need to correct for that for true image pix res
    fwhm = gaussian_sigma_to_fwhm*sigma

    # here I take values of evaluated model fit along center of image
    # these might be useful to show
    xctr_vals = []
    y=0
    for i in range(epsf.shape[1]):
        gaussval = gaussian.evaluate(x_mean,y,constant,amplitude,x_mean,y_mean,x_stddev,y_stddev,theta)
        xctr_vals.append(gaussval)
        y+=1
    yctr_vals = []
    x=0
    for i in range(epsf.shape[0]):
        gaussval = gaussian.evaluate(x,y_mean,constant,amplitude,x_mean,y_mean,x_stddev,y_stddev,theta)
        yctr_vals.append(gaussval)
        x+=1
    
    # here I am using the stddev in epsf to define levels n*sigma below the amplitude of the fit
    # is useful for contours
    #np.mean(psf.data),np.max(psf.data),np.min(psf.data),med=np.median(psf.data)
    std=np.std(epsf.data)
    levels=[amplitude-3*std,amplitude-2*std,amplitude-std]
    #plt.contour(psf.data,levels=levels)
    
    # may want to force xmean,ymean to be the ctr, constant to be zero, theta to be zero
    table = Table()
    table['constant'] = [constant]
    #table['amplitude'] = [amplitude]
    table['flux'] = [100] # if flux and amplitude flux is ignored, flux = amplitude*(2piSigma^2)
    table['x_mean'] = [x_mean]
    table['y_mean'] = [y_mean]
    table['x_stddev'] = [x_stddev/epsf.oversampling[0]]
    table['y_stddev'] = [y_stddev/epsf.oversampling[0]]
    table['theta'] = np.radians(np.array([theta]))
    
    shape=epsf.shape
    # making numpy array of model values in shape of epsf
    image1 = make_gaussian_sources_image(shape, table)
    image1 = image1.data
    resid = image1.data - epsf.data
    # turning model array into epsf obj for easy manipulation with the epsf
    #img_epsf = photutils.psf.EPSFStar(image1.data,cutout_center=(x_mean,y_mean))
    # for example the residual of gaussian model with the epsf...
    #resid = img_epsf.compute_residual_image(epsf)
    # idk what's happening with compute_residual_image but it isn't straight-forward subtraction of img_epsf - epsf
    # some parameter about the scale for registered epsf is being used, where it assumes img_epsf is a star, I really just can use a straight sub of the gauss model fit - epsf
    #resid = img_epsf.data - epsf.data 
    return gaussian,table,levels,xctr_vals,yctr_vals,image1,resid

def cut_target(file,saveas=None):
    """
    file should be fits.open("filename.fits")
    uses a header ra/dec to cut an image around target, in this case the lensing galaxy
    """

    hdu = file # where I'm going to overwrite the cutout data/hdr onto so it has same class for fits writeto
    dat = file.data
    hdr = file.header
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
    
    ra,dec=hdr['CAT-RA'],hdr['CAT-DEC']
    lensing_gal = SkyCoord(ra,dec,unit=(u.hourangle,u.deg))
    size = 50 # pixels
    
    cut = Cutout2D(dat,lensing_gal,size,wcs=wcs)
    cutwcs = cut.wcs
    if saveas:
        #pickle.dump(cut,open(saveas,"wb"))
        hdu.data = cut.data
        hdu.header.update(cut.wcs.to_header())
        hdu.writeto(saveas,overwrite=True)
    
    return hdu

def cut_plant(hdu,epsf,saveas=None):
    """
    create fits with planting SN into the cutout around lens galaxy
    """
    
    hdr = hdu.header
    ctrx,ctry = hdr['NAXIS1']/2, hdr['NAXIS2']/2
    ctrx,ctry = hdu.data.shape[0]/2, hdu.data.shape[0]/2
    locations = [(ctrx,ctry)]
    #epsf = scale(epsf,hdr,m50)
    
    a,b = plant2(hdu,hdr,epsf,locations,name=saveas)
    return

def cutfig(cut,saveas=None):
    wcs = cut.wcs
    fig, ax = plt.subplots(figsize=(6, 6),subplot_kw={'projection': wcs})
    matplotlib.rcParams.update({'font.size': 15,'xtick.labelsize':10,'ytick.labelsize':10})
    ax.imshow(zscale(cut.data))
    if saveas:
        plt.savefig(saveas,bbox_inches="tight")
        
    return


def phot(epsf,hdr):
    skybr=hdr['WMSSKYBR'] # mag/arcsec^2
    l1med=hdr['L1MEDIAN'] # median sky pixel value
    l1fwhm=hdr['L1FWHM'] # PSF FWHM arcsec
    pixscale=hdr['PIXSCALE'] # arcsec/pixel
    
    # [mag/arcsec^2] wmsskybr = -2.5log10(l1median/pixscale [value/pixel]/[arcsec/pixel]) + zp
    zp = skybr + 2.5*np.log10(l1med/pixscale/pixscale)
    
    """
    # kfo this is wrong
    # flux value from the PSF of symm 2d Gauss using l1fwhm; f ~ int dr dphi Ae^[(-r^2)/2sigma^2] = 2pi A sqrt[pi/2]/sigma
    A = np.max(epsf) # value
    sigma = gaussian_fwhm_to_sigma*(l1fwhm/pixscale) # pixel
    f = 2*np.pi*A*np.sqrt(np.pi/2)/sigma # value
    """
    f = np.sum(epsf)

    m = -2.5*np.log10(f) + zp
    
    return m

def analytic_flux(epsf,hdr):
    # flux value from the PSF of symm 2d Gauss using l1fwhm; f ~ int dr dphi Ae^[(-r^2)/2sigma^2] = 2pi A sqrt[pi/2]/sigma
    l1fwhm=hdr['L1FWHM'] # PSF FWHM arcsec
    pixscale=hdr['PIXSCALE'] # arcsec/pixel
    A = np.max(epsf) # value
    sigma = gaussian_fwhm_to_sigma*(l1fwhm/pixscale) # pixel
    f = 2*np.pi*A*np.sqrt(np.pi/2)/sigma # value
    return f

def sumepsfarray_flux(epsf):
    f = np.sum(epsf)
    return f

def threshold_amplitude(threshold,epsf):
    A = np.max(epsf) # value
    tmp = copy.copy(epsf)
    tmp *= threshold/A
    return tmp

def scale(epsf,hdr,mag):
    m = phot(epsf,hdr)
    # scale m to mag using mu: mag = -2.5log10(mu*f) + zp --> mu = 10^[(mag - m)/-2.5]
    mu = 10**((mag-m)/-2.5)
    scaled = epsf*mu
    return scaled

def plant2(image,hdr,epsf,locations,name='planted.fits'):
    """
    Add EPSF to fits difference image to represent SN
    """
    
    fakeSNmag = phot(epsf,hdr)
    
    # copying image so can leave original data untouched
    cpim = copy.copy(image.data)
    cphdr = copy.copy(hdr)
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
    
    # location should be list of pixels [(x1,y1),(x2,y2)...(xn,yn)]
    for pix in locations:
        pix = list(pix)
        revpix = copy.copy(pix)
        revpix.reverse()
        # index the image
        row,col=revpix
        nrows,ncols=epsf.shape
        # +2 in these to grab a couple more than needed, the correct shapes for broadcasting taken using actual psf.shapes
        rows=np.arange(int(np.round(row-nrows/2)),int(np.round(row+nrows/2))+2) 
        cols=np.arange(int(np.round(col-ncols/2)),int(np.round(col+ncols/2))+2) 
        rows = rows[:epsf.shape[0]]
        cols = cols[:epsf.shape[1]]
        cpim[rows[:, None], cols] += epsf
        np.float64(cpim)
    # inserting some new header values and writing to a fits image 
    cphdr['fakeSN']=True 
    #cphdr['fakeSNlocs']=str(locations) this is too long to stick in hdr 
    cphdr['NfakeSNe']=str(len(locations))
    cphdr['fakeSNmag'] = str(fakeSNmag)
    fits.writeto(name,cpim,cphdr,overwrite=True)
    plant_im = fits.open(name)[0]
    return plant_im,locations

def source_cat(image,nsigma=2,kernel_size=(3,3),npixels=5,deblend=False,contrast=.001,targ_coord=None):
    """
    the image should be fits.open('trim.fits'), is trimmed/aligned properly w reference/differences
    for some reason reference doesn't have the catalog ra of the target strong lensing galaxy in header
    will get a cat of properties for detected sources
    """
    # to be able to translate from ra/dec <--> pixels on image
    wcs,frame = WCS(image.header),image.header['RADESYS'].lower()
    hdr = image.header
    #L1mean,L1med,L1sigma,L1fwhm = hdr['L1MEAN'],hdr['L1MEDIAN'],hdr['L1SIGMA'],hdr['L1FWHM'] # counts, fwhm in arcsec 
    #pixscale,saturate,maxlin = hdr['PIXSCALE'],hdr['SATURATE'],hdr['MAXLIN'] # arcsec/pixel, counts for saturation and non-linearity levels

    # detect threshold uses sigma clipped statistics to get bkg flux and set a threshold for detected sources as objs above nsigma*bkg
    # bkg also available in the hdr of file, either way is fine  
    threshold = detect_threshold(image.data, nsigma=nsigma)
    sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3. pixels for kernel smoothing
    # optional ~ kernel smooths the image, using gaussian weighting with pixel size of 3
    kernel = Gaussian2DKernel(sigma, x_size=kernel_size[0], y_size=kernel_size[1])
    kernel.normalize()
    # make a segmentation map, id sources defined as n connected pixels above threshold (n*sigma + bkg)
    segm = detect_sources(image.data,
                          threshold, npixels=npixels, filter_kernel=kernel)
    # deblend useful for very crowded image with many overlapping objects...
    # uses multi-level threshold and watershed segmentation to sep local peaks as ind obj
    # use the same number of pixels and filter as was used on original segmentation
    # contrast is fraction of source flux local pk has to be consider its own obj
    if deblend:
        segm = deblend_sources(image.data, 
                                       segm, npixels=5,filter_kernel=kernel, 
                                       nlevels=32,contrast=contrast)

    # need bkg subtracted to do photometry using source properties
    boxsize=100
    bkg = Background2D(image.data,boxsize) # sigma-clip stats for background est over image on boxsize, regions interpolated to give final map 
    data_bkgsub = image.data - bkg.background
    cat = source_properties(data_bkgsub, segm,background=bkg.background,
                            error=None,filter_kernel=kernel)
    
    # going to id the target lensing galaxy from source catalog
    # since this is ideal detection location where strong lens could provide multi-im
    # this is going to be area where we will most want to plant and study 
    
    #CAT-RA  = 'blah'       / [HH:MM:SS.sss] Catalog RA of the object        
    #CAT-DEC = 'blah'       / [sDD:MM:SS.ss] Catalog Dec of the object
    if targ_coord == None:
        # the source images all have cat-ra cat-dec, will default grab target galaxy location from hdr
        ra = image.header['CAT-RA']
        dec = image.header['CAT-DEC']
    else:
        # if using the ref to detect source objs the target stuff isn't in there will need to provide tuple taken from source hdr 
        ra,dec = targ_coord # unpack

    lensing_gal = SkyCoord(ra,dec,unit=(u.hourangle,u.deg))
    pix_gal = astropy.wcs.utils.skycoord_to_pixel(lensing_gal,wcs)

    # TODO all sources of error including poisson from sources
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['cxx'].info.format = '.2f'
    tbl['cxy'].info.format = '.2f'
    tbl['cyy'].info.format = '.2f'
    tbl['gini'].info.format = '.2f'

    # going to add a column of surface brightness so we can plant into the obj shapes according to those
    surf_brightnesses = []
    for obj in tbl:
        unit = 1/obj['area'].unit
        surf_bright = obj['source_sum']/obj['area'].value # flux/pix^2
        surf_brightnesses.append(surf_bright) 
    surf_brightnesses = Column(surf_brightnesses,name='surface_brightness',unit=unit)
    tbl.add_column(surf_brightnesses)

    # take a look at the brightest or most elliptical objs from phot on segm objs detected
    tbl.sort('ellipticity') #
    elliptical=tbl[-10:]
    #tbl.sort('source_sum') ('surface_brightness') 

    # there is definitely a neater/cuter way to index table than this using loc to find obj of gal 
    tmp = tbl[tbl['xcentroid'].value > pix_gal[0]-10]
    tmp = tmp[tmp['xcentroid'].value < pix_gal[0]+10]
    tmp = tmp[tmp['ycentroid'].value > pix_gal[1]-10]
    targ_obj = tmp[tmp['ycentroid'].value < pix_gal[1]+10] 
    targ_sb = targ_obj['source_sum']/targ_obj['area']
    
    return cat,image,threshold,segm,targ_obj

def targ_lattice(cut_image,hdr):
    wcs,frame=WCS(hdr),hdr['RADESYS'].lower()
    NX = cut_image.shape[0]
    NY = cut_image.shape[1]
    xorigin,yorigin = cut_image.origin_original
    edge = 50 # pixels away from edge
    spacing = 100 # pixels between each location on lattice
    x = list(range(xorigin+edge,xorigin+NX-edge+1,spacing)) # +1 to make inclusive
    y = list(range(yorigin+edge,yorigin+NY-edge+1,spacing))
    pixels = list(itertools.product(x, y))
    locations = [] # skycoord locations that I will use to plant SNe across image  
    for i in range(len(pixels)):
        pix = pixels[i]
        location=astropy.wcs.utils.pixel_to_skycoord(pix[0],pix[1],wcs)
        locations.append(location)
    return locations,pixels

def target(image,targ_obj,ref=None,diff=None):
    if len(targ_obj) == 0:
        # the target isnt detected in image by my source_cat (photutils)...likely bad skymag bkg
        # going to use ref.fits to cut around the target and extract the parameters from these
        print('the target obj wasnt detected in source image, using ref image to get the target photutil params')
        #ref = my_data['ref.fits']
        wcs,frame = WCS(image.header),image.header['RADESYS'].lower()
        # the target strong lensing galaxy position
        ra=image.header['CAT-RA']
        dec=image.header['CAT-DEC']
        targ_coord = (ra,dec)
        # not cutting the ref on target to save computation time, get hdr error in source cat if do
        #coord = SkyCoord(ra,dec,unit=(u.hourangle,u.deg))        
        #pix=astropy.wcs.utils.skycoord_to_pixel(coord,wcs) # x,y pixel location
        #cut_ref = Cutout2D(ref.data,pix,25) # is 25 pixels big enough for any of the strong lens objects but small enough to avoid other objs?
        # photutils source properties to detect objs in image
        #cut_ref_catalog = source_cat(cut_ref)
        #cut_ref_cat,cut_ref_image,threshold,segm,targ_obj = source_catalog # unpacked to make a little clearer
        ref_catalog = source_cat(ref,targ_coord=targ_coord)
        ref_cat,ref_image,threshold,segm,targ_obj = ref_catalog # unpacked to make a little clearer
        # take useful photutil params for strong lensing galaxy target 
        # pixels and deg, sums ~ brightness in adu ~ for lco is straight counts (ie not yet rate isn't /exptime)
        equivalent_radius = targ_obj['equivalent_radius'][0].value
        xy = (targ_obj['xcentroid'][0].value,targ_obj['ycentroid'][0].value) 
        semimajor_axis, semiminor_axis = targ_obj['semimajor_axis_sigma'][0].value,targ_obj['semiminor_axis_sigma'][0].value
        orientation = targ_obj['orientation'][0].value 
    else:
        # the source image detected the targ_obj so take useful values already available (no need to get ref involved)
        # pixels and deg, sums ~ brightness in adu ~ for lco is straight counts (ie not yet rate isn't /exptime)
        equivalent_radius = targ_obj['equivalent_radius'][0].value
        xy = (targ_obj['xcentroid'][0].value,targ_obj['ycentroid'][0].value) 
        semimajor_axis, semiminor_axis = targ_obj['semimajor_axis_sigma'][0].value,targ_obj['semiminor_axis_sigma'][0].value
        orientation = targ_obj['orientation'][0].value 
    
    # cut around the image on target
    cut_targ = Cutout2D(image.data,xy,equivalent_radius*5)
    cuts = [cut_targ]
    if diff:
        cut_diff = Cutout2D(diff.data,xy,equivalent_radius*5)
        cuts.append(cut_diff)
    if ref:
        cut_ref = Cutout2D(ref.data,xy,equivalent_radius*5)
        cuts.append(cut_ref)

    # now going to grab (cutouts/patches) of boxes on galaxy 
    cut_xy = cut_targ.center_cutout
    shift_x = equivalent_radius*np.cos(orientation*np.pi/180)
    shift_y = equivalent_radius*np.sin(orientation*np.pi/180)

    # lets do a box on the ctr with length=width=radius 
    # the patch anchors on sw so shift the cut_xy 
    anchor_core = (cut_xy[0] - equivalent_radius/2, cut_xy[1] - equivalent_radius/2)
    # the patch (show in figures)
    box_core = matplotlib.patches.Rectangle(anchor_core,equivalent_radius,equivalent_radius,fill=None)
    # the cut (does sum for bkg)
    xy_core = xy # the center of galaxy in image
    cut_core = Cutout2D(image.data,xy_core,equivalent_radius)
    
    # shift box an equivalent radius along orientation from photutils creating next box 
    # assuming orientation ccw from x (east)
    # yes the boxes will overlap slightly unless orientation fully along x or y
    shift_x = equivalent_radius*np.cos(orientation*np.pi/180)
    shift_y = equivalent_radius*np.sin(orientation*np.pi/180)
    anchor_1 = (anchor_core[0]+shift_x,anchor_core[1]+shift_y)
    box_1 = matplotlib.patches.Rectangle(anchor_1,equivalent_radius,equivalent_radius,fill=None)
    # the cut (does sum for bkg)
    xy_1 = (xy[0]+shift_y,xy[1]+shift_y) 
    cut_1 = Cutout2D(image.data,xy_1,equivalent_radius)
    
    # similar shift one more time 
    anchor_2 = (anchor_core[0]+2*shift_x,anchor_core[1]+2*shift_y)
    box_2 = matplotlib.patches.Rectangle(anchor_2,equivalent_radius,equivalent_radius,fill=None)
    # the cut (does sum for bkg)
    xy_2 = (xy[0]+2*shift_y,xy[1]+2*shift_y) 
    cut_2 = Cutout2D(image.data,xy_2,equivalent_radius)
    
    bkg_core,bkg_1,bkg_2 = (cut_core,box_core),(cut_1,box_1),(cut_2,box_2)
    
    if diff or ref:
        # default diff None and ref None but if provided will return list of cuts order like [source,diff,ref]
        return targ_obj,cuts,bkg_core,bkg_1,bkg_2
    else:
        return targ_obj,cut_targ,bkg_core,bkg_1,bkg_2



def plant(image,psf,threshold,source_cat=None,hdr=None,mag=None,location=None,zp=None,plantname='planted.fits'):
    """
    the image should be the fits.open('difference.fits'), will add SN directly to here
    psf should be the epsf_builder(stars), ie a previous step is needed to check that have epsf which looks good
    source_cat is the catalog,targ_obj (strong lens galaxy), and segmentation image from photutils on image
    location should be a Skycoord(ra,dec) or if left as None will use the targ_obj strong lensing gal to place
    mag,zp (TODO need to know how to get proper zp so that scaling ePSF to correct mag; ePSF just means flux=1)
    """
    # unpack the source_cat so can use the targ_obj to place SN later if not given a location explicitly
    if source_cat:
        cat,orig_image,threshold,segm,targ_obj=source_cat # orig_image ~ meaning that pointing which source cat was run on not a diff
    
    if hdr:
        # I should never need this, hdr should always be none, want to set the plants using straight signal from difference not trying for a mag
        # if image is the diff, (or any of the cutouts) none of these are available, provide it explicitly from source hdr
        skymag=hdr['SKYMAG'] # computed [mag/arcsec^2]
        skybr=hdr['WMSSKYBR'] # meas
        pixscale=hdr['PIXSCALE'] # arcsec/pixel
        mean=hdr['L1MEAN'] # sigma-clipped mean bkg [counts]
        med=hdr['L1MEDIAN']
        sig=hdr['L1SIGMA']
        fwhm=hdr['L1FWHM']
        exptime=hdr['EXPTIME']

        #print('L1: mean,exptime,pixscale,fwhm',mean,exptime,pixscale,fwhm)
        # the sigma-clipped stats I think should be same as L1 
        mean_val, median_val, std_val = sigma_clipped_stats(image.data, sigma=2.)  
        #print('scs: mean_val,median_val,std_val',mean_val,median_val,std_val)
        if zp==None:
            # there should be an L1ZP, since there isn't I'm doing what I think makes sense to calculate it
            # I know it should be zp ~ 23, so hopefully that is about what we get
            # /exptime ~ data is in counts but skymag measured seems to be doing count rate, /pixscale since want /arcsec
            # ... I thought /pixscale^2 was correct but single pixscale is closer to 'expected': https://arxiv.org/pdf/1805.12220.pdf
            zp=skybr+2.5*np.log10(med/exptime/pixscale)
            # these are in the sdss system (r' filter is what is used in our observations) https://www.sdss.org/dr14/algorithms/fluxcal/
            # effectively same as AB system, u zp is off by .04 mag, z zp might be off by .02 mag but close enough for govt work
        
        if mag==None:
            # if don't tell it what mag we want SN, I'll make it 5 mags brighter than bkg sky
            mag = skybr-5
        
        # copying image and psf so can leave original data untouched
        cpim = copy.copy(image.data)
        psfmax = np.max(psf.data)
        print("psfmax ~ {} adu".format(psfmax))
        mu = 10**((mag-zp+2.5*np.log10(psfmax))/-2.5)*exptime*pixscale # factor to multiply psf to get given mag
        #print('mu ',mu)
    
    mu = np.median(threshold)/np.max(psf.data)
    cppsf = copy.copy(psf.data*mu) # amplitude of your epsf is the threshold value

    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
    lattice,boxes = False,False 
    if location==None:
        # use the targ obj to place SN
        x = [targ_obj['xcentroid']-targ_obj['equivalent_radius'],targ_obj['xcentroid']+targ_obj['equivalent_radius']]
        x = [i[0].value for i in x]
        x = np.linspace(x[0],x[1],100)
        x = np.random.choice(x)
        y = [targ_obj['ycentroid']-targ_obj['equivalent_radius'],targ_obj['ycentroid']+targ_obj['equivalent_radius']]
        y = [i[0].value for i in y]
        y = np.linspace(y[0],y[1],100)
        y = np.random.choice(y)
        pix = [x,y]
        revpix = copy.copy(pix)
        revpix.reverse()
        location=astropy.wcs.utils.pixel_to_skycoord(pix[0],pix[1],wcs)
    elif type(location)==tuple:
        # lattice was used to generate tuple w lists (skycoords,pixels), we want to plant many SNe across the image
        lattice = location
        # unpack the lists of lattice
        locations,pixels = lattice
        lattice = True 
    elif type(location)==list:
        # 3 boxes of lxw = req^2, starting ctr on target core and then shifted by req along orientation 
        boxes = True
    else:
        # give arb skycoord loc (ra/dec) and translate to pixels for plant
        pix=astropy.wcs.utils.skycoord_to_pixel(location,wcs) # x,y pixel location
        revpix = copy.copy(list(pix)) # row,col location for adding to data... y,x
        revpix.reverse()
    
    if boxes:
        # location is list of 3 pixel locations to plant to, one ctr on core of target, 2 more shifted req along orientation of target
        # the location list is unpacked from target fcn into the pixels in pipeline 
        for pix in location:
            pix = list(pix)
            revpix = copy.copy(pix)
            revpix.reverse()
            # indexes to add the psf to
            row,col=revpix
            nrows,ncols=cppsf.shape
            # +2 in these to grab a couple more than needed, the correct shapes for broadcasting taken using actual psf.shapes
            rows=np.arange(int(np.round(row-nrows/2)),int(np.round(row+nrows/2))+2) 
            cols=np.arange(int(np.round(col-ncols/2)),int(np.round(col+ncols/2))+2) 
            rows = rows[:cppsf.shape[0]]
            cols = cols[:cppsf.shape[1]]
            cpim[rows[:, None], cols] += cppsf
            np.float64(cpim)
            
        # inserting True fakeSN into hdr w the pix location
        cphdr = copy.copy(hdr)
        cphdr['fakeSN']=True 
        cphdr['fakeSN_loc']='boxes' 
        cphdr['NfakeSNe']=str(len(location))
        cphdr['fakeSNsigma']=str(threshold)
        cphdr['fakeZP']=str(zp)
        fits.writeto(plantname,cpim,cphdr,overwrite=True)
        print('{} SNe mag ~ {} (epsf*=mu ~ {}) planted in boxes by targ; zp ~ {}'.format(len(location),mag,mu,zp))
        plant_im = fits.open(plantname)[0]  
        return plant_im,location
    if lattice:
        # many locations to plant to
        for pix in pixels:
            pix = list(pix)
            revpix = copy.copy(pix)
            revpix.reverse()
            # indexes to add the psf to
            row,col=revpix
            nrows,ncols=cppsf.shape
            # +2 in these to grab a couple more than needed, the correct shapes for broadcasting taken using actual psf.shapes
            print('row,nrows,col,ncols',row,nrows,col,ncols)
            rows=np.arange(int(np.round(row-nrows/2)),int(np.round(row+nrows/2))+2) 
            cols=np.arange(int(np.round(col-ncols/2)),int(np.round(col+ncols/2))+2) 
            rows = rows[:cppsf.shape[0]]
            cols = cols[:cppsf.shape[1]]
            cpim[rows[:, None], cols] += cppsf
            np.float64(cpim)
            
        # inserting True fakeSN into hdr w the pix location
        cphdr = copy.copy(hdr)
        cphdr['fakeSN']=True 
        cphdr['fakeSN_loc']='lattice' 
        cphdr['NfakeSNe']=str(len(pixels))
        cphdr['fakeSNsigma']=str(threshold)
        cphdr['fakeZP']=str(zp)
        print('{} SNe mag ~ {} (epsf*=mu ~ {}) planted in lattice across image; zp ~ {}'.format(len(pixels),mag,mu,zp))
        #fits_write = None
        
        fits.writeto(plantname,cpim,cphdr,overwrite=True)
        plant_im = fits.open(plantname)[0]  
        return plant_im,pixels
    else:
        # single location for plant either using targ obj or provided skycoord
        # indexes for the lco data file to add the psf to
        row,col=revpix
        nrows,ncols=cppsf.shape
        # +2 in these to grab a couple more than needed, the correct shapes for broadcasting taken using actual psf.shapes
        rows=np.arange(int(np.round(row-nrows/2)),int(np.round(row+nrows/2))+2) 
        cols=np.arange(int(np.round(col-ncols/2)),int(np.round(col+ncols/2))+2) 
        rows = rows[:cppsf.shape[0]]
        cols = cols[:cppsf.shape[1]]
        cpim[rows[:, None], cols] += cppsf
        np.float64(cpim)
        # write the image with planted SN added to a new fits file (inserting True fakeSN into hdr)
        cphdr = copy.copy(hdr)
        cphdr['fakeSN']=True 
        cphdr['fakeSN_loc']=str(pix)
        plant_im = fits.writeto(plantname,cpim,cphdr,overwrite=True)
        print('SN mag ~ {} planted in image w bkg mag ~ {} at {} written to {}; zp ~ {}'.format(mag,skybr,location,plantname,zp))
        return plant_im 

def galaxy_lattice(image,cat):
    # typical lattice for planting in grid returns lists of skycoord locations & pixels
    # I will do the same to id locations of galaxies for planting
    hdr = image.header
    wcs,frame=WCS(hdr),hdr['RADESYS'].lower()
    # have single 4kx4k chip from wide field instrument
    NX = hdr['naxis1']
    NY = hdr['naxis2']
    # the cat has all source detections
    """
    The exposures aren't trimmed like the differences, they are rotated, just using crude fix, 4096 vs 2212 on trim 
    """
    ctr = 4096/2 # ctr of the full source cat img
    trimWidth = 2212/2 # +- along x&y for the source cat ctr to overlap w trimmed diff
    dx = dy = ctr - trimWidth
    lo,hi = ctr-trimWidth, ctr+trimWidth
    lo += 200
    hi += -200
    
    a = cat[cat['bbox_xmin'].value > lo]
    b = a[a['bbox_ymin'].value > lo]
    c = b[b['bbox_xmax'].value < hi]
    d = c[c['bbox_ymax'].value < hi]
    # taking 'large object' detections as equivalent radius
    large = d[d['equivalent_radius']>np.median(d['equivalent_radius'])]
    # want 'clear' signals taking these as max value from large
    clear = large[large['max_value']>np.median(large['max_value'])]
    # taking 'galaxy' detections from these large bright as elliptical
    clearell = clear[clear['eccentricity'] > np.median(clear['eccentricity'])]

    #xc = clearell['xcentroid'].value
    #yc = clearell['ycentroid'].value
    #req = clearell['equivalent_radius'].value
    x,y = [],[]
    xn,yn = [],[]
    pixels,npixels = [],[]
    for i in range(len(clearell)):
        xc,yc,req = clearell[i]['xcentroid'].value,clearell[i]['ycentroid'].value,clearell[i]['equivalent_radius'].value
        #updown = random.choice([-1,+1])
        #leftright = random.choice([-1,+1])
        #xi,yi = xc + np.random.uniform(xc,xc+leftright*req),np.random.uniform(yc,yc+updown*req)
        x.append(xc)
        xn.append(xc-dx)
        y.append(yc)
        yn.append(yc-dy)
        pixels.append((xc,yc))
        npixels.append((xc-dx,yc-dy))
    #pixels = list(itertools.product(x, y))
    #newpixels = list(itertools.product(xn,yn))
    locations = [] # skycoord locations that I will use to plant SNe across image  
    for i in range(len(pixels)):
        pix = pixels[i]
        location=astropy.wcs.utils.pixel_to_skycoord(pix[0],pix[1],wcs)
        locations.append(location)
    return locations,npixels,pixels

def lattice(image):
    hdr = image.header
    wcs,frame=WCS(hdr),hdr['RADESYS'].lower()
    # have single 4kx4k chip from wide field instrument
    NX = hdr['naxis1']
    NY = hdr['naxis2']
    edge = 100 # pixels away from edge
    spacing = 100 # pixels between each location on lattice
    x = list(range(0+edge,NX-edge+1,spacing)) # +1 to make inclusive
    y = list(range(0+edge,NY-edge+1,spacing))
    pixels = list(itertools.product(x, y))
    locations = [] # skycoord locations that I will use to plant SNe across image  
    for i in range(len(pixels)):
        pix = pixels[i]
        location=astropy.wcs.utils.pixel_to_skycoord(pix[0],pix[1],wcs)
        locations.append(location)
    return locations,pixels

def detection_efficiency(plant,cat):
    # provide the plant and detection cat run to find efficiency
    # unpack the plant (the image and locations)
    plant_im,pixels=plant 

    # unpack the detection catalog objs (cat,image,threshold,segm)
    catalog,image,threshold,segm,targ_obj = cat

    hdr=image.header
    #Nfakes=hdr['NfakeSNe']
    
    # use locations and a search radius on detections and plant locations to get true positives
    tbl = catalog.to_table()
    tbl_x,tbl_y = [i.value for i in tbl['xcentroid']], [i.value for i in tbl['ycentroid']]
    tbl_pixels = list(zip(tbl_x,tbl_y))
    tbl.add_column(Column(tbl_pixels),name='pix') # adding this for easier use indexing tbl later
    search = 5 # fwhm*n might be better criteria
    truths = []
    for pixel in tbl_pixels:
        for i in pixels:
            if pixel[0] > i[0] - search  and pixel[0] < i[0] + search and pixel[1] > i[1] - search and pixel[1] < i[1] + search:
                truths.append([i,pixel])
                #print(i,pixel)
            else:
                continue
    #print('{} source detections within search radius criteria'.format(len(truths)))
    # TODO: get the tbl_pixels which were outside the search radius criteria and return them as false positives
    
    # break truths into the plant pixels and det src pixel lists; easier to work w
    plant_pixels = []
    det_src_pixels = []
    for i in truths:
        plant_pix = i[0]
        det_src_pix = i[1]
        plant_pixels.append(plant_pix)
        det_src_pixels.append(det_src_pix)
    # the plant pixels which had multiple sources detected around it
    repeat_plant = [item for item, count in collections.Counter(plant_pixels).items() if count > 1]
    # the plant pixels which only had one source detected 
    single_plant = [item for item, count in collections.Counter(plant_pixels).items() if count == 1]
    N_plants_detected = len(single_plant) + len(repeat_plant)
    # adding nearby_plantpix col to src table; using None if source wasnt within the search radius of plant
    plant_col = []
    for i in tbl:
        tbl_x,tbl_y = i['xcentroid'].value,i['ycentroid'].value
        if (tbl_x,tbl_y) in det_src_pixels:
            idx = det_src_pixels.index((tbl_x,tbl_y))
            plant_col.append(plant_pixels[idx])
        else:
            plant_col.append(None)
    tbl.add_column(Column(plant_col),name='nearby_plantpix')
    
    # index table to grab false source detections
    false_tbl = tbl[tbl['nearby_plantpix']==None]
    truth_tbl = tbl[tbl['nearby_plantpix']!=None]
    
    single_truth_tbl,repeat_truth_tbl = [],[]
    for i in truth_tbl:
        if i['nearby_plantpix'] in repeat_plant:
            repeat_truth_tbl.append(i)
        else:
            single_truth_tbl.append(i)
    # should use a check on length rather than try/except below here
    # try/excepting is to avoid error for empty lists
    # mainly an issue on repeat truth tbl 
    try:
        single_truth_tbl = vstack(single_truth_tbl)
    except:
        pass
    try:
        repeat_truth_tbl = vstack(repeat_truth_tbl)
    except:
        pass            
    #print('Final: {} planted SNe, {} clean single detections, {} as multi-sources near a plant, {} false detections'.format(Nfakes,len(single_truth_tbl),len(repeat_truth_tbl),len(false_tbl)))
    #print('{} planted SNe had single clean source detected, {} planted SNe had multiple sources detected nearby, {} false detections'.format(len(single_plant),len(repeat_plant),len(false_tbl)))

    efficiency = N_plants_detected/len(pixels)

    #print('Detection efficiency (N_plants_detected/N_plants) ~ {} on mag ~ {} SNe'.format(efficiency,magfakes))
    return efficiency,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl


def dl_peter_ref(hdr,path=None,token='cb00e632ec494f78571af0b2f7db879e3546fb52'):
    """
    Peter gave me all the differences, however dont have refs
    """
    # origname ~ filename.fits
    origname = hdr['ORIGNAME']
    basename = origname.split('.')[0]
    print('basename:',basename)
    basename91 = basename[:-2]+'91'
    print('basename91:',basename91)
    # proposal
    propid = hdr['PROPID']
    date = hdr['DATE'][:10] # yyyy-mm-dd
    print(propid,date)
    """
    instrument = hdr['INSTRUME']
    obj = hdr['OBJECT']
    siteid = hdr['SITEID']
    telid = hdr['TELID']
    flt = hdr['FILTER']
    obstype = hdr['OBSTYPE']
    exp = hdr['EXPTIME']
    blkuid = hdr['BLKUID']
    dateobs = hdr['DATE-OBS']
    dayobs = hdr['DAY-OBS']
    """
    # these should probably be enough to get the images
    url = 'https://archive-api.lco.global/frames/?q=a&RLEVEL=91&PROPID={}&INSTRUME=&OBJECT=&SITEID=&TELID=&FILTER=&OBSTYPE=&EXPTIME=&BLKUID=&REQNUM=&basename=&start={}%2000%3A00&end={}%2023%3A59&id=&public='.format(propid,date,date)

    #print('url',url)
    tmp = requests.get(url,headers={'Authorization': 'Token {}'.format(token)})
    response = tmp.json()
    #print('response',response)
    frames = response['results']
    #print('frames:',frames)
    for frame in frames:
        print('frame:',frame)
        if path:
            file = os.path.join(path,frame['filename'])
        else:
            file = frame['filename']
        with open(file, 'wb') as f:
            f.write(requests.get(frame['url']).content) # ,headers={'Authorization': 'Token {}'.format(token)}).content

def f_efficiency(m,m50,alpha):
    #https://arxiv.org/pdf/1509.06574.pdf, strolger 
    return (1+np.exp(alpha*(m-m50)))**-1 


def df_stack():
    # step 1 get the data
    lco_path = '/work/oconnorf/efficiency_pipeline/lco/'
    peter_path = os.path.join(lco_path,'peter_diffs')
    peter_diffs = os.path.join(peter_path,'differences')
    output = peter_diffs+'/*/*/source/output/*csv'
    tmp = glob.glob(output)
    dfs =  [pd.read_csv(tmp[i]) for i in range(len(tmp))]
    df = pd.concat(dfs)
    df.to_csv('df.csv')
    # df = pd.read_csv('df.csv')
    print(len(df),df.keys())

import csv

def dl_headers(datetime,name,path=True,token='cb00e632ec494f78571af0b2f7db879e3546fb52'):
    """
    Need header for each datetime observation of every field
    """
    propid1 = 'LCO2019A-008'
    propid2 = 'LCO2019B-022'
    propid3 = 'LCO2020A-019'
    propid4 = 'LCO2020B-015'
    
    date = datetime.strftime('%Y-%m-%d')
    flt = 'rp'#hdr['FILTER']
    telid = '1m0a'#hdr['TELID']
    print(name,date)
    """
    propid = hdr['PROPID']
    date = hdr['DATE'][:10] # yyyy-mm-dd
    instrument = hdr['INSTRUME']
    obj = hdr['OBJECT']
    siteid = hdr['SITEID']
    obstype = hdr['OBSTYPE']
    exp = hdr['EXPTIME']
    blkuid = hdr['BLKUID']
    dateobs = hdr['DATE-OBS']
    dayobs = hdr['DAY-OBS']
    """
    
    url1 = 'https://archive-api.lco.global/frames/?q=a&RLEVEL=91&PROPID={}&INSTRUME=&OBJECT=&SITEID=&TELID={}&FILTER={}&OBSTYPE=&EXPTIME=&BLKUID=&REQNUM=&basename=&start={}%2000%3A00&end={}%2023%3A59&id=&public='.format(propid1,telid,flt,date,date)
    url2 = 'https://archive-api.lco.global/frames/?q=a&RLEVEL=91&PROPID={}&INSTRUME=&OBJECT=&SITEID=&TELID={}&FILTER={}&OBSTYPE=&EXPTIME=&BLKUID=&REQNUM=&basename=&start={}%2000%3A00&end={}%2023%3A59&id=&public='.format(propid2,telid,flt,date,date)
    url3 = 'https://archive-api.lco.global/frames/?q=a&RLEVEL=91&PROPID={}&INSTRUME=&OBJECT=&SITEID=&TELID={}&FILTER={}&OBSTYPE=&EXPTIME=&BLKUID=&REQNUM=&basename=&start={}%2000%3A00&end={}%2023%3A59&id=&public='.format(propid3,telid,flt,date,date)
    url4 = 'https://archive-api.lco.global/frames/?q=a&RLEVEL=91&PROPID={}&INSTRUME=&OBJECT=&SITEID=&TELID={}&FILTER={}&OBSTYPE=&EXPTIME=&BLKUID=&REQNUM=&basename=&start={}%2000%3A00&end={}%2023%3A59&id=&public='.format(propid4,telid,flt,date,date)

    print('urls',url1,url2,url3,url4)
    tmp1 = requests.get(url1,headers={'Authorization': 'Token {}'.format(token)}).json()
    tmp2 = requests.get(url2,headers={'Authorization': 'Token {}'.format(token)}).json()
    tmp3 = requests.get(url3,headers={'Authorization': 'Token {}'.format(token)}).json()
    tmp4 = requests.get(url4,headers={'Authorization': 'Token {}'.format(token)}).json()
    frames1 = tmp1['results']
    frames2 = tmp2['results']
    frames3 = tmp3['results']
    frames4 = tmp4['results']
    print(len(frames1),len(frames2),len(frames3),len(frames4))
    # only one of these frames proposals has results for the date 
    urls = [url1,url2,url3,url4]
    frames = [frames1,frames2,frames3,frames4]
    for i in range(len(frames)):
        f = frames[i]
        url = urls[i]
        if len(f) != 0:
            frames = f
            url = url
            break

    cwd = os.getcwd()
    hdrs = []
    for frame in frames:
        #print('frame:',frame)
        idframe = str(frame['id'])
        print(frame['OBJECT'],frame['DAY_OBS'],frame['PROPID'])
        tmp = url.split('frames')
        frameidurl = tmp[0] + 'frames/' + idframe + '/headers'
        frameidurl = frameidurl + tmp[1]
        frameidurl
        
        if path:
            headers_folder = os.path.join(cwd,'headers')
            if not os.path.exists(headers_folder): 
                os.makedirs(headers_folder)
            field_folder = os.path.join(headers_folder,name)
            if not os.path.exists(field_folder): 
                os.makedirs(field_folder)
            date_folder = os.path.join(field_folder,date)
            if not os.path.exists(date_folder):
                os.makedirs(date_folder)            
            file = os.path.join(date_folder,frame['filename'])
        else:
            file = frame['filename']
        
        print(file)
        basefile = file.split('.fits')[0]
        basefile+'.csv'
        print(basefile)
        w = csv.writer(open(basefile, "w"))
        header = requests.get(frameidurl).json()
        hdrs.append(header)
        for key,val in header.items():
            w.writerow([key,val])
        """
        with open(file, 'wb') as f:
            #f.write(requests.get(frame['url']).content) # ,headers={'Authorization': 'Token {}'.format(token)}).content
            #f.write(requests.get('https://archive-api.lco.global/frames/42/headers/').json())
        """
    #return  hdrs,url,frames

def cut_gaias(file,results,saveas=None):
    """
    file should be fits.open("filename.fits")
    results should be the table from gaia request
    use radec from gaia results to cut image on each star from file 
    """
    if not os.path.exists(saveas): 
        os.makedirs(saveas)
        
    hdu = file # where I'm going to overwrite the cutout data/hdr onto so it has same class for fits writeto
    dat = file.data
    hdr = file.header
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
    
    ra,dec=results['ra'],results['dec']
    # a ~ name ~ bGaiaDR2idx
    a = results['designation']
    
    gaia_stars = SkyCoord(ra,dec,unit=(u.hourangle,u.deg))
    size = 50 # pixels
    
    for i in range(len(gaia_stars)):
        # fix name designation
        ai = str(a[i])
        ai= ai.replace(" ","_")
        ai = ai.replace("'","")
        
        cut = Cutout2D(dat,gaia_stars[i],size,wcs=wcs)
        cutwcs = cut.wcs
        if saveas:
            #pickle.dump(cut,open(saveas,"wb"))
            hdu.data = cut.data
            hdu.header.update(cut.wcs.to_header())
            hdu.writeto(saveas+'/'+ai+'.fits',overwrite=True)
    
    return 

def peter_pipe():
    """
    This pipeline is to get df with header values and efficiency fit values for large group of differences
    They will be used in a griddata to interpolate efficiency values for images

    images from 96 night, 309 fields total available from all dates
    dictionary here has datename args to list of fields 
    datefields_dict = pickle.load(open("datefields_dict.pkl","rb"))
    dict_fields = datefields_dict[date_key]
    """

    # step 1 get the data
    lco_path = '/work/oconnorf/efficiency_pipeline/lco/'
    peter_path = os.path.join(lco_path,'peter_diffs')
    peter_diffs = os.path.join(peter_path,'differences')
    date_paths = [x for x in glob.glob(peter_diffs+'/*') if os.path.isdir(x)]
    # sbatch will want 0-95 array so it can index all the dates
    date_idx = int(sys.argv[1])
    date_path = date_paths[date_idx]
    date = os.path.basename(date_path) #date.split('/')[-1] 
    field_paths = [x for x in glob.glob(date_path+'/*') if os.path.isdir(x)]
    print(date_idx,date,len(field_paths))
    cnt = 0
    for field_path in field_paths:
        try:
            my_data = {}
            print("____________________________________________________")
            field = os.path.basename(field_path)
            print(field)
            source_folder = os.path.join(field_path,'source')
            output_folder = os.path.join(source_folder,'output')
            if not os.path.exists(output_folder): 
                os.makedirs(output_folder)

            tmp = glob.glob(source_folder+'/*cut*fits*')
            #print(tmp)
            for cut in tmp:
                #print(cut)
                lco_figures.cutout_figure(cut)
            # just tmp for cut figs
            # continue
            
            image_paths = glob.glob(source_folder+'/*fz')
            for image_path in image_paths:
                image = fits.open(image_path)[1]
                image_name = os.path.basename(image_path)
                filename = image_name.split('.')[0] # taking away .fits.fz
                saveas=image_path.split('.')[0]+"_cut.fits" # taking away .fits.fz
                # cutting around the target lens galaxy
                hdu = cut_target(image,saveas=saveas)
                #print(hdu.header['NAXIS1'],hdu.header['NAXIS2'])
                # grabbing previous available epsf data and m50 to plant SN in the cut
                sig_fit = glob.glob(output_folder+"/*sig_fit.pkl")
                sig_fit = pickle.load(open(sig_fit[0],"rb"))
                epsf = sig_fit['vanilla_epsf']
                m50 = sig_fit['m50']
                epsf = scale(epsf,hdu.header,m50)
                saveas=image_path.split('.')[0]+"_plantcut.fits" 
                cut_plant(hdu,epsf,saveas=saveas)

                if filename[0] == 'd':
                    diff_img = image
                else:
                    img = image
                my_data[filename] = image

            print('img,diff_img:',img,diff_img)
        except:
            print('this field_path had issue with get source img,diff_img',field_path)
            continue

        a = r,img = gaia_results(img) # fov query
        #b = good_stars,r,img = stars2(a) # cuts 
        cut_gaias(img,r,saveas=os.path.join(source_folder,'gaias'))
        continue


        # define some useful quantities
        hdr = img.header
        groupid,L1fwhm,pixscale,skybr = hdr['GROUPID'],hdr['L1fwhm'],hdr['pixscale'],hdr['WMSSKYBR'] # pixels, arcsec/pixels,mag/arcsec^2
        med,exptime,origname = hdr['L1MEDIAN'],hdr['EXPTIME'],hdr['ORIGNAME']
        moon = hdr['MOONSTAT'],hdr['MOONFRAC'],hdr['MOONDIST'],hdr['MOONALT']
        sol = hdr['SUNDIST'],hdr['SUNALT'],None,None # Nones so pd can read dictionary 
        print('filename ~ {} (groupid {}) has exptime ~ {}, L1fwhm ~ {} arcsec, pixscale ~ {} arcsec/pixel, moon stat/frac/dist/alt ~ {}/{}/{} /{}, sol dist/alt ~ {}, skybr {} mag/arcsec^2'.format(filename,groupid,exptime,L1fwhm,pixscale,moon[0],moon[1],moon[2],moon[3],sol,skybr))

        pickle_to = os.path.join(output_folder,origname[:-5]) # [:-5] to get rid of .fits
        print(pickle_to)

        # use the header l1fwhm to draw a symm 2d-gaussian; (amplitude 1 ctr on a (50x50) data array)
        vanilla_epsf = stock_epsf(L1fwhm)
        pickle.dump(vanilla_epsf,open(pickle_to+'L1FWHMepsf.pkl','wb'))

        nsigma,kernel_size,npixels,deblend,contrast,targ_coord = 3,(3,3),3,False,.001,None # npixels int(np.round(L1fwhm/pixscale))
        # key step defines threshold for detection, uses noise from scs of difference img
        # not sure exactly how/if can turn threshold pixel value into a magnitude
        # the difference image pixel data values are rescaled from those in exoposure during difference software
        threshold = detect_threshold(diff_img.data,nsigma=nsigma)
        threshold = np.median(threshold)
        print("Threshold~ {}, s/n ~ {} in difference considered detected".format(threshold,nsigma))
        hdrd = {} # fits table doesnt pickle right turn it into a dict
        for i in hdr:
            hdrd[i] = hdr[i] 

        # not an elegant soln but at a time crunch right now
        # many of the headers fail in pkl load when get past the l1pubdat looks like key that has bad values might be 'HISTORY'
        hdrd_fix = {}
        for i in hdrd:
            if i != 'L1PUBDAT':
                hdrd_fix[i] = hdrd[i]
            else:
                break

        d = {'threshold':threshold,'nsigma':nsigma,'hdr':hdrd_fix,'vanilla_epsf':vanilla_epsf}
        pickle.dump(d,open(pickle_to+'_hdrd.pkl','wb'))
        checkhdrd = pickle.load(open(pickle_to+'_hdrd.pkl','rb'))
        print("checkhdrd works")

        # run photutils source properties fcn to detect objs in exp img
        print('Source Catalog is a photutils source_properties using nsigma ~ {} (detection threshold above img bkg), gaussian kernel sized ~ {} pix, npixels ~ {} (connected pixels needed to be considered source), deblend ~ {} w contrast {}'.format(nsigma,kernel_size,npixels,deblend,contrast))
        source_catalog = source_cat(img,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
        cat,img,threshold,segm,targ_obj = source_catalog # unpacked to make a little clearer
        cat = cat.to_table() 

        # these should be tables, the photutil obj doesnt pkl properly
        print("type of cat:",type(cat))
        print("type of targ:",type(targ_obj))

        # pickle the source property table and add to dictionary
        pickle.dump(targ_obj, open(pickle_to+'_targ.pkl','wb'))
        pickle.dump(cat,open(pickle_to+'_cat.pkl','wb'))
        d['cat'] = cat
        d['targ'] = targ_obj
        
        # use gaia dr2 to id good stars to extract and build an epsf from
        results = gaia_results(img)
        gaia,img = results # unpacked 
        # extract good gaia stars from img for psf
        extracted_stars = stars(results)
        good_stars,img = extracted_stars # unpacked
        try:
            # use extracted stars to build epsf
            EPSF = ePSF(extracted_stars,oversampling=2)
            epsf_build,fitted_stars = EPSF # unpacked
            epsf_built = epsf_build.data # turn the photutils obj into its ndarray
            pickle.dump(epsf_built,open(pickle_to+'epsf_built.pkl','wb'))
            epsf_gaussian = gaussian2d(epsf_build)
            fit_gaussian,table,levels,xctr_vals,yctr_vals,epsf_2dGfit,resid = epsf_gaussian # unpacked
            d['epsf_built'] = epsf_built 
            """
            fit_gaussian ~ the best fitting 2d gaussian parameters to the built epsf (still oversampled)
            table ~ the parameters used generate 2dGfit array, same as fitted but with the oversampling taken out of the std
            """
            d['2dGfit'] = table    
            d['epsf_2dGfit'] = epsf_2dGfit  
        except:
            print("the epsf builder failed")
            epsf_built,table,epsf_2dGfit = None,None,None
            d['epsf_built'] = epsf_built
            d['2dGfit'] = table
            d['epsf_2dGfit'] = epsf_2dGfit
        
        print("The EPSFs are 50x50 data arrays")
        print("Vanilla EPSF: uses hdr L1FWHM to make symm-2d gaussian",type(vanilla_epsf))
        print("Built EPSF: uses gaia dr2 stars to create empirical fit",type(epsf_built))
        print("EPSF 2dGfit: does non-symm-2d gaussian analytical fit to the build",type(epsf_2dGfit))
        print("2dGfit: a table of the non-symm-2d gaussian parameters fit to the built",type(table))

        # to do: checks on the build/fit to decide whether they are okay to use in place of the vanilla epsf
        # for now: using the vanilla every single time for consistency 
        epsf = vanilla_epsf 

        # the plant 
        # generate list of epsf with different amplitudes set using diff image pixel-wise threshold values of 1,1.4,1.8...5,10 S/N
        # the epsf flux (np.sum(epsf)) expressed in amplitude (np.max(epsf)) is f=A*(2pi*Sigma^2); sigma the pixel stddev of data array
        a,b = np.arange(1,5.21,0.4),[10]
        sigmas = np.concatenate([a,b])
        thresholds,epsfs,mags = [],[],[]
        for i in sigmas:
            threshold = detect_threshold(diff_img.data,nsigma=i)
            threshold = np.median(threshold)
            thresholds.append(threshold)
            # scale the epsf so amplitude (max value at ctr) is equal the threshold value 
            epsfi = threshold_amplitude(threshold,epsf)
            f = np.sum(epsfi) 
            # if you want a check; flux = Amplitude*(2pi*Sigma^2)
            # fcheck = np.max(epsfi)*(2*np.pi*(hdr['L1FWHM]/hdr['PIXSCALE'])**2) 
            epsfs.append(epsfi)
            mi = phot(epsfi,hdr) # AB mag zp i have is coming from hdr param (mag/arcsec^2 sky brightness); to do exposures on gaia stars
            mags.append(mi)
        print("S/N (ratio pix data values ~ amplitude max epsf to median difference image)")
        print(sigmas)
        print("Thresholds (amplitude pix data values ~ e- counts)")
        print(thresholds)
        print("AB mag (flux is summed epsf, zp from hdr)")
        print(mags)

        # check that there are no issues with the difference image data
        if np.inf in mags:
            print("bad diff img data ~ there is an inf in the mags from neg s/n threshold, continuing without this one")
            continue
        elif np.nan in mags:
            print("bad diff img data ~ there is a nan in the mags from 0 s/n threshold, continuing without this one")
            continue

        # the lattice on image grid planting locations
        grid = lattice(diff_img)
        grid_locations,grid_pixels = grid
        grid_efficiencies = []

        """
        # identifying galaxies from source detection parameters, to plant in similar routine
        gals = galaxy_lattice(diff_img,cat)
        gal_locations,gal_npixels,gal_pixels = gals # npixels is the diff img, pixels is the untrimmed
        gal_efficiencies = []

        print(len(gal_npixels),gal_npixels[0])
        """
   
        for i in range(len(epsfs)):
            sigmai,thresholdi,mi,epsfi = sigmas[i],thresholds[i],mags[i],epsfs[i] 

            # create plant img in a grid
            plantname = '{}_lattice_plant_mag{:.2f}.fits'.format(pickle_to,mi)
            planted = plant2(diff_img,hdr,epsfi,grid_pixels,name=plantname)
            plant_im,pixels = planted # unpack
            # source properties of detected objs in fake img
            fakesource_cat = source_cat(plant_im,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
            fakecat,fakeimg,fakethreshold,fakesegm,faketarg_obj = fakesource_cat # unpacked to make a little clearer
            # detection efficiency  
            tmp = detection_efficiency(planted,fakesource_cat)
            efficiency,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl = tmp
            grid_efficiencies.append(efficiency) 
            
            """
            # create plant img into galaxies
            plantname = '{}_galaxies_plant_{}sigma.fits'.format(pickle_to,str(sigmai))
            planted = plant2(diff_img,hdr,epsfi,gal_npixels,name=plantname)
            plant_im,pixels = planted # unpack
            # source properties of detected objs in fake img
            fakesource_cat = source_cat(plant_im,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
            fakecat,fakeimg,fakethreshold,fakesegm,faketarg_obj = fakesource_cat # unpacked to make a little clearer
            # detection efficiency  
            tmp = detection_efficiency(planted,fakesource_cat)
            efficiency,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl = tmp
            gal_efficiencies.append(efficiency)
            """

        print(filename)
        print('grid_efficiencies: {}'.format(grid_efficiencies))
        #print('gal efficiencies: {}'.format(gal_efficiencies))
        print('peak pixel S/N: {}'.format(sigmas))
        print('mags: {}'.format(mags))

        d['grid_efficiencies'] = grid_efficiencies
        #d['gal_efficiencies'] = gal_efficiencies
        d['sigmas_thresholds'] = [sigmas,thresholds]
        d['mags'] = mags

        # use interp to get magnitude at which we have 50% detection efficiency 
        # need the values increasing along x for interp to work properly
        #gal_efficiencies = list(gal_efficiencies)
        grid_efficiencies,mags=list(grid_efficiencies),list(mags)
        #gal_efficiencies.reverse()
        #grid_efficiencies.reverse()
        #mags.reverse()

        # init guesses t50 from interp and alpha ~ 5 pretty steep drop
        m50 = np.interp(0.5,grid_efficiencies,mags)
        alpha = 5
        init_vals = [m50,alpha]  
        print("init vals (m50, alpha): {}".format(m50,alpha))
        try:
            best_vals, covar = curve_fit(f_efficiency, mags, grid_efficiencies, p0=init_vals, maxfev=5000) # fev function eval iters def 600
            print('fitted vals (m50, alpha): {}'.format(best_vals))
            m50,alpha = best_vals
        except:
            print("issue with curve fit the m50,alpha added to dict will be the init vals")
            pass

        d['m50'] = m50
        d['alpha'] = alpha

        #print(d)
        # pkl the pure dictionary 
        dictname = pickle_to + str(nsigma) + 'sig_fit.pkl'
        pickle.dump(d,open(dictname,'wb'))

        # planting/efficiency figures 
        # make figures
        try:
            lco_figures.detection_efficiency(mags,grid_efficiencies,m50,alpha,saveas=pickle_to+'_detection_efficiency.pdf')
            lco_figures.lattice_planted(mags,m50,pickle_to=pickle_to,saveas=pickle_to+'_plants.pdf')
        except:
            print("issue with the lattice planted or detection efficiency figure")
        
        """
        # try and turn it into a df
        df = pd.DataFrame(d)
        print(df)
        # recall name with the nsigma, threshold, and hdr at the beginning was pickled as name
        # dictname = pickle_to + str(nsigma) + 'sig.pkl'
        # renaming this which now also has the psfs and efficiency curve fit values
        dictname = pickle_to + str(nsigma) + 'sig_fitDF.pkl'
        pickle.dump(df,open(dictname,'wb')) # need to remember this is where m50,alpha go i.e. why we are. making the df
        """

        """
        efficiencies = []
        for mag in mags:
            # create plant img; zp None, using measure sky mag/arcsec^2 from L1 hdr to set
            plantname = '{}_planted_lattice_mag{}.fits'.format(pickle_to,str(mag))
            planted = plant(diff_img,epsf,source_cat=None,hdr=hdr,mag=mag,location=locations,zp=None,plantname=plantname)
            plant_im,pixels = planted # unpack

            # source properties of detected objs in fake img
            fakesource_cat = source_cat(plant_im,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
            fakecat,fakeimg,fakethreshold,fakesegm,faketarg_obj = fakesource_cat # unpacked to make a little clearer

            # detection efficiency  
            tmp = detection_efficiency(planted,fakesource_cat)
            efficiency,magfakes,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl = tmp
            efficiencies.append(efficiency)
            print(efficiency,magfakes)
            print('--------------------------------------------------------------')
            if len(efficiencies) > 2:
                if efficiencies[-1] < .01 and efficiencies[-2] < .01:
                    mags = mags[:len(efficiencies)]
                    print("we are at no detections cutting mag list to efficiencies length and breaking the loop")
                    break


        print(filename)
        print('efficiencies: {}'.format(efficiencies))
        print('mags: {}'.format(mags))
        # use interp to get magnitude at which we have 50% detection efficiency 
        # need the values increasing along x for interp to work properly
        efficiencies,mags=list(efficiencies),list(mags)
        efficiencies.reverse()
        mags.reverse()

        try:
            # init guesses m50 from interp and alpha ~ 5 pretty steep drop
            m50 = np.interp(0.5,efficiencies,mags)
            alpha = 5
            init_vals = [m50,alpha]  # for [m50,alpha]
            best_vals, covar = curve_fit(f_efficiency, mags, efficiencies, p0=init_vals)
            print('fitted vals (m50, alpha): {}'.format(best_vals))
            m50,alpha = best_vals
            print(m50,alpha,origname,exptime,[moon],L1fwhm,skybr)
            d = {'m50':m50,'alpha':alpha,'filename':origname,'exptime':exptime,'moon':[moon],'l1fwhm':L1fwhm,'skybr':skybr}
            print(d)
            df = pd.DataFrame(d)
            print(df)
            df.to_csv(pickle_to+'_df.csv')
        except:
            print("the fit failed, likely an issue with either bad psf or zp")

        rm = glob.glob(os.path.join(output_folder,'*fits'))
        for f in rm:
            os.remove(f) # remove all the planted fits free up space

    """
    
    """
    This block copied the difference and exposure for one of the observations for each field to a folder named source
    Keeps data more organized easier to run the efficiency measurement for each field when know the identical diff and exp
    Might want to return and run for the other observations not used and/or use move to reduce wasted storage space 

    for field_path in field_paths:
        print("____________________________________________________")
        source_folder = os.path.join(field_path,'source')
        if not os.path.exists(source_folder): 
            os.makedirs(source_folder)
        field = os.path.basename(field_path)
        print(field)
        image_paths = [x for x in glob.glob(field_path+'/*fz')]
        diff_paths = [x for x in glob.glob(field_path+'/d*fz')]
        exp_paths = list(set(image_paths)^set(diff_paths))
        for diff_path in diff_paths:
            diff = os.path.basename(diff_path)
            diff_image = fits.open(diff_path)[1]
            hdr = diff_image.header
            try:
                origname = hdr['ORIGNAME']
            except:
                print(diff,"fails origname")
                origname = None
            if origname:
                print(diff,"has origname",origname)
                dst = os.path.join(source_folder,diff)
                print("moving this diff to source folder",dst)
                shutil.copy(diff_path,source_folder) # move once ready to run for real
                break
        for exp_path in exp_paths:
            exp = os.path.basename(exp_path)
            exp_image = fits.open(exp_path)[1]
            hdr = exp_image.header
            if hdr['ORIGNAME'] == origname:
                print(exp,"has same origname as diff found",diff)
                dst = os.path.join(source_folder,exp)
                print("moving this exp to source folder",dst)
                shutil.copy(exp_path,source_folder) # move once ready to run for real
                break
            else:
                continue
        
        gotem = glob.glob(os.path.join(source_folder,'*'))
        print(gotem)
    """

    """
    This block was used to request.get all the reduced data exposure images for Peter's diffs 
    for i in fields:
        path = i # the field
        fieldname=i.split('/')[-1]
        images = [x for x in glob.glob(i+'/*fz')]
        for j in images:
            filename = j.split('/')[-1]
            image = fits.open(j)[1]
            list_vals = [image,path]
            my_data[filename] = list_vals #fits.open(j)[1]

    
    for key in my_data:
        image,path=my_data[key]
        hdr = image.header
        print(key,image,path)
        try:
            dl_peter_ref(hdr,path=path)    
        except:
            print(key,"failed dl")
    """


def lco_pipe():
    #print(sys.argv[0]) # the name of this command script  
    date_key = str(sys.argv[1]) # easy enough to change sbatch to get xx.xx date
    field_key = int(sys.argv[2]) # slurm array idx in sbatch that will be used to do the different fields 
    # lco_path ~ current working dir with scripts and sub-dirs of data  
    lco_path = '/work/oconnorf/efficiency_pipeline/lco/'
    # all the dates w lco data in the lco_path 
    all_dates = [x for x in glob.glob(lco_path+'/*') if os.path.isdir(x)]
    # your batch should have xx.xx date given so script knows which set of fields you want to do 
    date_path = os.path.join(lco_path,date_key+'/*')
    all_fields = [x for x in glob.glob(date_path) if os.path.isdir(x)]
    field = all_fields[field_key]
    # which date folder are we in and which field was this slurm idx job 

    # all the fits images needed, the trims, diffs, and ref 
    my_data = get_data(field)
    # a table that has the galaxy-galaxy strong lens system: id, magnification, lens_z, source_z, peakIa mag
    glsn = ascii.read('peakGLSN.csv')

    # each field should have a folder source_im (along w dia_out and dia_trim) 
    # in source_im is the image you want to do this for
    # ie the one that psf is measured on trim and SNe planted to diff of
    source_im = glob.glob(field+'/source_im/*fits')[0]
    source_output = os.path.join(field,'source_im/output') # where will be stick results of pipeline 
    filename=source_im.split('/')[-1]
    image = my_data[filename]
    diff_image = my_data['d_'+filename]
    ref_image = my_data['ref.fits']
    hdr = image.header
    groupid,L1fwhm,pixscale,skybr = hdr['GROUPID'],hdr['L1fwhm'],hdr['pixscale'],hdr['WMSSKYBR'] # pixels, arcsec/pixels,mag/arcsec^2
    med,exptime = hdr['L1MEDIAN'],hdr['EXPTIME']
    moon = hdr['MOONSTAT'],hdr['MOONFRAC'],hdr['MOONDIST'],hdr['MOONALT']
    zp=skybr+2.5*np.log10(med/exptime/pixscale)
    print('I am determining the zp from hdr as zp ~ skybr + 2.5*log10(L1MEDIAN/exptime/pixscale) ~ {}'.format(zp))
    glsnID = glsn[glsn['Source ID'] == groupid] # idx the glsn table using id 
    for i in hdr:
        print(i,hdr[i])
    
    print('filename ~ {} (groupid {}) has L1fwhm ~ {} pixels, pixscale ~ {} arcsec/pixel, moon stat/frac/dist/alt ~ {}/{}/{} /{}, skybr {} mag/arcsec^2'.format(filename,groupid,L1fwhm,pixscale,moon[0],moon[1],moon[2],moon[3],skybr))
    print('glsn ~ {}'.format(glsnID))
    print('\n')

    pickle_to = source_output + '/' + filename[:-5] # -5 get rid of .fits
    
    # photutils source properties to detect objs in image
    nsigma,kernel_size,npixels,deblend,contrast,targ_coord = 3,(3,3),int(np.round(L1fwhm/pixscale)),False,.001,None
    print('Source Catalog is a photutils source_properties using nsigma ~ {} (detection threshold above img bkg), gaussian kernel sized ~ {} pix, npixels ~ {} (connected pixels needed to be considered source), deblend ~ {} w contrast {}'.format(nsigma,kernel_size,npixels,deblend,contrast))
    source_catalog = source_cat(image,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
    cat,image,threshold,segm,targ_obj = source_catalog # unpacked to make a little clearer
    pickle.dump(cat.to_table(),open(pickle_to + '_source_cat.pkl','wb'))

    # get stars from the astroquery on gaia
    results = gaia_results(image)
    gaia,image = results # unpacked 
    # extract good gaia stars from image for psf
    extracted_stars = stars(results)
    good_stars,image = extracted_stars # unpacked
    # use extracted stars to build epsf
    EPSF = ePSF(extracted_stars,oversampling=2)
    epsf,fitted_stars = EPSF # unpacked
    pickle.dump(EPSF,open(pickle_to+'_epsf.pkl','wb'))
    # fit 2d gaussian to the epsf, see how 'non-gaussian' the actual psf is
    epsf_gaussian = gaussian2d(epsf)
    fit_gaussian,levels,xctr_vals,yctr_vals,image1,img_epsf,resid = epsf_gaussian # unpacked... levels list amplitude - sigma, ctr vals are gauss model sliced, image1 is array of values from gaussian fit in shape of epsf, img_epsf is epsf instance of it, resid is gauss - epsf 
    # make figures
    lco_figures.psf_and_gauss(epsf,epsf_gaussian,saveas=pickle_to+'_psf.pdf')
    lco_figures.used_stars(fitted_stars,saveas=pickle_to+'_stars.pdf')

    # target galaxy work, tuples cutting boxes around target (data,patch), how/if planting on cores might lower detection efficiency
    # also returns targ_obj again account for updates using ref (in the cases where empty targ_obj ie not detected in source)
    target_boxes = target(image,targ_obj,ref=ref_image,diff=diff_image) 
    targ_obj,cuts,bkg_core,bkg_1,bkg_2 = target_boxes # unpacked
    cut_targ,cut_diff,cut_ref = cuts # unpack cuts around target source,diff,and ref

    # measured psf is now going to be scaled to different magnitudes and planted in the difference image

    # key step to define threshold for detection, uses noise from scs of image
    threshold = detect_threshold(diff_image.data,nsigma=nsigma)
    threshold = np.median(threshold)
    threshold_mag = -2.5*np.log10(threshold/exptime/pixscale)+zp
    print("Threshold mag ~ {}, s/n ~ {} in difference considered detected".format(threshold_mag,nsigma))
    
    # does threshold change appreciably if consider more local region around target?
    xy,size = (hdr['CRPIX1'],hdr['CRPIX2']),400
    targ_region = Cutout2D(image.data,xy,size)
    targ_region_threshold = detect_threshold(diff_image.data,nsigma=nsigma)
    targ_region_threshold = np.median(threshold)
    targ_region_threshold_mag = -2.5*np.log10(threshold/exptime/pixscale)+zp
    print("Target region threshold mag ~ {}, s/n ~ {} in difference considered detected".format(targ_region_threshold_mag,nsigma))
    
    mags = np.arange(threshold_mag-2,threshold_mag+2,0.2) # zp ~ 23.5 # rough zp 
    locations = targ_lattice(targ_region,hdr)
    efficiencies = []
    for mag in mags:    
        plantname = '{}_planted_targetlattice_mag{}.fits'.format(pickle_to,str(mag))
        planted = plant(diff_image,epsf,source_cat=source_catalog,hdr=hdr,mag=mag,location=locations,zp=None,plantname=plantname)
        plant_im,pixels = planted # unpack

        # source properties of detected objs in fake image
        fakesource_cat = source_cat(plant_im,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
        fakecat,fakeimage,fakethreshold,fakesegm,faketarg_obj = fakesource_cat # unpacked to make a little clearer
        #pickle.dump(fakecat.to_table(),open(pickle_to+'_fakesource_cat.pkl','wb'))

        # detection efficiency  
        tmp = detection_efficiency(planted,fakesource_cat)
        efficiency,magfakes,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl = tmp
        efficiencies.append(efficiency)
        #pickle.dump(tmp,open(pickle_to+'_detection_efficiency_mag{}.pkl'.format(str(mag)),'wb'))
        print(efficiency,magfakes)
        print('--------------------------------------------------------------')

    print('efficiencies: {}'.format(efficiencies))
    print('mags: {}'.format(mags))
    # use interp to get magnitude at which we have 50% detection efficiency 
    # need the values increasing along x for interp to work properly
    efficiencies,mags=list(efficiencies),list(mags)
    efficiencies.reverse()
    mags.reverse()
    m50 = np.interp(0.5,efficiencies,mags)
    print('m50 ~ {}'.format(m50))

    # make figures
    lco_figures.detection_efficiency(mags,efficiencies,m50,target_boxes,skybr,zp,glsn=glsnID,saveas=pickle_to+'_target_detection_efficiency.pdf')
    #lco_figures.lattice_planted(mags,m50,pickle_to=pickle_to,saveas=pickle_to+'_plants.pdf')

    
    """
    # this loop (repeatedly) plants on target galaxy centroid; repeated so efficiency can be determined
    # the box plant wants a list of pixel coordinates, accessing from the target boxes here
    target_locations_orig = [bkg_core[0].center_original,bkg_1[0].center_original,bkg_2[0].center_original] # [0] so cut not patch
    target_locations_cutout = [bkg_core[0].center_cutout,bkg_1[0].center_cutout,bkg_2[0].center_cutout]
    # planting on target centroid at many magnitudes, I am not using the two shifts really only interested to see core
    # running detection repeatedly to determine efficiency
    efficiencies = []
    j = 0
    for mag in mags:
        efficiencies.append([])
        # planting into difference image
        box_plantname = '{}_planted_targ_mag{}.fits'.format(pickle_to,str(mag))
        box_planted_orig = plant(diff_image,epsf,source_catalog,hdr=hdr,mag=mag,location=[target_locations_orig[0]],zp=None,plantname=box_plantname)
        # unpack
        box_plant_im,box_pixels = box_planted_orig 
        # make target figures (similar to prev w source but now diff and plants)
        #lco_figures.target_image(box_plant_im,targ,saveas=pickle_to+'_target_plantdiff_mag{}.pdf'.format(str(mag)))
        # get a pdf showing image,ref,diff,fakeplant for SN w this mag
        lco_figures.view_targetplant(image,ref_image,diff_image,box_plant_im,target_boxes,zp,saveas=pickle_to+'targetplant_mag{}.pdf'.format(str(mag)))

        j += 1
        # The detection either works or doesn't for a plant there is no randomness in algorithm 
        # i.e. the catalog returned for an image is the same for a given detection
        # Therefor don't really need to do detection in range(0,N) but I do it twice anyways
        for i in range(0,2):
            # source properties of detected objs in fake image
            print(j,i)
            fakesource_cat = source_cat(box_plant_im,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
            fakecat,fakeimage,fakethreshold,fakesegm,faketarg_obj = fakesource_cat # unpacked to make a little clearer
            pickle.dump(fakecat.to_table(),open(pickle_to+'_fakesource_cat{}.pkl'.format(str(i)),'wb'))

            # detection efficiency  
            tmp = detection_efficiency(box_planted_orig,fakesource_cat)
            efficiency,magfakes,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl = tmp
            efficiencies[j-1].append(efficiency)
            pickle.dump(tmp,open(pickle_to+'_detection_efficiency{}_mag{}.pkl'.format(str(i),str(mag)),'wb'))
            print(efficiency,magfakes)
            print('--------------------------------------------------------------')
    avg_efficiencies=[]
    for i in efficiencies:
        efficiency = np.average(i)
        print(efficiency)
        avg_efficiencies.append(efficiency)
    avg_efficiencies,mags=list(avg_efficiencies),list(mags)
    avg_efficiencies.reverse()
    mags.reverse()
    m50 = np.interp(0.5,avg_efficiencies,mags)
    print('m50 ~ {}'.format(m50))  
    # make figures
    lco_figures.detection_efficiency(mags,avg_efficiencies,m50,target_boxes,skybr,zp,glsn=glsnID,saveas=pickle_to+'_target_detection_efficiency.pdf')
    """

    # lattice plant into the difference, a second way to do detection efficiency
    mags = np.arange(threshold_mag-2,threshold_mag+2,0.2) # zp ~ 23.5 # rough zp 
    locations = lattice(image)
    efficiencies = []
    for mag in mags:
        # create plant image; zp None, using measure sky mag/arcsec^2 from L1 hdr to set
        plantname = '{}_planted_lattice_mag{}.fits'.format(pickle_to,str(mag))
        planted = plant(diff_image,epsf,source_cat=source_catalog,hdr=hdr,mag=mag,location=locations,zp=None,plantname=plantname)
        plant_im,pixels = planted # unpack

        # source properties of detected objs in fake image
        fakesource_cat = source_cat(plant_im,nsigma=nsigma,kernel_size=kernel_size,npixels=npixels,deblend=deblend,contrast=contrast,targ_coord=None)
        fakecat,fakeimage,fakethreshold,fakesegm,faketarg_obj = fakesource_cat # unpacked to make a little clearer
        pickle.dump(fakecat.to_table(),open(pickle_to+'_fakesource_cat.pkl','wb'))

        # detection efficiency  
        tmp = detection_efficiency(planted,fakesource_cat)
        efficiency,magfakes,tbl,single_truth_tbl,repeat_truth_tbl,false_tbl = tmp
        efficiencies.append(efficiency)
        pickle.dump(tmp,open(pickle_to+'_detection_efficiency_mag{}.pkl'.format(str(mag)),'wb'))
        print(efficiency,magfakes)
        print('--------------------------------------------------------------')

    print(filename)
    print('efficiencies: {}'.format(efficiencies))
    print('mags: {}'.format(mags))
    # use interp to get magnitude at which we have 50% detection efficiency 
    # need the values increasing along x for interp to work properly
    efficiencies,mags=list(efficiencies),list(mags)
    efficiencies.reverse()
    mags.reverse()
    m50 = np.interp(0.5,efficiencies,mags)
    print('m50 ~ {}'.format(m50))

    # make figures
    lco_figures.detection_efficiency(mags,efficiencies,m50,target_boxes,skybr,zp,glsn=glsnID,saveas=pickle_to+'_detection_efficiency.pdf')
    lco_figures.lattice_planted(mags,m50,pickle_to=pickle_to,saveas=pickle_to+'_plants.pdf')


if __name__=="__main__":
    #print('lco pipe coming at ya')
    #lco_pipe()
    peter_pipe()
    #df_stack()
    """
    #import read_slate
    df = pd.read_csv('source_data.txt',delimiter='|',skiprows=1)

    df['NIa'] = pd.to_numeric(df['NIa'], errors='coerce')
    df['Ncc'] = pd.to_numeric(df['Ncc'], errors='coerce')
    df['e_NIa'] = pd.to_numeric(df['e_NIa'], errors='coerce')
    df['e_Ncc'] = pd.to_numeric(df['e_Ncc'], errors='coerce')

    firsts,finals, dates = [],[],[]
    for name in df[' Our Survey Name ']:
        tmp = read_slate.get_obs(name) # list of datetimes field name was observed
        dates.append(tmp) 
        firsts.append(np.min(tmp))
        finals.append(np.max(tmp))

    idx = int(sys.argv[1]) # 98 fields 0-97 sbatched
    name = df[' Our Survey Name '][idx]
    datetimes = dates[idx]
    for datetime in datetimes:
        print(name,datetime)
        dl_headers(datetime,name,path=True)
    """