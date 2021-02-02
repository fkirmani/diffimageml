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

import PIL
from PIL import Image
import cv2

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

    example_data['psfmodel1'] = os.path.abspath(os.path.join(
        example_data['dir'], "sky_image_1_TestEPSFModel.pkl"))

    example_data['cnninputdatadir1'] = os.path.abspath(os.path.join(
        example_data['dir'], "cnninputdata1"))


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
    
    print("cut_hdu")
    print("location,wcs",location,wcs)
    print("shape",dat.shape)
    cut = Cutout2D(dat,location,size,wcs=wcs) 
    cutwcs = cut.wcs
    cphdu.data = cut.data
    cphdu.header.update(cut.wcs.to_header())  
    
    if writetodisk:  
        cphdu.writeto(saveas,overwrite=True)

    self.postage_stamp = cphdu
    
    return cphdu

def get_lattice_positions(self, edge=100, spacing=100):
    """Function for constructing list of pixels in a grid over the image

    Parameters
    :edge : int
    "Gutter" pixels away from the edge of the image for the start/end of grid

    :spacing : int
    Number of pixels in x and y directions between each fake
    """
    hdu = self.diffim.sci
    #reads number of rows/columns from header and creates a grid of locations for planting
    hdr = hdu.header
    wcs,frame=WCS(hdr),hdr['RADESYS'].lower()
    
    NX = hdr['naxis1']
    NY = hdr['naxis2']
    x = list(range(0+edge,NX-edge+1,spacing)) # +1 to make inclusive
    y = list(range(0+edge,NY-edge+1,spacing))
    pixels = list(itertools.product(x, y))
    skycoords = [] # skycoord locations corresponding to the pixels  
    for i in range(len(pixels)):
        pix = pixels[i]
        skycoord=astropy.wcs.utils.pixel_to_skycoord(pix[0],pix[1],wcs)
        skycoords.append(skycoord)

    self.has_lattice = True # if makes it through lattice update has_lattice

    return np.array(pixels), np.array(skycoords)


def lco_epsf(self):
    """
    Another ePSF option besides building just use circular gaussian from
    lco header on the static sky search im
    At some point may want to have a class like telescope_psf where we can
    store pre-defined (i.e. not built) epsf
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
    epsfdata = photutils.datasets.make_gaussian_sources_image(shape, table,oversample=oversample)

    # make this produce a fittable2d psf model
    epsfmodel = photutils.psf.FittableImageModel(epsfdata, normalize=True)

    # TODO : we should include a normalization_correction to account for
    #  an "aperture correction" due to data outside the model

    self.has_lco_epsf = True # update bool if makes it through this function
    self.lco_epsf = epsfmodel

    return epsfmodel

def extract_psf_fitting_names(psf):
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


def write_to_catalog(columns , filename = "cat.ecsv" , column_names = None, overwrite = False ,  add_to = False):
    

    """
    
    Takes in a list of columns and writes them to a catalog file
    Includes handling to add to an existing catalog
    
    

    Parameters
    ----------
    
    columns: list or numpy array
        Array containing columns to be written in the catalog.
        Each column can be an Astropy column, list or a numpy array
        
    column_names: array
        Array containing labels for the provided columns to be incuded in the catalog
        Not necessary if the columns provided are already Astropy columns
        If none, we will not add labels to the columns
        
    filename: str
        If None, do not save to disk. If provided, save the catalog under this filename

    overwrite: boolean
        When True, overwrite an existing catalog
        Otherwise, will only save catalog if it does not already exist
        
        
    add_to: boolean
        If True, the provided columns will be appended to the given file.
        This is useful for producing a catalog with information stretching
        across many files.
    
    Returns
    _______
    
    catalog : Astropy Table : Table containing the input information
    
    
    """
    
    ##Prepare columns if necessary.

    if type(columns) != Table:
    
        for i in range(len(columns)):
            if type(columns[i]) == Column:
                continue
            else:
                if column_names == None:
                    columns[i] = Column(columns[i])
                else:
                    columns[i] = Column(columns[i] , name = column_names[i])
        catalog = Table(columns)
    else:
        catalog = columns
            
    file_format = "ascii.ecsv"
    
    if filename == None: ##Don't save to file, hust return catalog
        return catalog
    
    if not add_to: ##Generate new file or overwrite existing file
    
        if  overwrite: ##Writes (or overwrites) new file 
        
            catalog.write( filename , format = file_format , overwrite = True)
            
        else: ##Only write if file does not exist

            if os.path.exists(filename) and not overwrite:
                print ("Warning, file exists but overwrite flag is False. Will not save catalog")
                catalog = Table(columns)
                return catalog
                
            else:
                catalog.write( filename , format = file_format , overwrite = False)
        
    elif add_to: ##Add to existing file if possible
    
        if os.path.exists(filename): 
            ##File exists, so we add to the existing catalog
            
            current_catalog = read_catalog(filename)

            if len(current_catalog.columns) != len(new_table.columns):
                print ("Warning, mismatch in number of columns between current file and input data")
                print ("File has {} columns, but write function was provided with {} columns".format(len(current_catalog.columns), len(new_table.columns)))
                print ("Continuing without saving catalog")
                return new_table
            catalog = vstack([current_catalog , catalog])
            catalog.write(filename , format = file_format , overwrite = True)
            
        else:
            ##File does not exist, so we make one
            
            catalog.write(filename , format = file_format , overwrite = True)
         
    return catalog


def lco_xid_sdss_query():
    """
    Get sdss star properties, rmag and wcs for 112 target fields in our LCOLSS program.
    Needed if want to do a ZP calibration to the LCO data.

    Note requires Visiblity.csv stored locally. Matches ra/dec of lco targets from Visibility.csv 
    to stars from sdss database
    """

    from astroquery.sdss import SDSS
    visibility = ascii.read('Visibility.csv')

    Source_IDs = visibility['col1'] # Source IDs
    ra_deg = visibility['col2']
    dec_deg = visibility['col3'] 
    for idx in range(1,len(visibility[1:])+1):
        print("------------------------------")
        obj,ra,dec = Source_IDs[idx],ra_deg[idx],dec_deg[idx]
        print(idx,obj,ra,dec)
        
        """
        full_radius ~ pixscale * 2048 is arcsec from center of an LCO exposure image
        go to 90% of that radius to account for target ra/dec dithers i.e. not being perfectly centered and edge effects
        """
        full_radius = 0.389*(4096/2)    
        radius = 0.85*full_radius
        strradius = str(radius) + ' arcsec'
        print(radius,'ra ~ [{:.2f},{:.2f}], dec ~ [{:.2f},{:.2f}]'.format(float(ra)-radius/3600,float(ra)+radius/3600,float(dec)-radius/3600,float(dec)+radius/3600))
        fields = ['ra','dec','objid','run','rerun','camcol','field','r','mode','nChild','type','clean','probPSF',
                 'psfMag_r','psfMagErr_r'] 
        pos = SkyCoord(ra,dec,unit="deg",frame='icrs')
        xid = SDSS.query_region(pos,radius=strradius,fields='PhotoObj',photoobj_fields=fields) 
        Star = xid[xid['probPSF'] == 1]
        Gal = xid[xid['probPSF'] == 0]
        print(len(xid),len(Star),len(Gal))
        Star = Star[Star['clean']==1]
        print(len(Star))
        ascii.write(Star,f"{obj}_SDSS_CleanStar.csv")
        
        idx+=1

def LCO_PSF_PHOT(hdu,init_guesses):
    # im ~ np array dat, pixel [x0,y0] ~ float pixel position, sigma_psf ~ LCO-PSF sigma 
    x0,y0=init_guesses
    im = hdu.data
    hdr = hdu.header
    
    fwhm = hdr['L1FWHM']/hdr['PIXSCALE'] # PSF FWHM in pixels, roughly ~ 5 pixels, ~ 2 arcsec 
    sigma_psf = fwhm*gaussian_fwhm_to_sigma # PSF sigma in pixels
    
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()

    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    pos = Table(names=['x_0', 'y_0'], data=[[x0],[y0]]) # optionally give flux_0 has good aperture method for guessing though

    photometry = BasicPSFPhotometry(group_maker=daogroup,
                                     bkg_estimator=mmm_bkg,
                                     psf_model=psf_model,
                                     fitter=LevMarLSQFitter(),
                                     fitshape=(11,11))
    result_tab = photometry(image=im, init_guesses=pos)
    residual_image = photometry.get_residual_image()
    
    return result_tab

def table_header(hdu,idx=0):
    """
    Make a pandas data frame out of the fits header object
    """
    hdr = hdu.header
    d = {}
    for i in hdr:
        name = str(i)
        try:
            dat = float(hdr[i])
        except:
            dat = str(hdr[i])
        d[name] = dat
    df = pd.DataFrame(data=d,index=[idx])
    return df

def ZPimage(lco_phot_tab,matched_sdss,plot=True,saveas="zp.png",scs=True):
    """
    Make png images showing the determination of ZP.

    Parameters
    ___________
    lco_phot_tab ~ Astropy Table
        flux_fit, flux_unc from PSF-Phot on LCO image stars
    matched_sdss ~ Astropy Table
        psfMag_r, psfMagErr_r from matching sdss data

    Returns
    __________
    Weighted Average ~ ZP(star)
        ZP from each star using rmag and flux. 
    
    Linear Fit ~ rmag(-2.5log10flux)
        ZP from intercept.
    """
    try:
        assert(len(lco_phot_tab) == len(matched_sdss))
        print("{} data".format(len(lco_phot_tab)))
    except:
        print("Photometry and SDSS table aren't the same shape.")
        ZP,b = None,None
        return ZP,b

    flux_lco,flux_unc_lco = np.array(lco_phot_tab['flux_fit']),np.array(lco_phot_tab['flux_unc'])
    rmag,rmagerr = np.array(matched_sdss['psfMag_r']),np.array(matched_sdss['psfMagErr_r'])

    # clip negative fluxes 
    flux_clip = np.log10(lco_phot_tab['flux_fit'])
    indices = ~np.isnan(flux_clip)
    true_count = np.sum(indices)
    
    flux_lco,flux_unc_lco = flux_lco[indices],flux_unc_lco[indices]
    rmag,rmagerr = rmag[indices],rmagerr[indices]

    print("{} data after clipping bad fluxes".format(true_count))

    # uncertainties and weights
    lco_uncertainty = flux_unc_lco/flux_lco
    sdss_uncertainty = rmagerr
    uncertainties = []
    for i in range(true_count):
        uncertainties.append( np.sqrt(lco_uncertainty[i]**2 + sdss_uncertainty[i]**2) )
    print("median uncertainties {:.2f}, median sdss uncertainties {:.2f}, median lco uncertainties {:.2f}".format(np.median(uncertainties),np.median(sdss_uncertainty),np.median(lco_uncertainty)))
    weights = [1/i for i in uncertainties] 

    # ZP as weighted average of zp = m + 2.5 log10(f), for each star
    # m from sdss, f from psf-fit on lco  
    values = rmag + 2.5*np.log10(flux_lco)
    values = [i for i in values]
    ZP = util.weighted_average(weights,values)
    # clip remaining bad data using scs around weighted avg ZP
    if scs:
        zp_clip = sigma_clip(values,sigma=3)
        indices = ~zp_clip.mask
        true_count = np.sum(indices)
        values = np.array(values)[indices]
        weights = np.array(weights)[indices]
        ZP = util.weighted_average(weights,values)
        print("{} data after clipping around ZP from weighted average".format(true_count))
    print("ZP from weighted_average = {:.2f}".format(ZP))
    flux_lco,flux_unc_lco = flux_lco[indices],flux_unc_lco[indices]
    rmag,rmagerr = rmag[indices],rmagerr[indices]
    lco_uncertainty,sdss_uncertainty=lco_uncertainty[indices],sdss_uncertainty[indices]

    # weighted average plot
    if plot:
        matplotlib.rcParams.update({'font.size': 20,'xtick.labelsize':15,'ytick.labelsize':15})
        fig,ax = plt.subplots(figsize=(16,8))
        spacing = np.arange(0,true_count,1)
        uncertainties = [1/i for i in weights]
        ax.errorbar(spacing,values,yerr=uncertainties,marker='x',ls='',color='red',label='zp')
        ax.hlines(ZP,0,true_count,linestyle='--',label='ZP={:.1f}'.format(ZP),color='black')
        plt.xlabel("Star")
        plt.ylabel("$rmag_{sdss}$")
        plt.legend()
        plt.show()
        plt.legend()
        plt.savefig("weighted_average_"+saveas,bbox_inches='tight')
        plt.close()

    # ZP as intercept in linear fit 
    xi = -2.5*np.log10(flux_lco)
    xerr = lco_uncertainty
    fig,ax = plt.subplots(figsize=(16,8))
    m,b=np.polyfit(xi,rmag,1,w=weights)
    if plot:
        ax.errorbar(xi,rmag,xerr=xerr,yerr=rmagerr,marker='x',ls='',color='red',label='')
        x=np.linspace(np.min(xi),np.max(xi),100)
        y=m*x + b
        ax.plot(x,y,ls='--',color='black',label='ZP={:.1f}'.format(b))
        plt.show()
        plt.xlabel("$-2.5log10(f_{lco})$")
        plt.ylabel("$rmag_{sdss}$")
        plt.legend()
        plt.savefig("lin_fit_"+saveas,bbox_inches='tight')
        plt.close()
    print("ZP as interecept of linear fit = {:.2f}".format(b))

    try:
        assert(np.abs(ZP-b) < 0.2)
    except:
        # things that make you go hmmm
        print("weighted average and intercept ZPs disagree by more than 0.2 mag",ZP,b)
        ZP,b = None,None
        return ZP,b
    
    return ZP,b 

def lco_sdss_pipeline(self,threshold=10,writetodisk=False):
    """
    A pipeline to measure detection efficiency of fake planted point sources in LCO difference images. 
    The PSF-flux calibrated to AB magnitudes using ZP from sdss data. 

    Note assumes lco_xid_query has been run. i.e. that sdss data is stored locally ~ sdss_queries/target_SDSS_CleanStar.csv

    Parameters
    ---------------
    hdu : ~astropy.io.fits
        The LCO search image
    diffhdu : ~astropy.io.fits 
        The LCO difference image  
    threshold : float
        The S/N used in DAOStarFinder. Default 10

    Returns
    _________________
    df : pandas DataFrame
        Has columns with the header values and m50,alpha of efficiency.
        If pipeline fails df flag column returns idx corresponding to step of failure. 
    matched_lco : Astropy Table
        The stars in LCO image matched to known SDSS
    matched_sdss : Astropy Table
        The stars in SDSS matched to detected in LCO   
    lco_phot_tab : Astropy Table
        The LCO image photometry on the stars in matched_lco

    Steps in pipeline:
    1. Use DAOFIND to detect stars in the LCO images, https://photutils.readthedocs.io/en/stable/detection.html
    2. Use the hdu wcs to determine ra/dec of xcentroid&ycentroids for stars found
    3. Read the sdss_queries/target_SDSS_CleanStar.csv using hdu target (has sdss rmag and ra/dec, trimmed to good mags for ZP)
    4. Find DAO sources within 5 arcsec of SDSS sources
    5. Do Basic PSF-Photometry on stars in LCO matched to a SDSS, using L1FHWM and IntegratedGaussianPRF
    6. The ZP is the weighted average. weights of each ZP measurement using SDSS-rmags and LCO-fluxes fits 
    # I've ommitted 7,8 (if Fawad/SR ends up wanting them let me know)
    7. Add PSF to data at different mags and measure detection efficiencies
    8. Fit model of m50,alpha to efficiencies 
    """
    hdu = self.searchim.sci
    diffhdu = self.diffim.sci
    epsfmodel = lco_epsf(self)

    origname = hdu.header['ORIGNAME'].split("-e00")[0]
    print(origname)
    
    # 1. DAO
    print("\n")
    print("1. DAOStarFinder on exposure")
    data,hdr = hdu.data, hdu.header 
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)  
    print("mean {:.2f}, median {:.2f}, std {:.2f}".format(mean, median, std))  
    fwhm = hdr["L1FWHM"]
    print("threshold {:.1f},fwhm {:.2f} arcsec".format(threshold,fwhm))
    daofind = DAOStarFinder(fwhm=fwhm, threshold=5.*std)  
    sources = daofind(data - median)

    print("{} DAO sources".format(len(sources)))
    print(sources.columns)

    try:
        assert(fwhm >= 1.0 and fwhm <= 5.0)
    except:
        print("Image_Error 1. FWHM {:.2f} , typical is [1.5,3.5] arcsec strong peak at 2.".format(fwhm))
        print("ZP=m50=alpha=None")
        # make pd DF out of the header and store m50,alpha
        df = table_header(diffhdu,idx=origname)
        df['m50'] = None
        df['alpha'] = None
        df['ZP'] = None
        df['flag'] = 1
        print(df)
        if writetodisk:
            pickle.dump(df,open(f"{origname}_df.pkl","wb"))
        matched_lco,matched_sdss,lco_phot_tab = None,None,None
        return df,matched_lco,matched_sdss,lco_phot_tab
    
    # 2. LCO Skycoords
    print("\n")
    print("2. Ra/Dec of stars found in exposure using hdr wcs")
    lco_skycoords = []
    for i in range(len(sources)):
        pixel = [sources[i]['xcentroid'],sources[i]['ycentroid']]
        sky = pixtosky(hdu,pixel)
        lco_skycoords.append(sky)
    lco_skycoords = SkyCoord(lco_skycoords)

    # 3. Read-in SDSS
    print("\n")
    print("3. Reading in sdss star catalog for hdr object")
    obj = hdu.header['OBJECT']
    sdss = ascii.read(f"sdss_queries/{obj}_SDSS_CleanStar.csv")
    sdss_skycoords = SkyCoord(ra=sdss['ra'],dec=sdss['dec'],unit=units.deg)
    print("{} sdss".format(len(sdss)))
    print(sdss.columns)

    try:
        assert(len(sources) >= 0.1*len(sdss))
    except:
        print("Image_Error 3. DAO detected {} stars, < 10 percent of stars in sdss {}, something wrong with image",len(sources),len(sdss))
        print("ZP=m50=alpha=None")
        # make pd DF out of the header and store m50,alpha
        df = table_header(diffhdu,idx=origname)
        df['m50'] = None
        df['alpha'] = None
        df['ZP'] = None
        df['flag'] = 3
        print(df)
        if writetodisk:
            pickle.dump(df,open(f"{origname}_df.pkl","wb"))
        matched_lco,matched_sdss,lco_phot_tab = None,None,None
        return df,matched_lco,matched_sdss,lco_phot_tab
    
    # 4. Match SkyCoords
    print("\n")
    print("4. Determining stars in both lco DAO and sdss")
    matchcoord,catalogcoord = lco_skycoords,sdss_skycoords
    # shapes match matchcoord: idx into cat, min angle sep, unit-sphere distance 
    idx,sep2d,dist3d=match_coordinates_sky(matchcoord,catalogcoord)
    good_lcoidx,good_sdssidx,good_sep2d = [],[],[]
    matched_lco,matched_sdss = [],[]
    for i in range(len(sources)):
        if sep2d[i] < 5*units.arcsec:
            matched_lco.append(sources[i])
            matched_sdss.append(sdss[idx[i]])
            good_lcoidx.append(i)
            good_sdssidx.append(idx[i])
            good_sep2d.append(sep2d[i])
        else:
            pass

    try:
        assert(len(matched_lco) > 20)
        matched_lco,matched_sdss = vstack(matched_lco),vstack(matched_sdss)
        print("After matching (<5 arcsec separation), {} DAO sources, {} sdss".format(len(matched_lco),len(matched_sdss)))
        print("{:.2f} arcsec median separation".format(np.median([i.value*3600 for i in good_sep2d]))) 
    except:
        print("Image_Error 4. Matched < 20 stars, something wrong (possibly wcs), not enough to do photometry and calibrate ZP.")
        print("ZP=m50=alpha=None")
        # make pd DF out of the header and store m50,alpha
        df = table_header(diffhdu,idx=origname)
        df['m50'] = None
        df['alpha'] = None
        df['ZP'] = None
        df['flag'] = 4
        print(df)
        if writetodisk:
            pickle.dump(df,open(f"{origname}_df.pkl","wb"))
        lco_phot_tab = None
        return df,matched_lco,matched_sdss,lco_phot_tab   

    # 5. Photometry 
    print("\n")
    print("5. Doing Basic PSF-Photometry on the lco stars")
    lco_phots = []
    for i in range(len(matched_lco)):
        location,size = [matched_lco[i]['xcentroid'],matched_lco[i]['ycentroid']],50
        postage_stamp = cut_hdu(hdu,location,size)
        init_guess = postage_stamp.data.shape[0]/2,postage_stamp.data.shape[1]/2 # should be at center
        lco_psf_phot = LCO_PSF_PHOT(postage_stamp,init_guess)
        lco_phots.append(lco_psf_phot)
    try:
        assert(len(lco_phots) == len(matched_lco))
        lco_phot_tab = vstack(lco_phots)
        print("{} LCO PSF-Photometry".format(len(lco_phot_tab)))
        print(lco_phot_tab.columns)
        # write the successful matched stars & photometry into pkl
        pickle.dump(lco_phot_tab,open(f"{origname}_phot.pkl","wb"))
        pickle.dump(matched_lco,open(f"{origname}_match_lco.pkl","wb"))
        pickle.dump(matched_sdss,open(f"{origname}_match_sdss","wb"))
    except:
        print("Image Error 5. Photometry failed.")
        print("ZP=m50=alpha=None")
        # make pd DF out of the header and store m50,alpha
        df = table_header(diffhdu,idx=origname)
        df['m50'] = None
        df['alpha'] = None
        df['ZP'] = None
        df['flag'] = 5
        print(df)
        if writetodisk:
            pickle.dump(df,open(f"{origname}_df.pkl","wb"))
        lco_phot_tab = None
        return df,matched_lco,matched_sdss,lco_phot_tab 

    # 6. ZP as weighted average or linear intercept 
    # m from sdss, f from psf-fit on lco  
    print("\n")
    print("6. Getting ZP from weighted average or linear-intercept using sdss-rmags and lco-psf flux_fits")
    ZP,b = ZPimage(lco_phot_tab,matched_sdss,scs=True,plot=True,saveas=f"{origname}_zp.png")
    try:
        assert(ZP != None and b != None)
        print("ZP = {:.2f}, b = {:.2f}".format(ZP,b))
        ZP = b
        print("Using the linear intercept value as true ZP")
    except:
        print("Image Error 6. ZP calibration failed.")
        print("ZP=m50=alpha=None")
        # make pd DF out of the header and store m50,alpha
        df = table_header(diffhdu,idx=origname)
        df['m50'] = None
        df['alpha'] = None
        df['ZP'] = None
        df['flag'] = 6
        print(df)
        if writetodisk:
            pickle.dump(df,open(f"{origname}_df.pkl","wb"))
        return df,matched_lco,matched_sdss,lco_phot_tab 

    # make pd DF out of the header and store m50,alpha
    df = table_header(diffhdu,idx=origname)
    df['m50'] = None
    df['alpha'] = None
    df['ZP'] = ZP
    df['flag'] = 0
    print(df)
    if writetodisk:
        pickle.dump(df,open(f"{origname}_df.pkl","wb"))

    return df,matched_lco,matched_sdss,lco_phot_tab
    
def read_catalog(filename):
    '''
    Takes in the filename for a catalog
    
    Will return the Astropy Table stored in this file
    
    '''
    file_format = "ascii.ecsv"
    catalog = Table.read(filename , format = file_format)
    return catalog
    
def fits_rgb_png(mef , savefilename = None , rescale = False):
    '''
    Function to combine three fits images into a single RGB png file
    Assignes the difference image to be red, the search image to green
    and the template image to blue. Note that if rescale is set to False, then
    the images may appear blank, as the largest pixel values inside the fits images
    may be much less than the max allowed pixel value, so when viewed the images appear
    to be solid black. rescale = True eliminates this problem by rescaling the fits file
    data.
    
    Parameters
    __________
    
    mef: This can be a multi-extention fits file. Should include a difference image,
        search image and template image.
        Can also be a list of FitsImage objects or a FakePlanter object
        
    savefilename: str : If None, do not save resulting png file to disk. Otherwise, save
        resulting png to this filename
    
    rescale: boolean : If True, rescale data such that the largest pixel value in the triplet
        is rescaled to the largest allowed pixel value in the 16 bit format. Will preserve as much
        information as possible
    
    Returns
    _______
    
    Returns a numpy array containing the image data
    
    '''
    
    #The following generates an RGB image and saves it using PIL
    ##PIL Only handles 3x8 png images with RGB, so pixel data has to be manipulated to make it fit

    diff_data = mef[1].data
    search_data = mef[2].data
    templ_data = mef[3].data

    
    '''
    
    Not really necessary now that we support 16 bit color
    ##Shift the data so that the smallest pixel value is 0.
    ##Reduces the amount of rescaling necessary
    diff_shift = ( np.amin(diff_data) )
    search_shift = (np.amin(search_data) )
    templ_shift = ( np.amin(templ_data) )

    
    diff_data -= diff_shift
    search_data -= search_shift
    templ_data -= templ_shift
    '''



    largest_allowed_pix_value = 65535
    largest_pix_value = max( np.amax(diff_data) , np.amax(templ_data) , np.amax(search_data))
    
    if rescale:
        compression_factor = max( largest_pix_value / largest_allowed_pix_value , 0.0)
    else:
        compression_factor = max( largest_pix_value / largest_allowed_pix_value , 1.0) ##Only rescale in necessary
    
    print ("Data will be rescaled by a factor of {}".format(compression_factor))
    
    diff_data /= compression_factor
    search_data /= compression_factor
    templ_data /= compression_factor

    ##Check pixel values, print warning if we are overflowing max allowed value.
    ##If we are, png files will round any values greater than 255 to 255
    
    if np.amax(diff_data) > largest_allowed_pix_value:
        print ("Warning, max pixel value in diffs ({}) excedes max png pixel value".format(np.amax(diff_data)))
    if np.amax(search_data) > largest_allowed_pix_value:
        print ("Warning, max pixel value in search ({}) excedes max png pixel value".format(np.amax(search_data)))
    if np.amax(templ_data) > largest_allowed_pix_value:
        print ("Warning, max pixel value in templ ({}) excedes max png pixel value".format(np.amax(templ_data)))

    
    ### cv2 orders the colors as BGR, so we want template , search , diff
    

    bgrim = cv2.merge( ( templ_data.astype(np.uint16) , search_data.astype(np.uint16) , diff_data.astype(np.uint16)) )
    


    #Save outputs
    if savefilename != None:
    
        cv2.imwrite(savefilename , bgrim)
    
        
    return bgrim
