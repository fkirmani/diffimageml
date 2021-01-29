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


def PIL_IM(png,channel=None,writetodisk=False,saveas=None,show=False,pkl=False):
    """
    https://pillow.readthedocs.io/en/stable/handbook/tutorial.html
    Python Imaging Library (PIL) has many useful image processing tools for Image Class
    
    Notation...
    A. a difference image, has detection (plant or false positive)
    B. a 'search' image (typically a "new" single-epoch static sky image), has detection
    C. the template image (or 'reference'), does not have detection
    
    RGB...
    A secondary book-keeping technique employed for channels in which the ABC image data are stored
    Another method to cleanly identify which type of image is being looked at, should be no loss of data for ML
    A. R-channel png
    B. G-channel png
    C. B-channel png
    
    Parameters
    ----------
    png : str 
        filename of image which has detected sources to train on
    channel : str
        Default None attempts to use png (filename) to set. If provided needs to be str(r) or str(g) or str(b)
    writetodisk : bool
        Default False. True or False to save the png converted to given channel. 
    saveas : None or str
        Default None will use the png filename. For LCO taken from the header['MEF'] ~ FKNNN.png or FPNNN.png 
    pkl : bool
        Default False. Do we want option to save the opened PIL Image Class to disk as a pickle
    show : bool
        True or False to display the image
    Returns
    ---------
    PIL.PngImagePlugin.PngImageFile
    """

    # read in the image
    try:
        im = Image.open(png)
    except:
        print("Couldn't read png image provided. Please try again.")
        return
    
    # determine the mode (will probably be RGBA A ~ alpha/transparency)
    mode = im.mode
    print(mode)
    
    # put into RGB
    if  im.mode != "RGB":
        print("converting from {} to RGB".format(mode))
        rgb = im.convert("RGB")
    else:
        rgb = im
    
    # Define matrices to convert RGB to provided channel 
    """
    __________________________________________________
                    ## MATRIX ## 
           R           G            B       constants
     R: 1*oldRed + 0*oldGreen + 0*oldBlue +    C
     G: 0*oldRed + 1*oldGreen + 0*oldBlue +    C
     B: 0*oldRed + 0*oldGreen + 1*oldBlue +    C
     ________________________________________________
    """
    
    rmatrix = ( 1, 0, 0, 0, 
           0, 0, 0, 0, 
           0, 0, 0, 0) 
    gmatrix = ( 0, 0, 0, 0, 
           0, 1, 0, 0, 
           0, 0, 0, 0) 
    bmatrix = ( 0, 0, 0, 0, 
           0, 0, 0, 0, 
           0, 0, 1, 0) 
    
    # if not provided channel explicitly will attempt to use filename
    if channel == None:
        channel = png[0].lower()
        if channel != "r" or channel != "g" or channel != "b":
            print("Don't understand the channel. Please try again.")
            return
    # convert the image to given channel
    if channel.lower() == "r":
        img = rgb.convert("RGB",rmatrix)
    elif channel.lower() == "g":
        img = rgb.convert("RGB",gmatrix)
    elif channel.lower() == "b":
        img = rgb.convert("RGB",bmatrix)
    else:
        print("Don't understand the channel. Need to provide str(r) or str(g) or str(b). Please try again.")
        return
    
    if saveas == None:
        saveas = channel+'_'+os.path.basename(png)
    
    # save the converted channel png
    if writetodisk:
        img.save(saveas)
    
    # pickle im, the 0-level PIL.PngImagePlugin.PngImageFile read-in if want to do more processing
    if pkl:
        pklas = saveas[:-3] + "pkl"
        pickle.dump(im,open(pklas,"wb"))
    
    if show:
        img.show()
        
    return img


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



if __name__ == "__main__":
    import glob
    lco,lsst = False,True
    if lco:
        Apngs = glob.glob("test_data/cnninputdata2/class_lco/A/*png")
        Bpngs = glob.glob("test_data/cnninputdata2/class_lco/B/*png")
        Cpngs = glob.glob("test_data/cnninputdata2/class_lco/C/*png")
    if lsst:
        Apngs = glob.glob("test_data/cnninputdata2/class_lsst/A/*")
        Bpngs = glob.glob("test_data/cnninputdata2/class_lsst/B/*")
        Cpngs = glob.glob("test_data/cnninputdata2/class_lsst/C/*")

    for file in Apngs:
        print(file)
        PIL_IM(file,channel="r",writetodisk=True)
    for file in Bpngs:
        print(file)
        PIL_IM(file,channel="g",writetodisk=True)
    for file in Cpngs:
        print(file)
        PIL_IM(file,channel='b',writetodisk=True)

