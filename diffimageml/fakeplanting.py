import numpy as np
import os

from astroquery.gaia import Gaia

from astropy import units
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel
from astropy.io import ascii,fits
from astropy.nddata import Cutout2D,NDData
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm
from astropy.table import Table,Column,MaskedColumn,Row,vstack,setdiff,join
from astropy.wcs import WCS, utils as wcsutils

import photutils
from photutils.datasets import make_gaussian_sources_image
from photutils import Background2D, MedianBackground
from photutils import detect_sources, source_properties
from photutils.psf import EPSFModel, extract_stars
from photutils import EPSFBuilder, BoundingBox
from photutils import Background2D, MedianBackground
from photutils import EllipticalAperture, detect_threshold, deblend_sources

import itertools
import copy
import pickle


# astropy Table format for the gaia source catalog
_GAIACATFORMAT_ = 'ascii.ecsv'
_GAIACATEXT_ = 'ecsv'

# Column names for the magnitudes and S/N to use for selecting viable PSF stars
_GAIAMAGCOL_ =  'phot_rp_mean_mag'
_GAIASNCOL_ = 'phot_rp_mean_flux_over_error'

# Size of the box for each PSF star cutout (half width? or full?)
#  Does this also set the size of the resulting ePSF model?
_PSFSTARCUTOUTSIZE_ = 25 # pixels


class FakePlanterEPSFModel():
    """ A class for holding an effective PSF model.

    """
    def __init__(self):
        """

        """
        # TODO: zeropoint is measured in the FitsImage class
        #  maybe we should require it exists, then inherit the value here?
        self.zeropoint = 0
        self.epsf = None
        self.fitted_stars = None
        return

    def scaled_to_mag(self, mag):
        """Return a data array scaled to the given magnitude.
        Requires that a zeropoint has been set.
        """
        # TODO : add a check that zeropoint has been set by user
        return self.epsf.data * 10**(-0.4*(mag-self.zeropoint))


    def showepsfmodel(self):
        """ TODO: visualize the ePSF model"""
        norm = simple_norm(self.epsf.data, 'log', percent=99.)
        plt.imshow(self.epsf.data, norm=norm, origin='lower', cmap='viridis')
        plt.colorbar()
        return

    def writetofits(self):
        """TODO: write to a fits file"""
        #fits.writeto(name,epsf.data,hdr,overwrite=True)
        #         fits.writeto(plantname,image.data,hdr,overwrite=True)
        return


class FitsImage:
    """A class to hold a single FITS image and associated products
    such as a PSF model, and detected source catalog.
    """
    def __init__(self, fitsfilename):
        """
        Constructor for FitsImage class. 

        Parameters
        ----------
        fitsfilename : str
            Name of fits file to read (can be .fits or .fits.fz)

        Returns
        -------
        FitsImage : :class:`~diffimageml.FitsImage`
        """

        self.filename = fitsfilename
        self.read_fits_file(fitsfilename)
        self.psfstars = None
        self.psfmodel = None
        self.epsf = None

        self.sourcecatalog = None
        self.zeropoint = None
        return

    def read_fits_file(self,fitsfilename):
        """
        Read in a fits file. 

        Parameters
        ----------
        fitsfilename : str
            Name of fits file to read (can be .fits or .fits.fz)

        Returns
        -------
        hdulist : :class:`~astropy.io.fits.HDUList`
        sci :  the science array :class:`~astropy.io.fits.PrimaryHDU` (or similar)

        """
        self.hdulist = fits.open(fitsfilename)
        self.filename = fitsfilename
        if 'SCI' in self.hdulist:
            self.sci = self.hdulist['SCI']
        else:
            for i in range(len(self.hdulist)):
                if self.hdulist[i].data is not None:
                    self.sci = self.hdulist[i]

        # image World Coord System
        self.wcs = WCS(self.sci.header)

        # Sky coordinate frame
        # TODO : Not sure we can expect the RADESYS keyword is always present.
        # Maybe there's an astropy function to get this in a more general way?
        self.frame = self.sci.header['RADESYS'].lower()

        return self.hdulist, self.sci


    def pixtosky(self,pixel):
        """
        Given a pixel location returns the skycoord
        """
        hdu = self.sci
        hdr = hdu.header
        wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
        xp,yp = pixel
        sky = wcsutils.pixel_to_skycoord(xp,yp,wcs)
        return sky

    def skytopix(self,sky):
        """
        Given a skycoord (or list of skycoords) returns the pixel locations
        """
        hdu = self.sci
        hdr = hdu.header
        wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
        pixel = wcsutils.skycoord_to_pixel(sky,wcs)
        return pixel



    def has_detections(self):
        """Check if a list of detected sources exists """
        return self.sourcecatalog is not None

    def detect_sources(self,nsigma=2,kfwhm=2.0,npixels=5,deblend=False,contrast=.001):
        """Detect sources (transient candidates) in the diff image using
        the astropy.photutils threshold-based source detection algorithm.

        Parameters
        ----------
        nsgima : float
            SNR required for pixel to be considered detected
        kfwhm : float
            FWHM of Circular Gaussian Kernel convolved on data to smooth noise
        npixels : int
            Number of connected pixels which are detected to give source
        deblend : bool
            Will use multiple levels/iterations to deblend single sources into multiple
        contrast : float
            If deblending the flux ratio required for local peak to be considered its own object

        Returns
        -------
        self.sourcecatalog: :class:`~photutils.segmentation.properties.SourceCatalog`
        """
        # TODO

        # record the locations and fluxes of candidate sources in an
        # external source catalog file (or a FITS extension)

        # if a fake is detected, mark it as such in the source catalog

        # if a fake is not detected, add it to the source catalog
        # (as a false negative)

        # maybe separate?: run aperture photometry on each fake source
        # maybe separate?: run PSF fitting photometry on each fake source
        # to be able to translate from ra/dec <--> pixels on image

        hdr = self.sci.header
        wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
        #L1mean,L1med,L1sigma,L1fwhm = hdr['L1MEAN'],hdr['L1MEDIAN'],hdr['L1SIGMA'],hdr['L1FWHM'] # counts, fwhm in arcsec 
        #pixscale,saturate,maxlin = hdr['PIXSCALE'],hdr['SATURATE'],hdr['MAXLIN'] # arcsec/pixel, counts for saturation and non-linearity levels
        # if bkg None: detect threshold uses sigma clipped statistics to get bkg flux and set a threshold for detected sources
        # bkg also available in the hdr of file, either way is fine  
        # threshold = detect_threshold(hdu.data, nsigma=nsigma)
        # or you can provide a bkg of the same shape as data and this will be used
        boxsize=100
        bkg = Background2D(self.sci.data,boxsize) # sigma-clip stats for background est over image on boxsize, regions interpolated to give final map 
        threshold = detect_threshold(self.sci.data, nsigma=nsigma,background=bkg.background)
        ksigma = kfwhm * gaussian_fwhm_to_sigma  # FWHM pixels for kernel smoothing
        # optional ~ kernel smooths the image, using gaussian weighting
        kernel = Gaussian2DKernel(ksigma)
        kernel.normalize()
        # make a segmentation map, id sources defined as n connected pixels above threshold (n*sigma + bkg)
        segm = detect_sources(self.sci.data,
                              threshold, npixels=npixels, filter_kernel=kernel)
        # deblend useful for very crowded image with many overlapping objects...
        # uses multi-level threshold and watershed segmentation to sep local peaks as ind obj
        # use the same number of pixels and filter as was used on original segmentation
        # contrast is fraction of source flux local pk has to be consider its own obj
        if deblend:
            segm = deblend_sources(self.sci.data, 
                                           segm, npixels=5,filter_kernel=kernel, 
                                           nlevels=32,contrast=contrast)
        # need bkg subtracted to do photometry using source properties
        data_bkgsub = self.sci.data - bkg.background
        cat = source_properties(data_bkgsub, segm,background=bkg.background,
                                error=None,filter_kernel=kernel)

        self.sourcecatalog = cat 
        return self.sourcecatalog
        
    def detect_host_galaxies(self , target_x , target_y , pixel_coords = True):
        """Detect sources  in the sky image using the astropy.photutils threshold-based
         source detection algorithm to get data on the host galaxies.  
         '''

        Parameters
        ----------

        target_x : float
            Either the x pixel coordinate or the ra for the host galaxy
        target_y : float
            Either the y pixel coordinate or the dec for the host galaxy
        pixel_coords: bool
            If true, input coordinates are assumed to be pixel coords
        Sky coordinates will be assumed to be in degrees if units are not provided
            
        Returns
        -------
        
        self.hostgalaxies : array : contains information on all host galaxies in the image
        """
        
        if not pixel_coords:
            ##Set units to degrees unless otherwise specified
            if type(pixel_coords) != units.quantity.Quantity:
                target_x *= units.deg
                target_y *= units.deg
            C = SkyCoord(target_x,target_y)
            ##Convert to pixel coordinates
            pix = self.skytopix(C)
            target_x = pix[0]
            target_y = pix[1]
        ##TODO
        
        ##Add support to identify galaxies in the image
        
        
        if not self.has_detections():
            self.detect_sources()
        hostgalaxies = []
        for i in self.sourcecatalog:
            x=i.xcentroid.value
            y=i.ycentroid.value
            if abs(x - target_x) < 10  and abs(y - target_y) < 10:
                hostgalaxies.append(i)
                break
        
        self.hostgalaxies = hostgalaxies
        return self.hostgalaxies

    def fetch_gaia_sources(self, save_suffix='GaiaCat', overwrite=False,
                           verbose=False):
        """Using astroquery, download a list of sources from the Gaia
         catalog that are within the bounds of this image.

        Parameters
        ----------

        save_suffix: str
            If None, do not save to disk. If provided, save the Gaia source
            catalog to an ascii text file named as
            <rootname_of_this_fits_file>_<save_suffix>.<_GAIACATEXT_>

        overwrite: boolean
            When True, fetch from the remote Gaia database even if a local
            copy exists.  Write over the local file with the results from the
            remote db.

        self.gaia_catalog : Astropy Table : contains information on all
        Gaia sources in the image

        """
        #  when save_suffix is provided, check first to see if a
        #  catalog exists, and load the sources from there
        if save_suffix:
            root = os.path.splitext(os.path.splitext(self.filename)[0])[0]
            savefilename = root + '_' + save_suffix + '.' + _GAIACATEXT_
            if os.path.isfile(savefilename) and not overwrite:
                print("Gaia catalog {} exists. \n".format(savefilename) + \
                      "Reading without fetching.")
                self.read_gaia_sources(save_suffix=save_suffix)
                return

        # coord of central reference pixel
        ra_ref = self.sci.header['CRVAL1']
        dec_ref = self.sci.header['CRVAL2']
        coord = SkyCoord(ra_ref, dec_ref, unit=(units.deg, units.deg))

        ## Compute the pixel scale in units of arcseconds, from the CD matrix
        #cd11 = self.sci.header['CD1_1'] # deg/pixel
        #cd12 = self.sci.header['CD1_2'] # deg/pixel
        #cd21 = self.sci.header['CD2_1'] # deg/pixel
        #cd22 = self.sci.header['CD2_2'] # deg/pixel
        #cdmatrix = [[cd11,cd12],[cd21,cd22]]
        #pixelscale = np.sqrt(np.abs(np.linalg.det(cdmatrix))) * u.deg
        pixelscale = np.sqrt(wcsutils.proj_plane_pixel_area(self.wcs))

        # compute the width and height of the image from the NAXIS keywords
        naxis1 = self.sci.header['NAXIS1']
        naxis2 = self.sci.header['NAXIS2']
        width = naxis1 * pixelscale * units.deg
        height = naxis2 * pixelscale * units.deg

        # Do the search. Returns an astropy Table
        full_gaia_source_table = Gaia.query_object_async(
            coordinate=coord, width=width, height=height)

        # isolate the parameters of interest: ra,dec,r_mag
        racol = Column(data=full_gaia_source_table['ra'], name='ra')
        deccol = Column(data=full_gaia_source_table['dec'], name='dec')
        magcol = MaskedColumn(data=full_gaia_source_table[_GAIAMAGCOL_],
                              name='mag')

        sncol = MaskedColumn(data=full_gaia_source_table[_GAIASNCOL_],
                              name='signal_to_noise')

        # add columns  x and y (pixel locations on image)
        #sky_positions= []
        pixel_positions=[]
        for i in range(len(full_gaia_source_table)):
            sky_pos = SkyCoord(ra=racol[i], dec=deccol[i],
                               unit=units.deg, frame=self.frame)
            #sky_positions.append(sky_pos)
            pixel_pos = wcsutils.skycoord_to_pixel(sky_pos, self.wcs)
            pixel_positions.append(pixel_pos)
        xcol = Column([pos[0] for pos in pixel_positions], name='x')
        ycol = Column([pos[1] for pos in pixel_positions], name='y')

        # create a minimalist Table
        self.gaia_source_table = Table([racol,deccol,xcol,ycol,magcol,sncol])

        if verbose:
            print('There are {} stars available within fov '
                  'from gaia results queried'.format(
                len(self.gaia_source_table)))

        if save_suffix:
            if os.path.exists(savefilename):
                os.remove(savefilename)
            # TODO : make more space-efficient as a binary table?
            self.gaia_source_table.write(
                savefilename, format=_GAIACATFORMAT_)
            self.gaia_source_table.savefilename = savefilename

        return


    def read_gaia_sources(self, save_suffix='GaiaCat'):
        """Read in an existing catalog of sources from the Gaia
         database that are within the bounds of this image.

        Requires that fetch_gaia_sources() has previously been run,
        with save_suffix provided to save the catalog as an ascii
        text file named as
        <rootname_of_this_fits_file>_<save_suffix>.<_GAIACATEXT_>

        Parameters
        ----------

        save_suffix: str
            The suffix of the Gaia source catalog filename.
        """
        root = os.path.splitext(os.path.splitext(self.filename)[0])[0]
        catfilename = root + '_' + save_suffix + '.' + _GAIACATEXT_
        if not os.path.isfile(catfilename):
            print("Error: {} does not exist.".format(catfilename))
            return -1
        self.gaia_source_table = Table.read(
            catfilename, format=_GAIACATFORMAT_)
        return 0


    def measure_zeropoint(self):
        """Measure the zeropoint of the image, using a set of
        known star locations and magnitudes, plus photutils aperture
        photometry of those stars. """
        # TODO : measure the zeropoint
        return


    def extract_psf_stars(self, verbose=False):
        """
        Extract postage-stamp image cutouts of stars from the image, for use
        in building an ePSF model

        Parameters
        ----------

        verbose: bool : verbose output
        """
        gaiacat = self.gaia_source_table
        image = self.sci


        # Define bounding boxes for the extractions so we can remove
        # any stars with overlaps. We want stars without overlaps
        # so the PSF construction doesn't require any deblending.
        # TODO : allow user to set the overlap size, or set based on FWHM
        bboxes = []
        for i in gaiacat:
            x = i['x']
            y = i['y']
            size = 25
            ixmin, ixmax = int(x - size/2), int(x + size/2)
            iymin, iymax = int(y - size/2), int(y + size/2)

            bbox = BoundingBox(ixmin=ixmin, ixmax=ixmax, iymin=iymin, iymax=iymax)
            bboxes.append(bbox)
        bboxes = Column(bboxes)
        gaiacat.add_column(bboxes,name='bbox')

        # using the bbox of each star from results to determine intersections,
        # dont want confusion of multi-stars for ePSF
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
        for i in gaiacat:
            if i['bbox'] in intersections:
                #tmp.remove(i)
                row=j
                rows.append(row)
            j+=1
        gaiacat.remove_rows(rows)
        if verbose:
            print('{} stars, after removing intersections'.format(len(gaiacat)))

        # Limit to just stars with very good S/N
        gaiacat_trimmed = gaiacat[gaiacat['signal_to_noise']>100]
        if verbose:
            print('restricting extractions to stars w/ S/N > 100' 
                  'we have {} to consider'.format(len(gaiacat_trimmed)))

        # TODO? sort by the strongest signal/noise in r' filter
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
        psfstars_extracted = extract_stars(nddata, catalogs=gaiacat,
                                           size=_PSFSTARCUTOUTSIZE_)
        # using the bbox of each star from results to determine intersections,
        # we don't want confusion of blended stars in our ePSF
        intersections = []
        for i,obj1 in enumerate(psfstars_extracted.bbox):
            for j in range(i+1,len(psfstars_extracted.bbox)):
                obj2 = psfstars_extracted.bbox[j]
                if obj1.intersection(obj2):
                    #print(obj1,obj2)
                    # these are the ones to remove
                    intersections.append(obj1)
                    intersections.append(obj2)
        # use the intersections found to remove stars
        tmp = [i for i in psfstars_extracted] # get a list of stars rather than single photutils obj with all of them
        for i in tmp:
            if i.bbox in intersections:
                tmp.remove(i)
        if verbose:
            print('{} stars, after removing intersections'.format(len(tmp)))

        # note ref.fits doesn't have saturate and maxlin available
        # the image should be just one of the trims
        for i in tmp:
            if np.max(i.data) > saturate:
                tmp.remove(i)
            elif np.max(i.data) > maxlin:
                tmp.remove(i)

        if verbose:
            print('removed stars above saturation or non-linearity level'
                  '~ {}, {} ADU; now have {}'.format(
                saturate,maxlin,len(tmp)))
        psf_stars_selected = photutils.psf.EPSFStars(tmp)

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
        self.psfstars = psf_stars_selected

        return

    def build_epsf_model(self, oversampling=2,
                         verbose=False, save_suffix=None):
        """Build an effective PSF model from a set of stars in the image
        Uses a list of star locations (from Gaia)  which are below
        non-linearity/saturation

        Parameters
        ----------

        oversampling: int : the oversampling scale for the PSF model. See the
          photutils ePSF model documentation for details.

        verbose: bool : verbose output

        save_suffix: str
            The suffix for the epsf model output filename.
            If set to None, then no output file is generated

        """
        # TODO: check for existence of gaia source table and fetch/read it if needed
        #starcoordinates = fitsimage.gaia_source_table

        self.extract_psf_stars(verbose=verbose)
        assert(self.psfstars is not None)

        # TODO: accommodate other header keywords to get the stats we need
        hdr = self.sci.header
        L1mean = hdr['L1MEAN'] # for LCO: counts
        L1med  = hdr['L1MEDIAN'] # for LCO: counts
        L1sigma = hdr['L1SIGMA'] # for LCO: counts
        L1fwhm = hdr['L1FWHM'] # for LCO: fwhm in arcsec
        pixscale = hdr['PIXSCALE'] # arcsec/pixel
        saturate = hdr['SATURATE'] # counts (saturation level)
        maxlin = hdr['MAXLIN'] # counts (max level for linear pixel response)

        # oversampling chops pixels of each star up further to get better fit
        # this is okay since stacking multiple ...
        # however more oversampled the ePSF is, the more stars you need to get
        # smooth result
        # LCO is already oversampling the PSFs, the fwhm ~ 2 arcsec while
        # pixscale ~ 0.4 arcsec; should be able to get good ePSF measurement
        # without any oversampling
        # ePSF basic x,y,sigma 3 param model should be easily obtained if
        # consider that 3*pixscale < fwhm
        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=10,
                                   progress_bar=True)
        epsf, fitted_stars = epsf_builder(self.psfstars)

        self.epsf = epsf
        self.fitted_stars = fitted_stars

        if self.zeropoint is None:
            self.measure_zeropoint()

        # TODO : Can we save this ePSF model as a fits extension instead?
        if save_suffix:
            # Write out the ePSF model
            # TODO: make a function for generating output file names
            rootfilename = os.path.splitext(
                os.path.splitext(self.filename)[0])[0]
            epsf_filename = rootfilename + '_' + save_suffix + '.pkl'
            pickle.dump( self.epsf, open( epsf_filename, "wb" ) )

        return

    def load_epsfmodel_from_pickle(self, save_suffix):
        """Read in an ePSF model from a pickle file

        Parameters
        ----------

        save_suffix: str
            The suffix for the epsf model filename to be read.
        """
        rootfilename = os.path.splitext(
            os.path.splitext(self.filename)[0])[0]
        epsf_filename = rootfilename + '_' + save_suffix + '.pkl'
        self.epsf = pickle.load(open( epsf_filename, "rb" ) )
        return


class FakePlanter:
    """A class for handling the FITS file triplets (diff,search,ref),
    planting fakes, detecting fakes, and creating sub-images and
    catalogs for use in training+validation of machine learning
    algorithms
    """

    def __init__(self, diffim_fitsfilename,
                 searchim_fitsfilename=None,
                 templateim_fitsfilename=None):
        """Read in a triplet of three FITS files that hold
        A. a difference image
        B. a 'search' image (typically a "new" single-epoch static sky image)
        C. the template image (or 'reference')

        The diff image was constructed as the subtraction of the search minus
        the template:   A = B - C
        Generally this is not a straight subtraction operation, as we apply
        PSF convolution or other data modification with something like the
        Alard & Lupton or ZOGY algorithm.

        """
        # Read in the three fits files that hold the diff images
        self.diffim = FitsImage(diffim_fitsfilename)
        if searchim_fitsfilename:
            self.searchim = FitsImage(searchim_fitsfilename)
        if templateim_fitsfilename:
            self.templateim = FitsImage(templateim_fitsfilename)

        # has_fakes False until run plant_fakes
        self.has_fakes = False
        # has_lco_epsf False until run lco_epsf
        self.has_lco_epsf = False
        
        return

    @property
    def has_epsfmodel(self):
        """True if both the diffim and searchim have an ePSF model.
        Otherwise False.
        """
        if ( self.diffim.psfmodel is not None and
            self.searchim.psfmodel is not None ):
            return ( type(self.diffim.psfmodel) == FakePlanterEPSFModel and
                     type(self.searchim.psfmodel) == FakePlanterEPSFModel)
        return False

    def build_epsf_model(self):
        """Function for constructing an effective point spread function model
        from the stars in the static sky image.
        """
        # TODO : absorb build_ePSF.py module to here
        # identify stars in the static sky image by making a query to
        # the online Gaia database

        # build an ePSF model from those stars, add it as an extension to
        # the input fits image (optionally save the modified fits image to disk)

        # optional?: record pre-existing info about the image + measurements
        # of the ePSF model in the pipeline log file: FWHM, zeropoint
        return

    def has_fakes(self):
        """Check if fake stars have been planted in the image"""
        return self.has_fakes

    def plant_fakes(self,epsf,locations,SCA=None,writetodisk=False,saveas="planted.fits"):
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

        hdu = self.diffim.hdu # the fits opened difference image hdu

        # copying so can leave original data untouched
        cphdu = hdu.copy()
        cpim = cphdu.data
        cphdr = cphdu.header
        
        wcs,frame = WCS(cphdr),cphdr['RADESYS'].lower()
        
        # location should be list of pixels [(x1,y1),(x2,y2)...(xn,yn)]
        n = 0
        for pix in locations:
            pix = list(pix)
            xp,yp = pix
            sky = wcsutils.pixel_to_skycoord(xp,yp,wcs)
            idx = str(n).zfill(3) 
            cphdr['FK{}X'.format(idx)] = xp
            cphdr['FK{}Y'.format(idx)] = yp
            cphdr['FK{}RA'.format(idx)] = str(sky.ra.hms)
            cphdr['FK{}DEC'.format(idx)] = str(sky.dec.dms)

            if SCA:
                # SCA ~ scaling factor for epsf, epsf*sca, needs to be list of floats same length as locations 
                sca = SCA[n]
                epsfn = epsf*sca
            else:
                # SCA ~ None, all the same brightness of input epsf
                sca = 1
                epsfn = epsf*sca
            cphdr['FK{}SCA'.format(idx)] = sca
            cphdr['FK{}F'.format(idx)] = np.sum(epsfn)

            # TO-DO, once have actual epsf classes will be clearer to fill the model
            cphdr['FK{}MOD'.format(idx)] = "NA"

            revpix = copy.copy(pix)
            revpix.reverse()
            row,col=revpix
            nrows,ncols=epsf.shape
            # +2 in these to grab a couple more than needed, the correct shapes for broadcasting taken using actual psf.shapes
            rows=np.arange(int(np.round(row-nrows/2)),int(np.round(row+nrows/2))+2) 
            cols=np.arange(int(np.round(col-ncols/2)),int(np.round(col+ncols/2))+2) 
            rows = rows[:epsf.shape[0]]
            cols = cols[:epsf.shape[1]]
            cpim[rows[:, None], cols] += epsfn
            np.float64(cpim)

            n+=1
        
        # inserting some new header values
        cphdr['fakeSN']=True 
        cphdr['N_fake']=str(len(locations))
        cphdr['F_epsf']=str(np.sum(epsf))
        
        if writetodisk:
            fits.writeto(saveas,cpim,cphdr,overwrite=True)
        
        self.has_fakes = True # if makes it through this plant_fakes update has_fakes

        return cphdu
