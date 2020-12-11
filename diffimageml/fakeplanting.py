import numpy as np

from astropy import units as u
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy.io import ascii,fits
from astropy.wcs import WCS, utils as wcsutils
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm
from astropy.convolution import Gaussian2DKernel

from photutils import Background2D, MedianBackground, detect_threshold,detect_sources,source_properties
from photutils.psf import EPSFModel

class FakePlanterEPSFModel(EPSFModel):
    """ A class for holding an effective PSF model."""
    def __init__(self):
        self.fluxarray = np.array([]) # array of flux density values (e-/s/cm2)

        return

    def scaled_to_mag(self, mag):
        """Return a data array scaled to the given magnitude.
        Requires that a zeropoint has been set.
        """
        return self.fluxarray * 10**(-0.4*(mag-self.zeropoint))

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
        self.hdulist, self.hdu = self.read_fits_file(fitsfilename)
        self.psfmodel = None
        self.sourcecatalog = None
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
        hdu : :class:`~astropy.io.fits.PrimaryHDU` (or similar)

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

        # TODO
        #
        #Let's make sure that this is 
        #true in general, or change if not.
        #
        if fitsfilename.endswith('fz'):
            return self.hdulist, self.hdulist[1]
        else:
            return self.hdulist, self.hdulist[0]
        # TODO : remove the return statemens after confirming that no calling
        # functions need them.



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

        hdr = self.hdu.header
        wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
        #L1mean,L1med,L1sigma,L1fwhm = hdr['L1MEAN'],hdr['L1MEDIAN'],hdr['L1SIGMA'],hdr['L1FWHM'] # counts, fwhm in arcsec 
        #pixscale,saturate,maxlin = hdr['PIXSCALE'],hdr['SATURATE'],hdr['MAXLIN'] # arcsec/pixel, counts for saturation and non-linearity levels
        # if bkg None: detect threshold uses sigma clipped statistics to get bkg flux and set a threshold for detected sources
        # bkg also available in the hdr of file, either way is fine  
        # threshold = detect_threshold(hdu.data, nsigma=nsigma)
        # or you can provide a bkg of the same shape as data and this will be used
        boxsize=100
        bkg = Background2D(self.hdu.data,boxsize) # sigma-clip stats for background est over image on boxsize, regions interpolated to give final map 
        threshold = detect_threshold(self.hdu.data, nsigma=nsigma,background=bkg.background)
        ksigma = kfwhm * gaussian_fwhm_to_sigma  # FWHM pixels for kernel smoothing
        # optional ~ kernel smooths the image, using gaussian weighting
        kernel = Gaussian2DKernel(ksigma)
        kernel.normalize()
        # make a segmentation map, id sources defined as n connected pixels above threshold (n*sigma + bkg)
        segm = detect_sources(self.hdu.data,
                              threshold, npixels=npixels, filter_kernel=kernel)
        # deblend useful for very crowded image with many overlapping objects...
        # uses multi-level threshold and watershed segmentation to sep local peaks as ind obj
        # use the same number of pixels and filter as was used on original segmentation
        # contrast is fraction of source flux local pk has to be consider its own obj
        if deblend:
            segm = deblend_sources(self.hdu.data, 
                                           segm, npixels=5,filter_kernel=kernel, 
                                           nlevels=32,contrast=contrast)
        # need bkg subtracted to do photometry using source properties
        data_bkgsub = self.hdu.data - bkg.background
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
            
        Returns
        -------
        
        self.hostgalaxies : array : contains information on all host galaxies in the image
        """
        
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

    def fetch_gaia_sources(self, save_suffix=None):
        #TODO: set default save_suffix='GaiaCat'):
        """Using astroquery, download a list of sources from the Gaia
         catalog that are within the bounds of this image.

        Parameters
        ----------

        save_suffix: str
            If None, do not save to disk. If provided, save the Gaia source
            catalog to an ascii text file named as
             <name_of_this_fits_file>_<save_suffix>.txt

        Sets
        -------
        self.gaia_catalog : Astropy Table : contains information on all
        Gaia sources in the image

        """
        # TODO : when save_suffix is provided, check first to see if a
        #  catalog exists, and load the sources from there

        # coord of central reference pixel
        ra_ref = self.sci.header['CRVAL1']
        dec_ref = self.sci.header['CRVAL2']
        coord = SkyCoord(ra_ref, dec_ref, unit=(u.hourangle,u.deg))

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
        width = naxis1 * pixelscale * u.deg
        height = naxis2 * pixelscale * u.deg

        # Do the search. Returns an astropy Table
        self.gaia_source_table = Gaia.query_object_async(
            coordinate=coord, width=width, height=height)

        # TODO : saving to file not yet debugged
        if save_suffix:
            savefilename = os.path.split_ext(self.fitsfilename)[0] +\
                           '_' + save_suffix + '.txt'
            if os.path.exists(savefilename):
                os.remove(savefilename)
            # TODO : make more space-efficient as a binary table?
            self.gaia_source_table.write(
                savefilename, format='ascii.fixed_width')
            self.gaia_source_table.savefilename = savefilename

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
        return


    def plant_fakes(self):
        """Function for planting fake stars in the diff image.
        """
        # TODO : absorb plant_fakes.py module to here
        # using the ePSF model embedded in the fits file, plant a grid
        # of fakes or plant fakes around galaxies with varying magnitudes
        # (fluxes), mimicking strong-lensing sources

        # write info into the fits header for each planted fake, including
        # the location and total flux

        # optional:  write out the modified image with fakes planted as
        # a new fits file record in the image db that fakes have been
        # planted in the image

        return
