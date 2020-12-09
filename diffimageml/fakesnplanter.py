from astropy.io import fits
from photutils.psf import EPSFModel
import numpy as np
from astroquery.gaia import Gaia

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
        self.hdulist = fits.read(fitsfilename)
        self.psfmodel = None
        self.sourcecatalog = None
        return

class FakePlanter:
    """A class for handling the FITS file triplets (diff,search,ref),
    planting fakes, detecting fakes, and creating sub-images and
    catalogs for use in training+validation of machine learning
    algorithms
    """

    def __init__(self, diffimfitsfilename,
                 searchimfitsfilename=None,
                 templateimfitsfilename=None):
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
        self.diffim = FitsImage(diffimfitsfilename)
        if searchimfitsfilename:
            self.searchim = FitsImage(searchimfitsfilename)
        if templateimfitsfilename:
            self.templateim = FitsImage(templatefitsfilename)
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


    def has_detections(self):
        """Check if a list of detected sources exists """
        return


    def detect_sources(self):
        """Detect sources (transient candidates) in the diff image using
        the astropy.photutils threshold-based source detection algorithm.
        """
        # TODO : absorb detect_sources.py module to here

        # use an astropy threshold detection algorithm to identify transient
        # source candidates in the diff image fits file

        # record the locations and fluxes of candidate sources in an
        # external source catalog file (or a FITS extension)

        # if a fake is detected, mark it as such in the source catalog

        # if a fake is not detected, add it to the source catalog
        # (as a false negative)

        # maybe separate?: run aperture photometry on each fake source
        # maybe separate?: run PSF fitting photometry on each fake source
        return

