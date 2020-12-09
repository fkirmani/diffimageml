from astropy import fits

class fakesnplanter(object):
    """A class for handling the FITS file diff images,
    planting fakes, detecting fakes, and creating sub-images and
    catalogs for use in training+validation of machine learning
    algorithms
    """

    def __init__(self, diffimfitsfilename):
        """Read in a FITS file that holds a difference image."""
        #TODO: read in a fits file that holds a diff image
        #TODO: add attributes pointing to the associated static
        # sky 'search' image and template. Maybe read them in as well?
        return

    def has_epsf_model(self):
        """Check if an ePSF model exists as a FITS extension."""
        return

    def build_epsf_model(self):
        """Function for constructing an effective point spread function model
        from the stars in the static sky image.
        """
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

    