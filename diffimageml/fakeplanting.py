import numpy as np
import scipy

import os, collections


from astropy import units
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel
from astropy.io import ascii,fits
from astropy.nddata import Cutout2D,NDData
from astropy.stats import (sigma_clip, sigma_clipped_stats,
                           gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm)
from astropy.table import Table,Column,MaskedColumn,Row,vstack,setdiff,join
from astropy.wcs import WCS, utils as wcsutils
from astropy.visualization import ZScaleInterval,simple_norm
zscale = ZScaleInterval()

import photutils
from photutils.datasets import make_gaussian_sources_image
from photutils import Background2D, MedianBackground
from photutils import detect_sources, source_properties
from photutils.psf import EPSFModel, extract_stars
from photutils import EPSFBuilder, BoundingBox
from photutils import Background2D, MedianBackground
from photutils import EllipticalAperture, detect_threshold, deblend_sources
from photutils import CircularAperture , aperture_photometry , CircularAnnulus

import itertools
import copy
import pickle

import matplotlib
from matplotlib import pyplot as plt, cm
from mpl_toolkits.axes_grid1 import ImageGrid

import warnings
from astropy.wcs import FITSFixedWarning
##Supress FITSFixedWarnings
warnings.simplefilter("ignore" , category = FITSFixedWarning)
#local
from util import *
#from .util import *

# astropy Table format for the gaia source catalog
_GAIACATFORMAT_ = 'ascii.ecsv'
_GAIACATEXT_ = 'ecsv'

# Column names for the magnitudes and S/N to use for selecting viable PSF stars
_GAIAMAGCOL_ =  'phot_rp_mean_mag'
_GAIAFLUXCOL_ =  'phot_rp_mean_flux'
_GAIAFLUXERRCOL_ =  'phot_rp_mean_flux_error'
_GAIASNCOL_ = 'phot_rp_mean_flux_over_error'

# astropy Table format for the fake SN source catalog
_FSNCATFORMAT_ = 'ascii.ecsv'
_FSNCATEXT_ = 'ecsv'

# Size of the box for each PSF star cutout (half width? or full?)
#  Does this also set the size of the resulting ePSF model?
_PSFSTARCUTOUTSIZE_ = 25 # pixels
_MAX_N_PLANTS_ = 999


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

        self.psfstars = None
        self.fitted_stars = None
        self.psfmodel = None
        self.epsf = None

        self.hdulist = None
        self.sci = None

        self.wcs = None
        self.frame = None

        self.sci_with_fakes = None
        self.fakes_posflux_table = None

        self.sourcecatalog = None
        self.hostgalaxies = None
        self.zeropoint = None
        self.stellar_phot_table = None
        self.gaia_source_table = None

        self.read_fits_file(fitsfilename)

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

    @property
    def has_detections(self):
        """Check if a list of detected sources exists """
        return self.sourcecatalog is not None

    def detect_sources(self,nsigma=2,kfwhm=2.0,npixels=5,deblend=False,contrast=.001, **kwargs):
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
        boxsize=200
        bkg = Background2D(self.sci.data,boxsize) # sigma-clip stats for background est over image on boxsize, regions interpolated to give final map 
        threshold = detect_threshold(self.sci.data, nsigma=nsigma,background=bkg.background)
        ksigma = kfwhm * gaussian_fwhm_to_sigma  # FWHM pixels for kernel smoothing
        # optional ~ kernel smooths the image, using gaussian weighting
        kernel = Gaussian2DKernel(ksigma)
        kernel.normalize()
        # make a segmentation map, id sources defined as n connected pixels above threshold 
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

        # TODO the detection parameters into meta of table
        meta = {'detect_params':{"nsigma":nsigma,"kfwhm":kfwhm,"npixels":npixels,
                                                "deblend":deblend,"contrast":contrast}}

        self.sourcecatalog = cat

        # TODO : identify indicies of extended sources and make a property
        #  of the class that just gives an index into the source catalog
        #for i in self.sourcecatalog:
        #    if i.ellipticity > 0.35: ##Identifies Galaxies
        ##        if i.area.value < 8 and cut_cr: ##Removes cosmic rays
        #            continue
        #        xcol.append(i.centroid[1])
        #        ycol.append(i.centroid[0])
        #        source_propertiescol.append(i)
        # hostgalaxies = Table([xcol , ycol , source_propertiescol] , names = ("x" , "y" , "Source Properties"))
        # self.hostgalaxies = hostgalaxies
        # return self.hostgalaxies

        return self.sourcecatalog
    

    def detect_host_galaxies(self , ellipticity_cut = 0.35 , cut_cr = True , edges = True,**kwargs):
        """Detect sources  in the sky image using the astropy.photutils threshold-based
         source detection algorithm to get data on the host galaxies. Will attempt to identify
         the galaxies in the image
         '''

        Parameters
        ----------

        ellipticity_cut : float : We will select galaxies to be objects with an ellipticity
        greater than ellipticity_cut.
        
        cur_cr : boolean : If true, performs an additional cut on the number of pixels in the source
        in order to reduce the number of artifacts that get flagged as galaxies.

        edges: boolean : Default true, performs a cut of sources detected nearby the edges of image  
            
        Returns
        -------
        
        self.hostgalaxies : array : contains information on all host galaxies in the image
        """
        
        xcol = []
        ycol = []
        source_propertiescol = []
        shape = self.sci.data.shape
        
        if not self.has_detections:
            self.detect_sources(**kwargs)
        for i in self.sourcecatalog:
            if i.ellipticity > 0.35: ##Identifies Galaxies
                if i.area.value < 8 and cut_cr: ##Removes cosmic rays
                    continue
                if i.centroid[1].value < 50 or i.centroid[0].value < 50 and edges: ##Removes objects next to edges
                    continue
                if i.centroid[1].value > shape[0] - 50 or i.centroid[0].value > shape[1] - 50 and edges:
                    continue
                xcol.append(i.centroid[1])
                ycol.append(i.centroid[0])
                source_propertiescol.append(i)
        hostgalaxies = Table([xcol , ycol , source_propertiescol] , names = ("x" , "y" , "Source Properties"))


        self.hostgalaxies = hostgalaxies
        return self.hostgalaxies

    
    def plant_fakes_in_sci(self, psfmodel, posflux, subshape=None,
                           preserve_original=False,
                           writetodisk=False, save_suffix="withfakes"):
        """
        Add PSF/PRFs ("fakes") to the image data array.
        Also update the header to record the pixel positions and fluxes for
        each fake.

        Parameters
        ----------
        psfmodel : `astropy.modeling.Fittable2DModel` instance
            PSF/PRF model to be substracted from the data.

        posflux : Array-like of shape (3, N) or `~astropy.table.Table`
            Positions and fluxes for the objects to add.  If an array,
            it is interpreted as ``(x, y, flux)``  If a table, the columns
            'x_fit', 'y_fit', and 'flux_fit' must be present.

        subshape : length-2 or None
            The shape of the region around the center of the location to
            add the PSF to.  If None, add to the whole image.

        preserve_original : bool
            if True, store a copy of the unmodified original sci HDU as
            self.sci_orig

        writetodisk :  bool
            if True - write out the modified image data as fits files.
            Separate fits files are written for the search image and the
            difference image.

        save_suffix : str
            suffix to use for the output fits files. Each filename is defined
            as  <original_fits_filename_root>_<save_suffix>.fits


        Returns
        -------
        hdu_withfakes : FITS HDU object
            Holds the image data array with the PSF(s) added, and the header
            with new cards carrying the fake star meta-data.
        """

        if preserve_original:
            # making a copy to preserve original untouched sci data + hdr
            self.sci_orig = self.sci.copy()

        data = self.sci.data
        hdr = self.sci.header

        # TODO:  already have self.wcs and self.frame ??
        wcs,frame = WCS(hdr),hdr['RADESYS'].lower()

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

        # Set up constants across the loop
        # TODO: Do we need a copy of the psf model here? Do we need a new copy
        # for each fake that gets planted ?
        psfmodelcopy = psfmodel.copy()
        xname, yname, fluxname = extract_psf_fitting_names(psfmodel)
        indices = np.indices(data.shape)

        # TODO : now that we've got an option to preserve_original do we need copies here?
        subbeddata = data.copy()
        addeddata = data.copy()
        
        nfakes_planted = 0
        if subshape is None:
            indicies_reversed = indices[::-1]

            for row in posflux:
                getattr(psfmodelcopy, xname).value = row['x_fit']
                getattr(psfmodelcopy, yname).value = row['y_fit']
                getattr(psfmodelcopy, fluxname).value = row['flux_fit']

                xp,yp,flux_fit = row['x_fit'],row['y_fit'],row['flux_fit']
                sky = wcsutils.pixel_to_skycoord(xp,yp,wcs)
                idx = str(nfakes_planted).zfill(3)
                hdr['FK{}X'.format(idx)] = xp
                hdr['FK{}Y'.format(idx)] = yp
                hdr['FK{}RA'.format(idx)] = str(sky.ra.hms)
                hdr['FK{}DEC'.format(idx)] = str(sky.dec.dms)
                hdr['FK{}F'.format(idx)] = flux_fit
                # TO-DO, once have actual epsf classes will be clearer to fill the model
                hdr['FK{}MOD'.format(idx)] = "NA"
                nfakes_planted += 1

                #subbeddata -= psfmodel(*indicies_reversed)
                addeddata += psfmodelcopy(*indicies_reversed)
        else:
            for row in posflux:
                x_0, y_0 = row['x_fit'], row['y_fit']

                # float dtype needed for fill_value=np.nan
                y = extract_array(indices[0].astype(float), subshape, (y_0, x_0))
                x = extract_array(indices[1].astype(float), subshape, (y_0, x_0))

                getattr(psfmodelcopy, xname).value = x_0
                getattr(psfmodelcopy, yname).value = y_0
                getattr(psfmodelcopy, fluxname).value = row['flux_fit']

                xp,yp,flux_fit = row['x_fit'],row['y_fit'],row['flux_fit']
                sky = wcsutils.pixel_to_skycoord(xp,yp,wcs)
                idx = str(nfakes_planted).zfill(3)
                hdr['FK{}X'.format(idx)] = xp
                hdr['FK{}Y'.format(idx)] = yp
                hdr['FK{}RA'.format(idx)] = str(sky.ra.hms)
                hdr['FK{}DEC'.format(idx)] = str(sky.dec.dms)
                hdr['FK{}F'.format(idx)] = flux_fit
                # TO-DO, once have actual epsf classes will be clearer to fill the model
                hdr['FK{}MOD'.format(idx)] = "NA"
                nfakes_planted += 1
                
                addeddata = add_array(addeddata, psfmodelcopy(x, y), (y_0, x_0))

        # update the data array with all fakes in it
        self.sci.data = addeddata

        # inserting some new header values
        hdr['HASFAKES'] = True
        hdr['NFAKES'] = nfakes_planted
        hdr['PSF_FLUX'] = getattr(psfmodel, fluxname).value
        
        if writetodisk:
            fits.writeto(save_suffix, self.sci.data, hdr, overwrite=True)

        self.fake_source_table = posflux

        return

    @property
    def has_fakes(self):
        """True if this FitsImage has fakes in it.
        Note: only checks the image header.
        """
        if 'HASFAKES' in self.sci.header:
            if self.sci.header['HASFAKES']:
                if self.sci.header['NFAKES']>0:
                    if 'FK000X' in self.sci.header:
                        if self.sci.header['FK000X'] is not None:
                            return True
        return False


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
        from astroquery.gaia import Gaia
        full_gaia_source_table = Gaia.query_object_async(
            coordinate=coord, width=width, height=height)

        # isolate the parameters of interest: ra,dec,r_mag
        racol = Column(data=full_gaia_source_table['ra'], name='ra')
        deccol = Column(data=full_gaia_source_table['dec'], name='dec')
        magcol = MaskedColumn(data=full_gaia_source_table[_GAIAMAGCOL_],
                              name='mag')

        # Compute a magnitude error (yes, its asymmetric. OK)
        flux=full_gaia_source_table[_GAIAFLUXCOL_]
        fluxerr=full_gaia_source_table[_GAIAFLUXERRCOL_]
        magerr = 1.086 * fluxerr/flux
        magerrcol = MaskedColumn(data=magerr, name='magerr')

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
        self.gaia_source_table = Table(
            [racol, deccol, xcol, ycol, magcol, magerrcol, sncol])

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


    def plot_gaia_sources(self, magmin=12, magmax=18):
        """Show the locations of Gaia sources on the image.
        """
        # get the x,y pixel locations of all the sources in the image
        try:
            assert(self.gaia_source_table is not None)
        except assertionerror:
            print("No Gaia source table. Run fetch_gaia_sources()")
            return

        medpixval = np.median(self.sci.data)
        sigmapixval = np.std(self.sci.data)
        plt.imshow(self.sci.data, cmap=cm.Greys, interpolation='nearest',
                   aspect='equal', origin='lower', vmin=medpixval-sigmapixval,
                   vmax=medpixval+sigmapixval)

        mag = self.gaia_source_table['mag']
        imaglim = np.where( (magmin<mag) & (mag<magmax))[0]
        xsources = self.gaia_source_table['x'][imaglim]
        ysources = self.gaia_source_table['y'][imaglim]
        plt.plot(xsources, ysources, marker='o', ms=10, mfc='None',
                 mec='cyan', ls=' ', alpha=0.5)
        plt.title("Locations of Gaia Sources with {}<mag<{}".format(
            magmin,magmax))

        # TODO : plot diamonds over sources selected for PSF fitting

        # TODO : show cut-outs of stars used for PSF fitting
        return



    def do_stellar_photometry(self , gaia_catalog = None):
        """Takes in a source catalog for stars in the image from Gaia. Will perform
        aperture photometry on the sources listed in this catalog.

        Parameters
        ----------

        gaia_catalog: Astropy Table : Contains information on Gaia sources in the image
            You can also provide a catalog from detect_sources. If None, will use self.gaia_source_table

        self.stellar_phot_table : Astropy Table : Table containing the measured magnitudes
        for the stars in the image obtained from the Gaia catalog.
        
        """
        
        ##TODO: Add something to handle overstaturated sources
        ##We currently just ignore anything brighter than m = 16 to avoid saturated sources
        
        if not self.gaia_source_table and not gaia_catalog:
            print ("Warning: No catalog provided for aperture photometry.")
            print ("Run fetch_gaia_sources or providea source catalog")
            return

        elif not gaia_catalog:
            gaia_catalog = self.gaia_source_table

        positions = []
        
        for i in gaia_catalog:

            if 'mag' in gaia_catalog.colnames and i['mag'] < 16:
                continue
            if 'x' in gaia_catalog.colnames:
                positions.append( ( i['x'] , i['y'] ) ) ##Pixel coords for each source
            else:
                positions.append((i['xcentroid'].value , i['ycentroid'].value))
        
        ##Set up the apertures

        pixscale = self.sci.header["PIXSCALE"]
        FWHM = self.sci.header["L1FWHM"]
        
        aperture_radius = 2 * FWHM / pixscale
        apertures = CircularAperture(positions, r= aperture_radius)

        annulus_aperture = CircularAnnulus(positions, r_in = aperture_radius + 5 , r_out = aperture_radius + 10)
        annulus_masks = annulus_aperture.to_mask(method='center')
        
        ##Background subtraction using sigma clipped stats.
        ##Uses a median value from the annulus
        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(self.sci.data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _ , median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
            
        ##Perform photometry and subtract out background
        bkg_median = np.array(bkg_median)
        phot = aperture_photometry(self.sci.data, apertures)
        phot['annulus_median'] = bkg_median
        phot['aper_bkg'] = bkg_median * apertures.area
        
        
        phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
        
        phot['mag'] = -2.5 * np.log10( phot['aper_sum_bkgsub'] )
        
        self.stellar_phot_table = phot
        
        return


    def plot_stellar_photometry(self):
        """Simple plot of the stellar photometry results"""
        # TODO: update to show a comparison of aperture vs psf photometry

        try:
            assert(self.stellar_phot_table is not None)
        except:
            print("Missing measured stellar photometry. "
                  " Run do_stellar_photometry")
            return -1

        flux = self.stellar_phot_table['aper_sum_bkgsub']
        fluxerr = np.sqrt(self.stellar_phot_table['aper_bkg'])
        measured_mag = self.stellar_phot_table['mag']
        xphot = self.stellar_phot_table['xcenter']
        yphot = self.stellar_phot_table['ycenter']

        plt.errorbar(measured_mag+25, flux, fluxerr, ls=' ', marker='o')
        ax = plt.gca()
        ax.invert_xaxis()
        ax.set_yscale('log')
        plt.ylabel('Measured flux [counts]')
        plt.xlabel('Measured magnitude, assuming zpt=25')

        return


    def measure_zeropoint(self, showplot=False):
        """Measure the zeropoint of the image, using a set of
        known star locations and magnitudes, plus photutils aperture
        photometry of those stars.

        NOTE: currently using made-up data!!
        """
        try:
            assert(self.stellar_phot_table is not None)
        except:
            print("Missing measured stellar photometry. "
                  " Run do_stellar_photometry")
            return -1

        try:
            assert(self.gaia_source_table is not None)
        except:
            print("Missing Gaia catalog photometry. "
                  " Run fetch_gaia_sources")
            return -1

        star_flux = self.stellar_phot_table['aper_sum_bkgsub']
        star_flux_err = np.sqrt(self.stellar_phot_table['aper_bkg'])
        measured_mag = self.stellar_phot_table['mag']

        xphot = self.stellar_phot_table['xcenter']
        yphot = self.stellar_phot_table['ycenter']
        xcat = self.gaia_source_table['x']
        ycat = self.gaia_source_table['y']

        # Find the nearest Gaia catalog source for each measured star
        icat = []
        for i in range(len(xphot)):
            dist = np.sqrt((xphot[i].value - xcat)**2 +
                           (yphot[i].value - ycat)**2)
            icat.append(dist.argmin())
        star_mag = self.gaia_source_table['mag'][icat]
        star_mag_err = self.gaia_source_table['magerr']

        # mask non-positive flux measurements and those with S/N<20
        #star_flux_ma = np.ma.masked_less_equal(star_flux, 0, copy=True)
        ivalid = np.where( (star_flux>0) &
                           (np.abs(star_flux/star_flux_err)>20))
        nvalid = len(ivalid)

        # measure the zeropoint from each star
        zpt_fit = star_mag[ivalid] + 2.5 * np.log10(star_flux[ivalid])
        zpt_fit_err = np.sqrt(star_mag_err[ivalid]**2 +
                              (1.086 * star_flux_err[ivalid]
                               / star_flux[ivalid])**2 )

        # A dizzying array of ways to compute the zeropoint for the image
        zpt_mean_sc, zpt_median_sc, zpt_stdev_sc = sigma_clipped_stats(zpt_fit)
        #zpt_weighted_mean = np.average( zpt_fit, weights=1/zpt_fit_err**2)
        #zpt_fit_sigclipped = sigma_clip(zpt_fit, masked=True)
        #zpt_fit_err_sigclipped = zpt_fit_err[~zpt_fit_sigclipped.mask]
        #zpt_weighted_mean_sigclipped = np.average(
        #    zpt_fit_sigclipped[~zpt_fit_sigclipped.mask],
        #    weights=1/zpt_fit_err_sigclipped**2)

        self.zeropoint = zpt_median_sc

        if showplot:
            ax = plt.gca()
            plt.errorbar(star_mag[ivalid], zpt_fit, zpt_fit_err,
                         marker='.', ls=' ', color='k',
                         label='_nolabel_')
            skiptheselines = """
            ax.axhline(np.average(zpt_fit), color='darkorange',
                       label='{:.2f} naive mean, unclipped'.format(
                           np.average(zpt_fit)
                       ))

            ax.axhline(zpt_weighted_mean, color='red',
                       label='{:.2f} inv-var-wgtd mean, unclipped'.format(
                           zpt_weighted_mean
                       ))
            ax.axhline(zpt_mean_sc, color='teal',
                       label='{:.2f} sigma-clipped mean'.format(
                           zpt_mean_sc
                       ))
            ax.axhline(zpt_weighted_mean_sigclipped, color='blue',
                       label='{:.2f} sigma-clipped weighted mean'.format(
                           zpt_weighted_mean_sigclipped
                       ))
            """
            ax.axhline(zpt_median_sc, color='g',
                       label='{:.2f} sigma-clipped median'.format(
                           zpt_median_sc
                       ))
            plt.xlabel('Stellar Magnitude from Catalog')
            plt.ylabel('Inferred Zero Point')
            ax.legend(loc='best')

        return


    def extract_psf_stars(self, SNthresh=100, verbose=False):
        """
        Extract postage-stamp image cutouts of stars from the image, for use
        in building an ePSF model

        Parameters
        ----------

        SNthresh: float:  signal to noise threshold. Only stars with
        S/N > SNthresh are used for PSF construction.

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
        gaiacat_trimmed = gaiacat[gaiacat['signal_to_noise']>SNthresh]
        if verbose:
            print('restricting extractions to stars w/ S/N > {}' 
                  'we have {} to consider'.format(
                SNthresh, len(gaiacat_trimmed)))

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
                         verbose=False, save_suffix=None, overwrite=False):
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

        overwrite: bool
            If True, overwrite any existing ePSF model saved as a .pkl file
            If False, and a .pkl exists with the name indicated by save_suffix,
            just read that in without remaking the PSF model.

        """
        # check for existence of pre-made PSF model and load it if desired
        rootfilename = os.path.splitext(
            os.path.splitext(self.filename)[0])[0]
        if save_suffix is not None and overwrite == False:
            epsf_filename = rootfilename + '_' + save_suffix + '.pkl'
            if os.path.isfile(epsf_filename):
                self.load_epsfmodel_from_pickle(save_suffix=save_suffix)
                return


        # check for existence of gaia source table and fetch/read it if needed
        catfilename = rootfilename + '_' + save_suffix + '.' + _GAIACATEXT_
        if os.path.isfile(catfilename):
            try:
                self.read_gaia_sources(save_suffix=save_suffix)
            except:
                print("Tried to read existing Gaia source table... failed.")
        if self.psfstars is None:
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


    def plot_epsf_model(self):
        try :
            assert(self.epsf is not None)
        except:
            print("No ePSF model exists. Run build_epsf_model()")
            return -1
        plt.imshow(self.epsf.data, interpolation='Nearest', origin='lower')
        plt.colorbar()


    def write_fakesn_catalog(self , save_suffix = "fakecat" , overwrite = False , add_to = False, add_to_filename = None):
        
    
        """
        
        Writes information for fake sources into a fake source catalog
        Will include the locations and epsf models for all of the fake sources
        
        

        Parameters
        ----------

        save_suffix: str
            If None, do not save to disk. If provided, save the fake source
            catalog to an ascii text file named as
            <rootname_of_this_fits_file>_<save_suffix>.<_GAIACATEXT_>

        overwrite: boolean
            When True, overwrite an existing fake sn catalog
            Otherwise, will only save catalog if it does not already exist
            
            
        add_to_filename: str
            If None, this is ignored. If provided, the souce catalog from this
            image will be appended to the given file. Designed to create a catalog
            containing fake sources from multiple images. Will still produce a catalog
            for this image using save_suffix, unless save_suffix = None
        
        self.fakesncat : Astropy Table : Contains information on the fake sources and
        their host galaxies
        
        """
        
        
        fakes = []
        file_header = self.hdulist[0].header
        
        # If we are not adding to an existing file
        if save_suffix != None:
            root = os.path.splitext(os.path.splitext(self.filename)[0])[0]
            savename = root + "_" + save_suffix + "." + _FSNCATEXT_
            
            ##If file exists and not overwrite, exit now.
            if os.path.exists(savename) and not overwrite:
                print ("Warning: Fake SN catalog exists. Will not overwrite, so we won't save the catalog.")
                savename = None
                
        elif save_suffix == None: ##Don't save catalog for this image
            savename = None
            
        
        RA = []
        DEC = []
        SCA = []
        F = []
        MOD = []
        X = []
        Y = []
        for i in file_header.keys():
             if i[0:2] == "FK" and int(i[2:5]) not in fakes: #Identify header entries for fake SN
                N = i[2:5]
                fakes.append(int(N))
                RA.append(file_header["FK" + str(N) + "RA"])
                DEC.append(file_header["FK" + str(N) + "DEC"])
                SCA.append(file_header["FK" + str(N) + "SCA"])
                F.append(file_header["FK" + str(N) + "MOD"])
                MOD.append(file_header["FK" + str(N) + "MOD"])
                X.append(file_header["FK" + str(N) + "X"])
                Y.append(file_header["FK" + str(N) + "Y"])

        racol = Column(RA , name = "ra")
        deccol = Column(DEC , name = "dec")
        scacol = Column(SCA , name = "sca")
        fcol = Column(F , name = "F")
        modcol = Column(MOD , name = "mod")
        xcol = Column(X , name = "x")
        ycol = Column(Y , name = "y")
        
        
        if savename != None: ##Writes (or overwrites) new file 
        
            self.fakesncat = Table([racol , deccol , scacol , fcol , modcol , xcol , ycol])
            self.fakesncat.write( savename , format =_FSNCATFORMAT_ , overwrite = True)
            
        elif add_to_filename != None:
        
            if os.path.exists(add_to_filename): 
                ##File exists, so we add to the existing catalog
                
                self.read_fakesn_catalog(filename = add_to_filename)
                new_table = Table([racol , deccol , scacol , fcol , modcol , xcol , ycol])
                combined_table = vstack([self.fakesncat , new_table])
                combined_table.write(add_to_filename , format = _FSNCATFORMAT_ , overwrite = True)
                
            else:
                ##File does not exist, so we make one
                
                self.fakesncat = Table([racol , deccol , scacol , fcol , modcol , xcol , ycol])
                self.fakesncat.write( add_to_filename , format =_FSNCATFORMAT_ , overwrite = True)
             
        return
        
    def read_fakesn_catalog(self , save_suffix = "fakecat" , filename = None):
        """
        
        Reads in a fake source catalog
        

        Parameters
        ----------

        save_suffix: str
            If provided, read the fake sourc catalog named as
            <rootname_of_this_fits_file>_<save_suffix>.<_GAIACATEXT_>
            Will be ignored if a filename is provided

        filename: str
            If provided, will read in a catalog with this filename. Overwrites
            any save_suffix that is provided
        
        self.fakesncat : Astropy Table : Contains information on the fake sources and
        their host galaxies
        
        """
        
        if filename != None:
            root = os.path.splitext(os.path.splitext(self.filename)[0])[0]
            readname = root + "_" + save_suffix + "." + _FSNCATEXT_
        else:
            readname = filename
        
        self.fakesncat = Table.read(readname , format =_FSNCATFORMAT_)
        
    def write_hostgalaxy_catalog(self , filename , overwrite = False , add_to = False):
        
        '''
        Will record all host galaxy information into a host galaxy
        catalog
        
        Parameters
        __________
        
        filename: str
            File name for the catalog.
            
        overwrite: boolean
            If true, overwrite any existing catalog with the same filename
            Otherwise we will not overwrite the existing file.
            
        add_to: boolead
            If True we append the host galaxy data onto an existing catalog.
            Otherwise we create a new file or overwrite existing file
        '''
            
        if self.hostgalaxies == None:
            ##Generate host galaxy catalog first if necessary
            
            self.detect_host_galaxies()
            
            
        write_to_catalog(self.hostgalaxies, filename = filename , overwrite = overwrite , add_to = add_to)
        

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

        # has_lco_epsf False until run lco_epsf
        self.has_lco_epsf = False
        # detection_efficiency None until calculated
        self.detection_efficiency = None
        # detected fake sources None until determined
        self.plant_detections = None
        return

    @property
    def has_fakes(self):
        """Returns a list of the component images (templateim, searchim,
        diffim) that have fakes planted.
        If none have fakes, then returns an empty list.
        """
        hasfakeslist =  []
        for im in [self.templateim, self.searchim, self.diffim]:
            if im.has_fakes:
                hasfakeslist.append(im)
        return(hasfakeslist)


    @property
    def has_epsfmodel(self):
        """True if both the diffim and searchim have an ePSF model.
        Otherwise False.
        """
        if ( self.diffim.psfmodel is not None and
            self.searchim.psfmodel is not None ):
            return ( type(self.diffim.psfmodel) == EPSFModel and
                     type(self.searchim.psfmodel) == EPSFModel)
        return False


        # optional?: record pre-existing info about the image + measurements
        # of the ePSF model in the pipeline log file: FWHM, zeropoint
        return

    def generate_lens_parameters(self,phi_func=None,d_func=None,NImage=2):
        """
        Generates phi and d lens parameters based on a given function or a uniform
        distribution

        Parameters
        ----------
        phi_func: function
            Accepts NImage as an argument and returns an iterable of size NImage
            containing values of phi
        d_func: function
            Accepts NImage as an argument and returns an iterable of size NImage
            containing values of d
        NImage: int
            Number of images in the system

        Returns
        -------
        phi,d: lists
        """
        
        if phi_func is None:
            phi = np.random.uniform(0,360,size=NImage) # TODO uniform default?
        else:
            phi = phi_func(NImage)

        if d_func is None:
            d = np.random.uniform(0,10,size=NImage) # TODO Make real choice about distance
        else:
            d = d_func(NImage)

        return phi,d

    def set_fake_positions_at_galaxies(self, phi_deg, d_pix,
                                       fluxes=None, galaxy_indices=None):
        """
        Define the fake source pixel positions in each image of the triplet,
        using detected galaxies in the image.  Each fake is placed at a
        distance d relative to each galaxy's center and angle phi
        relative to the orientation of the galaxy's semi-major axis.

        Parameters
        ----------
        phi_deg : List or array (must be 1D)
            Fake source angles CCW from galaxy semimajor axis. degs
        d_pix : List or array (must be 1D)
            Fake source distances from host center. pixels
        fluxes: Default None will generate list same length as positions of constant flux = 10**4
            List or Array (must be 1D same length as positions) will set the fluxes 
        galaxy_indices : List or array (must be 1D)
            Index to the self.hostgalaxies table, indicating which galaxy
            to plant each SN in.  For mimicking lensed SN doubles, just repeat the
            same galaxy index twice, for quads: 4 times, etc.
            Default of None will select galaxies randomly from the
            hostgalaxies table


        Returns
        -------
        posfluxlist : list of three Tables (`~astropy.table.Table`)
                Each table gives x,y Positions and fluxes for the fake sources.
                The columns are labeled 'x_fit', 'y_fit', and 'flux_fit'
                suitable for feeding to the add_psf function.
                posfluxlist[0] is the diff image posflux table
                posfluxlist[1] is the search image posflux table
                posfluxlist[2] is the template image posflux table
        """
        # TODO : need to check that the host galaxy orientation is what we
        #  think it is.  i.e., is it in deg from the +x direction? or from N?

        # the host galaxy properties will be taken from the template which
        # should have the best detections
        hostgalaxies = self.templateim.hostgalaxies
        Ngalaxies = len(hostgalaxies)
        Nfakes = len(phi_deg)

        if galaxy_indices is None:
            # pick galaxies at random, with replacement
            # (allows a small chance of doubles)
            rng = np.random.default_rng()
            galaxy_indices = rng.choice(Ngalaxies, Nfakes, replace=True)

        # the pixel location from the centroid of detection on template
        x_template = np.array([xpix.value
                               for xpix in hostgalaxies["x"][galaxy_indices] ])
        y_template = np.array([ypix.value
                               for ypix in hostgalaxies["y"][galaxy_indices] ])

        # the search/diff locations will use their corresponding pixel locations for this sky location
        # needs to be included in the case that search or diff isn't sized the same as template
        template_wcs = self.templateim.wcs
        search_wcs = self.searchim.wcs
        diff_wcs = self.diffim.wcs

        sky_location = wcsutils.pixel_to_skycoord(
            x_template, y_template, template_wcs)
        search_location = wcsutils.skycoord_to_pixel(sky_location,search_wcs)
        x_search,y_search = search_location
        diff_location = wcsutils.skycoord_to_pixel(sky_location,diff_wcs)
        x_diff, y_diff = diff_location

        # Not needed?  Semi-major and semi-minor axis lengths
        #a = np.array([apx.value for apx in
        #              hostgalaxies["semimajor_axis_sigma"][galaxy_indices]] )
        #b = np.array([bpx.value for bpx in
        #              hostgalaxies["semiminor_axis_sigma"][galaxy_indices]] )

        # Orientation is the angle of each galaxy's semimajor axis, a,
        # in degrees
        # NOTE: I'm not sure if this is relative to the +x axis or
        # relative to North.
        orientation = units.deg * np.array(
            [srcprop.orientation.value for srcprop
             in hostgalaxies["Source Properties"][galaxy_indices] ])

        # Compute the offsets from galaxy center position
        delta_x = d_pix * np.cos((orientation + phi_deg * units.deg))
        delta_y = d_pix * np.sin((orientation + phi_deg * units.deg))

        # Apply the offsets to get the x,y locations where the fakes
        # will go (separately for each image in the triplet)
        xfake_temp = x_template + delta_x
        yfake_temp = y_template + delta_y
        xfake_search = x_search + delta_x
        yfake_search = y_search + delta_y
        xfake_diff = x_diff + delta_x
        yfake_diff = y_diff + delta_y

        # if user doesn't provide a list of fluxes, set all as a constant flux
        # TODO : set this to a constant mag, when the zeropoint is defined?
        if fluxes is None:
            fluxes = [10**3 for i in range(Nfakes)]

        # put into table ready for entry as photutils subtract_psf posflux arg
        meta = {"phi_deg":phi_deg,"d_pix":d_pix,"galaxy_indices":galaxy_indices,"delta_x":delta_x,"delta_y":delta_y}
        posflux_template = Table(data=[xfake_temp, yfake_temp, fluxes],
                                 names=["x_fit", "y_fit", "flux_fit"],meta=meta)
        posflux_search = Table(data=[xfake_search, yfake_search, fluxes],
                               names=["x_fit", "y_fit", "flux_fit"],meta=meta)
        posflux_diff = Table(data=[xfake_diff, yfake_diff, fluxes],
                             names=["x_fit", "y_fit", "flux_fit"],meta=meta)
            
        self.templateim.lensed_locations = posflux_template
        self.searchim.lensed_locations = posflux_search
        self.diffim.lensed_locations = posflux_diff

        return [posflux_diff,posflux_search,posflux_template]

    def plot_lensed_locations(self,writetodisk=False,saveas="lensed_locations.pdf"):
        """
        mpl figure showing cutout on host galaxy with ellipse, and lensed locations
        cutout taken from template image
        """
        
        try:
            assert(self.templateim is not None)
        except assertionerror:
            print("No template image. Provide a templateim_fitsfilename to FakePlanter")
            return

        try: 
            assert(self.templateim.hostgalaxies is not None)
        except assertionerror:
            print("No hostgalaxies. Run detect_host_galaxies()")
            
        try:
            assert(self.templateim.lensed_locations is not None)
        except assertionerror:
            print("No lensed locations. Run set_fake_positions_at_galaxies()")
        
        # going to draw patches of arcs showing explicitly what phi,d are w respect to on a galaxy
        # setting the arc_lw 
        arc_lw = 6

        # the host's detect_sources properties
        hostgalaxies = self.templateim.hostgalaxies

        # the lensed locations
        lensed_locations = self.templateim.lensed_locations
        galaxy_indices = lensed_locations.meta['galaxy_indices']
        delta_x = lensed_locations.meta['delta_x'].value
        delta_y = lensed_locations.meta['delta_y'].value
        phi_deg = lensed_locations.meta['phi_deg']
        d_pix = lensed_locations.meta['d_pix']
        
        # TODO this selection of which galaxy we want to plot could be more general...
        # just taking the first galaxy which is where quad lensed location is set in this nb example
        galaxy_idx = galaxy_indices[0]
        hostgalaxy = hostgalaxies[galaxy_idx]["Source Properties"].to_table()
        
        # cut hdu
        x = hostgalaxy["xcentroid"][0].value # pix
        y = hostgalaxy["ycentroid"][0].value # pix
        location = (x,y)
        size = 50
        cut = cut_hdu(self.templateim,location,size)
        # when placing patch with ellipse on the cut, it is centered on location of the hostgalaxy
        cut_xy = (size/2,size/2)

        #xtheta ytheta defined analytically for segm image using variance then partial theta ~ 0 gives an ellipse 
        a = hostgalaxy["semimajor_axis_sigma"][0].value # pix
        b = hostgalaxy["semiminor_axis_sigma"][0].value # pix
        orientation = hostgalaxy["orientation"][0].value # deg a-axis ccw from +x
        
        # patches for lensing galaxy
        # usually isophotal limit well represented by R ~ 3
        R = 3
        # mpl ellipse patch wants ctr in pixels, width and height as full lengths along image x,y
        # angle is rotation in deg ccw of semimajor ~ a with respect to +x 
        width = R*2*a 
        height = R*2*b 
        ellipse = matplotlib.patches.Ellipse(cut_xy,width,height,angle=orientation,fill=None)
        # mpl arrow patch wants x,y tail starts, dx,dy, tail lengths
        ga_dx = 10*np.cos(orientation*np.pi/180)
        ga_dy = 10*np.sin(orientation*np.pi/180)
        gal_arrow = matplotlib.patches.Arrow(cut_xy[0],cut_xy[1],ga_dx,ga_dy,width=1.0,color='white')
        ga_x,ga_y = cut_xy[0]+ga_dx,cut_xy[1]+ga_dy
        # mpl arc patch wants xy ctr, width/height lenths of horizontal/vertical axes, 
        # angle deg ccw +x, theta1 and theta2 ccw from angle 
        gal_arcsize=5
        gal_arc = matplotlib.patches.Arc(cut_xy,gal_arcsize,gal_arcsize,angle=0,theta1=0,theta2=orientation,color='white',lw=arc_lw)
        
        # patches for SN lensed locations
        circles,arrows,arcs = [],[],[]
        colors = ['red','blue','green','orange'] # four colors should be enough, no more than quad SN on galaxy
        arcsize = gal_arcsize + 2
        for i in range(len(lensed_locations)):
            xy = (cut_xy[0]+delta_x[i],cut_xy[1]+delta_y[i])
            circles.append(matplotlib.patches.Circle(xy,radius=3,fill=None,color=colors[i]))
            arrows.append(matplotlib.patches.Arrow(cut_xy[0],cut_xy[1],delta_x[i],delta_y[i],color=colors[i]))
            if phi_deg[i] > 0:
                arcs.append(matplotlib.patches.Arc(cut_xy,arcsize,arcsize,angle=orientation,theta1=0,theta2=phi_deg[i],color=colors[i],lw=arc_lw))
            else:
                arcs.append(matplotlib.patches.Arc(cut_xy,arcsize,arcsize,angle=orientation+phi_deg[i],theta1=0,theta2=np.abs(phi_deg[i]),color=colors[i],lw=arc_lw))

            arcsize += 2
            
        # get to plotting
        fig,ax=plt.subplots(figsize=(10,10))
        ax.imshow(zscale(cut.data),origin='lower',cmap=cm.Greys)
        ax.add_patch(ellipse) # ellipse a little questionable at the moment, mpl patch is vague on orientation 
        ax.add_patch(gal_arrow)
        ax.add_patch(gal_arc)
        bbox,fontsize=dict(facecolor='white', alpha=0.5),12
        plt.text(ga_x,ga_y,r'$\theta = {:.1f}$'.format(orientation),bbox=bbox,fontsize=fontsize)

        for i in range(len(lensed_locations)):
            ax.add_patch(circles[i])
            ax.add_patch(arrows[i])
            ax.add_patch(arcs[i])
            xy = (cut_xy[0]+delta_x[i],cut_xy[1]+delta_y[i])
            plt.text(xy[0],xy[1],'$\phi = {:.1f}, $ \n $d = {:.1f} $'.format(phi_deg[i],d_pix[i]),bbox=bbox,fontsize=fontsize)
            
        # show the hline of +x axis which is what ellipse orientation from detect_sources is relative to
        plt.hlines(cut_xy[1],0,size,colors='black',linestyles='--')
        plt.xlim(0,size)
        plt.ylim(0,size)
        plt.show()
        if writetodisk:
            plt.savefig(saveas,bbox_inches="tight")
        return


    def plant_fakes_triplet(self, posfluxlist, psfmodel='epsf',
                            writetodisk=False, save_suffix="withfakes", **kwargs):
        """Function for planting fake stars in the diff image and the search
        image.  Using the ePSF model defined by the search image, adds fake
        PSFs at the x,y pixel positions and flux scales given in posfluxtable.
        This could be positioned around detected galaxy positions (i.e., if the
        posfluxtable was generated using set_fake_positions_at_galaxies).

        The diff image fits header is updated to carry meta-data for each
         planted fake, including the location and total flux.

        Parameters
        ----------
        posfluxlist : list of three Tables (`~astropy.table.Table`)
                Each table gives x,y Positions and fluxes for the fake sources.
                The columns are labeled 'x_fit', 'y_fit', and 'flux_fit'
                suitable for feeding to the add_psf function.
                posfluxlist[0] is the diff image posflux table
                posfluxlist[1] is the search image posflux table
                posfluxlist[2] is the template image posflux table

        psfmodel : str or `astropy.modeling.Fittable2DModel` instance
            PSF/PRF model to be substracted from the data.
            If a str, must be the name of a PSF model currently defined as a
            property of the searchim (e.g. 'epsf')

        writetodisk :  bool
            if True - write out the modified image data as fits files.
            Separate fits files are written for the search image and the
            difference image.

        save_suffix : str
            suffix to use for the output fits files. Each filename is defined
            as  <original_fits_filename_root>_<save_suffix>.fits

        Returns
        -------
        Nothing.  Fakes are in self.diffim and self.searchim

        """
        if type(psfmodel) is str:
            psfmodel = self.searchim.__getattribute__(psfmodel)

        # TODO : this function was written to expect two separate tables of
        #  fake positions (x,y,flux) for the sci image and the diff image.
        #  Shouldn't these be identical?   See issue #110
        #
        #  Would be better to get RA,Dec and let this plant_fakes_triplet
        #  function handle the wcs conversions ?
        # Maybe need a flag in plant_fakes_in_sci that allows RA,Dec instead
        # of x,y ?

        self.diffim.plant_fakes_in_sci(psfmodel, posfluxlist[0], **kwargs)
        self.searchim.plant_fakes_in_sci(psfmodel, posfluxlist[1], **kwargs)

        # TODO : add writing to disk
        if writetodisk:
            print("Oops. We haven't written this yet.")
        return


    def plot_fakes(self, fake_indices, cutoutsize=50, showtriplets=True):
        """
        Show small cutouts of the fakes planted in the diffim and searchim

        Parameters
        ----------
        fake_indices : List-like
            indices for the fake sources, corresponding to the 'FKnnnX' and
            'FKnnnY' cards (and associated) that were written into the header
            of the 'sci' image for the diffim and/or searchim FitsImage objects

        cutoutsize : int
            number of pixels on a side for the image to be shown. We cut it in
            half and use the integer component, so if an odd number or float is
            provided it is rounded down to the preceding integer.

        showtriplets : bool


        """
        halfwidth = int(cutoutsize/2)

        # set up a grid of axes with the appropriate size
        nfakes = len(fake_indices)
        gridsize1 = np.int(np.sqrt(nfakes))
        gridsize2 = int(nfakes / gridsize1)
        if gridsize1*gridsize2 < nfakes:
            gridsize2 += 1

        fig = plt.figure(1, (3.*gridsize2, 3.*gridsize1))
        grid = ImageGrid(fig, 111, nrows_ncols=(gridsize1, gridsize2),
                         axes_pad=0.1,
                         )
        for i, idx in zip(range(nfakes), fake_indices):
            # get the fake x,y location and flux
            x = self.diffim.sci.header[ f'FK{idx:03d}X' ]
            y = self.diffim.sci.header[ f'FK{idx:03d}Y' ]
            flux = self.diffim.sci.header[ f'FK{idx:03d}F' ]

            # grab some pixels for the fake source
            cutout = self.diffim.sci.data[
                     int(y)-halfwidth:int(y)+halfwidth,
                     int(x)-halfwidth:int(x)+halfwidth]

            # set the vmin,vmax for scaling
            vmax = np.min( [np.std(cutout) * 5, np.max(cutout)] )
            vmin = np.max( [np.median(cutout) - np.std(cutout) * 3,
                            np.min(cutout)] )

            # show us the pixels!
            ax = grid[i]
            ax.imshow( cutout, vmin=vmin, vmax=vmax, cmap=cm.Greys,
                       interpolation='nearest', origin='lower',
                       aspect='equal')

            # label it with the x,y,flux values
            ax.text(0.05, 0.95, f'{x:.1f}, {y:.1f}, {flux:.1e}',
                    ha='left', va='top', color='r', transform=ax.transAxes)

        return

    def png_MEF(self,MEF,show=True,writetodisk=False,saveas=None,singleABC=True,separateABC=False):
        """
        Create PNG for MEF fits file

        A. a difference image, has detection (plant or false positive)
        B. a 'search' image (typically a "new" single-epoch static sky image), has detection
        C. the template image (or 'reference'), does not have detection

        Parameters
        ----------
        MEF: str or 'astropy.io.fits.hdu.hdulist.HDUList' 
            If string will attempt fits.open(MEF)
            Otherwise should be already opened single multi-extension fits file
        show : bool
            True or False to display the image
        writetodisk : bool
            True or False to save the png to disk
        saveas : None or str
            default None will use the header['MEF'] ~ FKNNN.png or FPNNN.png 
        singleABC : bool
            3x1 image of A,B,C as single png
        separateABC : bool
            3 separate images one png for each A, B, and C 
        """

        # TODO self.MEF for plants and false positives as part of our FakePlanter Class?
        # I think requiring MEF as input here is okay since want it to be specific
        # plants_MEF and FP_MEF would return list-like  

        if type(MEF) == str:
            try:
                MEF = fits.open(MEF)
            except:
                print("Couldn't read MEF provided.")
                return

        # assert we have MEF 
        try:
            assert(MEF[0].header['MEF'] != None)
        except:
            print("No MEF. Provide an MEF to FakePlanter using plants_MEF or FP_MEF.")
            return

        if saveas == None:
            saveas = MEF[0].header['MEF'] + ".png"

        if singleABC:
            fig,ax=plt.subplots(1,3)
            ax[0].imshow(MEF[1].data)
            ax[1].imshow(MEF[2].data)
            ax[2].imshow(MEF[3].data)
            if writetodisk:
                plt.savefig(saveas)
            if show:
                plt.show()

        if separateABC:
            plt.imshow(MEF[1].data)
            if writetodisk:
                plt.savefig("A"+saveas)
            if show:
                plt.show()

            plt.imshow(MEF[2].data)
            if writetodisk:
                plt.savefig("B"+saveas)
            if show:
                plt.show()

            plt.imshow(MEF[3].data)
            if writetodisk:
                plt.savefig("C"+saveas)
            if show:
                plt.show()

        return




    def plants_MEF(self,fake_indices,cutoutsize=50,writetodisk=False,saveas=None):
        """
        Create MEF Fits file ~ triplet of cutouts around planted FKnnn sources  
        primary data empty, primary header has FKnnn indicating which cutout
        returns list like with MEF for each of the list like FKnnn fake_indices provided 
            
        A. a difference image, has plant
        B. a 'search' image (typically a "new" single-epoch static sky image), has plant
        C. the template image (or 'reference'), does not have plant
        
        Parameters
        ----------
        fake_indices : List-like
            indices for the fake sources, corresponding to the 'FKnnnX' and
            'FKnnnY' cards (and associated) that were written into the header
            of the 'sci' image for the diffim and/or searchim FitsImage objects

        cutoutsize : int
            number of pixels on a side for the image to be shown. We cut it in
            half and use the integer component, so if an odd number or float is
            provided it is rounded down to the preceding integer.
        """
        
        # assert have A,B,C
        try:
            assert(self.diffim is not None)
        except assertionerror:
            print("No difference image. Provide a differenceim_fitsfilename to FakePlanter")
            return
        try:
            assert(self.searchim is not None)
        except assertionerror:
            print("No search image. Provide a searchim_fitsfilename to FakePlanter")
            return
        try:
            assert(self.templateim is not None)
        except assertionerror:
            print("No template image. Provide a templateim_fitsfilename to FakePlanter")
            return
        
        # assert A,B have fakes
        try:
            assert(self.diffim.has_fakes==True)
        except:
            print("No fakes in difference image. Try to run plant_fakes_triplet()")
        try:
            assert(self.searchim.has_fakes==True)
        except:
            print("No fakes in search image. Try to run plant_fakes_triplet()")
        
        # the search/diff locations will use their corresponding pixel locations for this sky location
        # needs to be included in the case that search or diff isn't sized the same as template
        # template doesn't have associated FKNNNX/Y locations since not planting to template
        template_wcs = self.templateim.wcs
        search_wcs = self.searchim.wcs
        diff_wcs = self.diffim.wcs

        nfakes = len(fake_indices)
        
        MEFS = []
        for i, idx in zip(range(nfakes), fake_indices):
            
            # get the fake x,y locations
            xdiff = self.diffim.sci.header[ f'FK{idx:03d}X' ]
            ydiff = self.diffim.sci.header[ f'FK{idx:03d}Y' ]
            diff_location = (xdiff,ydiff)
            xsearch = self.searchim.sci.header[ f'FK{idx:03d}X' ]
            ysearch = self.searchim.sci.header[ f'FK{idx:03d}Y' ]
            search_location = (xsearch,ysearch)
            sky_location = wcsutils.pixel_to_skycoord(
                xdiff, ydiff, diff_wcs)
            template_location = wcsutils.skycoord_to_pixel(sky_location,template_wcs)
            x_template,y_template = template_location
            # and flux
            flux = self.diffim.sci.header[ f'FK{idx:03d}F' ]
            
            # grab cutouts
            cutdiff = cut_hdu(self.diffim,diff_location,cutoutsize)
            cutsearch = cut_hdu(self.searchim,search_location,cutoutsize)
            cuttemp = cut_hdu(self.templateim,template_location,cutoutsize)
            
            # create MEF
            primary = fits.PrimaryHDU(data=None,header=None)
            primary.header["Author"] = "Kyle OConnor"
            primary.header["MEF"] = f'FK{idx:03d}'
            try:
                assert(fits.CompImageHDU == type(self.diffim.postage_stamp)==type(self.searchim.postage_stamp)==type(self.templateim.postage_stamp))
            except:
                self.diffim.postage_stamp = fits.CompImageHDU(data=self.diffim.postage_stamp.data,header=self.diffim.postage_stamp.header)
                self.searchim.postage_stamp = fits.CompImageHDU(data=self.searchim.postage_stamp.data,header=self.searchim.postage_stamp.header)
                self.templateim.postage_stamp = fits.CompImageHDU(data=self.templateim.postage_stamp.data,header=self.templateim.postage_stamp.header)

            new_hdul = fits.HDUList([primary, self.diffim.postage_stamp,self.searchim.postage_stamp,self.templateim.postage_stamp])
            if writetodisk:
                if saveas == None:
                    new_hdul.writeto(f'FK{idx:03d}.fits', overwrite=True)
                else:
                    new_hdul.writeto(f'{saveas}_FK{idx:03d}.fits', overwrite=True)
            MEFS.append(new_hdul)

        return MEFS

    def FP_MEF(self,false_positives = None,cutoutsize=50,writetodisk=False,saveas=None):
        """
        Create MEF Fits file ~ triplet of cutouts around false_positives 
        primary data empty, primary header has FPX and FPY indicating which cutout on diff
        returns list like with MEF for each false positives in row of the Table like provided 
            
        A. a difference image, has plants
        B. a 'search' image (typically a "new" single-epoch static sky image), has plants
        C. the template image (or 'reference'), does not have plants
        
        Parameters
        ----------
        false_positives : Astropy Table
            detected source catalog complete with photometry (detect_sources on clean diff not the planted)

        cutoutsize : int
            number of pixels on a side for the image to be shown. We cut it in
            half and use the integer component, so if an odd number or float is
            provided it is rounded down to the preceding integer.
        """
        
        if false_positives == None:
            false_positives = self.find_false_positives(edges=True)
        
        # assert have A,B,C
        try:
            assert(self.diffim is not None)
        except assertionerror:
            print("No difference image. Provide a differenceim_fitsfilename to FakePlanter")
            return
        try:
            assert(self.searchim is not None)
        except assertionerror:
            print("No search image. Provide a searchim_fitsfilename to FakePlanter")
            return
        try:
            assert(self.templateim is not None)
        except assertionerror:
            print("No template image. Provide a templateim_fitsfilename to FakePlanter")
            return
        
        # assert A,B have fakes
        try:
            assert(self.diffim.has_fakes==True)
        except:
            print("No fakes in difference image. Try to run plant_fakes_triplet()")
        try:
            assert(self.searchim.has_fakes==True)
        except:
            print("No fakes in search image. Try to run plant_fakes_triplet()")
        
        # the search/diff locations will use their corresponding pixel locations for this sky location
        # needs to be included in the case that search or diff isn't sized the same as template
        # template doesn't have associated FKNNNX/Y locations since not planting to template
        template_wcs = self.templateim.wcs
        search_wcs = self.searchim.wcs
        diff_wcs = self.diffim.wcs
        
        MEFS = []
        for i in range(len(false_positives)):
            
            # get the fake x,y locations
            xdiff = false_positives[i]['x']
            ydiff = false_positives[i]['y']
            diff_location = (xdiff,ydiff)
            sky_location = wcsutils.pixel_to_skycoord(
                xdiff, ydiff, diff_wcs)
            search_location = wcsutils.skycoord_to_pixel(sky_location,search_wcs)
            xsearch,ysearch = search_location
            template_location = wcsutils.skycoord_to_pixel(sky_location,template_wcs)
            x_template,y_template = template_location
            # and mag
            mag = false_positives[i]['mag']
            
            # grab cutouts
            cutdiff = cut_hdu(self.diffim,diff_location,cutoutsize)
            cutsearch = cut_hdu(self.searchim,search_location,cutoutsize)
            cuttemp = cut_hdu(self.templateim,template_location,cutoutsize)
            
            # create MEF
            primary = fits.PrimaryHDU(data=None,header=None)
            primary.header["Author"] = "Kyle OConnor"
            primary.header["MEF"] = f'FP{i:03d}'
            try:
                assert(fits.CompImageHDU == type(self.diffim.postage_stamp)==type(self.searchim.postage_stamp)==type(self.templateim.postage_stamp))
            except:
                self.diffim.postage_stamp = fits.CompImageHDU(data=self.diffim.postage_stamp.data,header=self.diffim.postage_stamp.header)
                self.searchim.postage_stamp = fits.CompImageHDU(data=self.searchim.postage_stamp.data,header=self.searchim.postage_stamp.header)
                self.templateim.postage_stamp = fits.CompImageHDU(data=self.templateim.postage_stamp.data,header=self.templateim.postage_stamp.header)
                
            new_hdul = fits.HDUList([primary, self.diffim.postage_stamp,self.searchim.postage_stamp,self.templateim.postage_stamp])
            if writetodisk:
                if saveas == None:
                    new_hdul.writeto(f'FP{i:03d}.fits', overwrite=True)
                else:
                    new_hdul.writeto(f'{saveas}_FP{i:03d}.fits', overwrite=True)
            MEFS.append(new_hdul)

        return MEFS

    def has_fakes(self):
        """Check if fake stars have been planted in the image"""
        return self.has_fakes


    def find_plant_detections(self , image_with_fakes = None):

        """
        Builds catalog of succesfully detected fake sources

        Parameters
        ----------
        image_with_fakes : A diff image with fakes planted. If None, will assume that it is self.diffim
            In this case we will use detect sources with the default parameters

        Returns
        -------
        self.plant_detections : Astropy Table : Contains the x and y positions for each planted fake, and
            the detect column will contain a 1 if the source is recovered succesfully and a 0 otherwise
        """

        if image_with_fakes == None:
            image_with_fakes = self.diffim

        if not image_with_fakes.has_detections:
            image_with_fakes.detect_sources()

        fakeID , fakeposition = self.get_fake_locations(image_with_fakes)

        pixscale = image_with_fakes.sci.header["PIXSCALE"]
        FWHM = image_with_fakes.sci.header["L1FWHM"]
        radius = FWHM / pixscale

        detect = []
        x = []
        y = []
        magnitudes = []

        for i in fakeposition:
            x.append(i[0])
            y.append(i[1])
            d = 0
            for k in image_with_fakes.sourcecatalog:
                if np.sqrt( (k.centroid[1].value - i[0]) ** 2 + (k.centroid[0].value - i[1]) ** 2) < radius:
                    d = 1
                    break
            detect.append(d)

        if not image_with_fakes.stellar_phot_table:
            image_with_fakes.do_stellar_photometry(image_with_fakes.sourcecatalog.to_table())
        if not self.searchim.gaia_source_table:
            self.searchim.fetch_gaia_sources()
        if not self.searchim.stellar_phot_table:
            self.searchim.do_stellar_photometry(self.searchim.gaia_source_table)
        if not self.searchim.zeropoint:
            self.searchim.measure_zeropoint()

        for i in range(len(x)):
            xposition = x[i]
            yposition = y[i]

            found = False
            for k in image_with_fakes.stellar_phot_table:

                if np.sqrt( (k['xcenter'].value - xposition) ** 2 + (k['ycenter'].value - yposition) ** 2) < radius:
                    if np.isnan(k['mag']):
                        magnitudes.append(k['mag'])
                    else:
                        magnitudes.append(k['mag'] + self.searchim.zeropoint)
                    found = True
                    break
            if found:
                continue
            magnitudes.append(999)

        self.plant_detections = Table([x , y , detect , magnitudes] , names = ('x','y','detect' , 'mag'))
        
        return self.plant_detections

    def find_false_positives(self , clean_diff = None, edges = False):
        """
        Runs aperture photometry on the false positives.

        Parameters
        ----------
        clean_diff : If None, will use self.diffim. Otherwise should be a FitsImage object
            containing a clean difference image corresponding to self.searchim
        edges: If False,will return all detected sources. Otherwise if True will exclude objects 
            having ctr that is within 50 pixels of an edge (ie ignore edge effects and restricts to useful FP to cutout for training ML).

        Returns
        -------
        false_positives : Astropy Table : detected source catalog complete with photometry


        """


        if not clean_diff:
            clean_diff = self.diffim

        shape = clean_diff.sci.data.shape

        ##radius for matching sources across catalogs
        pixscale = clean_diff.sci.header["PIXSCALE"]
        FWHM = clean_diff.sci.header["L1FWHM"]
        radius = FWHM / pixscale

        ##Find zeropoint (and other quantities) if necessary
        if not self.searchim.gaia_source_table:
            self.searchim.fetch_gaia_sources()
        if not self.searchim.stellar_phot_table:
            self.searchim.do_stellar_photometry(self.searchim.gaia_source_table)
        if not self.searchim.zeropoint:
            self.searchim.measure_zeropoint()


        if not clean_diff.sourcecatalog:
            clean_diff.detect_sources()

        if not clean_diff.stellar_phot_table:
            clean_diff.do_stellar_photometry(clean_diff.sourcecatalog.to_table())

        x = []
        y = []
        mag = []

        for i in clean_diff.sourcecatalog.to_table():
            xcenter = i["xcentroid"].value
            ycenter = i["ycentroid"].value

            if xcenter < 50 or ycenter < 50 and edges: ##Removes objects next to edges
                continue
            if xcenter > shape[0] - 50 or ycenter > shape[1] - 50 and edges:
                continue

            x.append(xcenter)
            y.append(ycenter)
            found = False
            for k in clean_diff.stellar_phot_table:
                if np.sqrt( (xcenter - k['xcenter'].value) ** 2 + (ycenter - k['xcenter'].value ) ** 2 ) < radius:
                    mag.append(k['mag'] + self.searchim.zeropoint)
                    found = True
                    break
            if not found:
                mag.append(999)

        return Table([x , y , mag ] , names = ('x' , 'y' , 'mag'))


    def confusion_matrix(self , fp_detections=None , low_mag_lim = None , high_mag_lim = None):
        """Function for creating confusion matrix of detections vs plants
        Clean 
        low_mag_lim : Will ignore sources with magnitudes less than this
        high_mag_lim : Will ignore sources with magnitudes greater than this
        """

        #TO-DO decide what the confusion_matrix shoud look like
        #imagining using the plant fits header for the plant detections and the threshold parameters for fp detections

        #plant_detections a yet to be defined property
        #will be something like a catalog/file with rows for each planted object 
        #plants have a col for detect ~ 1 is detection (TP), 0 is non-detection (FN)

        if self.plant_detections == None:
            self.find_plant_detections()

        plants = self.plant_detections
        
        #fp_detections is the same type of catalog/file from plants detection but run using the clean diff
        #default None will run it here
        #need to make sure detection flag gets updated in plant hdr during the efficiency function once thats in here
        
        TP = [] # detected plant
        FN = [] # not detected plant
        FP = [] # detected, but not a plant, (all the detections on clean diffim)
        TN = None # not detected not a plant, (no meaning)


        for i in plants:
            if low_mag_lim != None and (i['mag'] < low_mag_lim or np.isnan(i['mag'])):
                continue
                
            if high_mag_lim != None and (i['mag'] > high_mag_lim or np.isnan(i['mag'])):
                continue
            if i['detect'] == 1:
                TP.append(i)
            elif i['detect'] == 0:
                FN.append(i)

        if len(TP) != 0:
            TP = vstack(TP)
        else:
            TP = []
        if len(FN) != 0:
            FN = vstack(FN)
        else:
            FN = []
        
        if fp_detections:
            FP = fp_detections
        else:
            # TO-DO set the parameters in detect_sources using vals from run on the plant 
            # self.detection_vals = [nsigma,kfwhm,npixels,deblend,contrast]
            if low_mag_lim == None and high_mag_lim == None:
                FP = self.find_false_positives()
            else:

                false_positives = self.find_false_positives()
                FP = []
                for i in false_positives:
                    if high_mag_lim != None and i['mag'] > high_mag_lim:
                        continue
                    elif low_mag_lim != None and i['mag'] < low_mag_lim:
                        continue
                    FP.append(i)

                if len(FP) != 0:
                    FP = vstack(FP)
                else:
                    FP = []

        ##Filter out true positives in false positive list
        real_positives = [] ##List of indices to remove from FP
        for i in range(len(FP)):
            if FP[i] in TP:
                real_positives.append(i)
        if FP != []:
            FP.remove_rows(real_positives)

        return [TP,FN,FP,TN]
    
    def get_fake_locations(self,image_with_fakes=None):
        """Returns a list of fakeIDs and (x,y) pixel locations for the
        specified image.  The info for each fake is read from the
        'sci' attribute of the specified FitsImage object.  The 'sci' attribute
        is a fits HDU object, and the info for each fake is extracted from
        the header keywords (starting with 'FK').

        Parameters
        ----------
        image_with_fakes : `~fakeplanting.FitsImage`
            A FitsImage object containing the planted fake sources in it.
            (default self.diffim)

        """
        if image_with_fakes is None:
            image_with_fakes = self.diffim
        hdr = image_with_fakes.sci.header
        fake_plant_x_keys = [key for key in hdr.keys() if \
                             key.startswith('FK') and key.endswith('X')]
        fake_plant_x = []
        fake_plant_y = []
        fakeIDs = []
        for key in fake_plant_x_keys:
            fake_id_str = key[2:2+len(str(_MAX_N_PLANTS_))]
            fakeIDs.append(int(fake_id_str))
            fake_plant_x.append(hdr['FK%sX'%fake_id_str])
            fake_plant_y.append(hdr['FK%sY'%fake_id_str])
        fake_positions = np.array([fake_plant_x,fake_plant_y]).T
        return fakeIDs, fake_positions


    def set_fake_detection_header(self,image_with_fakes,detection_table=None,outfilename=None):
        if detection_table is None:
            detection_table = self.detection_table

        for row in detection_table:
            image_with_fakes.header['FK%sDET'%row['fakeID']] = row['detected']
        if isinstance(outfilename,str):
            fits.writeto(outfilename,image_with_fakes,image_with_fakes.header,overwrite=True)
        return image_with_fakes
        

    @property
    def has_detection_efficiency(self):
        return self.detection_efficiency is not None

    def calculate_detection_efficiency(self,image_with_fakes=None,
                fake_plant_locations=None,source_catalog=None,gridSize = 2,**kwargs):
        """
        Given a difference image with fake sources planted and a detected 
        source catalog, will calculate the detection efficiency.

        Parameters
        ----------
        image_with_fakes : `~fakeplanting.FitsImage`
            A fits image class containing the planted fake sources (default self.diffim)
        fake_plant_locations : list or `~numpy.ndarray`
            2D array containing the x,y locations of the fake sources (default read self.diffim.sci.header)
        source_catalog : :class:`~photutils.segmentation.properties.SourceCatalog`
            Detected source catalog 

        Returns
        -------
        detection_efficiency : float
        detection_table : `~astropy.table.Table` with ID,xy-locations,detected (1 or 0)
        """
        if image_with_fakes is None:
            image_with_fakes = self.diffim

        if source_catalog is None:
            if isinstance(image_with_fakes,FitsImage):
                if image_with_fakes.has_detections:
                    source_catalog = image_with_fakes.sourcecatalog
                else:
                    source_catalog = self.diffim.detect_sources(**kwargs)
            else:
                raise RuntimeError("If image_with_fakes is not of type FitsImage, must provide a source_catalog.")

        if fake_plant_locations is None:
            fake_plant_ids,fake_plant_locations = self.get_fake_locations(image_with_fakes)

        # use locations and a search radius on detections and plant locations to get true positives
        tbl = source_catalog.to_table()
        tbl_x,tbl_y = [i.value for i in tbl['xcentroid']], [i.value for i in tbl['ycentroid']]
        tbl_pixels = list(zip(tbl_x,tbl_y))        
        search = gridSize # fwhm*n might be better criteria

        truths = []
        binary_detection_dict = {key:0 for key in fake_plant_ids}
        for pixel in tbl_pixels:
            for ind in range(len(fake_plant_locations)):
                i = fake_plant_locations[ind]
                if pixel[0] > i[0] - search  and pixel[0] < i[0] + search and pixel[1] > i[1] - search and pixel[1] < i[1] + search:
                    truths.append([tuple(i),pixel])
                    binary_detection_dict[fake_plant_ids[ind]] = 1
                    break # TODO Think about multiple detections
                else:
                    continue

        plant_pixels = []
        det_src_pixels = []
        for i in truths:
            plant_pix = i[0]
            det_src_pix = i[1]
            if plant_pix not in plant_pixels:
                plant_pixels.append(plant_pix)
                det_src_pixels.append(det_src_pix)
        
        N_plants_detected = len(plant_pixels)
        efficiency = N_plants_detected/len(fake_plant_locations)
        binary_detection = [binary_detection_dict[fkID] for fkID in fake_plant_ids]
        detection_table = Table([fake_plant_ids,fake_plant_locations[:,0],fake_plant_locations[:,1],binary_detection],
                    names=['fakeID','pixX','pixY','detected'])
        
        if isinstance(image_with_fakes,FitsImage):
            image_with_fakes = self.set_fake_detection_header(image_with_fakes = image_with_fakes.sci,detection_table = detection_table)
        else:
            image_with_fakes = self.set_fake_detection_header(image_with_fakes = image_with_fakes,detection_table = detection_table)
        self.detection_efficiency = efficiency
        self.detection_table = detection_table
        self.diffim = image_with_fakes
        return self.detection_efficiency,self.detection_table
