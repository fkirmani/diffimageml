"""
===================
PSF Model Construction
===================

Demonstration of building an ePSF model from Gaia stars in an LCO image.
"""
#import sys
import os
import diffimageml
#from diffimageml import util

#from diffimageml.test_diffimageml import _SRCDIR_

#_SRCDIR_ = os.path.abspath(os.path.join(
#    os.path.dirname(os.path.abspath(__file__)),'..'))

_EXAMPLEDATADIR_ = diffimageml.get_example_data_dir()

# Hard coding the test data filenames
_DIFFIM1_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'diff_pydia_1.fits.fz'))
_FAKEDIFFIM1_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'diff_pydia_1_fakegrid.fits'))
_SEARCHIM1_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'sky_image_1.fits.fz'))
_TEMPLATEIM1_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'template_1.fits.fz'))


hide="""
_DIFFIM2_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'diff_pydia_2.fits.fz'))
_FAKEDIFFIM2_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'diff_pydia_2_fakegrid.fits'))
_SEARCHIM2_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'sky_image_2.fits.fz'))
_TEMPLATEIM2_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'template_2.fits.fz'))
"""

###############################################################
# Explain here what this example does

assert(os.path.isfile(_SEARCHIM1_))
searchim = diffimageml.FitsImage(_SEARCHIM1_)

# ## Fetch a catalog of stars in the image from the Gaia db
# and show fetched gaia stars
searchim.fetch_gaia_sources(overwrite=False)
searchim.plot_gaia_sources(magmin=12, magmax=18)


# ## Do Photometry of the Gaia Stars
searchim.do_stellar_photometry(searchim.gaia_source_table)

# show photometry of the gaia stars
searchim.plot_stellar_photometry()


# ## Measure the zero point for this image from the Gaia stars
searchim.measure_zeropoint(showplot=True)


# ## Build an ePSF Model from the Gaia stars that are not saturated
# TODO : show the ePSF model building

