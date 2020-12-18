"""
===================
PSF Model Construction
===================

Demonstration of building an ePSF model from Gaia stars in an LCO image.
"""

import os
import diffimageml
from matplotlib import pyplot as plt


# Hard coding the test data filenames
_EXAMPLEDATADIR_ = diffimageml.get_example_data_dir()

_DIFFIM1_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'diff_pydia_1.fits.fz'))
_FAKEDIFFIM1_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'diff_pydia_1_fakegrid.fits'))
_SEARCHIM1_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'sky_image_1.fits.fz'))
_TEMPLATEIM1_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'template_1.fits.fz'))

_DIFFIM2_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'diff_pydia_2.fits.fz'))
_FAKEDIFFIM2_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'diff_pydia_2_fakegrid.fits'))
_SEARCHIM2_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'sky_image_2.fits.fz'))
_TEMPLATEIM2_ = os.path.abspath(os.path.join(
    _EXAMPLEDATADIR_, 'template_2.fits.fz'))


###############################################################

# %%
# PSF Model Construction Overview
# -------------------------------
#
# To construct a PSF model we go through the following steps
#
# * Load an example fits image
#
# * Fetch a catalog of stars in the image from the Gaia db
#
# * Do photometry of the Gaia stars
#
# * Measure the zero point for the image.
#
# * Build an ePSF model from the Gaia stars

assert(os.path.isfile(_SEARCHIM1_))
searchim = diffimageml.FitsImage(_SEARCHIM1_)

###############################################################
# Fetch a catalog of stars in the image from the Gaia db
# (or read in a saved local copy)

searchim.fetch_gaia_sources(overwrite=False)
searchim.plot_gaia_sources(magmin=12, magmax=18)


###############################################################
# Do photometry of the Gaia stars, within a user-specified
# magnitude range
searchim.do_stellar_photometry(searchim.gaia_source_table)

searchim.plot_stellar_photometry()

###############################################################
# Measure the zero point for this image from the Gaia stars
searchim.measure_zeropoint(showplot=True)


###############################################################
# Build an ePSF Model from the Gaia stars that are not saturated
searchim.build_epsf_model(verbose=False, save_suffix='TestEPSFModel')
searchim.plot_epsf_model()


