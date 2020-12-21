"""
===================
PSF Model Construction
===================

Demonstration of building an ePSF model from Gaia stars in an LCO image.
"""

import os
import diffimageml
from matplotlib import pyplot as plt

# Get a dict with test data filenames
example_data_dict = diffimageml.get_example_data()
searchim1 = example_data_dict['searchim1']

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

assert(os.path.isfile(searchim1))
searchim = diffimageml.FitsImage(searchim1)

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


