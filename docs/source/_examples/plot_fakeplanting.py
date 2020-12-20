"""
===================
Fake Source Planting
===================

Demonstration of planting fake PSFs in diff images to mimic
strongly-lensed supernovae.
"""

import os
import diffimageml
from matplotlib import pyplot as plt

###############################################################

# %%
# Fake Source Planting Overview
# -------------------------------
#
# From a trio of images (template, search, diff), we
#
# #. find galaxies in the template image
#
# #. build a PSF model from the search image
#
# #. plant Fake PSFs in the diff image near the galaxy locations


###############################################################
# Setup 1: Get the Data
# ---------------------
# Load in a trio of fits images from the example data dir
# Pull them together into a FakePlanter triplet

example_data_dict = diffimageml.get_example_data()
assert(os.path.isfile(example_data_dict['diffim1']))
assert(os.path.isfile(example_data_dict['searchim1']))
assert(os.path.isfile(example_data_dict['templateim1']))

fakeplantertrio = diffimageml.FakePlanter(
    example_data_dict['diffim1'],
    searchim_fitsfilename=example_data_dict['searchim1'],
    templateim_fitsfilename=example_data_dict['templateim1'])


###############################################################
# Setup 2: Make the PSF Model
# ---------------------
#
# Measure the zero point and build the ePSF model from Gaia stars
# See the other example for details and plots.
# In practice, this code will load an existing ePSF model from the
# example data directory.

fakeplantertrio.searchim.fetch_gaia_sources(overwrite=False)
fakeplantertrio.searchim.do_stellar_photometry(
    fakeplantertrio.searchim.gaia_source_table)
fakeplantertrio.searchim.measure_zeropoint(showplot=False)
fakeplantertrio.searchim.build_epsf_model(verbose=False, save_suffix='TestEPSFModel')


###############################################################
# Plant Fakes
# --------
#
# Here we plant just 10 very bright fakes
epsfmodel = fakeplantertrio.searchim.epsf
hostgalcat = fakeplantertrio.templateim.detect_host_galaxies()
fakeplantertrio.plant_fakes_in_diffim(epsfmodel, posfluxtable=fakeloc)



###############################################################
# Make a trio of postage stamps for each fake
# --------
#
# Show the trio of fakes for each


