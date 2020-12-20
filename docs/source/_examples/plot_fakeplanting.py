"""
===================
Fake Source Planting
===================

Demonstration of planting fake PSFs in diff images to mimic
strongly-lensed supernovae.
"""

import os
import numpy as np
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

print("FakePlanter Trio constructed.")
assert(fakeplantertrio.searchim.has_fakes == False)
assert(fakeplantertrio.diffim.has_fakes == False)
print("  (No fakes yet)")


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
fakeplantertrio.searchim.build_epsf_model(
    verbose=False, save_suffix='TestEPSFModel')


###############################################################
# Plant Fakes
# --------
#
# Here we plant just 10 very bright fakes

# detect sources in the template image, identify likely galaxies
fakeplantertrio.templateim.detect_sources()
hostgaltable = fakeplantertrio.templateim.detect_host_galaxies()


# Make 10 locations for random fakes (each relative to a galaxy center point)
Nfakes = 10
phi = np.random.uniform(0, 360, Nfakes)
d = np.random.uniform(0, 5, Nfakes)
fluxes = np.random.uniform(10**2, 10**4, Nfakes)

# fix the positions of the fakes in x,y coordinates on the diff image
# This returns three tables: one each for the diffim, searchim, and templateim
fake_positions_and_fluxes = fakeplantertrio.set_fake_positions_at_galaxies(
    phi, d, fluxes)

# Grab the existing ePSF model from the search image
epsfmodel = fakeplantertrio.searchim.epsf

# Plant the fakes
fakeplantertrio.plant_fakes_triplet(
    fake_positions_and_fluxes, psfmodel=epsfmodel,
    writetodisk=False, save_suffix="planted.fits")

print("Fake planting is done.")
assert(fakeplantertrio.diffim.has_fakes==True)
assert(fakeplantertrio.searchim.has_fakes==True)
print(" has_fakes is True, True!")


###############################################################
# Display Fakes
# --------
#
# Show a few examples of fakes from the diff image, using a few
# random indices from the list of fakes

fakeIDs, fake_positions =  fakeplantertrio.get_fake_locations()

rng = np.random.default_rng()
fakeids_to_show = rng.choice(fakeIDs, 3)
print(fakeids_to_show)


###############################################################
# TODO:  Show us the fakes!!
# --------
#

# fakeplantertrio.plot_fakes(fake_indices=fakeids_to_show)



###############################################################
# Make a trio of postage stamps for each fake
# --------
#
# Show the trio of fakes for each


