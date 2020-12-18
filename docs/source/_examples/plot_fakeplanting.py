"""
===================
Fake Source Planting
===================

Demonstration of planting fake PSFs in diff images to mimic
strongly-lensed supernovae.
"""
		
###############################################################
# Explain here what this example does
import os
import diffimageml

HIDE = """

_SEARCHIM1_ = '../../test_data/sky_image_1.fits.fz'
assert(os.path.isfile(_SEARCHIM1_))
searchim = diffimageml.FitsImage(_SEARCHIM1_)

# ## Fetch a catalog of stars in the image from the Gaia db
searchim.fetch_gaia_sources(overwrite=False)

# ## Do Photometry of the Gaia Stars
searchim.do_stellar_photometry(searchim.gaia_source_table)

# show photometry of the gaia stars
# searchim.plot_stellar_photometry()


# ## Measure the zero point for this image from the Gaia stars
searchim.measure_zeropoint(showplot=False)


# ## Build an ePSF Model from the Gaia stars that are not saturated

# Find galaxies in the template image

# ## Plant the ePSF model as fake lensed SNe around galaxies in the images
"""




