import astropy
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm
from astropy.table import Table,Column,Row,vstack,setdiff,join
from astropy.nddata import Cutout2D,NDData
import astropy.units as u
import os 
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval,simple_norm
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord


import photutils
from photutils.datasets import make_gaussian_sources_image

import numpy as np
import itertools
import copy

import matplotlib
from astropy.visualization import ZScaleInterval,simple_norm
zscale = ZScaleInterval()
import pickle
import numpy as np

import diffimageml
from diffimageml import fakeplanting, util

import PIL
import sys

###Most of the following is a repeat from examples/planting_lensing_galaxy.ipynb

use_example_dataset = 1

# get one of the two test data-sets provided
exampledata = util.get_example_data()

# initialize the FakePlanter Class 
if use_example_dataset==1:
    fp = fakeplanting.FakePlanter(exampledata['diffim1'],
                                  searchim_fitsfilename=exampledata['searchim1'],
                                  templateim_fitsfilename=exampledata['templateim1'])
elif use_example_dataset==2:
    fp = fakeplanting.FakePlanter(exampledata['diffim2'],
                                  searchim_fitsfilename=exampledata['searchim2'],
                                  templateim_fitsfilename=exampledata['templateim2'])

if len(sys.argv) > 1:
    if len(sys.argv) != 4:
        print ("Warning, insufficient number of input files provided")
        sys.exit()
    else:
        fp = fakeplanting.FakePlanter(sys.argv[1] , 
                                      searchim_fitsfilename=sys.argv[2],
                                      templateim_fitsfilename=sys.argv[3])
            
                                  
# Galaxies are identified using cuts on ellipticity/area to remove stars/cosmic rays from detected sources
fp.has_detections = False
fp.templateim.detect_host_galaxies(ellipticity_cut = 0.35 , cut_cr = True, edges = True, nsigma=10)
print("{} galaxies detected in the template".format(len(fp.templateim.hostgalaxies)))

print("For purpose of displaying positions (phi,d) for planting fakes around lensing host, selecting the first galaxy") 
galaxy_idx = 0
host = fp.templateim.hostgalaxies[galaxy_idx]

# Show the properties of the selected galaxy
print (host["Source Properties"].to_table())


# quad imaged SN
phi_deg = [45,-45,135,-135]
d_pix = [10,10,10,10]
fluxes = [2*10**3,3*10**3,4*10**3,5*10**3]
galaxy_indices = [galaxy_idx for i in range(len(phi_deg))]

print("Setting four locations around the galaxy (all of similar flux will be easily detectable)")

fake_positions_and_fluxes = fp.set_fake_positions_at_galaxies(phi_deg,d_pix,fluxes=fluxes,galaxy_indices=galaxy_indices)
print(fp.templateim.lensed_locations.meta)
#fp.templateim.lensed_locations

print("Theta (orientation from source_properties) is direction of the galaxy semimajor axis a, CCW with respect to +x")
print("Phi is CCW with respect to the semimajor axis")
print("d is pixels from galaxy centroid")

fp.plot_lensed_locations()

build_epsf = pickle.load(open(exampledata['psfmodel1'],"rb"))

# Plant the fakes
fp.plant_fakes_triplet(
    fake_positions_and_fluxes, psfmodel=build_epsf,
    writetodisk=False, save_suffix="planted.fits",preserve_original=False)

print("Fake planting is done.")
assert(fp.diffim.has_fakes==True)
assert(fp.searchim.has_fakes==True)
print("has_fakes is True, True!")

fake_indices=[0,1,2,3]
fp.plot_fakes(
    fake_indices=fake_indices)
    
"""
Going to use the remainder of the detected galaxies now to plant some more fake SNe
"""
print("{} galaxies detected in the template".format(len(fp.templateim.hostgalaxies)))
print("Recall from above, galaxy_idx = 0, we selected the first galaxy to use for planting our quad-SN demo")
print("We will now also include a single planted SN (solo) around each of the other detected galaxies (setting locations/fluxes at random)")
print('\n')

# We start by concatenating these host galaxies (with random parameters) onto the 
# pre-defined (see above) posflux table of the of the quad SN
hosts = fp.templateim.hostgalaxies[galaxy_idx+1:]
phi_deg = np.concatenate([phi_deg,np.random.uniform(low=0,high=180,size=len(hosts))])
d_pix = np.concatenate([d_pix,np.random.uniform(low=0,high=20,size=len(hosts))])
fluxes = np.concatenate([fluxes,np.random.uniform(low=10**3,high=10**4,size=len(hosts))])
galaxy_indices = np.concatenate([galaxy_indices,[i+1 for i in range(len(hosts))]])

fake_positions_and_fluxes = fp.set_fake_positions_at_galaxies(
    phi_deg,d_pix,fluxes=fluxes,galaxy_indices=galaxy_indices)

# Uncomment to see the randomly selected angles phi and pixel offsets d.
# print(fp.templateim.lensed_locations.meta)

# Show the final table of lensed transient positions + fluxes.
fp.templateim.lensed_locations  

# Plant the fakes
fp.plant_fakes_triplet(
    fake_positions_and_fluxes, psfmodel=build_epsf,
    writetodisk=False, save_suffix="planted.fits",preserve_original=False)

print("Fake planting is done.")
assert(fp.diffim.has_fakes==True)
assert(fp.searchim.has_fakes==True)
print("has_fakes is True, True!")

# Create the MEFS for all the fake plants
fake_indices = [i for i in range(len(fake_positions_and_fluxes[0]))]
MEFS = fp.plants_MEF(fake_indices,cutoutsize=50,writetodisk=True)

i = 3
for mef in MEFS[3:8]:
    # This creates a PNG for each image in the triplet, shows it,
    # then closes it.  No saving to disk.  
    print (i)
    i += 1
    fp.png_MEF(mef, show=True, writetodisk=False)
    
mef = MEFS[3] ##Random choice for testing purposes
    

#The following generates an RGB image and saves it using PIL
##PIL Only handles 3x8 png images with RGB, so pixel data has to be manipulated to make it fit

diff_data = mef[1].data
search_data = mef[2].data
templ_data = mef[3].data


diff_shift = ( np.amin(diff_data) )
search_shift = (np.amin(search_data) )
templ_shift = ( np.amin(templ_data) )


diff_data -= diff_shift
diff_data /= 2 ##Compress data. Works for exammple set one, needs to be generalized to work with other data

search_data -= search_shift
search_data /= 2

templ_data -= templ_shift
templ_data /= 2

##Check pixel values
if np.amax(diff_data) > 255:
    print ("Warning, max pixel value in diffs ({}) excedes max png pixel value".format(np.amax(diff_data)))
if np.amax(search_data) > 255:
    print ("Warning, max pixel value in search ({}) excedes max png pixel value".format(np.amax(search_data)))
if np.amax(templ_data) > 255:
    print ("Warning, max pixel value in templ ({}) excedes max png pixel value".format(np.amax(templ_data)))


##Build PIL images from fits data
diff=PIL.Image.fromarray(diff_data)
search=PIL.Image.fromarray(search_data)
templ=PIL.Image.fromarray(templ_data)

composite=PIL.Image.merge('RGB' , (diff.convert('L'),search.convert('L'),templ.convert('L'))) ##RGB Image


#Save outputs
diff.convert('L').save("diff1.png")
search.convert('L').save("search1.png")
templ.convert('L').save("templ1.png")
composite.save("test_data/LCO_example_1.png")

