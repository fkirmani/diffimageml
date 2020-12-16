# diffimageml

[![Documentation Status](https://readthedocs.org/projects/diffimageml/badge/?version=latest)](http://diffimageml.readthedocs.org/en/latest/?badge=latest)

Applying machine learning for transient detection/classification in difference images.


Product Goal: a general-purpose difference image analysis pipeline that can identify multiply-imaged transients using machine learning. 

Inputs: difference images as FITS files.  Associated source catalogs from the static-sky (template) images and the difference images. 

Outputs: categorization score for any transient candidates in the image using three classes
0 - not a real transient
1 - single transient
2 - multiply-imaged transient candidate

Target datasets:  Strong lensing systems observed with Las Cumbres Observatory and the Hubble Space Telescope; simulated images for the Rubin Observatory LSST and Roman Space Telescope.

### Acknowledgements

Makes use of Gaia data: https://www.cosmos.esa.int/web/gaia-users/credits
