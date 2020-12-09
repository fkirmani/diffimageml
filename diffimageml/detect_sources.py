from astropy.io import ascii,fits
from astropy.wcs import WCS
from photutils import Background2D, MedianBackground, detect_threshold,detect_sources,source_properties
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm
from astropy.convolution import Gaussian2DKernel



def test_source_detection(hdu):
    return(detect_sources(hdu))


def detect_sources(hdu,nsigma=2,kfwhm=2.0,npixels=5,deblend=False,contrast=.001,targ_coord=None):
    """
    catalog of properties for detected sources using the extended source photometry on the bkg subtracted data
    """
    # to be able to translate from ra/dec <--> pixels on image
    hdr = hdu.header
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
    #L1mean,L1med,L1sigma,L1fwhm = hdr['L1MEAN'],hdr['L1MEDIAN'],hdr['L1SIGMA'],hdr['L1FWHM'] # counts, fwhm in arcsec 
    #pixscale,saturate,maxlin = hdr['PIXSCALE'],hdr['SATURATE'],hdr['MAXLIN'] # arcsec/pixel, counts for saturation and non-linearity levels
    # if bkg None: detect threshold uses sigma clipped statistics to get bkg flux and set a threshold for detected sources
    # bkg also available in the hdr of file, either way is fine  
    # threshold = detect_threshold(hdu.data, nsigma=nsigma)
    # or you can provide a bkg of the same shape as data and this will be used
    boxsize=100
    bkg = Background2D(hdu.data,boxsize) # sigma-clip stats for background est over image on boxsize, regions interpolated to give final map 
    threshold = detect_threshold(hdu.data, nsigma=nsigma,background=bkg.background)
    ksigma = kfwhm * gaussian_fwhm_to_sigma  # FWHM pixels for kernel smoothing
    # optional ~ kernel smooths the image, using gaussian weighting
    kernel = Gaussian2DKernel(ksigma)
    kernel.normalize()
    # make a segmentation map, id sources defined as n connected pixels above threshold (n*sigma + bkg)
    segm = detect_sources(hdu.data,
                          threshold, npixels=npixels, filter_kernel=kernel)
    # deblend useful for very crowded image with many overlapping objects...
    # uses multi-level threshold and watershed segmentation to sep local peaks as ind obj
    # use the same number of pixels and filter as was used on original segmentation
    # contrast is fraction of source flux local pk has to be consider its own obj
    if deblend:
        segm = deblend_sources(hdu.data, 
                                       segm, npixels=5,filter_kernel=kernel, 
                                       nlevels=32,contrast=contrast)
    # need bkg subtracted to do photometry using source properties
    data_bkgsub = hdu.data - bkg.background
    cat = source_properties(data_bkgsub, segm,background=bkg.background,
                            error=None,filter_kernel=kernel)

    return cat 

if __name__=='__main__':
    hdu1 = fits.open("../test_data/sky_image_1.fits.fz")[1]
    print(test_source_detection(hdu1))