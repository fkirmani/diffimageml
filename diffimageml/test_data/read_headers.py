from astropy.io import fits
import sys

def print_header_info(fname):
	'''
	This function takes in the name of a fits or fits.fz file.
	Will print assorted bits of information from the file header
	'''
	
	hdu = fits.open(fname)
	

	
	if "fits.fz" in fname:
		##Handles compressed images
		n = 1
		
	else:
		##Handles fits files
		n = 0
	
	
	print ("Date of Observation is {}".format(hdu[n].header['Date']))
	print ("Source Name is {}".format(hdu[n].header['Object']))
	print ("FWHM for this image is {}".format(hdu[n].header['L1FWHM']))
	print ("Exposure time is {} seconds".format(hdu[n].header['EXPTIME']))
	
	
if __name__ == "__main__":
	print_header_info(sys.argv[1])
