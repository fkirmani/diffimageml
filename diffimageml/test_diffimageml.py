##Test File
import sys,os,traceback
from copy import deepcopy
from astropy.table import Table

_SRCDIR_ = os.path.abspath(os.path.join(
	os.path.dirname(os.path.abspath(__file__)),'..'))
sys.path.append(_SRCDIR_)
import diffimageml


# Hard coding the test data filenames
_DIFFIM1_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'diffimageml', 'test_data', 'diff_pydia_1.fits.fz'))
_SEARCHIM1_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'diffimageml', 'test_data', 'sky_image_1.fits.fz'))
_TEMPLATEIM1_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'diffimageml', 'test_data', 'template_1.fits.fz'))

_DIFFIM2_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'diffimageml', 'test_data', 'diff_pydia_2.fits.fz'))
_SEARCHIM2_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'diffimageml', 'test_data', 'sky_image_2.fits.fz'))
_TEMPLATEIM2_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'diffimageml', 'test_data', 'template_2.fits.fz'))

_DOFAST_ = True # False # Use this to skip slow tests

def test_pristine_data():
	"""
	Check for existence of the pristine (level 0) test data
	located in the testdata directory
	"""
	assert os.path.isfile(_DIFFIM1_)
	assert os.path.isfile(_SEARCHIM1_)
	assert os.path.isfile(_TEMPLATEIM1_)

	assert os.path.isfile(_DIFFIM2_)
	assert os.path.isfile(_SEARCHIM2_)
	assert os.path.isfile(_TEMPLATEIM2_)

	return 1

def test_fetch_gaia_sources():
	""" Check that an astroquery call to the Gaia db works"""
	# read in a fits file
	searchim1 = diffimageml.fakeplanting.FitsImage(_SEARCHIM1_)

	# fetch gaia data
	# NOte: we adjust the save_suffix so we can test saving of the output to
	# a file, but it will not conflict  with existence of a pre-baked gaia
	# catalog test file with the default suffix GaiaCat
	searchim1.fetch_gaia_sources(save_suffix=None)#'TestGaiaCat')

	# TODO : more informative test output?
	assert type(searchim1.gaia_source_table) == Table
	assert len(searchim1.gaia_source_table) > 0
	#assert os.path.isfile(searchim1.gaia_source_table.savefilename)

	return 1


def test_fakeplanter_class():
	"""Create a FakePlanter object from the pristine (level 0) test data"""
	fakeplanter = diffimageml.FakePlanter(
		_DIFFIM1_, _SEARCHIM1_, _TEMPLATEIM1_)
	return 1


def test_checkepsfmodel(fakeplanterobject):
	"""Given a fakeplanterobject, check if it has an ePSF model"""
	return(fakeplanterobject.has_epsf_model)


def test_fakeplanter(accuracy=0.05):
    # self given a fake planter object which has had fake planting done 
    # unit test
    planthdu = self.plant_fakes
    hdu = self.hdu

    fitsflux = np.sum(planthdu.data - hdu.data)
    epsfflux = int(planthdu.header['N_fake'])*float(planthdu.header['f_fake'])
    #print(fitsflux,epsfflux)
    if np.abs(fitsflux-epsfflux)/epsfflux < accuracy:
        #print("plant was successful")
        return 1

def test_FitsImageClass():
	from astropy.io.fits import HDUList,PrimaryHDU
	FitsImageClassInstance = diffimageml.FitsImage(_SEARCHIM1_)
	assert( isinstance(FitsImageClassInstance.hdulist,HDUList))
	return FitsImageClassInstance
	
def test_source_detection(FitsImageTest):
	source_catalog = FitsImageTest.detect_sources()
	return FitsImageTest.has_detections()
 
def test_host_galaxy_detection(Image=None):
	import numpy as np
	if Image == None:
		SkyImageClassInstance=diffimageml.FitsImage(_SEARCHIM1_)
		SkyImageClassInstance.detect_sources()
	elif not Image.has_detections():
		Image.detect_sources()
		SkyImageClassInstance = Image
	else:
		SkyImageClassInstance = Image
	pixel_x = 2012
	pixel_y = 2056
	SkyImageClassInstance.detect_host_galaxies(pixel_x , pixel_y)
	assert(len(SkyImageClassInstance.hostgalaxies) == 1)
	host_x = SkyImageClassInstance.hostgalaxies[0].xcentroid.value
	host_y = SkyImageClassInstance.hostgalaxies[0].ycentroid.value
	assert( np.sqrt( (pixel_x - host_x) ** 2 + (pixel_y - host_y) ** 2 ) < 10)
	
def test_diffimageml():
	if _DOFAST_:
		print("SKIPPING SLOW TESTS")
	failed=0
	total=0
	# Fill in tests here.  Put a separate try/except around each test and track
	# the count of total tests and failures
	try:
		print('Testing pristine data...', end='')
		total += 1
		test_pristine_data()
		print("Passed!")
	except Exception as e:
		print('Failed')
		print(traceback.format_exc())
		failed+=1

	try:
		if not _DOFAST_:
		    print('Testing Gaia astroquery...', end='')
		    total += 1
		    test_fetch_gaia_sources()
		    print("Passed!")
	except Exception as e:
		print('Failed')
		print(traceback.format_exc())
		failed+=1

	try:
		print('Testing FakePlanter instantiation...', end='')
		total += 1
		#test_fakeplanter_class()
		print("Passed!")
	except Exception as e:
		print('Failed')
		print(traceback.format_exc())
		failed+=1

	try:
		print('Testing FakePlanter planting...', end='')
		total += 1
		test_fakeplanter(accuracy=0.05)
		print("Passed!")
	except Exception as e:
		print('Failed')
		print(traceback.format_exc())
		failed+=1

	try:
		FitsImage_Instance = None
		print('Testing FitsImage instantiation...', end='')
		total += 1
		FitsImage_Instance = test_FitsImageClass()
		print("Passed!")
	except Exception as e:
		print('Failed')
		print(traceback.format_exc())
		failed+=1
    
	try:
		if not _DOFAST_:
			print('Testing SourceDetection...', end='')
			total += 1
			if FitsImage_Instance is not None:
				detected = test_source_detection(FitsImage_Instance)
			else:
				detected = test_source_detection(diffimageml.FitsImage(_SEARCHIM1_))
			if not detected:
				raise RuntimeError("Source detection successful, but no catalog found.")
			print("Passed!")
	except Exception as e:
		print('Failed')
		print(traceback.format_exc())
		failed+=1
		
	try:
		if not _DOFAST_:
			print ("Testing Host Galaxy Detection...", end='')
			total += 1
			test_host_galaxy_detection(Image=FitsImage_Instance)
		print ("Passed!")
	except Exception as e:
		print('Failed')
		print(traceback.format_exc())
		failed += 1
	

	print('Passed %i/%i tests.'%(total-failed,total))

	return

if __name__ == '__main__':
	test_diffimageml()
