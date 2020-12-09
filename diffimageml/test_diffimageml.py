##Test File
import sys,os,traceback
from copy import deepcopy
_SRCDIR_ = os.path.abspath(os.path.join(
	os.path.dirname(os.path.abspath(__file__)),'..'))
sys.path.append(_SRCDIR_)
import diffimageml

# Hard coding the test data filenames
_DIFFIM1_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'test_data', 'diff_pydia_1.fits.fz'))
_SEARCHIM1_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'test_data', 'sky_image_1.fits.fz'))
_TEMPLATEIM1_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'test_data', 'template_1.fits.fz'))

_DIFFIM2_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'test_data', 'diff_pydia_2.fits.fz'))
_SEARCHIM2_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'test_data', 'sky_image_2.fits.fz'))
_TEMPLATEIM2_ = os.path.abspath(os.path.join(
	_SRCDIR_, 'test_data', 'template_2.fits.fz'))



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

def test_fakeplanter_class():
	"""Create a FakePlanter object from the pristine (level 0) test data"""
	fakeplanter = diffimageml.FakePlanter(
		_DIFFIM1_, _SEARCHIM1_, _TEMPLATEIM1_)
	return 1


def test_checkepsfmodel(fakeplanterobject):
	"""Given a fakeplanterobject, check if it has an ePSF model"""
	return(fakeplanterobject.has_epsf_model)

def test_FitsImageClass():
	from astropy.io.fits import HDUList,PrimaryHDU
	FitsImageClassInstance = diffimageml.FitsImage(_SEARCHIM1_)
	assert( isinstance(FitsImageClassInstance.hdulist,HDUList))
	return FitsImageClassInstance
	
def test_source_detection(FitsImageTest):
	source_catalog = FitsImageTest.detect_sources()
	return FitsImageTest.has_detections()

def test_diffimageml():
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
		print('Testing FakePlanter instantiation...', end='')
		total += 1
		test_fakeplanter_class()
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


	print('Passed %i/%i tests.'%(total-failed,total))

	return

if __name__ == '__main__':
	test_diffimageml()
