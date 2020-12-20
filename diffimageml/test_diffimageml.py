##Test File
import sys,os,traceback,pickle,unittest,warnings
import numpy as np
from astropy.table import Table

_SRCDIR_ = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),'..'))
sys.path.append(_SRCDIR_)
import diffimageml

# Hard coding the test data filenames
_DIFFIM1_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'diff_pydia_1.fits.fz'))
_FAKEDIFFIM1_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'diff_pydia_1_fakegrid.fits'))
_SEARCHIM1_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'sky_image_1.fits.fz'))
_TEMPLATEIM1_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'template_1.fits.fz'))

_DIFFIM2_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'diff_pydia_2.fits.fz'))
_FAKEDIFFIM2_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'diff_pydia_2_fakegrid.fits'))
_SEARCHIM2_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'sky_image_2.fits.fz'))
_TEMPLATEIM2_ = os.path.abspath(os.path.join(
    _SRCDIR_, 'diffimageml', 'test_data', 'template_2.fits.fz'))

_GOFAST_ = True # Use this to skip slow tests

class TestDataExistence(unittest.TestCase):
    """
        Check for existence of the pristine (level 0) test data
        located in the testdata directory
    """
    def test_pristine_difference1(self):
        self.assertTrue(os.path.isfile(_DIFFIM1_))
    def test_pristine_difference2(self):
        self.assertTrue(os.path.isfile(_DIFFIM2_))
    def test_pristine_search1(self):
        self.assertTrue(os.path.isfile(_SEARCHIM1_))
    def test_pristine_search2(self):
        self.assertTrue(os.path.isfile(_SEARCHIM2_))
    def test_pristine_template1(self):
        self.assertTrue(os.path.isfile(_TEMPLATEIM1_))
    def test_pristine_template2(self):
        self.assertTrue(os.path.isfile(_TEMPLATEIM2_))


class TestPlanter(unittest.TestCase):

    def setUp(self):
        """Create a FakePlanter object from the pristine (level 0) test data"""
        self.fakeplanterobject = diffimageml.FakePlanter(
            _DIFFIM1_, _SEARCHIM1_, _TEMPLATEIM1_)
        

    @unittest.skipIf(_GOFAST_,"Skipping slow `test_fakeplanter`")
    def test_fakeplanter(self,accuracy=0.05):
        epsf = diffimageml.util.lco_epsf(self.fakeplanterobject)
        locations = diffimageml.util.get_lattice_positions(self.fakeplanterobject)
        pixels,skycoords = locations

        # TODO: this is gonna need debugging
        pre_imdata = self.fakeplanterobject.diffim.sci.data
        post_im = self.fakeplanterobject.plant_fakes_in_diffim(epsf, pixels)
        post_imdata = post_im.data

        fitsflux = np.sum(post_imdata - pre_imdata)
        # TODO: this should have SCA to stay general if plants are scaled differently
        epsfflux = int(post_im.header['N_fake'])*float(post_im.header['F_epsf'])
        self.assertTrue(np.abs(fitsflux-epsfflux)/epsfflux < accuracy)


class TestFitsImage(unittest.TestCase):
    def setUp(self):
        self.FitsImageClassInstance = diffimageml.FitsImage(_SEARCHIM1_)
        # OK to run this even in _GOFAST_ mode b/c it will load a pre-baked cat
        self.FitsImageClassInstance.fetch_gaia_sources(save_suffix='TestGaiaCat')

    @unittest.skipIf(_GOFAST_,"Skipping slow `test_fetch_gaia_sources`")
    def test_fetch_gaia_sources(self):
        """ Check that an astroquery call to the Gaia db works"""
        self.assertEqual(type(self.FitsImageClassInstance.gaia_source_table) == Table)
        self.assertTrue(len(self.FitsImageClassInstance.gaia_source_table) > 0)

    @unittest.skipIf(_GOFAST_,"Skipping slow `test_photometry_of_stars`")
    def test_photometry_of_stars(self):
        # TODO: must also get gaia sources, but that's a separate test, should
        #  do them in series, and pass the object along?
        self.FitsImageClassInstance.do_stellar_photometry(self.FitsImageClassInstance.gaia_source_table)
        self.assertTrue(self.FitsImageClassInstance.stellar_phot_table is not None)

    @unittest.skipIf(_GOFAST_,"Skipping slow `test_build_epsf_model`")
    def test_build_epsf_model(self,verbose=True):
        """Check construction of an ePSF model from Gaia stars.
        """
        # Build the ePSF model and save to disk
        self.FitsImageClassInstance.build_epsf_model(
            verbose=verbose, save_suffix='TestEPSFModel')

        self.assertTrue(self.FitsImageClassInstance.epsf is not None)
        self.assertTrue(self.FitsImageClassInstance.epsf.data.sum()>0)

        # read in the ePSF model we just created
        self.FitsImageClassInstance.read_epsf_model(save_suffix='TestEPSFModel')
        self.assertTrue(self.FitsImageClassInstance.epsf is not None)
        self.assertTrue(self.FitsImageClassInstance.epsf.data.sum()>0)

    def test_measure_zeropoint(self):
        """Check measuring of zeropoint from known stars in the image"""
        self.FitsImageClassInstance.do_stellar_photometry(
            self.FitsImageClassInstance.gaia_source_table)
        self.FitsImageClassInstance.measure_zeropoint()
        self.assertTrue(self.FitsImageClassInstance.zeropoint is not None)
        
    def tearDown(self):
        self.FitsImageClassInstance.hdulist.close()

class TestSourceDetection(unittest.TestCase):
    def setUp(self):
        self.FitsImageClassInstance = diffimageml.FitsImage(_SEARCHIM1_)
        self.FakePlanterClassInstance = diffimageml.FakePlanter(
            _FAKEDIFFIM2_)

        if not _GOFAST_:
            self.FitsImageClassInstance.detect_sources()

    @unittest.skipIf(_GOFAST_,"Skipping slow `test_source_detection`")
    def test_source_detection(self):
        return FitsImageTest.has_detections()

    @unittest.skipIf(_GOFAST_,"Skipping slow `test_detection_efficiency`")
    def test_detection_efficiency(self):
        eff,eff_table = self.FakePlanterClassInstance.calculate_detection_efficiency(
                        source_catalog=self.FitsImageClassInstance.sourcecatalog)
        self.assertTrue(type(eff)==float)
        self.assertTrue(len(eff_table)>0)

    @unittest.skipIf(_GOFAST_,"Skipping slow `test_host_galaxy_detection`")
    def test_host_galaxy_detection(self):
        pixel_x = 2012
        pixel_y = 2056
        ra = 17.3905276
        dec = 15.0091647
        ###Tests detect host galaxies with pixel coords
        self.FitsImageClassInstance.detect_host_galaxies()
        self.assertTrue(len(self.FitsImageClassInstance.hostgalaxies) >= 1)
        
        ##Make sure that target galaxy is flagged
        target = False
        for i in self.FitsImageClassInstance.hostgalaxies:
            if np.sqrt( (i['x'].value - pixel_x) ** 2 + (i['y'].value - pixel_y) ** 2 ) < 10:
                target = True
        
        self.assertTrue(target)

    def tearDown(self):
        self.FitsImageClassInstance.hdulist.close()


def test_loader(loader):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite


if __name__ == '__main__':
    #TEST LIST
    #test_cases = 'ALL'
    test_cases = [TestDataExistence,TestFitsImage]

    if test_cases == 'ALL':
        unittest.main()
    else:
        runner = unittest.TextTestRunner()
        runner.run(test_loader(unittest.TestLoader()))
