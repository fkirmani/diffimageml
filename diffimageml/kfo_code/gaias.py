"""
want to extract all of the gaia stars in FoV of each source image
place cutouts of each into a folder
"""
import requests

import random
import sys
import glob
import os
from optparse import OptionParser
parser = OptionParser()
(options,args)=parser.parse_args()
import shutil

import copy
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle
from matplotlib.colors import BoundaryNorm
import numpy as np
import itertools
import collections 
from scipy.optimize import curve_fit

import astropy
from astropy.io import ascii,fits
from astropy.table import vstack,hstack,Table,Column,Row,setdiff,join
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.units import Quantity
from astroquery.gaia import Gaia
from astropy.convolution import Gaussian2DKernel
from astropy.visualization import ZScaleInterval,simple_norm
zscale = ZScaleInterval()
from astropy.nddata import Cutout2D,NDData
from astropy.stats import sigma_clipped_stats,gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

import photutils
from photutils import find_peaks
from photutils.psf import extract_stars
from photutils import EPSFBuilder
from photutils import detect_threshold
from photutils import detect_sources
from photutils import deblend_sources
from photutils import source_properties, EllipticalAperture
from photutils import BoundingBox
from photutils import Background2D, MedianBackground

import pandas as pd
from photutils.psf import IntegratedGaussianPRF, DAOGroup, BasicPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter

def LCO_PSF_PHOT(hdu,init_guesses):
	# im ~ np array dat, pixel [x0,y0] ~ float pixel position, sigma_psf ~ LCO-PSF sigma 
	x0,y0=init_guesses
	im = hdu.data
	hdr = hdu.header
	
	fwhm = hdr['L1FWHM']/hdr['PIXSCALE'] # PSF FWHM in pixels, roughly ~ 5 pixels, ~ 2 arcsec 
	sigma_psf = fwhm*gaussian_fwhm_to_sigma # PSF sigma in pixels
	
	psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
	daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
	mmm_bkg = MMMBackground()
	fitter = LevMarLSQFitter()

	psf_model.x_0.fixed = True
	psf_model.y_0.fixed = True
	pos = Table(names=['x_0', 'y_0'], data=[[x0],[y0]]) # optionally give flux_0 has good aperture method for guessing though

	photometry = BasicPSFPhotometry(group_maker=daogroup,
									 bkg_estimator=mmm_bkg,
									 psf_model=psf_model,
									 fitter=LevMarLSQFitter(),
									 fitshape=(11,11))
	result_tab = photometry(image=im, init_guesses=pos)
	residual_image = photometry.get_residual_image()
	
	return result_tab


def gaia_results(image,saveas=None):
	# image wcs and frame, for conversions pixels/skycoord
	wcs,frame=WCS(image.header),image.header['RADESYS'].lower()
	# coord of strong lensing galaxy
	ra=image.header['CAT-RA']
	dec=image.header['CAT-DEC']
	coord = SkyCoord(ra,dec,unit=(u.hourangle,u.deg))

	# the pixscale is same along x&y and rotated to default (N up, E left) cdi_j ~ delta_ij
	cdi_i = image.header['CD1_1'] # deg/pixel
	naxis = image.header['NAXIS1'] # naxis1=naxis2
	radius = 3600*cdi_i*naxis/2 # approx 800 arcsec entire image
	radius*=.75 # do 3/4 of that
	# do the search
	r = Gaia.query_object_async(coordinate=coord, radius=radius*u.arcsec)

	if saveas:
		if not os.path.exists(saveas): 
			os.makedirs(saveas)
		pickle.dump(r,open(saveas+"/gaia_results.pkl","wb"))

	return r,image

def cut_gaias(file,results,size=None,saveas=None):
	"""
	file should be fits.open("filename.fits")
	results should be the table from gaia request
	use radec from gaia results to cut image on each star from file 
	"""

	if not os.path.exists(saveas): 
		os.makedirs(saveas)
		
	hdu = file # where I'm going to overwrite the cutout data/hdr onto so it has same class for fits writeto
	dat = file.data
	hdr = file.header
	wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
	
	ra,dec=results['ra'],results['dec']
	# a ~ name ~ bGaiaDR2idx
	a = results['designation']
	
	gaia_stars = SkyCoord(ra,dec,unit=(u.hourangle,u.deg))
	if size == None:
		size = 50 # pixels, ~ 20 arcsec, lxw box for cutout on star 
	
	d = {}
	for i in range(len(gaia_stars)):
		# fix name designation
		ai = str(a[i])
		ai= ai.replace(" ","_")
		ai = ai.replace("'","")
		
		cut = Cutout2D(dat,gaia_stars[i],size,wcs=wcs)
		cutwcs = cut.wcs
		if saveas:
			#pickle.dump(cut,open(saveas,"wb"))
			hdu.data = cut.data
			hdu.header.update(cut.wcs.to_header())
			hdu.writeto(saveas+'/'+ai+'.fits',overwrite=True)

		# this dictionary ties together the designation key to the results for that star and the cutout image file
		d[ai] = [results[i],ai+'.fits']

	#pickle.dump(d,open(saveas+'/'+'gaia_dict.pkl'))

	return 

def weighted_average(weights,values,plot=True,saveas=None):
	# zp ~ need to filter out nan/masked vals
	indices = ~np.isnan(values)
	true_count = np.sum(indices)
	avg = np.average(values[indices], weights=weights[indices])
	if plot:
		matplotlib.rcParams.update({'font.size': 20,'xtick.labelsize':15,'ytick.labelsize':15})
		fig,ax = plt.subplots(figsize=(16,8))
		spacing = np.arange(0,true_count,1)
		ax.errorbar(spacing,values[indices],yerr=weights[indices],marker='x',ls='',color='red',label='zp (gaia mag, lco flux)')
		ax.hlines(avg,0,true_count,linestyle='--',label='avg~{:.1f}'.format(avg),color='red')
		plt.legend()
		plt.savefig(saveas,bbox_inches='tight')

	return avg

def data_cut_gaias(file,result=None,saveas=None):
	"""
	file ~ cutout fits of star  
	result ~ DR2 catalog values for star

	Gaia stars are being used to build an ePSF and/or determine ZP from DR2 Grp mag
	Need to make some cuts to the stars which were in the FoV:
	Oversaturated or non-linear
	(Perhaps overlapping using bbox?)
	"""

	if not os.path.exists(saveas): 
		os.makedirs(saveas)
	
	hdu = fits.open(file)[0]
	dat = hdu.data
	hdr = hdu.header
	wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
	saturate,maxlin = hdr['SATURATE'],hdr['MAXLIN'] # counts for saturation and non-linearity levels
	
	if np.max(dat) < maxlin:
		shutil.copy(file,saveas)

	return

if __name__ == "__main__":
	sources = glob.glob("peter_diffs/differences/*/*/source/*fz") # ~300 image sets expsoure and diff 
	# exp not trimmed ~ 800 arcsec lxw FoV, diff trimmed ~ 400 arcsec lxw FoV
	source_data = pd.read_csv('source_data.txt',delimiter='|',skiprows=1) # 98 targets Shu properties
	# Gaia Query onto 98 targets (in order of source_data) ~ 50 stars from each, target names as keys
	resultsdict = pickle.load(open("query_resultsdict.pkl","rb")) 

	print("{} sources".format(len(sources)))
	diffs,exps = [],[]
	for i in sources:
		root = os.path.basename(i)
		if root[0] == 'd':
			diffs.append(i)
		else:
			exps.append(i)
	print("{} diffs, {} exps".format(len(diffs),len(exps)))

	dophot = True
	if dophot:
	
		# partitioning the exps into batches to run in parallel
		# 3 exposures /chunk that will have phot done on it with sbatch int arg
		chunks = [exps[x:x+3] for x in range(0, len(exps), 3)]
		print("{} chunks, {} exps/chunk".format(len(chunks),len(chunks[0])))
		chunk_idx = int(sys.argv[1])
		print("{} chunk_idx".format(chunk_idx))
		expschunk = chunks[chunk_idx]

		# open the exposures 
		# cutouts on all stars in each exposure, using gaia coord as ctr
		expims = []
		corrupts = []
		neededtargets = []
		for exp in expschunk:
			# opening exposures
			try:
				hdu = fits.open(exp)[1]
				expims.append(hdu)
			except:
				corrupts.append(exp)
				print("exposure {} is corrupt continuing".format(exp))
				continue

			dat = hdu.data
			hdr = hdu.header
			wcs,frame=WCS(hdr),hdr['RADESYS'].lower()
			target = hdr['OBJECT']
			try:
				r = resultsdict[target] # Gaia results
			except:
				print("{} isnt available in resultsdict".format(target))
				neededtargets.append(target)
				continue

			saveas = exp.split("source")[0] + "source/gaias"
			# cutting out gaia stars in exposures, gaia coord is ctr, 50 pixels ~ 20 arcsec box lxw
			#cut_gaias(hdu,r,size=50,saveas=saveas)

			lco_phot_tab = []
			for i in r:
				position=SkyCoord(ra=i['ra'],dec=i['dec'],unit=u.deg,frame=frame)
				pixel=skycoord_to_pixel(position,wcs)
				try:
					phot_tab = LCO_PSF_PHOT(hdu,pixel)
				except:
					# assuming it doesn't fail into exception on first result
					phot_tab = [None for i in range(len(phot_tab))]
					phot_tab['x_0'] = pixel[0]
					phot_tab['y_0'] = pixel[1]

				lco_phot_tab.append(phot_tab)
			lco_phot_tab = vstack(lco_phot_tab)
			pickle.dump(lco_phot_tab,open(saveas+"/lco_phot_tab.pkl","wb"))

		print('--------------------------------------------------------------')

		final_new_menu = list(dict.fromkeys(neededtargets))
		print("{} needed targets for query".format(final_new_menu))
		print("{} exposure images, {} corrupts".format(len(expims),len(corrupts)))
	
	hstacked = False
	if hstacked:
		gaias = glob.glob("peter_diffs/differences/*/*/source/gaias")
		print("{} gaias".format(len(gaias)))
		for g in gaias:
			try:
				lco_phot_file = glob.glob(g+"/lco_phot_tab*")
				lco_phot_tab = pickle.load(open(lco_phot_file[0],"rb"))
				gcut_files = glob.glob(g+"/*fits")
				hdu = fits.open(gcut_files[0])[1]
				hdr = hdu.header
				target = hdr['OBJECT']
				r = resultsdict[target] 
				gtab = hstack([r,lco_phot_tab])
				pickle.dump(gtab,open(g+"/gtab.pkl","wb"))
			except:
				print(g)
				continue

	if chunk_idx == 1:
		# don't need all ~ 100 chunks running this piece to get zp in parrallel
		# all the tables from photometry are loaded in glob here 
		gtabs = glob.glob("peter_diffs/differences/*/*/source/gaias/gtab*")
		print("{} gtabs".format(len(gtabs)))
		for g in gtabs:

			gtab = pickle.load(open(g,"rb"))

			flux_fit = gtab['flux_fit']
			flux_uncertainty = gtab['flux_unc']
			flux_accuracy = flux_fit/flux_uncertainty

			grp_mag = gtab['phot_rp_mean_mag']
			grp_flux = gtab['phot_rp_mean_flux'] 
			grp_flux_accuracy = gtab['phot_rp_mean_flux_over_error']

			# grp_mag = -2.5 log10( flux_fit ) + zp
			zps = grp_mag + 2.5*np.log10(flux_fit)
			zp_accuracy = np.sqrt(flux_accuracy**2 + grp_flux_accuracy**2)
			gtab.add_columns([zps,zp_accuracy],names=['zp','zp_accuracy'])

			zptab = os.path.join(os.path.dirname(g),"zptab.pkl")
			pickle.dump(gtab,open(zptab,"wb"))
			
			"""
			ETC warns of saturation at everything brighter than mag = 15.2 (and 16) for 300s (and 600s) exposure
			The Query gives 50 stars around every target... I should tune it to get useful gaia-rp mag objects
			To return an unlimited number of rows set Gaia.ROW_LIMIT to -1.
			Gaia.ROW_LIMIT = -1
			"""
			tmp = gtab[gtab['phot_rp_mean_mag'] >= 16]
			zpval = os.path.join(os.path.dirname(g),"ZP.pkl")
			zpplot = os.path.join(os.path.dirname(g),"ZP.pdf")
			ZP = weighted_average(tmp['zp_accuracy'],tmp['zp'],plot=True,saveas=zpplot)
			print(ZP)
			pickle.dump(ZP,open(zpval,"wb"))
		
		
		

	"""
	# this block would do the request on the images (i.e. using target name/coord in header)
	# better of just using results from available gaia request, previously run for all the targets 
	corrupts = [] # couldn't open the exposure source image
	requestError500 = [] # internal server error on gaia request, requests.exceptions.HTTPError: Error 500:
	noGaias = [] # couldn't cut any gaia stars out of the exposure
	for exp in exps:
		print("---------------------------------------------------")
		try:
			img = fits.open(exp)[1]
		except:
			corrupts.append(exp)
			print("exposure {} is corrupt continuing".format(exp))
			continue
		try:
			r,img = gaia_results(img)
			print("exposure {} has {} gaia stars in FoV".format(exp,len(r)))
		except:
			requestError500.append(exp)
			print("exposure {} had a gaia request error".format(exp))
			continue
		saveas = exp.split("source")[0] + "source/gaias"
		try:
			cut_gaias(img,r,saveas=saveas)
		except:
			noGaias.append(exp)
			print("exposure {} has no stars in FoV continuing".format(exp))
			continue

	print("\n")
	print("{} corrupt exposure source files:".format(len(corrupts)))
	print(corrupts)

	print("{} exposure source files with no gaia stars in FoV:".format(len(noGaias)))
	print(noGaias)

	print("{} exposure source files that hit a gaia request error".format(len(requestError500)))
	print(requestError500)
	print("\n")
	"""


