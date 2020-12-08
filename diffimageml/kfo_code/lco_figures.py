import glob
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
from matplotlib.patches import Circle
from matplotlib.colors import BoundaryNorm
import numpy as np
from scipy.optimize import curve_fit

import astropy
from astropy.io import ascii,fits
from astropy.table import vstack,Table,Column,Row,setdiff,join
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
from photutils import Background2D, MedianBackground
bkg_estimator = MedianBackground()

# Suppress warnings. Relevant for astroquery. Comment this out if you wish to see the warning messages
import warnings
warnings.filterwarnings('ignore')

import lco_fakeSNpipeline	


def cutout_figure(file,saveas='.pdf'):
	matplotlib.rcParams.update({'font.size': 10})

	#print('inside cutout_figure')
	#print('file',file)
	name = file.split('.')[0]
	#print('name',name)
	saveas = name + saveas
	#print('saveas',saveas)

	hdu = fits.open(file)
	#print('hdu',hdu)
	try:
		hdu = hdu[1]
	except:
		hdu = hdu[0]

	#print('data',hdu.data)
	wcs = WCS(hdu.header)
	#print('wcs',wcs)

	plt.subplot(projection=wcs)
	plt.imshow(hdu.data, origin='lower')
	#plt.grid(color='white', ls='solid')
	plt.xlabel('Right Ascension')
	plt.ylabel('Declination')
	plt.savefig(saveas,bbox_inches='tight')


def psf_and_gauss(epsf,epsf_gaussian,saveas='lco_psf.pdf'):
	# take a look at the ePSF image built from stack and a fitted gaussian 
	fit_gaussian,levels,xctr_vals,yctr_vals,image1,img_epsf,resid = epsf_gaussian # unpacked... levels list amplitude - sigma, ctr vals are gauss model sliced, image1 is array of values from gaussian fit in shape of epsf, img_epsf is epsf instance of it, resid is gauss - epsf 
	constant,amplitude,x_mean,y_mean,x_stddev,y_stddev,theta=fit_gaussian.parameters
	matplotlib.rcParams.update({'font.size': 30})

	fig, ax = plt.subplots(2,2,figsize=(7.5, 7.5),gridspec_kw={'width_ratios': [3, 1],'height_ratios':[3,1]})
	#fig.add_subplot()
	#im1 = ax[0][0].imshow(zscale(epsf.data),cmap='gray')
	# works better with a lognormalization stretch to data 
	norm = simple_norm(epsf.data, 'log')
	im1 = ax[0][0].imshow(epsf.data,norm=norm,vmin=np.min(epsf.data),cmap='viridis')
	# Adding the colorbar
	cbaxes = fig.add_axes([-0.3, 0.1, 0.03, 0.8])  # This is the position for the colorbar
	ticks = [norm.vmax,norm.vmax/10,norm.vmax/100]
	ticks.append(norm.vmin)
	#print(ticks)
	cb = plt.colorbar(im1, cax = cbaxes,ticks=ticks,format='%.1e')
	#plt.colorbar(im1,ax=ax[0][0])

	tmp=np.arange(0,epsf.shape[0])
	# vertical slices along ctr i.e. x = 0... 
	# gaussian fit values of epsf
	ax[1][0].plot(tmp,yctr_vals)
	# epsf
	epsf.shape
	row_idx = np.array([i for i in range(epsf.shape[0])])
	col_idx = np.array([int(epsf.shape[0]/2)])
	epsf_ctrx = epsf.data[row_idx[:, None], col_idx]
	ax[1][0].scatter(row_idx,epsf_ctrx)
	ax[1][0].text(0.01, .01, r'$\sigma_x\sim{:.1f}$'.format(abs(x_stddev)), fontsize=25,rotation=0)
	# horizontal slice ... y=0 gaussian fit vals
	ax[0][1].plot(xctr_vals,tmp)
	row_idx = np.array([int(epsf.shape[0]/2)])
	col_idx = np.array([i for i in range(epsf.shape[0])])
	epsf_ctry = epsf.data[row_idx[:, None], col_idx]
	ax[0][1].scatter(epsf_ctry,col_idx)
	ax[0][1].text(0.01, 45, r'$\sigma_y\sim{:.1f}$'.format(abs(y_stddev)), fontsize=25,rotation=-90)

	#ax.set_xlabel('',fontsize=45)
	#ax[0][0].set_xticks([])
	#ax[0][0].set_yticks([])
	ax[1][0].set_xticks([])
	ax[1][0].set_yticks([])
	ax[0][1].set_xticks([])
	ax[0][1].set_yticks([])
	#ax2 = ax[0].twinx()
	"""
	# contours of the epsf for levels 1,2,and3 sigma (from np.std(data)) below gaussian fit amplitude
	ax[1][1].contour(epsf.data,levels=levels)
	ax[1][1].set_xlim(23,27)
	ax[1][1].set_ylim(23,27)
	#ax[1][1].set_xticks([15,35])
	#ax[1][1].set_yticks([15,35])
	ax[1][1].yaxis.tick_right()
	"""
	med=np.median(resid)
	std=np.std(resid)
	# define the colormap
	cmap = plt.get_cmap('YlGnBu') # YlGnBu
	# extract all colors from the map
	cmaplist = [cmap(i) for i in range(cmap.N)]
	# create the new map
	cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

	# define the bins and normalize and forcing 0 to be part of the colorbar!
	maximnorm = norm.vmax # the amplitude of the psf from normalization above, want to see residual relative to this value
	bounds = np.arange(np.min(resid),np.max(resid),std/10)
	idx=np.searchsorted(bounds,0)
	bounds=np.insert(bounds,idx,0)
	norm = BoundaryNorm(bounds, cmap.N,'log') 
	im2 = ax[1][1].imshow(resid,origin='lower',cmap=cmap,norm=norm) # ,vmin=vmin,vmax=vmax
	# zoom in to region 3sigma around ctr
	ax[1][1].set_xlim(x_mean-3*x_stddev,x_mean+3*x_stddev)
	ax[1][1].set_ylim(y_mean-3*y_stddev,y_mean+3*y_stddev)
	ticks = [norm.vmin,norm.vmax,0]

	cbaxes = fig.add_axes([1, 0.1, 0.03, 0.2])  # This is the position for the colorbar
	cb = plt.colorbar(im2,cax=cbaxes,ticks=ticks,format='%.1e') 

	plt.savefig(saveas,bbox_inches='tight')

def used_stars(fitted_stars,saveas='lco_stars.pdf'):
	# can either show the 'good' stars ie those used to build the epsf, or using i.compute_residual_image(epsf) to show how well the epsf fit each
	matplotlib.rcParams.update({'font.size': 15})
	tmp = fitted_stars.all_good_stars
	print(len(tmp))
	nrows,ncols=int(np.sqrt(len(tmp)))+1,int(np.sqrt(len(tmp)))+1
	fig, ax = plt.subplots(nrows,ncols,figsize=(7.5, 7.5))
	ax=ax.ravel()
	for i in range(len(tmp)):
		ax[i].imshow(zscale(tmp[i]))
	plt.savefig(saveas,bbox_inches='tight')


def efficiency(m,m50,alpha):
	#https://arxiv.org/pdf/1509.06574.pdf, strolger 
	return (1+np.exp(alpha*(m-m50)))**-1

def detection_efficiency(mags,efficiencies,m50,alpha,targ=None,skybr=None,zp=None,glsn=None,saveas='lco_detection_efficiency.pdf'):
	matplotlib.rcParams.update({'font.size': 30})
	fig, ax = plt.subplots(1,1,figsize=(5, 5))
	#fig.add_subplot()
	ax.scatter(mags,efficiencies,marker='o')
	ax.title.set_text('Detection Efficiency')
	try:
		ax.text(max(mags)+1,0.8, r'$skybr \sim {:.2f} mag/arcsec^2 $'.format(skybr), fontsize=25,rotation=0)
		ax.text(max(mags)+1,0.6, r'$maxtarget \sim {:.2f} mag$'.format(-2.5*np.log10(targ[0]['max_value'][0])+zp), fontsize=25,rotation=0)
		ax.text(max(mags)+1,0.4, r'$mintarget \sim {:.2f} mag$'.format(-2.5*np.log10(targ[0]['min_value'][0])+zp), fontsize=25,rotation=0)
		ax.text(max(mags)+1,0.2, r'$target \sim {:.2f} mag$'.format(-2.5*np.log10(targ[0]['source_sum'][0]/targ[0]['area'][0].value)+zp), fontsize=25,rotation=0)
	except:
		# targ,skybr,zp issue
		pass
	ax.set_xlabel('mag',fontsize=45)
	tmp = [m50-0.2,m50,m50+0.2]
	xticks = [float("{:.1f}".format(i)) for i in tmp]
	ax.set_xticks(xticks)
	ax.set_yticks([0,0.25,0.5,0.75,1])
	# lets fit smooth exp model to the efficiency data
	x = mags
	y = efficiencies
	# init guesses m50 from interp and alpha ~ 5 pretty steep drop
	#alpha = 5
	#init_vals = [m50,alpha]  # for [m50,alpha]
	#best_vals, covar = curve_fit(efficiency, x, y, p0=init_vals, maxfev=5000)
	#print('best_vals: {}'.format(best_vals))
	# should be feeding in the m50,alpha from fit in pipeline! not fitting again inside this figure
	mags = np.linspace(min(mags),max(mags),1000) # use many to get smooth model
	e = efficiency(mags,m50,alpha)
	plt.plot(mags,e)
	plt.vlines(m50,0,1,linestyle='--',color='black',label=r'm50 ~ {:.2f}, $\alpha$ ~ {:.1f}'.format(m50,alpha))
	if glsn:
		# predicted lensing mag applied to peak Ia for the strong lens
		pkIa = glsn['Peak Apparent Magnitude'][0]        
		plt.vlines(pkIa,0,1,linestyle='-',color='red',label=r'pkIa ~ {:.2f}'.format(pkIa))
	plt.legend(bbox_to_anchor=(1.5,-0.25))
	plt.savefig(saveas,bbox_inches='tight')

def lattice_planted(mags,m50,pickle_to,saveas='lco_plants.pdf'):
	# get a look at grid of SNe from clearly visible high detection rate to un-detected
	position,size=(1200,1200),450 # need to zoom in on the figures to get clean look at the planted SNe
	#Cutout2D(data,position,size)
	print(mags,m50)
	# get idx of planted mags that is nearest to the m50
	idx = min(range(len(mags)), key=lambda i: abs(mags[i]-m50))
	cutmag1=Cutout2D(fits.open(pickle_to+'_lattice_plant_mag{:.2f}.fits'.format(float(mags[idx+1])))[0].data,position,size)
	cutmag2=Cutout2D(fits.open(pickle_to+'_lattice_plant_mag{:.2f}.fits'.format(float(mags[idx])))[0].data,position,size)
	cutmag3=Cutout2D(fits.open(pickle_to+'_lattice_plant_mag{:.2f}.fits'.format(float(mags[idx-1])))[0].data,position,size)
	cutmag4=Cutout2D(fits.open(pickle_to+'_lattice_plant_mag{:.2f}.fits'.format(float(mags[idx-2])))[0].data,position,size)

	matplotlib.rcParams.update({'font.size': 30})
	fig, ax = plt.subplots(2,2,figsize=(10, 10))
	ax[0][0].imshow(zscale(cutmag1.data),cmap='gray')
	ax[0][1].imshow(zscale(cutmag2.data),cmap='gray')
	ax[1][0].imshow(zscale(cutmag3.data),cmap='gray')
	ax[1][1].imshow(zscale(cutmag4.data),cmap='gray')
	ax[0][0].title.set_text("AB mag {:.2f}".format(float(mags[idx+1])))
	ax[0][1].title.set_text("{:.2f}".format(float(mags[idx])))
	ax[1][0].title.set_text("{:.2f}".format(float(mags[idx-1])))
	ax[1][1].title.set_text("{:.2f}".format(float(mags[idx-2])))
	ax[0][0].set_xticks([])
	ax[0][0].set_yticks([])
	ax[1][0].set_xticks([])
	ax[1][0].set_yticks([])
	ax[0][1].set_xticks([])
	ax[0][1].set_yticks([])
	ax[1][1].set_xticks([])
	ax[1][1].set_yticks([])

	plt.savefig(saveas,bbox_inches='tight')

def view_targetplant(image,ref,diff,diffplant,target,zp,saveas='targetplant.pdf'):
	# take useful targ_obj values; comes from source_cat, is the photutils for the galaxy object
	# pixels and deg, sums ~ brightness in adu ~ for lco is straight counts (ie not yet rate isn't /exptime)
	targ_obj,cuts,bkg_core,bkg_1,bkg_2 = target # unpack
	cut_targ,cut_diff,cut_ref = cuts # assume target was provided diff and ref 
	
	(cut_core,box_core),(cut_1,box_1),(cut_2,box_2)=bkg_core,bkg_1,bkg_2 # unpack again

	# grab target parameters
	area = targ_obj['area'].value[0] # pix^2 isophotal area, the segmentation obj above threshold
	pixscale = image.header['pixscale'] # arcsec/pixel
	equivalent_radius = targ_obj['equivalent_radius'][0].value # radius of circular obj with area 
	xy = (targ_obj['xcentroid'][0].value,targ_obj['ycentroid'][0].value) 
	semimajor_axis, semiminor_axis = targ_obj['semimajor_axis_sigma'][0].value,targ_obj['semiminor_axis_sigma'][0].value #xtheta ytheta defined analytically for segm image using variance then partial theta ~ 0 gives an ellipse  
	orientation = targ_obj['orientation'][0].value # deg a-axis with respect to first image axis
	
	# cut around the image on target (already available/should be same as the cuts provided but doing using image provided so easy to understand in script)
	cut_im = Cutout2D(image.data,xy,equivalent_radius*5) 
	cut_ref = Cutout2D(ref.data,xy,equivalent_radius*5)
	cut_diff = Cutout2D(diff.data,xy,equivalent_radius*5)
	cut_diffplant = Cutout2D(diffplant.data,xy,equivalent_radius*5)
	
	cut_xy = cut_im.center_cutout
	
	# usually isophotal limit well represented by R ~ 3
	R = 3
	# mpl ellipse patch wants ctr in pixels, width and height as full lengths along image x,y
	# angle is rotation in deg ccw 
	# I'm assuming that angle means it wants the dir of semimajor ~ a with respect to x (no image shown descr vague)
	# this is a little questionable at the moment
	width = R*2*semimajor_axis*np.cos(orientation*np.pi/180)
	height = R*2*semimajor_axis*np.sin(orientation*np.pi/180)
	ellipse = matplotlib.patches.Ellipse(cut_xy,width,height,angle=orientation,fill=None)

	circle = matplotlib.patches.Circle(cut_xy,radius=equivalent_radius,fill=None)
	
	# bkg_i need to be re-calculated there is an error otherwise (I think due to passing mpl patch as kwarg)

	shift_x = equivalent_radius*np.cos(orientation*np.pi/180)
	shift_y = equivalent_radius*np.sin(orientation*np.pi/180)
	
	# lets do a box on the ctr with length=width=radius 
	# the patch anchors on sw so shift the cut_xy 
	anchor_core = (cut_xy[0] - equivalent_radius/2, cut_xy[1] - equivalent_radius/2)
	# the patch (show in figures)
	box_core = matplotlib.patches.Rectangle(anchor_core,equivalent_radius,equivalent_radius,fill=None)
	# the cut (does sum for bkg)
	xy_core = xy # the center of galaxy in image
	cut_core = Cutout2D(image.data,xy_core,equivalent_radius)

	# shift box an equivalent radius along orientation from photutils creating next box 
	# assuming orientation ccw from x (east)
	# yes the boxes will overlap slightly unless orientation fully along x or y
	anchor_1 = (anchor_core[0]+shift_x,anchor_core[1]+shift_y)
	box_1 = matplotlib.patches.Rectangle(anchor_1,equivalent_radius,equivalent_radius,fill=None)
	# the cut (does sum for bkg)
	xy_1 = (xy[0]+shift_y,xy[1]+shift_y) 
	cut_1 = Cutout2D(image.data,xy_1,equivalent_radius)
	
	# similar shift one more time 
	anchor_2 = (anchor_core[0]+2*shift_x,anchor_core[1]+2*shift_y)
	box_2 = matplotlib.patches.Rectangle(anchor_2,equivalent_radius,equivalent_radius,fill=None)
	# the cut (does sum for bkg)
	xy_2 = (xy[0]+2*shift_y,xy[1]+2*shift_y) 
	cut_2 = Cutout2D(image.data,xy_2,equivalent_radius)
	
	bkg_core,bkg_1,bkg_2 = (cut_core,box_core),(cut_1,box_1),(cut_2,box_2)
	
	# get to plotting
	fig,ax=plt.subplots(2,2,figsize=(10,10))
	ax[0][0].imshow(zscale(cut_im.data))
	#ellipse = matplotlib.patches.Ellipse(cut_xy,semimajor_axis,semiminor_axis,angle=orientation,fill=None)
	ax[0][0].add_patch(circle) # ellipse a little questionable at the moment, mpl patch is vague on orientation 
	ax[0][0].title.set_text('image')
	ax[0][1].imshow(zscale(cut_ref.data))
	ax[0][1].title.set_text('ref')
	ax[1][0].imshow(zscale(cut_diff.data))
	ax[1][0].title.set_text('diff')
	ax[1][1].imshow(zscale(cut_diffplant.data))
	ax[1][1].title.set_text('plantdiff')
	
	# do some markers showing centroid and req along orientation 
	xrange,yrange=np.linspace(cut_xy[0],cut_xy[0]+shift_x),np.linspace(cut_xy[1],cut_xy[1]+shift_y)
	#shift0 = ax[0][0].scatter(xrange,yrange,s=equivalent_radius**2,marker='.',color='black')
	core0 = ax[0][0].scatter(cut_xy[0], cut_xy[1],s=2*equivalent_radius**2, marker="*",color='white')
	#ax[0][0].legend([shift0,core0],['$r_{eq}(\Theta$)','core'])
	ax[0][0].legend([core0],['centroid'])
	# text that shows the boxes pixel sum, area, and flux ~ sum/area/exptime ... todo eventually get to mag/arcsec^2 and/or nsigma above sky bkg
	area = np.pi*equivalent_radius**2 
	exptime = image.header['exptime']

	ax[0][1].text(1.1,1.01,'Target: Area ~ {:.1f} $pix^2$ ~ {:.1f} $arcsec^2$ \n [a,b], theta ~ [{:.1f} {:.1f}] pix, {:.1f} deg (a ccw +x) \n max, min, avg ~ {:.2f} {:.2f} {:.2f} mag'.format(area,area*pixscale**2,semimajor_axis,semiminor_axis,orientation,-2.5*np.log10(targ_obj['max_value'][0])+zp,-2.5*np.log10(targ_obj['min_value'][0])+zp,-2.5*np.log10(targ_obj['source_sum'][0]/targ_obj['area'][0].value)+zp),
				 bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},transform=ax[1][1].transAxes,
				 verticalalignment='top', horizontalalignment='left',
		color='black', fontsize=15)

	ax[0][1].text(1.1,0.5,'SN: m ~ {:.2f}'.format(float(diffplant.header['fakeSNmag'])),
				 bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},transform=ax[1][1].transAxes,
				 verticalalignment='top', horizontalalignment='left',
		color='black', fontsize=15)
	plt.show()
	plt.savefig(saveas,bbox_inches='tight')
	plt.close('all')

def target_image(image,target,saveas='target_image.pdf'):
	# take useful targ_obj values; comes from source_cat, is the photutils for the galaxy object
	# pixels and deg, sums ~ brightness in adu ~ for lco is straight counts (ie not yet rate isn't /exptime)
	targ_obj,cuts,bkg_core,bkg_1,bkg_2 = target # unpack
	cut_targ,cut_diff,cut_ref = cuts # assume target was provided diff and ref 
	
	(cut_core,box_core),(cut_1,box_1),(cut_2,box_2)=bkg_core,bkg_1,bkg_2 # unpack again

	# grab target parameters
	equivalent_radius = targ_obj['equivalent_radius'][0].value
	xy = (targ_obj['xcentroid'][0].value,targ_obj['ycentroid'][0].value) 
	semimajor_axis, semiminor_axis = targ_obj['semimajor_axis_sigma'][0].value,targ_obj['semiminor_axis_sigma'][0].value
	orientation = targ_obj['orientation'][0].value 
	
	# cut around the image on target (already available/should be same as the cuts provided but doing using image provided so easy to understand in script)
	cut_im = Cutout2D(image.data,xy,equivalent_radius*5) 
	cut_xy = cut_im.center_cutout

	ellipse = matplotlib.patches.Ellipse(cut_xy,semimajor_axis,semiminor_axis,angle=orientation,fill=None)

	# bkg_i need to be re-calculated there is an error otherwise (I think due to passing mpl patch as kwarg)

	shift_x = equivalent_radius*np.cos(orientation*np.pi/180)
	shift_y = equivalent_radius*np.sin(orientation*np.pi/180)

	# lets do a box on the ctr with length=width=radius 
	# the patch anchors on sw so shift the cut_xy 
	anchor_core = (cut_xy[0] - equivalent_radius/2, cut_xy[1] - equivalent_radius/2)
	# the patch (show in figures)
	box_core = matplotlib.patches.Rectangle(anchor_core,equivalent_radius,equivalent_radius,fill=None)
	# the cut (does sum for bkg)
	xy_core = xy # the center of galaxy in image
	cut_core = Cutout2D(image.data,xy_core,equivalent_radius)
	
	# shift box an equivalent radius along orientation from photutils creating next box 
	# assuming orientation ccw from x (east)
	# yes the boxes will overlap slightly unless orientation fully along x or y
	shift_x = equivalent_radius*np.cos(orientation*np.pi/180)
	shift_y = equivalent_radius*np.sin(orientation*np.pi/180)
	anchor_1 = (anchor_core[0]+shift_x,anchor_core[1]+shift_y)
	box_1 = matplotlib.patches.Rectangle(anchor_1,equivalent_radius,equivalent_radius,fill=None)
	# the cut (does sum for bkg)
	xy_1 = (xy[0]+shift_y,xy[1]+shift_y) 
	cut_1 = Cutout2D(image.data,xy_1,equivalent_radius)
	
	# similar shift one more time 
	anchor_2 = (anchor_core[0]+2*shift_x,anchor_core[1]+2*shift_y)
	box_2 = matplotlib.patches.Rectangle(anchor_2,equivalent_radius,equivalent_radius,fill=None)
	# the cut (does sum for bkg)
	xy_2 = (xy[0]+2*shift_y,xy[1]+2*shift_y) 
	cut_2 = Cutout2D(image.data,xy_2,equivalent_radius)
	
	bkg_core,bkg_1,bkg_2 = (cut_core,box_core),(cut_1,box_1),(cut_2,box_2)
	
	"""
	# to take a look at the ellipse that photutils found
	fig,ax=plt.subplots()
	ellipse = matplotlib.patches.Ellipse(cut_xy,semimajor_axis,semiminor_axis,angle=orientation,fill=None)
	ax.imshow(zscale(cut_targ.data))
	ax.add_patch(patch)
	"""
	fig,ax=plt.subplots(2,2,figsize=(10,10))
	ax[0][0].imshow(zscale(cut_im.data))
	ax[0][0].add_patch(box_core)
	ax[0][0].add_patch(box_1)
	ax[0][0].add_patch(box_2)
	# do some markers showing core ctr and shift along orientation 
	xrange,yrange=np.linspace(cut_xy[0],cut_xy[0]+shift_x),np.linspace(cut_xy[1],cut_xy[1]+shift_y)
	shift0 = ax[0][0].scatter(xrange,yrange,s=equivalent_radius**2,marker='.',color='black')
	core0 = ax[0][0].scatter(cut_xy[0], cut_xy[1],s=2*equivalent_radius**2, marker="*",color='white')
	# the zoom in on on the cuts
	fcore = ax[0][1].imshow(zscale(cut_core.data),label='fcore')
	f1 = ax[1][0].imshow(zscale(cut_1.data),label='f1')
	f2 = ax[1][1].imshow(zscale(cut_2.data),label='f2')
	# legend detailing that the boxes are rectangular lxw = req and have three which start at ctr and shift req along theta 
	ax[0][0].legend([shift0,core0],['$r_{eq}(\Theta$)','core'])
	# text that shows the boxes pixel sum, area, and flux ~ sum/area/exptime ... todo eventually get to mag/arcsec^2 and/or nsigma above sky bkg
	area = equivalent_radius**2 
	try:
		exptime = image.header['exptime']
	except:
		# ref doesnt have the good stuff in hdr
		exptime = 300
	ax[0][1].text(0.1, 0.8, 'fcore ~ {:.1e} adu/s/pix^2'.format(np.sum(cut_core.data)/exptime/area),
		bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},transform=ax[0][1].transAxes,
				 verticalalignment='bottom', horizontalalignment='left',
		color='black', fontsize=15)
	ax[1][0].text(0.1, 0.8, 'f1 ~ {:.1e} adu/s/pix^2'.format(np.sum(cut_1.data)/exptime/area),
		bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},transform=ax[1][0].transAxes,
				 verticalalignment='bottom', horizontalalignment='left',
		color='black', fontsize=15)
	ax[1][1].text(0.1, 0.8, 'f2 ~ {:.1e} adu/s/pix^2'.format(np.sum(cut_2.data)/exptime/area),
		bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},transform=ax[1][1].transAxes,
				 verticalalignment='bottom', horizontalalignment='left',
		color='black', fontsize=15)
	ax[0][1].text(1.1,1.01,'fcore ~ $\sum_i p_i/exptime/area $ \n area ~ $r_{eq}^2 $',
				 bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},transform=ax[1][1].transAxes,
				 verticalalignment='top', horizontalalignment='left',
		color='black', fontsize=15)
	
	# todo get r_eq formatted to show getting key error because of {eq}, similar include theta
	#plt.show()
	plt.savefig(saveas,bbox_inches='tight')
	plt.close('all')


