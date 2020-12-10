import os 
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

def cut_hdu(self,location,size,writetodisk=False,saveas=None):
    """
    cutout size lxw ~ (dy x dx) box on fits file centered at a pixel or skycoord location
    if size is scalar gives a square dy=dx 
    updates hdr wcs keeps other info from original
    """
    hdu = self.hdu
    cphdu = hdu.copy()
    dat = cphdu.data
    hdr = cphdu.header
    wcs,frame = WCS(hdr),hdr['RADESYS'].lower()
        
    cut = Cutout2D(dat,location,size,wcs=wcs) 
    cutwcs = cut.wcs
    if writetodisk:
        cphdu.data = cut.data
        cphdu.header.update(cut.wcs.to_header())    
        cphdu.writeto(saveas,overwrite=True)
    
    return 