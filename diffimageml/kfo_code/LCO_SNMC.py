import pickle
import sncosmo
import pandas as pd
import numpy as np
import os
import glob
import read_slate
from numpy.random import choice

import sys
from optparse import OptionParser
parser = OptionParser()
(options,args)=parser.parse_args()

source_data = pd.read_csv('source_data.txt',delimiter='|',skiprows=1)
window = pickle.load(open("window.pkl","rb"))
headers = pickle.load(open("LCOGT_Hdrs.pkl","rb"))

# registers the sdssrp band to sncosmo for our observations used on lco 1m
# puts it in as 'sdssrp' 
sdssrpband()

def sdssrpband():
    import sncosmo
    from astropy.io import ascii
    import astropy.units as u
    # https://lco.global/observatory/instruments/filters/sdss-r/
    sdssrp = ascii.read(open("sdssrp.txt","rb"))
    sdssrp.rename_columns(['\lambda','[nm]'],['lambda [nm]','transmission'])
    wavelength = sdssrp['lambda [nm]']
    transmission = sdssrp['transmission']
    band = sncosmo.Bandpass(wavelength, transmission, name='sdssrp',wave_unit=u.nm)
    sncosmo.register(band)
    #print(band.minwave(),band.maxwave(),band.wave_eff)
    plot = False
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({'font.size': 15})
        fig,ax = plt.subplots(figsize=(10.5,8.5))
        ax.plot(wavelength,transmission)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Transmission")
        ax.set_xlim(525,725)
        plt.savefig('transmission_sdssrp.pdf',bbox_inches='tight')
    return band

"""
definition of the window
for name in df[' Our Survey Name ']:
    tmp = read_slate.get_obs(name) # list of datetimes field name was observed
    dates.append(tmp) 
    firsts.append(np.min(tmp))
    finals.append(np.max(tmp))
first = np.min(firsts)
final = np.max(finals)
window = final - first
print(first,final,window)
window.days ~ 652, start 2018-12-04, finish 2020-09-16 
d = {'first':first,'final':final,'window':window}
d
pickle.dump(d,open("window.pkl","wb"))
"""

# stole lens_extinction from code I wrote for Requiem where I was showing the color mag plot for different SN models
# I used Av ~ 1 = vobs - vint; dimmed by a mag in v-band from dust at lensing galaxy
# to do what is a typical Av value, Av ~ 1 had no defense/reference made for it 
# then used that to determine the dimming of other bands with a typical r_v value, could show vector in the color-mag space
# here rather than vector I'm guessing I'll just subtract the Ar from the luminosity function? 
def lens_extinction(source='salt2-extended',zs=1.95,zl=0.338,Av=1,dust=sncosmo.CCM89Dust(),t=0):
    # the source makes no difference, dust has same effect on every model
    # the model as it would look like for obs at the lens plane w no dust
    model=sncosmo.Model(source=source)
    model.set(z=zs-zl) 
    # the model as it would look for obs at lens w dust
    dust_model = sncosmo.Model(source=source,effects=[dust],effect_names=['mw'],effect_frames=['obs'])
    dust_model.set(z=zs-zl)
    r_v = 3.1 # AV/E(B-V) typical value for diffuse ISM
    ebv = Av/r_v
    dust_model.set(mwebv=ebv,mwr_v=r_v)
    """
    I want to know how the dust dimmed f160w & f105w in our frame ie zl~0.338 further
    f160w ~ 16000 & f105w ~ 10500 (Angstrom), they have widths about 1500 (Angstrom)
    lens observer at these wavelengths/(1+zl) should correspond to the wavelengths for us
    I can get flux density at each of these filters eff wavelength w & wo dust
    the filters aren't terribly wide and throughputs are flat, will approximate mags using just eff wavelength
    that should get me an approximate vec shift (E(105-160),A160) I want w Av ~ 1 in the lens plane
    """
    # the intrinsic flux densities for lens observer at the corresponding wavelengths to bands we see  
    f160 = model.flux(t,[16000/(1+zl)])
    f105 = model.flux(t,[10500/(1+zl)])
    # dusty flux densities
    dustyf160 = dust_model.flux(t,[16000/(1+zl)])
    dustyf105 = dust_model.flux(t,[10500/(1+zl)])
    
    # what do the extinctions look like, A ~ mobs - m
    A160 = -2.5*np.log10(dustyf160/f160)
    A105 = -2.5*np.log10(dustyf105/f105)
    # my color mag is (x,y) ~ (f105w-f160w,f160w)
    # I want to show vector (E(105-160),A160)
    # color excess 
    E = A105 - A160
    return (E[0],A160[0])


from scipy.special import gammaincinv as ginv

def poissonLimits( N, confidence=1 ):
    """   
    Adapted from P.K.G.Williams : 
    http://newton.cx/~peter/2012/06/poisson-distribution-confidence-intervals/
    Let's say you observe n events in a period and want to compute the k
    confidence interval on the true rate - that is, 0 < k <= 1, and k =
    0.95 would be the equivalent of 2sigma. Let a = 1 - k, i.e. 0.05. The
    lower bound of the confidence interval, expressed as a potential
    number of events, is
       scipy.special.gammaincinv (n, 0.5 * a)
    and the upper bound is
       scipy.special.gammaincinv (n + 1, 1 - 0.5 * a)
    
    The halving of a is just because the 95% confidence interval is made
    up of two tails of 2.5% each, so the gammaincinv function is really,
    once you chop through the obscurity, exactly what you want.
    INPUTS : 
      N : the number of observed events
      confidence : may either be a float <1, giving the exact 
          confidence limit desired (e.g.  0.95 or 0.99)
          or it can be an integer in [1,2,3], in which case 
          we set the desired confidence interval to match 
          the 1-, 2- or 3-sigma gaussian confidence limits
             confidence=1 gives the 1-sigma (68.3%) confidence limits
             confidence=2  ==>  95.44% 
             confidence=3  ==>  99.74% 
    """
    if confidence<1 : k = confidence
    elif confidence==1 : k = 0.6826
    elif confidence==2 : k = 0.9544
    elif confidence==3 : k = 0.9974
    else :
        print( "ERROR : you must choose nsigma from [1,2,3]")
        return( None )
    lower = ginv( N, 0.5 * (1-k) )
    upper = ginv( N+1, 1-0.5*(1-k) )
    return( lower, upper )

def f_asymmGauss(x,mu,sigma1,sigma2):
    """
    GENPEAK_SALT2c:    -0.054
    GENSIGMA_SALT2c:    0.043  0.101     # bifurcated sigmas
    GENRANGE_SALT2c:   -0.300  0.500     # color range
    GENPEAK_SALT2x1:     0.973
    GENSIGMA_SALT2x1:    1.472   0.222     # bifurcated sigmas
    GENRANGE_SALT2x1:    -3.0     2.0       # x1 (stretch) range
    """
    A = 1/(np.sqrt(np.pi/2)*(sigma1 + sigma2)) 
    if x < mu:
        return A*np.exp(-(x-mu)**2/(2*sigma1**2))
    else:
        return A*np.exp(-(x-mu)**2/(2*sigma2**2))

def f_efficiency(m,m50,alpha):
    #https://arxiv.org/pdf/1509.06574.pdf, strolger 
    return (1+np.exp(alpha*(m-m50)))**-1 

def SNMC(df,name,window=652/365,scale=1e6):
    """
    df should be the full data frame, with properties of all the targets
    it will be indexed with name for single target by 'our survey name' inside the function
    Using the SN rates with window of time to generate N SNe, [N/yr]*1.8 yr
    The window ~ final - first observation from read_slate
    """
    avg_NIa = np.mean(df['NIa'])
    avg_eNIa = np.mean(df['e_NIa'])
    avg_Ncc = np.mean(df['Ncc'])
    avg_eNcc = np.mean(df['e_Ncc'])
    targ = df[df[' Our Survey Name '] == name]
    zS,mu = targ['zS'],targ['mu']
    NIa = targ['NIa']
    Ncc = targ['Ncc']
    e_NIa = targ['e_NIa']
    e_Ncc = targ['e_Ncc']
    """
    if pd.isna(NIa):
        # use the avg if rate not available, should only apply to SWELLSJ0841+3824
        print('using avg SN rates for',name)
        NIa,e_NIa = avg_NIa,avg_eNIa
    """
    # scale up rates to large enough numbers to get multiple models to plant 
    NIa*=scale
    e_NIa*=scale
    Ncc*=scale
    e_Ncc*=scale
    # select a rate
    NIa = np.random.normal(NIa,e_NIa)
    if NIa < 0:
        NIa = 0

    # use window to get number from rates
    if type(window) == dict:
        window = window['window']
        window = window.days/365
    nIa = NIa*window # NIa [/yr], window ~ 652 days, n [expected number of Ia in window]
    ncc = Ncc*window
    
    # dusts
    """
    for a review of dust http://www.astronomy.ohio-state.edu/~pogge/Ast871/Notes/Dust.pdf
    Observationally, RV ranges between 2 and 6, but most often one finds the interstellar extinction law
    assumed by people as adopting one of two “typical” values for RV:
    RV=3.1, typical of the Diffuse ISM.
    RV=5, typical of dense (molecular) clouds. 
    """
    mwdust = sncosmo.CCM89Dust()
    mwdust.set(ebv=0,r_v=3.1) # these are the defaults of CCM89Dust() 
    hostdust = sncosmo.CCM89Dust()
    hostdust.set(ebv=0,r_v=3.1)
    import sfdmap
    dustmap=sfdmap.SFDMap('/home/oconnorf/miniconda/lib/python3.7/site-packages/mwdust/dust_maps')
    # using  Schlegel, Finkbeiner & Davis (1998) MW dust map to set EBV
    # get ra and dec of the SN ~ target
    ra = df['ra']
    dec = df['dec']
    print(ra,dec)
    #ra,dec=0.0,0.0
    ebv = dustmap.ebv(ra,dec)
    mwdust.set(ebv=ebv)

    # Luminosity Functions: Goldstein 2019 Table 1, MB Vega, https://arxiv.org/pdf/1809.10147.pdf
    band,sys = 'bessellb','vega'
    MIa,sigmaIa = -19.23,0.1
    MIIp,sigmaIIp = -16.9,1.12
    MIIL,sigmaIIL = -17.46,0.38
    MIIn,sigmaIIn = -19.05,0.5
    MIbc,sigmaIbc = -17.51,0.74
    # Luminosity Functions: Converted to AB Mag System, http://www.astronomy.ohio-state.edu/~martini/usefuldata.html, Blanton et al. 2007
    # B-Band m_AB - m_Vega ~ -0.09
    band,sys = 'bessellb','ab'
    dm = -0.09
    MIa += dm
    MIIp += dm
    MIIL += dm
    MIIn += dm
    MIbc += dm
    
    # fractions core collapse: Eldridge 2013, https://arxiv.org/abs/1301.1975 
    # Volume limited ~ 30 Mpc couple hundred transients in the sample
    fracIb,fracIc,fracIIp,fracIIL,fracIIn,fracIIb,fracIIpec= 0.09,0.17,0.55,0.03,.024,0.121,0.01
    # my combinations of eldridges subtypes to goldsteins
    fracIbc = fracIb + fracIc
    fracIIL = fracIIL + fracIIpec + fracIIb 
    # if you want the check on CC fractions
    fracs = [fracIIp,fracIIL,fracIIn,fracIbc]
    # np.sum(fracs)
    """
    Another possible source for fractions...
    Fractions core collapse: Smith 2010, https://arxiv.org/pdf/1006.3899.pdf 
    Volume limited ~ 60 Mpc sample from LOSS
    """

    # Stretch and Color Ia parameter distributions
    c = np.linspace(-0.3,0.5,1000)
    cdist = [f_asymmGauss(ci,-0.054,.043,0.101) for ci in c]
    dc = (max(c) - min(c))/len(c)
    cps = np.array(cdist)*dc
    tmp = (1 - np.sum(cps))/len(c)
    cps = [i+tmp for i in cps]

    x1 = np.linspace(-3.0,2.0,1000)
    x1dist = [f_asymmGauss(xi,0.973,1.472,0.222) for xi in x1]
    dx1 = (max(x1)-min(x1))/len(x1)
    x1ps = np.array(x1dist)*dx1
    tmp = (1-np.sum(x1ps))/len(x1)
    x1ps = [i+tmp for i in x1ps]

    
    # The models
    Iamodels = []
    CCmodels = []
    IIpmodels,IILmodels,IInmodels,Ibcmodels = [],[],[],[]
    # The mags if want to see those (without magnification)
    magIas = [] 
    magIIps,magIILs,magIIns,magIbcs = [],[],[],[]
    cs,x1s=[],[]
    for i in range(int(nIa)):
        model = sncosmo.Model(source='salt2-extended',
        effects=[hostdust, mwdust],effect_names=['host', 'mw'],effect_frames=['rest', 'obs']) # Ia
        magIa = np.random.normal(MIa,sigmaIa)
        magIas.append(magIa)
        # lensing magnification
        mabs = magIa - 2.5*np.log10(mu)
        t0 = np.random.uniform(0,int(window*365))
        # stretch
        list_of_candidates,number_of_items_to_pick,probability_distribution=x1,1,x1ps
        x1draw = choice(list_of_candidates, number_of_items_to_pick,p=probability_distribution)
        x1s.append(x1draw)
        # color 
        list_of_candidates,number_of_items_to_pick,probability_distribution=c,1,cps
        cdraw = choice(list_of_candidates, number_of_items_to_pick,p=probability_distribution)
        cs.append(cdraw)
        # set the values
        model.set(z=zS, t0=t0, c=cdraw, x1=x1draw)
        model.set_source_peakabsmag(mabs,band,sys)
        Iamodels.append(model)
    for i in range(int(fracIIp*ncc)):
        model = sncosmo.Model(source='s11-2005lc',
        effects=[hostdust, mwdust],effect_names=['host', 'mw'],effect_frames=['rest', 'obs']) # IIp
        magIIp = np.random.normal(MIIp,sigmaIIp)
        magIIps.append(magIIp)
        mabs = magIIp - 2.5*np.log10(mu)
        t0 = np.random.uniform(0,int(window*365))
        model.set(z=zS, t0=t0)
        model.set_source_peakabsmag(mabs,band,sys) 
        IIpmodels.append(model)
    CCmodels.append(IIpmodels)
    for i in range(int(fracIIL*ncc)):
        model = sncosmo.Model(source='nugent-sn2l',
        effects=[hostdust, mwdust],effect_names=['host', 'mw'],effect_frames=['rest', 'obs']) # IIL
        magIIL = np.random.normal(MIIL,sigmaIIL)
        magIILs.append(magIIL)
        mabs = magIIL - 2.5*np.log10(mu)
        t0 = np.random.uniform(0,int(window*365))
        model.set(z=zS, t0=t0)
        model.set_source_peakabsmag(mabs,band,sys)  
        IILmodels.append(model)
    CCmodels.append(IILmodels)
    for i in range(int(fracIIn*ncc)):
        model = sncosmo.Model(source='nugent-sn2n',
        effects=[hostdust, mwdust],effect_names=['host', 'mw'],effect_frames=['rest', 'obs']) # IIn
        magIIn = np.random.normal(MIIn,sigmaIIn) 
        magIIns.append(magIIn)
        mabs = magIIn -2.5*np.log10(mu)
        t0 = np.random.uniform(0,int(window*365))
        model.set(z=zS, t0=t0)
        model.set_source_peakabsmag(mabs,band,sys)  
        IInmodels.append(model)
    CCmodels.append(IInmodels)
    for i in range(int(fracIbc*ncc)):
        model = sncosmo.Model(source='nugent-sn1bc',
        effects=[hostdust, mwdust],effect_names=['host', 'mw'],effect_frames=['rest', 'obs']) # Ibc
        magIbc = np.random.normal(MIbc,sigmaIbc)
        magIbcs.append(magIbc)
        mabs = magIbc -2.5*np.log10(mu)
        t0 = np.random.uniform(0,int(window*365))
        model.set(z=zS, t0=t0)
        model.set_source_peakabsmag(mabs,band,sys)  
        Ibcmodels.append(model)
    CCmodels.append(Ibcmodels)

    return [Iamodels,CCmodels]



def SNpeaks(model,band='sdssrp',magsys='ab'):
    mintime = model.mintime()
    maxtime = model.maxtime()
    mags =[]
    times = np.arange(mintime,maxtime,3) # few days spacing
    for t in times:
        m = model.bandmag(band,magsys,t)
        mags.append(m)

    a = np.array(mags)
    peak = np.nanmin(np.nanmin(a[a != -np.inf]))
    return peak


def SNexpected(df,name,scale=1e6,SN=None,window=window,headers=headers,efficiency=None):
    dates = read_slate.get_obs(name)
    print(name,len(dates),'observations in window', window)

    if SN == None:
        SN = SNMC(df,name,scale)
    Iamodels,CCmodels=SN
    IIpmodels,IILmodels,IInmodels,Ibcmodels = CCmodels
    
    phaseshifts = [i - window['first'] for i in dates]    
    phaseshifts = [i.days for i in phaseshifts] 
    
    """
    This may be worth doing in a seperaete function, want to set efficiency curve for each observation using hdr keys
    # get the headers for name/dates
    name2 = " '" + name + "'"
    hdrs = headers[headers[" 'OBJECT'"] == name2]
    # need to hexbin for efficiencies before I go here and try and set the actual efficiency fit using hdr params 
    for date in dates:
        day_obs = hdrs[" 'DAY_OBS'"]
        # use date to grab hdr of day obs then use that to set efficiency
    """
    
    # efficiency placeholder 
    alpha = 5
    if efficiency == None:
        # https://exposure-time-calculator.lco.global/
        sigmam50 = 0.3
        # 300s vals 3sigma
        m50_300s = 22.4
        # 600s vals 3sigma
        m50_600s = 22.8
        
    band = 'sdssrp'
    magsys = 'ab'
    
    Iadetections = []
    # Ia detections should be the length of Iamodels, with either a zero or one depending on detection
    # detection ruled as any model which had mag < m50_i for any shift_i 
    print('---------------------------------------------')
    print('beginning Ia models', len(Iamodels))
    for i in range(len(Iamodels)):
        model = Iamodels[i]
        modeldict = {}
        for j in range(len(model.param_names)):
            modeldict[model.param_names[j]] = model.parameters[j]
        peak = model.bandmag(band,magsys,modeldict['t0'])
        quickdict = {'modeli':i,'pk':[peak,modeldict['t0']]}
        Iadetections.append([quickdict,modeldict]) # i 0 -> nSN, pk rp mag, t0 of pk, model dictionary  
        obsdicts = [] # list for dictionary of each observation 
        obsidx = 0
        for shift in phaseshifts:
            mag = model.bandmag(band,magsys,shift) # mag may be pd.isna() or np.inf 
            m50_i = np.random.normal(m50_300s,sigmam50)
            eta_i = f_efficiency(mag,m50_i,alpha)
            obsdict = {'obsi':obsidx,'m50':m50_i,'tobs':shift,'magobs':mag} 
            # maybe eventually do something with the actual efficiency right now just going to call above m50 detection
            if mag < m50_i:
                obsdict['detection'] = 1
            else:
                obsdict['detection'] = 0
            obsdicts.append(obsdict)
            obsidx += 1
        Iadetections[i].append(obsdicts)
        result = {}
        for j in obsdicts:
            if j['detection'] == 1:
                result['detected'] = 1
                break
            else:
                pass
            result['detected'] = 0
        Iadetections[i].append(result)
        
    ndetects = 0
    for i in Iadetections:
        r = i[-1]
        if r['detected'] == 1:
            ndetects += 1
        else:
            continue
    print(ndetects,'detections')
    # starting out dictionary for the expected number of detected SNe in window
    # not scaling down here including that factor in dict though
    ndetectSNe = {'name':name,'window':window,'scale':scale,'Ia':ndetects}
        
    print('---------------------------------------------')
    IIpdetections = []
    print('beginning IIp models', len(IIpmodels))
    for i in range(len(IIpmodels)):
        model = IIpmodels[i]
        modeldict = {}
        for j in range(len(model.param_names)):
            modeldict[model.param_names[j]] = model.parameters[j]
        peak = model.bandmag(band,magsys,modeldict['t0'])
        quickdict = {'modeli':i,'pk':[peak,modeldict['t0']]}
        IIpdetections.append([quickdict,modeldict]) # i 0 -> nSN, pk rp mag, t0 of pk, model dictionary  
        obsdicts = [] # list for dictionary of each observation 
        obsidx = 0
        for shift in phaseshifts:
            mag = model.bandmag(band,magsys,shift) # mag may be pd.isna() or np.inf 
            m50_i = np.random.normal(m50_300s,sigmam50)
            eta_i = f_efficiency(mag,m50_i,alpha)
            obsdict = {'m50':m50_i,'tobs':shift,'magobs':mag} 
            # maybe eventually do something with the actual efficiency right now just going to call above m50 detection
            if mag < m50_i:
                obsdict['detection'] = 1
            else:
                obsdict['detection'] = 0
            obsdicts.append(obsdict)
            obsidx += 1
        IIpdetections[i].append(obsdicts)
        result = {}
        for j in obsdicts:
            if j['detection'] == 1:
                result['detected'] = 1
                break
            else:
                pass
            result['detected'] = 0
        IIpdetections[i].append(result)
    
    ndetects = 0
    for i in IIpdetections:
        r = i[-1]
        if r['detected'] == 1:
            ndetects += 1
        else:
            continue
    print(ndetects,'detections')
    ndetectSNe['IIp'] = ndetects
    
    print('---------------------------------------------')
    IILdetections = []
    print('beginning IIL models', len(IILmodels))
    for i in range(len(IILmodels)):
        model = IILmodels[i]
        modeldict = {}
        for j in range(len(model.param_names)):
            modeldict[model.param_names[j]] = model.parameters[j]
        peak = model.bandmag(band,magsys,modeldict['t0'])
        quickdict = {'modeli':i,'pk':[peak,modeldict['t0']]}
        IILdetections.append([quickdict,modeldict]) # i 0 -> nSN, pk rp mag, t0 of pk, model dictionary  
        obsdicts = []
        obsidx = 0
        for shift in phaseshifts:
            mag = model.bandmag(band,magsys,shift) # mag may be pd.isna() or np.inf 
            m50_i = np.random.normal(m50_300s,sigmam50)
            eta_i = f_efficiency(mag,m50_i,alpha)
            obsdict = {'m50':m50_i,'tobs':shift,'magobs':mag} 
            # maybe eventually do something with the actual efficiency right now just going to call above m50 detection
            if mag < m50_i:
                obsdict['detection'] = 1
            else:
                obsdict['detection'] = 0
            obsdicts.append(obsdict)
            obsidx += 1
        IILdetections[i].append(obsdicts)
        result = {}
        for j in obsdicts:
            if j['detection'] == 1:
                result['detected'] = 1
                break
            else:
                pass
            result['detected'] = 0
        IILdetections[i].append(result)
    
    ndetects = 0
    for i in IILdetections:
        r = i[-1]
        if r['detected'] == 1:
            ndetects += 1
        else:
            continue
    print(ndetects,'detections')
    ndetectSNe['IIL'] = ndetects
    
    print('---------------------------------------------')
    IIndetections = []
    print('beginning IIn models', len(IInmodels))
    for i in range(len(IInmodels)):
        model = IInmodels[i]
        modeldict = {}
        for j in range(len(model.param_names)):
            modeldict[model.param_names[j]] = model.parameters[j]
        peak = model.bandmag(band,magsys,modeldict['t0'])
        quickdict = {'modeli':i,'pk':[peak,modeldict['t0']]}
        IIndetections.append([quickdict,modeldict]) # i 0 -> nSN, pk rp mag, t0 of pk, model dictionary  
        obsdicts = []
        obsidx = 0
        for shift in phaseshifts:
            mag = model.bandmag(band,magsys,shift) # mag may be pd.isna() or np.inf 
            m50_i = np.random.normal(m50_300s,sigmam50)
            eta_i = f_efficiency(mag,m50_i,alpha)
            obsdict = {'m50':m50_i,'tobs':shift,'magobs':mag} 
            # maybe eventually do something with the actual efficiency right now just going to call above m50 detection
            if mag < m50_i:
                obsdict['detection'] = 1
            else:
                obsdict['detection'] = 0
            obsdicts.append(obsdict)
            obsidx += 1
        IIndetections[i].append(obsdicts)
        result = {}
        for j in obsdicts:
            if j['detection'] == 1:
                result['detected'] = 1
                break
            else:
                pass
            result['detected'] = 0
        IIndetections[i].append(result)
                
    ndetects = 0
    for i in IIndetections:
        r = i[-1]
        if r['detected'] == 1:
            ndetects += 1
        else:
            continue
    print(ndetects,'detections')
    ndetectSNe['IIn'] = ndetects

    print('---------------------------------------------')
    Ibcdetections = []
    print('beginning Ibc models', len(Ibcmodels))
    for i in range(len(Ibcmodels)):
        model = Ibcmodels[i]
        modeldict = {}
        for j in range(len(model.param_names)):
            modeldict[model.param_names[j]] = model.parameters[j]
        peak = model.bandmag(band,magsys,modeldict['t0'])
        quickdict = {'modeli':i,'pk':[peak,modeldict['t0']]}
        Ibcdetections.append([quickdict,modeldict]) # i 0 -> nSN, pk rp mag, t0 of pk, model dictionary  
        obsdicts = []
        obsidx = 0
        for shift in phaseshifts:
            mag = model.bandmag(band,magsys,shift) # mag may be pd.isna() or np.inf 
            m50_i = np.random.normal(m50_300s,sigmam50)
            eta_i = f_efficiency(mag,m50_i,alpha)
            obsdict = {'m50':m50_i,'tobs':shift,'magobs':mag} 
            # maybe eventually do something with the actual efficiency right now just going to call above m50 detection
            if mag < m50_i:
                obsdict['detection'] = 1
            else:
                obsdict['detection'] = 0
            obsdicts.append(obsdict)
            obsidx += 1
        Ibcdetections[i].append(obsdicts)
        result = {}
        for j in obsdicts:
            if j['detection'] == 1:
                result['detected'] = 1
                break
            else:
                pass
            result['detected'] = 0
        Ibcdetections[i].append(result)
                
    ndetects = 0
    for i in Ibcdetections:
        r = i[-1]
        if r['detected'] == 1:
            ndetects += 1
        else:
            continue
    print(ndetects,'detections')
    ndetectSNe['Ibc'] = ndetects
    
    CCdetections = [IIpdetections,IILdetections,IIndetections,Ibcdetections]
    
    return ndetectSNe,Iadetections,CCdetections


def yields_fracs_peaks_tobs(result):
    """
    The result pkl with every lightcurve is too large to shuffle from hyperion to local
    This fcn is giving from my result data:the yields and fracs of detected SNe and the distribution of peak magnitudes for each lens and tobs
    """
    ndetectSNe,snia,sncc = result[0],result[1],result[2]
    snIIp,snIIL,snIIn,snIbc = sncc

    fracIa = ndetectSNe['Ia']/len(snia)
    yieldIa = ndetectSNe['Ia']/ndetectSNe['scale']
    yieldCC = (ndetectSNe['IIp'] + ndetectSNe['IIL'] + ndetectSNe['IIn'] + ndetectSNe['Ibc'])/ndetectSNe['scale']
    yieldIIp = ndetectSNe['IIp']/ndetectSNe['scale']
    yieldIIL = ndetectSNe['IIL']/ndetectSNe['scale']
    yieldIIn = ndetectSNe['IIn']/ndetectSNe['scale']
    yieldIbc = ndetectSNe['Ibc']/ndetectSNe['scale']
    fracIIp = ndetectSNe['IIp']/len(snIIp)
    fracIIL = ndetectSNe['IIL']/len(snIIL)
    fracIIn = ndetectSNe['IIn']/len(snIIn)
    fracIbc = ndetectSNe['Ibc']/len(snIbc)
    fracCC = (ndetectSNe['IIp'] + ndetectSNe['IIL'] + ndetectSNe['IIn'] + ndetectSNe['Ibc'])/(len(snIIp)+len(snIIL)+len(snIIn)+len(snIbc))

    pkIas = [i[0]['pk'][0] for i in snia] # rband (sdssrp) peak
    pkIIps = [i[0]['pk'][0] for i in snIIp]
    pkIILs = [i[0]['pk'][0] for i in snIIL]
    pkIIns = [i[0]['pk'][0] for i in snIIn]
    pkIbcs = [i[0]['pk'][0] for i in snIbc]

    observations = snia[0][2] # iamodel0 (still has all the observations) 
    #print(observations)
    tobs = [i['tobs'] for i in observations]

    # start adding these to the dictionary
    ndetectSNe['tobs'] = tobs
    ndetectSNe['fracIa'] = fracIa
    ndetectSNe['fracCC'] = fracCC
    ndetectSNe['fracIIp'] = fracIIp
    ndetectSNe['fracIIL'] = fracIIL
    ndetectSNe['fracIIn'] = fracIIn
    ndetectSNe['fracIbc'] = fracIbc
    ndetectSNe['yieldIa'] = yieldIa
    ndetectSNe['yieldCC'] = yieldCC
    ndetectSNe['yieldIIp'] = yieldIIp
    ndetectSNe['yieldIIL'] = yieldIIL
    ndetectSNe['yieldIIn'] = yieldIIn
    ndetectSNe['yieldIbc'] = yieldIbc
    ndetectSNe['pkIa'] = pkIas
    ndetectSNe['pkIIp'] = pkIIps
    ndetectSNe['pkIIL'] = pkIILs
    ndetectSNe['pkIIn'] = pkIIns
    ndetectSNe['pkIbc'] = pkIbcs

    return ndetectSNe



def plot_peaks(data,saveas='peaks.pdf'):
    import matplotlib
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    matplotlib.rcParams.update({'font.size': 15})
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5)) # 2,1 r,c
    sns.distplot(data['pkIa'], hist=False, kde=True, color="midnightblue", label="pkIa",ax=ax)
    sns.distplot(data['pkIIp'], hist=False, kde=True, color="red", label="pkIIp",ax=ax)
    sns.distplot(data['pkIIL'], hist=False, kde=True, color="yellow", label="pkIIL", ax=ax)
    sns.distplot(data['pkIIn'], hist=False, kde=True, color="green", label="pkIIn", ax=ax)
    sns.distplot(data['pkIbc'], hist=False, kde=True, color="blue", label="pkIbc", ax=ax)

    ax.set_xlabel('Peak $m_r$ (AB)')
    ax.legend(loc='upper right')

    plt.savefig(saveas,bbox_inches="tight")


if __name__ == "__main__":
    scale = 1e6
    source_data = pd.read_csv('source_data.txt',delimiter='|',skiprows=1)
    source_data['NIa'] = pd.to_numeric(source_data['NIa'], errors='coerce')
    source_data['Ncc'] = pd.to_numeric(source_data['Ncc'], errors='coerce')
    source_data['e_NIa'] = pd.to_numeric(source_data['e_NIa'], errors='coerce')
    source_data['e_Ncc'] = pd.to_numeric(source_data['e_Ncc'], errors='coerce')

    # these avg SN rate errors are how I'm doing the +- on final detection numbers 
    avg_eNIa = np.mean(source_data['e_NIa'])
    avg_eNcc = np.mean(source_data['e_Ncc'])
    avg_Ncc = np.mean(source_data['Ncc'])
    avg_NIa = np.mean(source_data['NIa'])
    percent_eNIa = avg_NIa/avg_eNIa
    percent_eNcc = avg_Ncc/avg_eNcc

    print("Average Percent SN Rate Errors")
    print("Ia {}, CC {}".format(percent_eNIa,percent_eNcc))

    window = pickle.load(open("window.pkl","rb"))
    headers = pickle.load(open("LCOGT_Hdrs.pkl","rb"))
    output_folder = "SNoutputs"
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)
    
    run_simulation = False
    if run_simulation:
        # these lines will run the SN simulation and target observation/detection fcns 
        # should have batch args 0-97 for the 98 sources 
        source_data_idx = int(sys.argv[1])
        name = source_data.iloc[source_data_idx][' Our Survey Name ']
        pickle_to = os.path.join(output_folder,name)
        print(source_data_idx,pickle_to)
        SN = SNMC(source_data,name,window=window,scale=scale)
        pickle.dump(SN,open(pickle_to+"_SN.pkl","wb"))
        results = SNexpected(source_data,name,scale=scale,SN=SN,window=window)
        pickle.dump(results,open(pickle_to+"_results.pkl","wb"))
    
    make_peaks = True
    if make_peaks:
    # these lines will make peaks from the SN lightcurves for each target
    # ie how are the peaks for each SN distributed at each target
        sns = glob.glob("SNoutputs/*SN.pkl")
        for m in sns:
            pickle_to = m.split('.')[0] + 'peaks.pkl'
            mi = pickle.load(open(m,'rb'))
            ias,ccs = mi[0],mi[1]
            ps,ls,ns,ibcs = ccs[0],ccs[1],ccs[2],ccs[3]
            ias = ias[:100]
            ps = ps[:100]
            ls = ls[:100]
            ns = ns[:100]
            ibcs = ibcs[:100]
            pkIas = [SNpeaks(i,band='sdssrp',magsys='ab') for i in ias] #SNpeaks(model,band='sdssrp',magsys='ab'):
            pkIIps = [SNpeaks(i,band='sdssrp',magsys='ab') for i in ps]
            pkIILs = [SNpeaks(i,band='sdssrp',magsys='ab') for i in ls]
            pkIIns = [SNpeaks(i,band='sdssrp',magsys='ab') for i in ns]
            pkIbcs = [SNpeaks(i,band='sdssrp',magsys='ab') for i in ibcs]
            snpeaks = {'pkIa':pkIas,'pkIIp':pkIIps,'pkIIL':pkIILs,'pkIIn':pkIIns,'pkIbc':pkIbcs}
            pickle.dump(snpeaks,open(pickle_to,"wb"))
    
    group_peaks = True
    if group_peaks:
        # these lines group all the peaks from every target
        # ie how are the peaks for each SN distributed over every target's z/mu in our survey
        peaks = glob.glob("SNoutputs/*SNpeaks.pkl")
        pkIas,pkIIps,pkIILs,pkIIns,pkIbcs = [],[],[],[],[]
        for pk in peaks:
            tmp = pickle.load(open(pk,"rb"))
            pkIas.append(np.min(tmp['pkIa']))
            pkIIps.append(np.min(tmp['pkIIp']))
            pkIILs.append(np.min(tmp['pkIIL']))
            pkIIns.append(np.min(tmp['pkIIn']))
            pkIbcs.append(np.min(tmp['pkIbc']))
        d = {'pkIa':pkIas,'pkIIp':pkIIps,'pkIIL':pkIILs,'pkIIn':pkIIns,'pkIbc':pkIbcs}
        pickle_to = "SNoutputs/allSNpeaks.pkl"
        pickle.dump(d,open(pickle_to,"wb"))
        # make a pdf of the result
        plot_peaks(d)
    
    print_results = True
    if print_results:
        # these are used to print out some results
        rs = glob.glob("SNoutputs/*results.pkl")
        results = []
        for r in rs:
            ri = pickle.load(open(r,'rb'))
            # this update adds some new useful stuff to results
            update = yields_fracs_peaks_tobs(ri)
            pickle.dump(update,open(r[:-4]+'_update.pkl','wb')) # r[:-4] to drop .pkl from r in glob of this lens result pkl
            results.append(ri)
        print(len(results))
        ndetectSNe = [r[0] for r in results] 
        print(len(ndetectSNe))
        ndetectSNs,ndetectIas,ndetectCCs = [],[],[]
        ndetectIIps,ndetectIILs,ndetectIIns,ndetectIbcs = [],[],[],[]
        percent_detectSNs,percent_detectIas,percent_detectCCs = [],[],[]
        for r in results:
            # unpack results so can determine the detection percentages and do some final expected number stuff  
            ndetectSNe = r[0]
            Iadetections = r[1]
            CCdetections = r[2]
            [IIpdetections,IILdetections,IIndetections,Ibcdetections] = CCdetections
         
            ndetectIas.append(ndetectSNe['Ia']/scale)
            ndetectcc = (ndetectSNe['IIp'] + ndetectSNe['IIL'] + ndetectSNe['IIn'] + ndetectSNe['Ibc'])/scale
            ndetectCCs.append(ndetectcc)
            ndetectSNs.append(ndetectSNe['Ia']/scale+ndetectcc)
            ndetectIIps.append(ndetectSNe['IIp']/scale)
            ndetectIILs.append(ndetectSNe['IIL']/scale)
            ndetectIIns.append(ndetectSNe['IIn']/scale)
            ndetectIbcs.append(ndetectSNe['Ibc']/scale)

            percent_detectIIp = ndetectSNe['IIp']/len(IIpdetections)
            percent_detectIIL = ndetectSNe['IIL']/len(IILdetections)
            percent_detectIIn = ndetectSNe['IIn']/len(IIndetections)
            percent_detectIbc = ndetectSNe['Ibc']/len(Ibcdetections)
            percent_detectCCsubtypes = [percent_detectIIp,percent_detectIIL,percent_detectIIn,percent_detectIbc]
            percent_detectCC = (ndetectSNe['IIp']+ndetectSNe['IIL']+ndetectSNe['IIn']+ndetectSNe['Ibc'])/(len(IIpdetections)+len(IILdetections)+len(IIndetections)+len(Ibcdetections))
            percent_detectSNs.append((ndetectSNe['Ia']+ndetectSNe['IIp']+ndetectSNe['IIL']+ndetectSNe['IIn']+ndetectSNe['Ibc'])/(len(Iadetections)+len(IIpdetections)+len(IILdetections)+len(IIndetections)+len(Ibcdetections)))
            percent_detectIas.append(ndetectSNe['Ia']/len(Iadetections))
            percent_detectCCs.append([percent_detectCC,percent_detectCCsubtypes])

        avgndetectIa = np.mean(ndetectIas)
        avgndetectCC = np.mean(ndetectCCs)
        avgndetectSNe = np.mean(ndetectSNs)
        totIa = np.sum(ndetectIas)
        totCC = np.sum(ndetectCCs)
        totSN = totIa + totCC
        totSN = np.sum(ndetectSNs)

        totIIp = np.sum(ndetectIIps)
        totIIL = np.sum(ndetectIILs)
        totIIn = np.sum(ndetectIIns)
        totIbc = np.sum(ndetectIbcs)
        avgndetectIIp = np.mean(ndetectIIps)
        avgndetectIIL = np.mean(ndetectIILs)
        avgndetectIIn = np.mean(ndetectIIns)
        avgndetectIbc = np.mean(ndetectIbcs)

        print("Total Detection Numbers:")
        print("SN {},Ia {},CC {}".format(totSN,totIa,totCC))
        print("IIp {}, IIL {}, IIn {}, Ibc {}".format(totIIp,totIIL,totIIn,totIbc))

        print("Total Detections Poisson Uncertainties (68 percent-1 sig limits)")
        pl = poissonLimits(totSN)
        pIa = poissonLimits(totIa)
        pCC = poissonLimits(totCC)
        pIIp = poissonLimits(totIIp)
        pIIL = poissonLimits(totIIL)
        pIIn = poissonLimits(totIIn)
        pIbc = poissonLimits(totIbc)
        print("SN {}, Ia {}, CC".format(pl,pIa,pCC))
        print("IIp {}, IIL {}, IIn {}, Ibc {}".format(pIIp,pIIL,pIIn,pIbc))

        print("Average Detection Numbers:")
        print("SN {}, Ia {}, CC {}".format(avgndetectSNe,avgndetectIa,avgndetectCC))
        print("IIp {}, IIL {}, IIn {}, Ibc {}".format(avgndetectIIp,avgndetectIIL,avgndetectIIn,avgndetectIbc))

        avg_percent_detectSN = np.mean(percent_detectSNs)
        avg_percent_detectIa = np.mean(percent_detectIas)
        avg_percent_detectCC = np.mean([i[0] for i in percent_detectCCs])
        avg_percent_detectIIp = np.mean([i[1][0] for i in percent_detectCCs])
        avg_percent_detectIIL = np.mean([i[1][1] for i in percent_detectCCs])
        avg_percent_detectIIn = np.mean([i[1][2] for i in percent_detectCCs])
        avg_percent_detectIbc = np.mean([i[1][3] for i in percent_detectCCs])

        print("Average Percent Detection Numbers")
        print("SN {}, Ia {}, CC {}".format(avg_percent_detectSN,avg_percent_detectIa,avg_percent_detectCC))
        print("IIp {}, IIL {}, IIn {}, Ibc {}".format(avg_percent_detectIIp,avg_percent_detectIIL,avg_percent_detectIIn,avg_percent_detectIbc))

