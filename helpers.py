import numpy as np
from astroquery.hitran import Hitran
from astropy import units as un
from astropy.constants import c, k_B, h, u
from molmass import Formula
from astropy import units as un
from scipy.optimize import curve_fit
import pdb as pdb

def convert_quantum_strings(hitran_data):
    '''
    Converts Vp, Vpp, Qp and Qpp quantum number strings to more useful format for analysis.
    Takes HITRAN values and saves them to new fields, e.g., 'Vp_HITRAN'
   
    Parameters
    ------------
    hitran_data : astropy table
    astropy table containing HITRAN data

    molecule_name : string
    Moleule name, e.g., 'CO'

    Returns
    ----------
    hitran_data : astropy table
    astropy table containing converted quantum number fields
    '''
    nlines=np.size(hitran_data)
    hitran_data.rename_column('Vp','Vp_HITRAN')
    hitran_data.rename_column('Vpp','Vpp_HITRAN')
    hitran_data.rename_column('Qp','Qp_HITRAN')
    hitran_data.rename_column('Qpp','Qpp_HITRAN')
    hitran_data['Vup']=np.zeros(nlines)
    hitran_data['Vlow']=np.zeros(nlines)
    hitran_data['Qup']=np.zeros(nlines)
    hitran_data['Qlow']=np.zeros(nlines)
    for i,myvp in enumerate(hitran_data['Vp_HITRAN']):
        if(hitran_data['molec_id'][i]==5):
            hitran_data['Vup'][i]=np.int(myvp)  #Upper level vibrational state
            hitran_data['Vlow'][i]=np.int(hitran_data['Vpp_HITRAN'][i])   #Lower level vibrational state
            type=(hitran_data['Qpp_HITRAN'][i].split())[0]   #Returns P or R  
            num=np.int((hitran_data['Qpp_HITRAN'][i].split())[1])
            hitran_data['Qlow'][i]=num  #Lower level Rotational state
            if(type=='P'): 
                hitran_data['Qup'][i]=num-1  #Upper level Rotational state for P branch
            if(type=='R'): 
                hitran_data['Qup'][i]=num+1  #Upper level Rotational state for R branch

    return hitran_data     


def strip_superfluous_hitran_data(hitran_data):
    '''
    Strips hitran_data astropy table of columns superfluous for IR astro spectroscopy

    Parameters
    ----------
    hitran_data : astropy table
    HITRAN data extracted by extract_hitran_data.  Contains all original columns from HITRAN.

    Returns    
    ----------
    hitran_data : astropy table
    HITRAN data stripped of some superfluous columns
    '''

    del hitran_data['sw']
    del hitran_data['gamma_air']
    del hitran_data['gamma_self']
    del hitran_data['n_air']
    del hitran_data['delta_air']
    del hitran_data['ierr1']
    del hitran_data['ierr2']
    del hitran_data['ierr3']
    del hitran_data['ierr4']
    del hitran_data['ierr5']
    del hitran_data['ierr6']
    del hitran_data['iref1']
    del hitran_data['iref2']
    del hitran_data['iref3']
    del hitran_data['iref4']
    del hitran_data['iref5']
    del hitran_data['iref6']
    del hitran_data['line_mixing_flag']
    return hitran_data        


def calc_linewidth(p,perr=None):
    '''
    Given Gaussian fit to Flux vs. wavelength in microns, find line width in km/s
   
    Parameters
    ----------
    p : numpy array
    parameters from Gaussian fit
    
    Returns                                                        
    ---------                                                             
    linewidth : float
    linewidth in km/s (FWHM) 

    '''
    linewidth_err=0*un.km/un.s
    linewidth=sigma_to_fwhm(p[2]/p[1]*c.value*1e-3*un.km/un.s)
    if(perr is not None): linewidth_err=sigma_to_fwhm(perr[2]/p[1]*c.value*1e-3*un.km/un.s)

    return (linewidth, linewidth_err)

def gauss3(x, a0, a1, a2):
    z = (x - a1) / a2
    y = a0 * np.exp(-z**2 / 2.)
    return y

def gauss4(x, a0, a1, a2, a3):
    z = (x - a1) / a2
    y = a0 * np.exp(-z**2 / 2.) + a3
    return y

def gauss5(x, a0, a1, a2, a3, a4):
    z = (x - a1) / a2
    y = a0 * np.exp(-z**2 / 2.) + a3 + a4 * x
    return y

def line_fit(wave,flux,nterms=4,p0=None,bounds=None):
    '''
    Take wave and flux values and perform a Gaussian fit

    Parameters
    ----------
    wave : numpy array
      Wavelength values in units of microns.  Should be an isolated line.  
    flux : numpy array
      Flux density values (F_nu) in units of Jy.  Should be an isolated line.

    Returns
    ---------
    linefit['parameters','yfit','resid'] : dictionary
       Dictionary containing fit parameters,fit values, and residual
    '''
    options={5:gauss5, 4:gauss4, 3:gauss3}
    fit_func=options[nterms]
    try:
        if(bounds is not None):
            fitparameters, fitcovariance = curve_fit(fit_func, wave, flux, p0=p0,bounds=bounds,absolute_sigma=True)
        else:
            fitparameters, fitcovariance = curve_fit(fit_func, wave, flux, p0=p0,absolute_sigma=True)
    except RuntimeError:
        print("Error - curve_fit failed")
        return -1
    perr = np.sqrt(np.diag(fitcovariance))
    fitoutput={"yfit":fit_func(wave,*fitparameters),"parameters":fitparameters,
               "covariance":fitcovariance,"resid":flux-fit_func(wave,*fitparameters),
               "parameter_errors":perr}
    return fitoutput

def extract_vup(hitran_data,vup_value):

    vbool=[]
    for myvp in hitran_data['Vp']:
        vbool.append(np.int(myvp)==vup_value)  
    out=hitran_data[vbool]
    return out

def calc_line_flux_from_fit(pfit, sigflux=None):
    '''
    Take parameters from line fit and compute a line flux and error

    Parameters
    ----------
    pfit : list
      fit parameters from Gaussian fit (pfit must have 3-5 elements)
    sigflux : float
      rms residual of fit, for computing error on line flux.

    Returns
    ---------
    (line flux, lineflux_err) : tuple of astropy quantities
       The line flux and the line flux error
    '''

    if(np.size(pfit)==5): [a0,a1,a2,a3,a4]=pfit
    if(np.size(pfit)==4): [a0,a1,a2,a3]=pfit
    if(np.size(pfit)==3): [a0,a1,a2]=pfit

#Add error catching for size of p<3 or >5

    nufit=c.value/(a1*1e-6)  #Frequency of line, in s-1
    lineflux=np.abs(a0)*1.e-26*np.sqrt(2*np.pi)*(np.abs(a2)*1.e-6*nufit**2./c.value)*un.W/un.m/un.m
    if(sigflux is not None):
        lineflux_err=1.e-26*np.sqrt(2.*np.pi)*1.e-6*nufit**2./c.value*np.abs(a2)*sigflux*un.W/un.m/un.m
    else:
        lineflux_err=None
    return (lineflux,lineflux_err)

def sigma_to_fwhm(sigma):
    '''
    Convert sigma to fwhm

    Parameters
    ----------
    sigma : float
       sigma of Gaussian distribution

    Returns
    ----------
    fwhm : float
       Full Width at Half Maximum of Gaussian distribution
    '''                        
    return  sigma*(2.*np.sqrt(2.*np.log(2.)))

def fwhm_to_sigma(fwhm):
    '''
    Convert fwhm to sigma

    Parameters
    ----------
    fwhm : float
       Full Width at Half Maximum of Gaussian distribution

    Returns
    ----------
    sigma : float
       sigma of Gaussian distribution
    '''                        

    return fwhm/(2.*np.sqrt(2.*np.log(2.)))

def wn_to_k(wn):
    '''                        
    Convert wavenumber to Kelvin

    Parameters
    ----------
    wn : AstroPy quantity
       Wavenumber including units

    Returns
    ---------
    energy : AstroPy quantity
       Energy of photon with given wavenumber

    '''              
    return wn.si*h*c/k_B

def extract_hitran_data(molecule_name, wavemin, wavemax, isotopologue_number=1, eupmax=None, aupmin=None):
    '''                                                               
    Extract data from HITRAN 
    Primarily makes use of astroquery.hitran, with some added functionality specific to common IR spectral applications
    Parameters 
    ---------- 
    molecule_name : string
        String identifier for molecule, for example, 'CO', or 'H2O'
    wavemin: float
        Minimum wavelength of extracted lines (in microns)
    wavemax: float
        Maximum wavelength of extracted lines (in microns)                   
    isotopologue_number : float, optional
        Number representing isotopologue (1=most common, 2=next most common, etc.)
    eupmax : float, optional
        Maximum extracted upper level energy (in Kelvin)
    aupmin : float, optional
        Minimum extracted Einstein A coefficient
    Returns
    ------- 
    hitran_data : astropy table
        Extracted data
    '''

    #Convert molecule name to number
    M = get_molecule_identifier(molecule_name)

    #Convert inputs to astroquery formats
    min_wavenumber = 1.e4/wavemax
    max_wavenumber = 1.e4/wavemin

    #Extract hitran data using astroquery
    tbl = Hitran.query_lines(molecule_number=M,isotopologue_number=isotopologue_number,min_frequency=min_wavenumber / un.cm,max_frequency=max_wavenumber / un.cm)

    #Do some desired bookkeeping, and add some helpful columns
    tbl.rename_column('nu','wn')
    tbl['nu']=tbl['wn']*c.cgs.value   #Now actually frequency of transition
    tbl['eup_k']=(wn_to_k((tbl['wn']+tbl['elower'])/un.cm)).value
    tbl['wave']=1.e4/tbl['wn']       #Wavelength of transition, in microns
    tbl.rename_column('global_upper_quanta','Vp')
    tbl.rename_column('global_lower_quanta','Vpp')
    tbl.rename_column('local_upper_quanta','Qp')
    tbl.rename_column('local_lower_quanta','Qpp')

    #Extract desired portion of dataset
    ebool = np.full(np.size(tbl), True, dtype=bool)  #default to True
    abool = np.full(np.size(tbl), True, dtype=bool)  #default to True
    #Upper level energy
    if(eupmax is not None):
        ebool = tbl['eup_k'] < eupmax
    #Upper level A coeff
    if(aupmin is not None):
        abool = tbl['a'] > aupmin
     #Combine
    extractbool = (abool & ebool)
    hitran_data=tbl[extractbool]

    hitran_data['a'].unit = '/ s'
    hitran_data['wn'].unit = '/ cm'
    hitran_data['nu'].unit = 'Hz'
    hitran_data['eup_k'].unit = 'K'
    hitran_data['wave'].unit = 'micron'
    hitran_data['elower'].unit = '/ cm'

    #Return astropy table
    return hitran_data

def get_global_identifier(molecule_name,isotopologue_number=1):
    '''                                                                                                                                
    For a given input molecular formula, return the corresponding HITRAN *global* identifier number.
    For more info, see https://hitran.org/docs/iso-meta/ 
                                                                                                                                       
    Parameters                                                                                                                         
    ----------                                                                                                                         
    molecular_formula : str                                                                                                            
        The string describing the molecule.              
    isotopologue_number : int, optional
        The isotopologue number, from most to least common.                                                                              
                                                                                                                                       
    Returns                                                                                                                            
    -------                                                                                                                            
    G : int                                                                                                                            
        The HITRAN global identifier number.                                                                                        
    '''

    mol_isot_code=molecule_name+'_'+str(isotopologue_number)

    trans = { 'H2O_1':1, 'H2O_2':2, 'H2O_3':3, 'H2O_4':4, 'H2O_5':5, 'H2O_6':6, 'H2O_7':129,
               'CO2_1':7,'CO2_2':8,'CO2_3':9,'CO2_4':10,'CO2_5':11,'CO2_6':12,'CO2_7':13,'CO2_8':14,
               'CO2_9':121,'CO2_10':15,'CO2_11':120,'CO2_12':122,
               'O3_1':16,'O3_2':17,'O3_3':18,'O3_4':19,'O3_5':20,
               'N2O_1':21,'N2O_2':22,'N2O_3':23,'N2O_4':24,'N2O_5':25,
               'CO_1':26,'CO_2':27,'CO_3':28,'CO_4':29,'CO_5':30,'CO_6':31,
               'CH4_1':32,'CH4_2':33,'CH4_3':34,'CH4_4':35,
               'O2_1':36,'O2_2':37,'O2_3':38,
               'NO_1':39,'NO_2':40,'NO_3':41,
               'SO2_1':42,'SO2_2':43,
               'NO2_1':44,
               'NH3_1':45,'NH3_2':46,
               'HNO3_1':47,'HNO3_2':117,
               'OH_1':48,'OH_2':49,'OH_3':50,
               'HF_1':51,'HF_2':110,
               'HCl_1':52,'HCl_2':53,'HCl_3':107,'HCl_4':108,
               'HBr_1':54,'HBr_2':55,'HBr_3':111,'HBr_4':112,
               'HI_1':56,'HI_2':113,
               'ClO_1':57,'ClO_2':58,
               'OCS_1':59,'OCS_2':60,'OCS_3':61,'OCS_4':62,'OCS_5':63,
               'H2CO_1':64,'H2CO_2':65,'H2CO_3':66,
               'HOCl_1':67,'HOCl_2':68,
               'N2_1':69,'N2_2':118,
               'HCN_1':70,'HCN_2':71,'HCN_3':72,
               'CH3Cl_1':73,'CH3CL_2':74,
               'H2O2_1':75,
               'C2H2_1':76,'C2H2_2':77,'C2H2_3':105,
               'C2H6_1':78,'C2H6_2':106,
               'PH3_1':79,
               'COF2_1':80,'COF2_2':119,
               'SF6_1':126,
               'H2S_1':81,'H2S_2':82,'H2S_3':83,
               'HCOOH_1':84,
               'HO2_1':85,
               'O_1':86,
               'ClONO2_1':127,'ClONO2_2':128,
               'NO+_1':87,
               'HOBr_1':88,'HOBr_2':89,
               'C2H4_1':90,'C2H4_2':91,
               'CH3OH_1':92,
               'CH3Br_1':93,'CH3Br_2':94,
               'CH3CN_1':95,
               'CF4_1':96,
               'C4H2_1':116,
               'HC3N_1':109,
               'H2_1':103,'H2_2':115,
               'CS_1':97,'CS_2':98,'CS_3':99,'CS_4':100,
               'SO3_1':114,
               'C2N2_1':123,
               'COCl2_1':124,'COCl2_2':125}
 
    return trans[mol_isot_code]

#Code from Nathan Hagen
#https://github.com/nzhagen/hitran
def translate_molecule_identifier(M):
    '''                                                                                                            
    For a given input molecule identifier number, return the corresponding molecular formula.                      
                                                                                                                   
    Parameters                                                                                                     
    ----------                                                                                                     
    M : int                                                                                                        
        The HITRAN molecule identifier number.                                                                     
                                                                                                                   
    Returns                                                                                                        
    -------                                                                                                        
    molecular_formula : str                                                                                        
        The string describing the molecule.                                                                        
    '''

    trans = { '1':'H2O',    '2':'CO2',   '3':'O3',      '4':'N2O',   '5':'CO',    '6':'CH4',   '7':'O2',     '8':'NO',
              '9':'SO2',   '10':'NO2',  '11':'NH3',    '12':'HNO3', '13':'OH',   '14':'HF',   '15':'HCl',   '16':'HBr',
             '17':'HI',    '18':'ClO',  '19':'OCS',    '20':'H2CO', '21':'HOCl', '22':'N2',   '23':'HCN',   '24':'CH3Cl',
             '25':'H2O2',  '26':'C2H2', '27':'C2H6',   '28':'PH3',  '29':'COF2', '30':'SF6',  '31':'H2S',   '32':'HCOOH',
             '33':'HO2',   '34':'O',    '35':'ClONO2', '36':'NO+',  '37':'HOBr', '38':'C2H4', '39':'CH3OH', '40':'CH3Br',
             '41':'CH3CN', '42':'CF4',  '43':'C4H2',   '44':'HC3N', '45':'H2',   '46':'CS',   '47':'SO3'}
    return(trans[str(M)])

#Code from Nathan Hagen
#https://github.com/nzhagen/hitran
def get_molecule_identifier(molecule_name):
    '''                                                                                                                                
    For a given input molecular formula, return the corresponding HITRAN molecule identifier number.                                   
                                                                                                                                       
    Parameters                                                                                                                         
    ----------                                                                                                                         
    molecular_formula : str                                                                                                            
        The string describing the molecule.                                                                                            
                                                                                                                                       
    Returns                                                                                                                            
    -------                                                                                                                            
    M : int                                                                                                                            
        The HITRAN molecular identifier number.                                                                                        
    '''

    trans = { '1':'H2O',    '2':'CO2',   '3':'O3',      '4':'N2O',   '5':'CO',    '6':'CH4',   '7':'O2',     '8':'NO',
              '9':'SO2',   '10':'NO2',  '11':'NH3',    '12':'HNO3', '13':'OH',   '14':'HF',   '15':'HCl',   '16':'HBr',
             '17':'HI',    '18':'ClO',  '19':'OCS',    '20':'H2CO', '21':'HOCl', '22':'N2',   '23':'HCN',   '24':'CH3Cl',
             '25':'H2O2',  '26':'C2H2', '27':'C2H6',   '28':'PH3',  '29':'COF2', '30':'SF6',  '31':'H2S',   '32':'HCOOH',
             '33':'HO2',   '34':'O',    '35':'ClONO2', '36':'NO+',  '37':'HOBr', '38':'C2H4', '39':'CH3OH', '40':'CH3Br',
             '41':'CH3CN', '42':'CF4',  '43':'C4H2',   '44':'HC3N', '45':'H2',   '46':'CS',   '47':'SO3'}
    ## Invert the dictionary.                                                                                                          
    trans = {v:k for k,v in trans.items()}
    return(int(trans[molecule_name]))
