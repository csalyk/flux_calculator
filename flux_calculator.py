import numpy as np
import pandas as pd
from .helpers import extract_hitran_data, extract_vup, line_fit, calc_linewidth, fwhm_to_sigma
from .helpers import calc_line_flux_from_fit,strip_superfluous_hitran_data, convert_quantum_strings
import matplotlib.pyplot as plt
from astropy.constants import c
from astropy.table import Table
import pdb as pdb

def make_rotation_diagram(lineflux_data):
    '''                                                                                                 
    Take ouput from calc_fluxes and use it to make a rotation diagram    
                                                                                                        
    Parameters                                                                                          
    ---------                                                                                           
    lineflux_data: astropy Table
        Output from calc_fluxes
                                                                                                        
    Returns                                                                                             
    --------                                                                                            
    rot_table: astropy Table                                                                            
        Table of x and y values for rotation diagram.                                                   
                                                                                                        
    '''
    x=lineflux_data['eup_k']
    y=np.log(lineflux_data['lineflux']/(lineflux_data['wn']*lineflux_data['gup']*lineflux_data['a']))
    rot_table = Table([x, y], names=('x', 'y'),  dtype=('f8', 'f8'))
    rot_table['x'].unit = 'K'

    return rot_table


def calc_fluxes(wave,flux,hitran_data, fwhm_v=20., sep_v=40.,cont=1.,verbose=True,vet_fits=False,
                plot=False,v_dop=0):
    '''                                                                                     
                                                                                            
    Parameters                                                                              
    ---------                                                                               
    wave : numpy array                                                                      
        wavelength values, in microns                                                       
    flux : numpy array                                                                      
        flux density values, in units of Jy                                
    htiran_data : astropy table
        output from extract_hitran
    fwhm_v : float, optional - defaults to 8 km/s
        estimate of line width in km/s for line fitting input
    sep_v : float, optional - defaults to 40 km/s
        total width used for line fits in km/s
    cont : float, optional - defaults to 1.
        Continuum level, in Jy.
    verbose: bool, optional - defaults to True
        True prints out some messages during runtime.
    vet_fits: bool, optional - defaults to False
        If True, user is prompted to decide if fit is good or not.
    plot: bool, optional - defaults to False
        If True, data and fits are plotted.  If vet_fits=True, this gets set to True automatically.
    v_dop : float, optional (defaults to 0)
        Doppler shift in km/s of spectrum relative to vacuum.  Note that this makes no assumptions about
         reference frame.

    Returns                                                                                 
    --------                                                                                
    lineflux_data : astropy table
       Table holding both HITRAN data and fit parameters (including flux, line width, and Doppler shift) 
        for fit lines.  
                                                                                            
    '''
    if(vet_fits==True): 
        plot=True
    lineflux_data=convert_quantum_strings(hitran_data)
    lineflux_data=strip_superfluous_hitran_data(lineflux_data)

    nlines=np.size(lineflux_data)
    #Add new columns to astropy table to hold line fluxes and error bars
    lineflux_data['lineflux']=np.zeros(nlines)  
    lineflux_data['lineflux_err']=np.zeros(nlines)  
    lineflux_data['linewidth']=np.zeros(nlines)
    lineflux_data['linewidth_err']=np.zeros(nlines)
    lineflux_data['v_dop_fit']=np.zeros(nlines)  
    lineflux_data['v_dop_fit_err']=np.zeros(nlines)  
    lineflux_data['continuum']=np.zeros(nlines)  
    lineflux_data['continuum_err']=np.zeros(nlines)  
    goodfit_bool=[True]*nlines
    #Loop through HITRAN wavelengths
    for i,w0 in enumerate(lineflux_data['wave']):   
        #Perform Gaussian fit for each line
        #Calculate Doppler shift, line width, and line separation in microns
        wdop=v_dop*1e3/c.value*w0
        dw=sep_v*1e3/c.value*w0
        dw2=2*sep_v*1e3/c.value*w0
        sig_w=fwhm_to_sigma(fwhm_v*1e3/c.value*w0)
        mybool=((wave>(w0+wdop-dw)) & (wave<(w0+wdop+dw)) & np.isfinite(flux))
        myx=wave[mybool]
        myy=flux[mybool]
        if((len(myx) <= 5) & (verbose==True) ):
            print('Not enough data near ', w0+wdop, ' microns. Skipping.')
            goodfit_bool[i]=False
        if(len(myx) > 5):
            g=line_fit(myx,myy,nterms=4,p0=[0.1,w0+wdop,sig_w,cont])
            if(g!=-1):   #curve fit succeeded
                p=g['parameters']
                perr=g['parameter_errors']
                resid=g['resid']
                sigflux=np.sqrt(np.mean(resid**2.))
                (lineflux,lineflux_err)=calc_line_flux_from_fit(p,sigflux=sigflux)
                lineflux_data['lineflux'][i]=lineflux.value
                lineflux_data['lineflux_err'][i]=lineflux_err.value
                lineflux_data['linewidth'][i]=np.abs((calc_linewidth(p,perr=perr))[0].value)
                lineflux_data['linewidth_err'][i]=np.abs((calc_linewidth(p,perr=perr))[1].value)
                lineflux_data['v_dop_fit'][i]=(p[1]-w0)/w0*c.value*1e-3   #km/s
                lineflux_data['v_dop_fit_err'][i]=(perr[1])/w0*c.value*1e-3   #km/s
                lineflux_data['continuum'][i]=(p[3])   #Jy
                lineflux_data['continuum_err'][i]=(perr[3])   #Jy

                if(plot==True):
                    fig=plt.figure(figsize=(10,3))
                    ax1=fig.add_subplot(111)
                    ax1.plot(wave,flux,'C0',linestyle='steps-mid',label='All data')
                    ax1.plot(myx,myy,'C1',linestyle='steps-mid',label='Fit data')
                    ax1.plot(myx,g['yfit'],'C2',label='Fit')
                    ax1.axvline(w0+wdop,color='C3',label='Line center')
                    ax1.set_xlim(np.min(myx)-dw2,np.max(myx)+dw2)
                    ax1.set_xlabel(r'Wavelength [$\mu$m]')
                    ax1.set_ylabel(r'F$_\nu$ [Jy]')
                    ax1.legend()
                    plt.show(block=False)
                    plt.close()    
                user_input=None
                if(vet_fits==True):
                    user_input=input("Is this fit okay? [y or n]")
                    while((user_input!='y') & (user_input!='n')):
                        user_input=input("Is this fit okay? Please enter y or n.") 
                if(user_input=='n'): 
                    goodfit_bool[i]=False
            if(g==-1):   #curve fit failed
                goodfit_bool[i]=False
                if(plot==True):
                    fig=plt.figure(figsize=(10,3))
                    ax1=fig.add_subplot(111)
                    ax1.plot(wave,flux,'C0',linestyle='steps-mid',label='All data')
                    ax1.plot(myx,myy,'C1',linestyle='steps-mid',label='Fit data')
                    ax1.axvline(w0+wdop,color='C3',label='Line center')
                    ax1.set_xlim(np.min(myx)-dw2,np.max(myx)+dw2)
                    ax1.set_xlabel(r'Wavelength [$\mu$m]')
                    ax1.set_ylabel(r'F$_\nu$ [Jy]')
                    ax1.legend()
                    plt.show(block=False)
                    plt.pause(0.5)
                    plt.close()    

    lineflux_data['lineflux'].unit = 'W / m2'
    lineflux_data['lineflux_err'].unit = 'W / m2'
    lineflux_data['linewidth'].unit = 'km / s'
    lineflux_data['linewidth_err'].unit = 'km / s'
    lineflux_data['v_dop_fit'].unit = 'km / s'
    lineflux_data['v_dop_fit_err'].unit = 'km / s'
    lineflux_data['continuum'].unit = 'Jy'
    lineflux_data['continuum_err'].unit = 'Jy'

    lineflux_data=lineflux_data[goodfit_bool]

    return lineflux_data
    
