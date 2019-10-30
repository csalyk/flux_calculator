# flux_calculator
flux_calculator is a set of python codes to compute line fluxes from an IR spectrum.
Code is created with infrared medium/high-resolution spectroscopy in mind.  Users interested
in other applications should proceed with caution.

Users are requested to let the developer know if they are using the code.  Code has been
tested for only a few use cases, and users utilize at their own risk.

# Requirements
Requires internet access to utilize astroquery.hitran and access HITRAN partition function files

Requires the molmass and astropy packages

## Functions
extract_hitran_data extracts relevant data from HITRAN database

calc_fluxes calculates and writes fluxes

extract_vup extracts certain vup values from HITRAN dataset
## Usage

```python
from flux_calculator import extract_hitran_data, calc_fluxes, extract_vup

#Read in HITRAN data
out_all=extract_hitran_data('CO',4.6,5.2)  #astropy table
lineflux_data=extract_vup(out_all,1)

#Read in spectral data
data=pd.read_csv('reduced_spectra/nirspec_lkha330_glue.dat', header=26,names=['wave','flux'],
                 skipinitialspace=True,sep=' ')
wave=data['wave']
flux=data['flux']

out=calc_fluxes(wave,flux,lineflux_data,fwhm_v=17., sep_v=80., cont=1.12,vet_fits=False, plot=True, v_dop=15.)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

