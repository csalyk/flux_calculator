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
from slabspec import write_flux_file

    out_all=extract_hitran_data('CO',4.6,5.2) 
    lineflux_data=extract_vup(out_all,1)
    size=np.size(lineflux_data)
    flux_table=calc_fluxes(wave,flux)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

