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

## Usage

```python
from slabspec import write_flux_file
flux_table=write_fluxes(wave,flux)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

