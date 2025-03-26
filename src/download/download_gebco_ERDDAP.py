import xarray as xr

# Functions outside this script
from WITOIL_iMagine.src.utils.utils import *

def gebco_errdap(
    min_lat,
    max_lat,
    min_lon,
    max_lon,
    output_name=str,
    ):
    
    ds = xr.open_dataset('https://erddap.cmcc-opa.eu/erddap/griddap/Surf_f204_4c2a_5962')

    ds = ds.sel(latitude = slice(min_lat,max_lat), longitude = slice(min_lon,max_lon))

    # Rename variables only if they exist in the dataset
    ds = Utils.rename_netcdf_variables_mdk3(ds)

    ds.to_netcdf(output_name)