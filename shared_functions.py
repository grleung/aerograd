import os
import xarray as xr
import pandas as pd
import numpy as np
from run_params import p00,cp, rgas

def get_var(filePath, variables):
    #This function gets a variable/variables from the given file

    if os.path.exists(filePath):
        ds = xr.open_dataset(filePath)[variables]#,engine='h5netcdf').

        if len(ds.dims)==3:
            ds = ds.rename_dims({'phony_dim_2':'z', 'phony_dim_1':'y', 'phony_dim_0':'x'})
        elif len(ds.dims)==2:
            ds = ds.rename_dims({'phony_dim_1':'y', 'phony_dim_0':'x'})
        elif 'phony_dim_3' in ds.dims:
            ds = ds.rename_dims({'phony_dim_3':'patch','phony_dim_2':'z','phony_dim_1':'y', 'phony_dim_0':'x'})
        elif 'phony_dim_4' in ds.dims:
            ds = ds.rename_dims({'phony_dim_4':'k','phony_dim_2':'z','phony_dim_1':'y', 'phony_dim_0':'x'})

        return(ds)
    else:
        print(f'Error: file {filePath} does not exist.')

def calc_tcon(ds):
    ds = ds.assign(TCON = ds.RTP - ds.RV)
    ds = ds.assign(TCONnorain = ds.RCP + ds.RSP + ds.RPP)
    return(ds)

def calc_press(ds):
    ds = ds.assign(P = p00 * (ds.PI / cp) ** (cp/rgas))
    return(ds)


