import pandas as pd
import numpy as np
import os
import h5py
import xarray as xr
from jug import TaskGenerator

variables = ['ACCPR',
            'SFLUX_T','SFLUX_R',#'PCPRR',#'RSHORT','RLONG','RLONGUP',
           'AODT']

runName = 'grad.absc.mid'
dataPath= f'/camp2e/gleung/aerograd/{runName}/'
anaPath = f'/camp2e/gleung/aerograd-analysis/{runName}/'

dr = 'mean_surf_flux'
if not os.path.isdir(f"{anaPath}{dr}"):
    os.mkdir(f"{anaPath}{dr}")

paths = [f"{dataPath}{p}" for p in sorted(os.listdir(dataPath)) if (p.startswith('a-L') and p.endswith('.h5'))]
print(paths)

@TaskGenerator
def take_mean_cross_section(v, paths, savePath):
    out = np.zeros((len(paths),998))
    for i, p in enumerate(paths):
        with h5py.File(p) as f:
            ovar = f[v][1:999,1:999]
            #ovar = f[v][:,:]
            out[i,:] = np.nanmean(ovar, axis=1)

    xr.DataArray(out, name=v, dims = ['time','y']).to_dataframe().to_pickle(f"{savePath}{v}.pkl")
    
for v in variables:
    take_mean_cross_section(v, paths, f"{anaPath}{dr}/")
