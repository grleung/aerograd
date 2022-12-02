import pandas as pd
import numpy as np
import os
import h5py
import xarray as xr
from jug import TaskGenerator

runs = ['grad.1000.absc','nograd.1000.absc','grad.1000','nograd.1000','grad.500','nograd.500']

@TaskGenerator
def calc_cloud_cover(paths, savePath):
    out = np.zeros((len(paths),998))
    for i, p in enumerate(paths):
        with h5py.File(p) as f:
            rtp = f['RTP'][1:119,1:999,1:999]
            rv = f['RV'][1:119,1:999,1:999]
                
            ovar = rtp - rv
            
            ovar = np.where(ovar>=1E-5, ovar, 0)
            ovar = np.count_nonzero(ovar, axis=0)
            out[i,:] = np.count_nonzero(ovar, axis=1)/(998)
            
    xr.DataArray(out, name='cld_cover', dims = ['time','y']).to_dataframe().to_pickle(f"{savePath}cld_cover.pkl")
    
for runName in runs:
    dataPath= f'/camp2e/gleung/aerograd/{runName}/'
    anaPath = f'/camp2e/gleung/aerograd-analysis/{runName}/'

    dr = 'mean_surf_flux'
    if not os.path.isdir(f"{anaPath}{dr}"):
        os.mkdir(f"{anaPath}{dr}")

    paths = [f"{dataPath}{p}" for p in sorted(os.listdir(dataPath)) if (p.startswith('a-L') and p.endswith('.h5'))]
    print(paths)

    calc_cloud_cover(paths, f"{anaPath}{dr}/")
