import pandas as pd
import numpy as np
import os
import h5py
import xarray as xr
from jug import TaskGenerator

runName = 'grad.absc.mid'
dataPath= f'/camp2e/gleung/aerograd/{runName}/'
anaPath = f'/camp2e/gleung/aerograd-analysis/{runName}/'

dr = 'mean_cross_section'
if not os.path.isdir(f"{anaPath}{dr}"):
    os.mkdir(f"{anaPath}{dr}")

paths = [f"{dataPath}{p}" for p in sorted(os.listdir(dataPath)) if (p.startswith('a-L') and p.endswith('.h5'))]
print(paths)
ccnname = 'BRC1MP'
#ccnname = 'CCCMP'
variables = ['RV','RCP','RSP','RPP',
	'THETA','WC','UC','VC',ccnname,
             'TCON',
             'FTHRD','LWDN','LWUP','SWDN','SWUP']


@TaskGenerator
def take_mean_cross_section(v, paths, savePath):
    out = np.zeros((len(paths),118,998))
    for i, p in enumerate(paths):
        print(p)
        with h5py.File(p) as f:
            if v == 'TCON':
                rtp = f['RTP'][1:119,1:999,1:999]
                rv = f['RV'][1:119,1:999,1:999]
                
                ovar = rtp - rv
            else:
                ovar = f[v][1:119,1:999,1:999]
            
            out[i,:,:] = np.nanmean(ovar, axis=2) #2 for mean over x, 1 for mean over y
        
    xr.DataArray(out, name=v, dims = ['time','z','y']).to_dataframe().to_pickle(f"{savePath}{v}.pkl")
    
for v in variables:
    take_mean_cross_section(v, paths, f"{anaPath}{dr}/")
