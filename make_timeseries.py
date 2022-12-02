import pandas as pd
import numpy as np
import datetime as dt
from run_params import dx,dy,nx,ny

runs = ['grad.1000','nograd.1000',
        'grad.500','nograd.500',
        'grad.1000.absc','nograd.1000.absc']

anaPath = f"/camp2e/gleung/aerograd-analysis/"

def timeseries_features(run, pcp_thresh=0.1, hours=12):
    df = pd.read_hdf(f"{anaPath}{run}/tobac-out/w_features_track.h5",'table')
    df['mean_pcprr'] = df.feature.map(pd.read_pickle(f"{anaPath}{run}/tobac-out/pcprr_features.pkl").set_index('feature').mean_pcprr)
    df['maxcell_pcprr'] = df.cell.map(df.groupby('cell').mean_pcprr.mean())
    
    #minimum rainthreshold
    df = df[df.maxcell_pcprr*3600>pcp_thresh]
    
    #lifetime threshold
    df = df[df.lifetime>dt.timedelta(minutes=5)]
    
    df = df.groupby('frame').feature.count()
    df = df[df.index<=hours*12 + 1]
    
    df = df.append(pd.Series(np.zeros(df.index[0]))).sort_index()
    
    return(df)

def timeseries_totalrain(run, hours=12):
    df = pd.read_pickle(f"{anaPath}{run}/mean_surf_flux/ACCPR.pkl").reset_index()
    df = df.groupby('time').ACCPR.mean()

    df = df[df.index<12*hours +1] * (dx * dy *nx * ny)/1E9
    
    return(df)

def timeseries_cloudcover(run, hours=12):
    df = pd.read_pickle(f"{anaPath}{run}/mean_surf_flux/cld_cover.pkl").reset_index()
    df = df.groupby('time').cld_cover.mean()

    df = df[df.index<12*hours +1]
    
    return(df)

for run in runs:
    df = pd.DataFrame()
    df['RainingUpdrafts'] = timeseries_features(run)
    df['AccRainingUpdrafts'] = df.RainingUpdrafts.cumsum()
    df['AccRain'] = timeseries_totalrain(run)
    df['CloudCover'] = timeseries_cloudcover(run)
    
    df.to_pickle(f"{anaPath}{run}/timeseries.pkl")
