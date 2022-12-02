import os
import pandas as pd 
import numpy as np
import datetime as dt
import time
from shared_functions import get_var
from run_params import *
from jug import TaskGenerator

runs = ['grad.1000','nograd.1000',
        'grad.500','nograd.500',
        'grad.1000.absc','nograd.1000.absc']

anaPath = f"/camp2e/gleung/aerograd-analysis/"

@TaskGenerator
def mean_cross_section(run, ccn_name, hours0=0, hours=12, 
                       out_resx = 0.25, winds_resz = 0.25, winds_resx = 4):
    tcon = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/RCP.pkl").RCP+pd.read_pickle(f"{anaPath}{run}/mean_cross_section/RSP.pkl").RSP+pd.read_pickle(f"{anaPath}{run}/mean_cross_section/RPP.pkl").RPP
    tcon = tcon.reset_index()
    
    data = tcon.copy()
    data.columns = ['time','z','y','TCON']
    data = data[(data.time>hours0) & (data.time<= hours*12 + 1)]
    
    for var in ['WC','VC',ccn_name,'SWDN','SWUP','LWDN','LWUP','FTHRD']:
        df = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/{var}.pkl").reset_index()
        df = df[(df.time>hours0) & (df.time<= hours*12 + 1)]
    
        data[var] = df[var]

    data['y'] = ((data.y+1)*dy)/1000
    data['y'] = data.y

    #absolute value of horizontal distance from center of gradient
    data['abs_hdist'] = abs(data.y - 50)
    data.loc[data.y-50 < 0, 'VC'] = -data[data.y-50 < 0].VC

    #bin horizontal distance
    data['abs_hdist'] = 0.05 * (data.abs_hdist//0.05)
    
    data = data.groupby(['time','z','abs_hdist']).mean().reset_index()
    plot = data.groupby(['z','abs_hdist']).mean().reset_index()
    
    winds = plot.copy()
    winds['alt'] = (winds.z+1).map(alt/1000)
    winds = winds.groupby([winds_resx*round(winds.abs_hdist//winds_resx),
                           winds_resz*round(winds.alt//winds_resz)]).mean()[['VC','WC']]
    winds = winds.reset_index()

    winds = winds[(winds.alt>0) & (winds.alt<=14)]
    winds.to_pickle(f"{anaPath}{run}/mean_wind-{hours0}-{hours}.pkl")
    
    plot = plot.reset_index()
    plot['abs_hdist'] = out_resx*round(plot.abs_hdist//out_resx)
    plot = plot.groupby(['abs_hdist', 'z']).mean().reset_index()

    plot.to_pickle(f"{anaPath}{run}/mean_cross_section-{hours0}-{hours}.pkl")
    
@TaskGenerator
def mean_cross_surf(run, ccn_name, hours0=0, hours=12, 
                    resx = 0.5):
    data = pd.DataFrame()
    
    for var in ['AODT','SFLUX_T','SFLUX_R','ACCPR','cld_cover']:
        df = pd.read_pickle(f"{anaPath}{run}/mean_surf_flux/{var}.pkl").reset_index()
        df = df[(df.time>hours0) & (df.time<= hours*12 + 1)]
        
        if '_T' in var:
            data['y'] = df.y
            df[var] = df[var]*cp
        elif '_R' in var:
            df[var] = df[var]*lv
        
        data[var] = df[var]
        
        
    data['y'] = ((data.y+1)*dy/1000) + 0.05
    data['abs_hdist'] = resx*(abs(data.y-50)//resx)
    
    data = data.groupby('abs_hdist').mean()
    
    #surface CCN
    df = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/{ccn_name}.pkl")
    df = df[(df.index.get_level_values(1)==0)].reset_index()
    df = df[(df.time>hours0) & (df.time<= hours*12 + 1)]
    df['y'] = ((df.y+1)*dy/1000 )+ 0.05
    df['abs_hdist'] = resx*(abs(df.y-50)//resx)
    df = df.groupby('abs_hdist').mean()
    data['SurfCCN'] = df[ccn_name]
        
    
    data.to_pickle(f"{anaPath}{run}/mean_surf_flux-{hours0}-{hours}.pkl")

@TaskGenerator
def mean_feature_count(run, hours0=0, hours=12, 
                    resx = 5, pcp_thresh=0.1):
    #raining features
    df = pd.read_hdf(f"{anaPath}{run}/tobac-out/w_features_track.h5",'table')
    df = df[(df.frame>hours0) & (df.frame<= hours*12 + 1)]
        
    df['mean_pcprr'] = df.feature.map(pd.read_pickle(f"{anaPath}{run}/tobac-out/pcprr_features.pkl").set_index('feature').mean_pcprr)
    df['maxcell_pcprr'] = df.cell.map(df.groupby('cell').mean_pcprr.mean())
    
    #minimum rainthreshold
    df = df[df.maxcell_pcprr*3600>pcp_thresh]
    
    #lifetime threshold
    df = df[df.lifetime>dt.timedelta(minutes=5)]
    
    df['abs_hdist'] = resx*(abs(((df.hdim_1 + 1) * dx/1000) + 0.05 - 50)//resx)
    
    bins = np.arange(0,50+resx,resx)
    
    out = pd.Series(np.histogram(df.abs_hdist, bins=bins)[0])
    out.index = (bins[1:]+bins[:-1])/2
    
    out.to_pickle(f"{anaPath}{run}/mean_feature_count-{hours0}-{hours}.pkl")
                  
for run in runs:
    if 'absc' in run:
        ccn_name = 'BRC1MP'
    else:
        ccn_name = 'CCCMP'
        
    #mean_cross_section(run,ccn_name)
    mean_cross_surf(run,ccn_name)
    mean_feature_count(run)
    

