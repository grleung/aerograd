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
    tcon.columns = ['TCON']
    tcon = tcon.reset_index()
    
    data = tcon
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
    
    mean = data.groupby(['time','z','abs_hdist']).mean().reset_index()
    plot = mean.groupby(['z','abs_hdist']).mean().reset_index()
    
    winds = plot.copy()
    winds['alt'] = (winds.z+1).map(alt/1000)
    winds = winds.groupby([resx*round(winds_resx.absx//winds_resx),
                           winds_resz*round(winds.alt//winds_resz)]).mean()[['VC','WC']]
    winds = winds.reset_index()

    winds = winds[(winds.alt>0) & (winds.alt<=14)]

    winds.to_pickle(f"{anaPath}{run}/mean_wind-{hours0}-{hours}.pkl")
    

    plot = plot.reset_index()
    plot = plot.groupby([out_resx*round(plot.absx//out_resx), 'z']).mean()[['TCON',
                                                                    'WC',ccn_name,
                                                                    'SWDN','SWUP',
                                                                    'LWDN','LWUP',
                                                                    'FTHRD']].reset_index()

    plot.to_pickle(f"{anaPath}{run}/mean_cross_section-{hours0}-{hours}.pkl")
    
    
@TaskGenerator
def mean_cross_surf(run, hours0=0, hours=12, 
                    resx = 0.25):
    data = pd.DataFrame()
    
    for var in ['AODT','SFLUX_T','SFLUX_R']:
        df = pd.read_pickle(f"{anaPath}{run}/mean_surf_flux/{var}.pkl").reset_index()
        df = df[(df.time>hours0) & (df.time<= hours*12 + 1)]
        
        if '_T' in var:
            data['y'] = df.y
            df[var] = df[var]*cp
        elif '_R' in var:
            df[var] = df[var]*lv
        
        data[var] = df[var]
        
    data['y'] = (((data.y+1)*dy)/1000) + 0.05
    data['abs_hdist'] = resx*(abs(data.y-50)//resx)
    
    data = data.groupby('abs_hdist').mean()

    data.to_pickle(f"{anaPath}{run}/mean_surf_flux-{hours0}-{hours}.pkl")
    
for run in runs:
    if 'absc' in run:
        ccn_name = 'BRC1MP'
    else:
        ccn_name = 'CCCMP'
        
    mean_cross_section(run,ccn_name)
    mean_cross_surf(run)

