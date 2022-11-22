import os 
import xarray as xr
import iris
import numpy as np
import pandas as pd
import xarray as xr
from run_params import *
from shared_functions import get_var
import tobac
import time
from jug import TaskGenerator
import datetime as ddt
dataPath = f"/camp2e/gleung/aerograd/"
anaPath = f"/camp2e/gleung/aerograd-analysis/"

runs = ['nograd.500.norad','grad.500.norad','nograd.1000.norad','grad.1000.norad']

#Feature Detection
parameters_features={}
parameters_features['position_threshold']='weighted_diff'
parameters_features['min_distance']=0
parameters_features['sigma_threshold']=1
parameters_features['threshold']=[1,3,5] #m/s
parameters_features['n_erosion_threshold']=0
parameters_features['n_min_threshold']=10
parameters_features['PBC_flag'] = 'both'
parameters_features['vertical_coord'] = 'altitude'
#Tracking
parameters_linking={}
parameters_linking['method_linking']='predict'
parameters_linking['adaptive_stop']=0.2
parameters_linking['adaptive_step']=0.95
parameters_linking['extrapolate']=0
parameters_linking['order']=1
parameters_linking['subnetwork_size']=100
parameters_linking['memory']=0
parameters_linking['time_cell_min']=5*60
parameters_linking['method_linking']='predict'
parameters_linking['v_max']=10
parameters_linking['d_min']=2000  
parameters_linking['min_h1']=0  
parameters_linking['max_h1']=1000  
parameters_linking['min_h2']=0  
parameters_linking['max_h2']=1000  
parameters_linking['PBC_flag']='both' 
parameters_linking['vertical_coord'] = 'altitude'  

nx = 1000
ny = 1000
@TaskGenerator
def run_tobac(run):
    paths = [f"{dataPath}{run}/{p}" for p in 
             sorted(os.listdir(f"{dataPath}{run}")) 
             if p.startswith('a-L-') and p.endswith('.h5')]

    lat = iris.load(paths[0],'GLAT')[0]
    lon = iris.load(paths[0],'GLON')[0]

    xs = iris.coords.DimCoord(np.arange(0,nx*dx,dx), 
                              standard_name='projection_x_coordinate',
                             units = 'metre')
    ys = iris.coords.DimCoord(np.arange(0,ny*dy,dy), 
                              standard_name='projection_y_coordinate',
                             units = 'metre')
    zs = iris.coords.DimCoord(np.arange(0,nz,1), 
                              standard_name='model_level_number')

    lat = iris.coords.AuxCoord(lat.data,
                        standard_name='latitude',
                        units='degrees')

    lon = iris.coords.AuxCoord(lon.data,
                        standard_name='longitude',
                        units='degrees')
    
    altitude = iris.coords.AuxCoord(alt.values,
                        standard_name='altitude',
                        units='metre')

    times = iris.coords.DimCoord(np.arange(0,len(paths)*5,5), 
                              standard_name='time', 
                                               units = 'minutes since 2019-09-16 00:00:00')
    
    ws = iris.load(paths,'WC')

    for w,p in zip(ws, paths):
        w.add_aux_coord(iris.coords.AuxCoord(
                                pd.to_datetime(p.split('/')[-1][4:-6]),
                                standard_name='time'))

    ws = iris.cube.CubeList(ws).merge()[0] 

    ws.remove_coord('time')
    ws.add_dim_coord(times,0)
    ws.add_dim_coord(zs,1)
    ws.add_dim_coord(ys,2)
    ws.add_dim_coord(xs,3)
    ws.add_aux_coord(altitude, data_dims=1)
    ws.add_aux_coord(lat, data_dims=[2,3])
    ws.add_aux_coord(lon, data_dims=[2,3])
 
    dxy, dt = tobac.get_spacings(ws)
    
    Features=tobac.feature_detection_multithreshold(ws,dxy,**parameters_features)
    Track=tobac.linking_trackpy(Features,w,dt=dt,dxy=dxy,**parameters_linking)
    
    Track['lifetime'] = Track.cell.map(Track.groupby('cell').time_cell.max())
    
    Features = Features[Features.feature.isin(Track[(Track.lifetime>=ddt.timedelta(minutes=5))].feature)]
    
    Features.to_hdf(f"{anaPath}{run}/tobac-out/w_features.h5",'table')
    
    Track.to_hdf(f"{anaPath}{run}/tobac-out/w_features_track.h5",'table')

for run in runs:
    if not os.path.exists(f"{anaPath}{run}/tobac-out/"):
        os.mkdir(f"{anaPath}{run}/tobac-out/")
    
    run_tobac(run)
