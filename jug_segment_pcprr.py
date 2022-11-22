import os 
import xarray as xr
import iris
import numpy as np
import pandas as pd
import xarray as xr
from run_params import *
import tobac
import time
from jug import TaskGenerator
import trackpy as tp
import logging
import datetime as ddt
from scipy.ndimage import labeled_comprehension

dataPath = f"/camp2e/gleung/aerograd/"
anaPath = f"/camp2e/gleung/aerograd-analysis/"

runs = ['grad.1000.norad','nograd.1000.norad','grad.500.norad','nograd.500.norad']

parameters_segmentation={}
parameters_segmentation['method']='watershed'
parameters_segmentation['threshold']= 0.01/3600  # mm/hr converted rain rate
parameters_segmentation['PBC_flag']='both' 
@TaskGenerator
def run_segment(run):
    if not os.path.exists(f"{anaPath}{run}/tobac-out/"):
        os.mkdir(f"{anaPath}{run}/tobac-out/")
    
    Features = pd.read_hdf(f"{anaPath}{run}/tobac-out/w_features.h5",'table')
    Track = pd.read_hdf(f"{anaPath}{run}/tobac-out/w_features_track.h5",'table')

    paths = [f"{dataPath}{run}/{p}" for p in 
             sorted(os.listdir(f"{dataPath}{run}")) 
             if p.startswith('a-L-') and p.endswith('.h5') 
             and (pd.to_datetime(p.split('/')[-1][4:-6]) in Features.time.unique())]
    
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

    times = [(pd.to_datetime(p.split('/')[-1][4:-6])-ddt.datetime(year=2019,month=9,day=16))/ddt.timedelta(minutes=1) for p in sorted(paths)]
    times = iris.coords.DimCoord(times, 
                              standard_name='time', 
                                               units = 'minutes since 2019-09-16 00:00:00')
    
    
    #Segmentation
    pcprr = iris.load(paths,'PCPRR')

    for w,p in zip(pcprr, paths):
        w.add_aux_coord(iris.coords.AuxCoord(
                                pd.to_datetime(p.split('/')[-1][4:-6]),
                                standard_name='time'))

    pcprr = iris.cube.CubeList(pcprr).merge()[0] 

    pcprr.remove_coord('time')
    pcprr.add_dim_coord(times,0)
    pcprr.add_dim_coord(ys,1)
    pcprr.add_dim_coord(xs,2)
    pcprr.add_aux_coord(lat, data_dims=[1,2])
    pcprr.add_aux_coord(lon, data_dims=[1,2])

    dxy, dt = tobac.get_spacings(pcprr)
    mask,Features_pcprr=tobac.segmentation_2D(Features,
                                              pcprr,dxy,
                                              **parameters_segmentation)

    #Calculating Parameters
    if not (mask.coord("projection_x_coordinate").has_bounds()):
        mask.coord("projection_x_coordinate").guess_bounds()

    if not (mask.coord("projection_y_coordinate").has_bounds()):
        mask.coord("projection_y_coordinate").guess_bounds()

    area = np.outer(
        np.diff(mask.coord("projection_x_coordinate").bounds, axis=1),
        np.diff(mask.coord("projection_y_coordinate").bounds, axis=1),
            )

    Features_pcprr['raining_area'] = labeled_comprehension(
        area, mask.data, Features_pcprr["feature"], np.sum, area.dtype, np.nan
        )

    Features_pcprr['mean_pcprr'] = labeled_comprehension(
        pcprr.data, mask.data, Features_pcprr["feature"], np.mean, pcprr.data.dtype, np.nan
    )

    Features_pcprr['sum_pcprr'] = labeled_comprehension(
        pcprr.data, mask.data, Features_pcprr["feature"], np.sum, pcprr.data.dtype, np.nan
    )

    Features_pcprr.to_pickle(f"{anaPath}{run}/tobac-out/pcprr_features.pkl")    
    
for run in runs:
    if (os.path.exists(f"{anaPath}{run}/tobac-out/w_features_track.h5")) & (os.path.exists(f"{anaPath}{run}/tobac-out/w_features.h5")):
        run_segment(run)
