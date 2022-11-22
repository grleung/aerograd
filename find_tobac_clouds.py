import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import os
import iris
from jug import TaskGenerator

from run_params import *
from shared_functions import get_var

import tobac
import time

dataPath = f"/camp2e/gleung/aerograd/"
anaPath = f"/camp2e/gleung/aerograd-analysis/"

runs = ['emit.diurn']#,'sulf.3000']

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

@TaskGenerator
def find_clouds(run):
    Features = pd.read_hdf(f"{anaPath}{run}/tobac-out/w_features.h5",'table')
    Track= pd.read_hdf(f"{anaPath}{run}/tobac-out/w_features_track.h5",'table')

    mask = xr.open_dataset(f"{anaPath}{run}/tobac-out/tcon_3d_segmentation_mask.nc")

    Clouds = pd.DataFrame(columns=[])

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

    times = iris.coords.DimCoord(np.arange(0,len(paths)*5,5),
                              standard_name='time',
                                               units = 'minutes since 2019-09-16 00:00:00')

    for t in mask.time.values:
        path = pd.to_datetime(t).strftime('a-L-%Y-%m-%d-%H%M%S-g1.h5')
        print(path)

        ws = iris.load(f"{dataPath}{run}/{path}",'WC')[0]

        ws.add_dim_coord(zs,0)
        ws.add_dim_coord(ys,1)
        ws.add_dim_coord(xs,2)
        ws.add_aux_coord(lat, data_dims=[1,2])
        ws.add_aux_coord(lon, data_dims=[1,2])

        ws = xr.DataArray.from_iris(ws)

        submask = mask.sel(time=pd.to_datetime(t)).segmentation_mask

        feats = np.delete(np.unique(submask.data),0)
        feats = Track[(Track.feature.isin(feats)) & 
                      (Track.time_cell>=dt.timedelta(minutes=5))].feature.values


        for f in feats:
            x = submask.where(submask==f).dropna('model_level_number',how='all').dropna('projection_y_coordinate',how='all').dropna('projection_x_coordinate',how='all')

            clevs = x.dropna('model_level_number', how='all').model_level_number.values

            #subset the vertical velocity for the horizontal extent of the condensate mask
            sub = ws.sel(projection_x_coordinate=(x.projection_x_coordinate.values),
                     projection_y_coordinate=(x.projection_y_coordinate.values))
            #find vertical level of maximum w within the horizontal extent of condesnate mask
            wlev = sub.argmax(dim=...).get('model_level_number').values

            if 0 in clevs: #not including the surface
                clevs = np.delete(clevs,0)

            if (len(consecutive(clevs))) == 1: #it's one contiguous column
                if (wlev >= clevs.min()) & (wlev <= clevs.max()): #make sure that the wmax is inside the cloud area -- if not, doesn't get included
                    Clouds.loc[f,'top'] = clevs.max()
                    Clouds.loc[f,'bot'] = clevs.min()

                    Clouds.loc[f,'CTH']= alt[clevs.max()]/1000
                    Clouds.loc[f,'CBH'] = alt[clevs.min()]/1000

                    Clouds.loc[f,'dvertical'] = False
                    Clouds.loc[f,'maxW'] = ws.where(~x.isnull()).max(...).values
            else: #if two vertically stacked condensate features
                print(f, consecutive(clevs))
                print(wlev)

                clevs = consecutive(clevs)
                clevs2 = [c for c in clevs if wlev in c]

                if len(clevs2)==1:
                    clevs2 = clevs2[0]
                    #clouds[(clouds.segmentation_mask==f) & (~clouds.index.get_level_values(0).isin(clevs))] = 0

                    Clouds.loc[f,'top'] = clevs2.max()
                    Clouds.loc[f,'bot'] = clevs2.min()

                    Clouds.loc[f,'CTH']= alt[clevs2.max()]/1000
                    Clouds.loc[f,'CBH'] = alt[clevs2.min()]/1000

                    #gives the minimum separation of another cloud layer from this one
                    Clouds.loc[f,'dvertical'] = True
                    Clouds.loc[f,'maxW'] = ws.where(~x.isnull()).max(...).values
                else:
                    print(clevs)
                    print(len(clevs))

    Clouds['cell'] = Clouds.index.map(Track.set_index('feature').cell)
    Clouds['time_cell'] = Clouds.index.map(Track.set_index('feature').time_cell)
    Clouds['time'] = pd.to_datetime(Clouds.index.map(Track.set_index('feature').timestr))

    Clouds.to_hdf(f"{anaPath}{run}/tobac-out/cloud_features.h5",'table')
    
for run in runs:
    find_clouds(run)
