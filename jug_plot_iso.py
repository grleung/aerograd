import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import os
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib as mpl
from PIL import Image 
from jug import TaskGenerator

runName = 'grad.diurn-open'#rce.source.nowind.day'
#paths (change for this machine)
dataPath= f'/camp2e/gleung/aerograd/{runName}/'
anaPath = f'/camp2e/gleung/aerograd-analysis/{runName}/'

if not os.path.isdir(anaPath):
    os.mkdir(anaPath)

from palettable.cmocean.sequential import Deep_20
deep = mcolors.ListedColormap(Deep_20.mpl_colors)
lite = True

if lite:  
    paths = [f"{dataPath}{p}" for p in sorted(os.listdir(dataPath)) if p.startswith('a-L') and p.endswith('-g1.h5')]
else:
    paths = [f"{dataPath}{p}" for p in sorted(os.listdir(dataPath)) if p.startswith('a-A') and p.endswith('-g1.h5')]

from run_params import *
from shared_functions import *

get_var = TaskGenerator(get_var)
alt = alt/1000	

#isosurface plotting parameters (change for this model run)
elev = 20
azim = 45
sp = 0.1 #resolution of isosurfaces 


def cbar(x0,y0,dx,dy, plot, fig, label, orientation='vertical', ticks=[], ticker=mticker.MaxNLocator(), log=False):
    cax = fig.add_axes([x0, y0, dx,dy])
    
    if log:
        cbar = plt.colorbar(plot, cax=cax, ticks=ticks, orientation=orientation)
        if orientation == 'vertical':
            cbar.ax.yaxis.set_major_locator(mticker.LogLocator())
        else:
            cbar.ax.xaxis.set_major_locator(mticker.LogLocator())

    else:
        if len(ticks)!=0:
            if abs(min(ticks))<0.001:
                cbar = plt.colorbar(plot, cax=cax, ticks=ticks, orientation=orientation, format='%1.1e')
            else:
                cbar = plt.colorbar(plot, cax=cax, ticks=ticks, orientation=orientation)
        
        else:
            cbar = plt.colorbar(plot, cax=cax, orientation=orientation)
    
            if orientation == 'vertical':
                cbar.ax.yaxis.set_major_locator(ticker)
            else:
                cbar.ax.xaxis.set_major_locator(ticker)
                
    '''if orientation == 'vertical':
        cbar.ax.yaxis.set_major_formatter(formatter)
    else:
        cbar.ax.xaxis.set_major_formatter(formatter)'''
    
    cbar.set_label(label)
    
    cbar.outline.set_visible(False)
    
@TaskGenerator
def plot_iso(ds, thresh, elev, azim, sp, figPath, name, save=True):
    tcon = ds['TCON'].to_dataframe().reset_index()
    tcon['alt'] = tcon['z'].map(alt)
    tcon['x'] = tcon['x']*dx / 1000
    tcon['y'] = tcon['y']*dy / 1000

    tcon = tcon.set_index(['x','y','alt']).drop(columns=['z'])
    tcon = tcon.to_xarray().to_array().values[0]

    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(111, projection='3d')

    #scaling
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))

    try:
        verts, faces, normals, values = measure.marching_cubes(tcon, thresh, spacing=(sp,sp,sp))
        ax.plot_trisurf(verts[:,0],verts[:,1], faces, verts[:,2], color='gray',lw=1, zorder=1000)

    except:
        print('no surf')
        pass
    print('surfs calc')
    #This plots a total condensate isosurface for value = val and a grid of surface potential temperature


    surf = ds['CCCMP'].where(ds.z==1, drop=True).squeeze('z').stack(pos=('x','y'))
    
    c = ax.scatter(surf.x*sp*1.15 - 90, surf.y*sp*1.15 - 90, zs=-25,
                  c = surf.values*1000,cmap=deep,alpha=1, marker = 'o', s=0.5,
                   vmin=3E-6, vmax=3E-5,zorder=0)
    
    print('surf done')
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(0, dy*ny/1000)
    ax.set_ylim(0, dx*nx/1000)
    ax.set_zlim(0, 19)

    ax.set_xlabel('y (km)')
    ax.set_ylabel('x (km)')
    ax.set_zlabel('Altitude (km)')

    fig.subplots_adjust(top=2, bottom=0)
    
    '''surf = ds.sel(z=1).mean(dim='x')
    surf = surf.assign(dy = dy * surf.y)
    surf = surf.swap_dims({'y':'dy'})
    surf = surf.assign(Grad = surf.CCCMP.differentiate('dy'))
    surf = surf.to_dataframe()

    ax.scatter(0,surf.index/(1000), zs = (2*(surf.Grad/surf.Grad.max())**2).rolling(50,center=True).mean(), color='black')
    '''
    cbar(ax.get_position().x0,ax.get_position().y0+0.05,ax.get_position().x1-ax.get_position().x0,0.05, c, fig, 'Surface Aerosol Mixing Ratio (g kg$^{-1}$)', orientation='horizontal')    
    
    ax.text2D(0.55, 1.4, f'0.01 g/kg Total Condensate Mixing Ratio Isosurface (gray) \n{name[4:14]} {name[15:19]} UTC',fontsize=12, ha='center', va='center', transform=fig.transFigure)
     
    if save==True:
        plt.savefig(f"{figPath}{name}.png", dpi=300, bbox_inches='tight',pad_inches=0)

        #crops image manually because matplotlib 3d plotting won't allow for getting rid of whitespace
        im = Image.open(f"{figPath}{name}.png")
        width, height = im.size
        im1 = im.crop((0, height*0.25, width, height*1))
        im1.save(f"{figPath}{name}.png")

        im.close()
        im1.close()
    else:
        plt.show()

    plt.close('all')

@TaskGenerator
def calc_tcon(ds):
    ds = ds.assign(TCON=ds.RTP - ds.RV)
    return(ds)

if not os.path.isdir(f"{anaPath}isosurface-totcon/"):
    os.mkdir(f"{anaPath}isosurface-totcon/")

dr = f"{anaPath}isosurface-totcon/"

#plotting total condensate isosurfaces and integrated total condensate plan views        
for p in paths:
    print(p)
    name = p.split('/')[-1].split('.')[0]

    if True:#not os.path.exists(f"{dr}{name}"):
        ds = get_var(f"{dataPath}/{p.split('/')[-1]}", ['CCCMP','RTP','RV'])
        ds = calc_tcon(ds)
        plot_iso(ds, 1E-5, elev, azim, sp, dr, name, save=True)
   
    print(name)

