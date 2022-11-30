import os
import pandas as pd 
import numpy as np
import datetime as dt
import time
from shared_functions import get_var
from run_params import *

h0 = 4
h =12

run = 'grad.absc.mid'
ccn_name = 'BRC1MP'

anaPath = f"/camp2e/gleung/aerograd-analysis/"

wc = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/WC.pkl").reset_index()
uc = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/VC.pkl").reset_index()
ccn = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/{ccn_name}.pkl").reset_index()
ccn = ccn[ccn.time<=h*12+1]
wc = wc[wc.time<=h*12 + 1]
uc = uc[uc.time<=h*12 + 1]

tcon = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/RCP.pkl").RCP+pd.read_pickle(f"{anaPath}{run}/mean_cross_section/RSP.pkl").RSP+pd.read_pickle(f"{anaPath}{run}/mean_cross_section/RPP.pkl").RPP
tcon.columns = ['TCON']
tcon = tcon.reset_index()
tcon = tcon[tcon.time<=h*12 + 1]

swdn = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/SWDN.pkl").reset_index()
swdn = swdn[swdn.time<=h*12 + 1]

swup = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/SWUP.pkl").reset_index()
swup = swup[swup.time<=h*12 + 1]


lwdn = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/LWDN.pkl").reset_index()
lwdn = lwdn[lwdn.time<=h*12 + 1]

lwup = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/LWUP.pkl").reset_index()
lwup = lwup[lwup.time<=h*12 + 1]


fthrd = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/FTHRD.pkl").reset_index()
fthrd = fthrd[fthrd.time<=h*12 + 1]

data = tcon.copy()
data['WC'] = wc.WC
data[ccn_name] = ccn[ccn_name]
data['VC'] = uc.VC
data['y'] = ((data.y+1)*dx)/1000
data['y'] = data.y+0.05

data.columns = ['time','z','y',
                'TCON',
                'WC',ccn_name,'VC']
data['absx'] = abs(data.y-50)
data.loc[data.y-50<0,'VC'] = -data[data.y-50<0].VC 
data['SWDN'] = swdn.SWDN
data['SWUP'] = swup.SWUP
data['LWDN'] = lwdn.LWDN
data['LWUP'] = lwup.LWUP
data['FTHRD'] = fthrd.FTHRD
data['absx'] = 0.05 * (data.absx//0.05)


data = data[(data.time>h0*12) & (data.time <= 12*h + 1)]
mean = data.groupby(['time','z','absx']).mean()
mean = mean.reset_index()
plot = mean.groupby(['z','absx']).mean().reset_index()

resz = 0.25
resx = 4

winds = plot.copy()
winds['alt'] = (winds.z+1).map(alt/1000)
winds = winds.groupby([resx*round(winds.absx//resx), resz*round(winds.alt//resz)]).mean()[['VC','WC']]
winds = winds.reset_index()

winds = winds[(winds.alt>0) & (winds.alt<=14)]

winds.to_pickle(f"{anaPath}{run}/mean_wind-{h0}-{h}.pkl")
resz = 1
resx = 0.25

plot = plot.reset_index()
plot = plot.groupby([resx*round(plot.absx//resx), 'z']).mean()[['TCON',
                                                                'WC',ccn_name,
                                                                'SWDN','SWUP',
                                                                'LWDN','LWUP',
                                                                'FTHRD']]
plot = plot.reset_index()

plot.to_pickle(f"{anaPath}{run}/mean_cross_section-{h0}-{h}.pkl")

aod = pd.read_pickle(f"{anaPath}{run}/mean_surf_flux/AODT.pkl").reset_index()
shf = pd.read_pickle(f"{anaPath}{run}/mean_surf_flux/SFLUX_T.pkl").reset_index() 
lhf = pd.read_pickle(f"{anaPath}{run}/mean_surf_flux/SFLUX_R.pkl").reset_index()

surf = aod.copy()

surf['SHF'] = shf.SFLUX_T * cp
surf['LHF'] = lhf.SFLUX_R * lv
resx = 0.25

suba = surf[(surf.time<=12*h+1) & (surf.time>12*h0)]
suba['y'] = (((suba.y+1)*dx)/1000) + 0.05
suba['absx'] = resx * (abs(suba.y - 50) // resx)

suba = suba.groupby('absx').mean()

suba.to_pickle(f"{anaPath}{run}/mean_surf_flux-{h0}-{h}.pkl")

