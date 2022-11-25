import os
import pandas as pd 
import numpy as np
import datetime as dt
import time
from shared_functions import get_var
from run_params import *

h = 12

run = 'nograd.1000.absc'
anaPath = f"/camp2e/gleung/aerograd-analysis/"

dx  = 100

wc = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/WC.pkl").reset_index()
uc = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/VC.pkl").reset_index()
ccn = pd.read_pickle(f"{anaPath}{run}/mean_cross_section/BRC1MP.pkl").reset_index()
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
#data = wc.copy()
data['BRC1MP'] = ccn.BRC1MP
data['VC'] = uc.VC
data['y'] = ((data.y+1)*dx)/1000
data['y'] = data.y+0.05
#data = data[(data.y>=50) & (data.y<=9950)]

data.columns = ['time','z','y',
                'TCON',
                'WC','VC']
data['absx'] = abs(data.y-50)
data.loc[data.y-50<0,'VC'] = -data[data.y-50<0].VC 
data['SWDN'] = swdn.SWDN
data['SWUP'] = swup.SWUP
data['LWDN'] = lwdn.LWDN
data['LWUP'] = lwup.LWUP
data['FTHRD'] = fthrd.FTHRD
data['absx'] = 0.05 * (data.absx//0.05)


data = data[(data.time <= 12*h + 1) & (data.time>12*h1)]
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

winds.to_pickle(f"{anaPath}{run}/mean_wind-{h}.pkl")
resz = 1
resx = 0.25

plot = plot.reset_index()
plot = plot.groupby([resx*round(plot.absx//resx), 'z']).mean()[['TCON',
                                                                'WC','BRC1MP',
                                                                'SWDN','SWUP',
                                                                'LWDN','LWUP',
                                                                'FTHRD']]
plot = plot.reset_index()

plot.to_pickle(f"{anaPath}{run}/mean_cross_section-{h}.pkl")
