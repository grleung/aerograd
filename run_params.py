import pandas as pd

nx = 1000
dx = 100
x = nx*dx/1000
ny = 1000
dy = 100
y = ny*dy/1000
nz = 120
dzrat = 1.015
dzmax = 200

dataPath = '/camp2e/gleung/aerograd/grad.500/'
nz = 120
p = f"{dataPath}/a-A-2019-09-16-000000-g1.h5"

def read_zs(dataPath, p, nz, var='ztn',varname='z'):
    save = False
    c = 0
    alts = []

    with open(f"{dataPath}/{p.split('/')[-1].split('.')[0][:-2]}head.txt") as f:
        lines = f.readlines()
        for line in lines:
            if f'__{var}' in line:
                save = True
                c = -2

            if save:
                if c>=0:
                    alts.append(float(line))
                c +=1

            if (c>nz-1):
                save=False
    
    alts = pd.Series(alts)
    alts.index.name = varname
    return(alts)

alt = read_zs(dataPath, p, nz)
dz = 1/read_zs(dataPath, p, nz, var ='dztn', varname='dz')
       
thresh = 1E-5 
g = 9.8065
eps = 0.622
cp = 1004
p00 = 100000
rgas = 287
lv = 2.5E6
