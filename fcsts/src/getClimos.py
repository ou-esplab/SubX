#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import pandas as pd

import os
from datetime import datetime, timedelta, date
import time
import argparse

import matplotlib.pyplot as plt
import proplot as pplt

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature

from subx_utils import *

# Eliminate Warnings
import warnings
warnings.filterwarnings("ignore")

# Set xarray to keep attributes
xr.set_options(keep_attrs=True)

# Get Start Time
start_time = time.time()

# Parse commend line arguments
#parser = argparse.ArgumentParser()
#parser.add_argument("--date",nargs='?',default=None,help="make subx forecasts based on this date")
#args = parser.parse_args()

# ### Models and Forecast Settings

_,subxclimos_list,_,_=initSubxModels()
model_labels=[item['group']+'-'+item['model'] for item in subxclimos_list]
nweeks=4

# ### File paths

url='http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/'
datatype='hindcast'
#hcstPath='/shared/subx/hindcast/'
hcstPath='/mcs/scratch/kpegion/subx/hindcast/'


print('PROCESSING CLIMOS FOR: ')

# Loop over all the SubX Models
ds_models_list=[]
for imodel,subx_model in enumerate(subxclimos_list):
    
    # Get the model, group, variables, and levels from the dictionary 
    varnames=subx_model['varnames']
    plevstrs=subx_model['plevstrs']
    model=subx_model['model']
    group=subx_model['group']
    
    print('===> '+group+'-'+model)
    
    # Loop over variables for this model
    ds_anoms_list=[]
    for varname,plevstr in zip(varnames,plevstrs):
        
        # Read Data
        baseURL=url+'.'+group+'/.'+model+'/.'+datatype+'/.dc9915/.'+varname
        inFname=baseURL+'/dods'  
        print(inFname)
        ds_tmp=xr.open_dataset(inFname,chunks={'S':'500MB'})
 
        # Remove hours,minutes,seconds from Initial Condition date
        #ds_tmp['S']=ds_tmp['S'].dt.floor('d')

        # Get list of startdates 
        startdates=ds_tmp['S'].values

        for s in startdates:
            tmp1,tmp2=s.astype(str).split('T')
            yyyy,mm,dd=tmp1.split('-')
            mmdd=mm+dd

            if ('P' in list(ds_tmp[varname].dims)):
                ds=ds_tmp.sel(S=s,P=int(plevstr)) 
            else:
                ds=ds_tmp.sel(S=s)
            print(ds)

            print("GETTING DATA VIA INGRID")
            ds_out=getDataViaIngrid(ds,baseURL)
            print(ds_out)

            # Rename coordinates & add attributes
            ds_out['S']=ds_out['S'].dt.floor('d')
            ds_out=ds_out.rename({'X':'lon','Y':'lat','L':'time'})  
            ds_out['time']=np.arange(len(ds_out['time']))
            ds_out['lon'].attrs['units']='degrees_east'
            ds_out['lat'].attrs['units']='degrees_north'
            ds_out=ds_out.squeeze(drop=True)

            # Set files & paths for writing
            climo_path=hcstPath+varname+plevstr+'/daily/climo/'+group+'-'+model
            print()
            print(climo_path)
            if not os.path.exists(climo_path):
                print("MAKING CLIMO PATH: ")
                os.makedirs(climo_path)

            climo_fname=climo_path+'/'+varname+'_'+group+'-'+model+'_'+mmdd+'.climo.p.nc'
            print(climo_fname)

            # Write Files
            print()
            print("WRITING DATA")
            ds_out.to_netcdf(climo_fname)


print()
print("TIME INFO: --- %s seconds ---" % (time.time() - start_time))

