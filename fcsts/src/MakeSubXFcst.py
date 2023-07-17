#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import pandas as pd

import os.path
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
parser = argparse.ArgumentParser()
parser.add_argument("--date",nargs='?',default=None,help="make subx forecasts based on this date")
args = parser.parse_args()

# ### Models and Forecast Settings

subxmodels_list,_,_,_,_=initSubxModels()
model_labels=[item['group']+'-'+item['model'] for item in subxmodels_list]
nweeks=4
interactive_vars=['tas','pr','zg']

# ### File paths

url='http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/'
datatype='forecast'
#hcstPath='/mcs/scratch/kpegion/subx/hindcast/'
hcstPath='/data/esplab/shared/model/initialized/subx/hindcast/'
#outPath='/mcs/scratch/kpegion/subx/figs_test/'
outPath='/data/esplab/shared/model/initialized/subx/forecast/weekly/'

# ### Fcst Date Handling

if (args.date):
    fcstdate,fcst_week_dates=getFcstDates(date=args.date)
else:
    fcstdate,fcst_week_dates=getFcstDates()

# ### Make a subx_fcst `xarray.Dataset` containing all models + MME for weeks 1-4

print('PROCESSING FCSTS FOR: ')

ds_models_list=[]
ds_modelsmin_list=[]
ds_modelsmax_list=[]

# Loop over all the SubX Models
for imodel,subx_model in enumerate(subxmodels_list):
    
    # Get the model, group, variables, and levels from the dictionary 
    varnames=subx_model['varnames']
    plevstrs=subx_model['plevstrs']
    model=subx_model['model']
    group=subx_model['group']
    
    print('===> '+group+'-'+model)
    
    # Loop over variables for this model
    ds_anoms_list=[]
    ds_emaxanoms_list=[]
    ds_eminanoms_list=[]
    
    for varname,plevstr in zip(varnames,plevstrs):
        
        # Read Data
        baseURL=url+'.'+group+'/.'+model+'/.'+datatype+'/.'+varname
        #inFname=baseURL+'/'+str(date.toordinal(fcstdate))+'/pop/dods' 
        inFname=baseURL+'/7000/pop/dods' 
        print(inFname)
        ds=xr.open_dataset(inFname)
        
        # Check if P is a coordinate and if so, select the Pressure Level
        if ('P' in list(ds[varname].dims)):
            ds=ds.sel(P=int(plevstr))
        
        # Remove hours,minutes,seconds from Initial Condition date
        ds['S']=ds['S'].dt.floor('d')
        
        # Identify dates in this forecast week
        dates_mask=np.isin(np.array(fcst_week_dates).astype('datetime64[ns]'),ds['S'])
        startdates=fcst_week_dates[dates_mask]
        
        # Make string versions of the dates in this forecast week
        sdates=startdates.strftime(('%Y%m%d'))
        mmdd=startdates.strftime(('%m%d'))
        
        # Select only the startdates in this fcst week
        ds=ds.sel(S=ds['S'].isin(np.array(startdates.values).astype('datetime64[ns]'))) 
        
        # Make sure there is data for this forecast week and this model, if not, skip
        if (ds['S'].size>0):
            
            # Drop any startdates with all missing
            ds=ds.dropna('S',how='all')
            
            # Make sure there is data for this forecast week and this model after dropping missing
            if (ds['S'].size>0):
                
                # Select the most recent available start date in this forecast week
                if group=='NRL': 
                    ds=preprocNRL(ds,startdates)
                elif group=='NCEP':
                    ds=preprocCFSv2(ds)
                else:
                    # Select the most recent non-missing start date in fcst week
                    ds=ds.sel(S=ds['S'][-1])
                    
                    # Get Subsetted Data Using Ingrid
                    ds=getDataViaIngrid(ds,baseURL)
                    
                    # Save the start date information to ic_dates
                    ds=ds.assign_coords({'ic_dates':pd.to_datetime(ds['S'].values).strftime('%Y%m%d')})

                # Create model as coordinates to keep track of which models are being used
                ds=ds.assign_coords({'model':group+'-'+model})    
                        
                # L got treated as a timeDelta on read in; convert to datetime64 based on start date
                ds['L']=ds['S'].values+ds['L'].dt.floor('d')
                
                # Rename coordinates & add attributes
                ds_out=ds.rename({'X':'lon','Y':'lat','L':'lead'})  
                ds_out['lon'].attrs['units']='degrees_east'
                ds_out['lat'].attrs['units']='degrees_north'
            
                # Get Climo and reset and re-name coordinates as needed to subtract from full field
                climo_fname=hcstPath+varname+plevstr+'/daily/climo/'+group+'-'+model+'/'+varname+'_'+group+'-'+model+'_'+mmdd[-1]+'.climo.p.nc'
                ds_clim=xr.open_dataset(climo_fname)
                ds_clim=ds_clim.rename({'time':'lead'})
                
            
                 # Special handling for GEFSv12 incorrect units for precip
                if (model=='GEFSv12'):
                    if (varname=='pr'):
                        ds_clim=ds_clim/86400.0
                    
                # Special handling for GEFSv12_CPC regarding leads
                elif (model=='GEFSv12_CPC'):
                    
                    # For this model, there are differences in the L dimension between the forecast
                    # and hindcast.  Align them to account for this.
                    leads=pd.date_range(ds['L'][0].values-pd.Timedelta(days=1),ds['L'][-1].values,freq='D')
                    ds_clim['lead']=leads
                    
                    if (varname=='pr'):
                        ds_clim=ds_clim/86400.0
                
                else:
                    ds_clim['lead']=ds['L']   
            
                # Calculate Ensemble Mean and Ensemble Mean Anomalies       
                ds_emean=ds_out.mean(dim='M').squeeze() 
                ds_anoms=ds_emean[varname]-ds_clim[varname]
                
                ds_emax=ds_out.max(dim='M').squeeze()
                ds_emaxanoms=ds_emax[varname]-ds_clim[varname]
                
                ds_emin=ds_out.min(dim='M').squeeze()
                ds_eminanoms=ds_emin[varname]-ds_clim[varname] 
                
                # Assign the number of ensemble members
                ds_anoms['nens']=len(ds['M'])
                ds_emaxanoms['nens']=len(ds['M'])
                ds_eminanoms['nens']=len(ds['M'])
                
                # Append anoms to list
                ds_anoms_list.append(ds_anoms)
                ds_emaxanoms_list.append(ds_emaxanoms)
                ds_eminanoms_list.append(ds_eminanoms)
                
            else:
                # Report that this model is missing
                print(model+' '+varname+' is missing for forecast date: ',fcstdate) 
        else:
            # Report that this model is missing
            print(model+' '+varname+' is missing for forecast date: ',fcstdate) 
    
    # Make list with all variables for this model and append to models list 
    if (ds_anoms_list):
        ds_models_list.append(xr.merge(ds_anoms_list))
        
    if (ds_emaxanoms_list):
        ds_modelsmax_list.append(xr.merge(ds_emaxanoms_list))
        
    if (ds_eminanoms_list):
        ds_modelsmin_list.append(xr.merge(ds_eminanoms_list))    


# Combine into dataset with all variables and all models
ds_models=xr.combine_nested(ds_models_list,concat_dim='model').persist()
ds_modelsmax=xr.combine_nested(ds_modelsmax_list,concat_dim='model').persist()
ds_modelsmin=xr.combine_nested(ds_modelsmin_list,concat_dim='model').persist() 

# Make Weekly Averages Consistent with CPC Forecast Week of Sat-Fri
ds_week=makeFCSTWeeks(ds_models,fcstdate,nweeks)
ds_weekmax=makeFCSTWeeks(ds_modelsmax,fcstdate,nweeks)
ds_weekmin=makeFCSTWeeks(ds_modelsmin,fcstdate,nweeks) 

# Make SubX-MME
ds_mme=ds_week.mean(dim='model')
ds_mme=ds_mme.assign_coords({'S':fcstdate,'model':'SUBX-MME',
                             'nens':ds_week['nens'].sum(),
                             'ic_dates':pd.to_datetime(fcstdate).strftime('%Y%m%d')})

ds_mme_max=ds_weekmax.max(dim='model')
ds_mme_max=ds_mme_max.assign_coords({'S':fcstdate,'model':'SUBX-MME',
                             'nens':ds_weekmax['nens'].sum(),
                             'ic_dates':pd.to_datetime(fcstdate).strftime('%Y%m%d')})

ds_mme_min=ds_weekmin.min(dim='model')
ds_mme_min=ds_mme_min.assign_coords({'S':fcstdate,'model':'SUBX-MME',
                             'nens':ds_weekmin['nens'].sum(),
                             'ic_dates':pd.to_datetime(fcstdate).strftime('%Y%m%d')})

# Combine dataset for individual models and MME into a single xarray.Dataset 
ds_subx_fcst=xr.concat([ds_week,ds_mme],dim='model').compute()
ds_subx_fcst_max=xr.concat([ds_weekmax,ds_mme_max],dim='model').compute()
ds_subx_fcst_min=xr.concat([ds_weekmin,ds_mme_min],dim='model').compute()

# Set global attributes
ds_subx_fcst=setattrs(ds_subx_fcst,fcstdate)
ds_subx_fcst_max=setattrs(ds_subx_fcst_max,fcstdate)
ds_subx_fcst_min=setattrs(ds_subx_fcst_min,fcstdate)

print()
print("SUBX FCST DATASET: ")
print(ds_subx_fcst)

# Write Files
print()
print("WRITING DATA")
subxWrite(ds_subx_fcst,fcstdate,'emean',outPath)
subxWrite(ds_subx_fcst_max,fcstdate,'emax',outPath)
subxWrite(ds_subx_fcst_min,fcstdate,'emin',outPath)

# Make Figures
figpath=outPath+fcstdate.strftime('%Y%m%d')+'/images/'
print()
print("MAKING STATIC FIGURES")
subxPlot(ds_subx_fcst,figpath)

# Make Interactive plots
print()
print("MAKING INTERACTIVE PLOTS")
subxIntPlot(ds_subx_fcst,interactive_vars,figpath)

# Print timing information
print()
print("TIME INFO: --- %s seconds ---" % (time.time() - start_time))

