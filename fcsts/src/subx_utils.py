import xarray as xr
import numpy as np
import pandas as pd
import os
import os.path
from datetime import datetime, timedelta, date

import matplotlib.pyplot as plt
import proplot as pplt

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature

import hvplot
import hvplot.xarray
import hvplot.pandas

import panel as pn
import panel.widgets as pnw

import geoviews
import geoviews.feature as gf

xr.set_options(keep_attrs=True)  

def initSubxModels():
    
    #all_varnames=['ua','ua','rlut','tas','ts','zg','va','va','pr','zg','uas','vas','psl']
    #all_plevstrs=['850','200','toa','2m','sfc','500','200','850','sfc','200','10m','10m','msl']
    #all_units=['ms-1','ms-1','Wm-2','degC','degC','m','ms-1','ms-1','mmday-1','m','ms-1','ms-1','hPa']
    
    all_varnames=['tas','pr','zg']
    all_plevstrs=['2m','sfc','500']
    all_units=['degC','mmday-1','m']
   
    sub1_varnames=['tas','pr','zg']
    sub1_plevstrs=['2m','sfc','500']
    sub1_units=['degC','mmday-1','m']
    
    ccsm4_dict={'model':'CCSM4','group':'RSMAS','varnames': all_varnames, 'plevstrs': all_plevstrs, 'plot_loc':2}
    geos_dict={'model':'GEOS_V2p1','group':'GMAO','varnames': all_varnames, 'plevstrs': all_plevstrs,'plot_loc':4}
    fim_dict={'model':'FIMr1p1','group':'ESRL','varnames': all_varnames, 'plevstrs': all_plevstrs,'plot_loc':1}
    #geps_dict={'model':'GEPS6','group':'ECCC','varnames': all_varnames, 'plevstrs': all_plevstrs,'plot_loc':6}
    geps_dict={'model':'GEPS7','group':'ECCC','varnames': all_varnames, 'plevstrs': all_plevstrs,'plot_loc':6}
    nrl_dict={'model':'NESM','group':'NRL','varnames': all_varnames, 'plevstrs': all_plevstrs,'plot_loc':5}
    gefs_dict={'model':'GEFSv12_CPC','group':'EMC','varnames': sub1_varnames, 'plevstrs': sub1_plevstrs,'plot_loc':3}
    #gefs_dict={'model':'GEFSv12','group':'EMC','varnames': sub1_varnames, 'plevstrs': sub1_plevstrs,'plot_loc':3}
    cfsv2_dict={'model':'CFSv2','group':'NCEP','varnames': sub1_varnames, 'plevstrs': sub1_plevstrs,'plot_loc':7}

    subxclimos_list=[fim_dict,ccsm4_dict,geos_dict,nrl_dict,geps_dict,gefs_dict,cfsv2_dict]
    subxmodels_list=[fim_dict,ccsm4_dict,geos_dict,nrl_dict,geps_dict,gefs_dict,cfsv2_dict]
   
    return subxmodels_list, subxclimos_list,all_varnames, all_plevstrs, all_units

def initPlotParams():

    # Dictionary defining parameters for plottting variables
    clevs_tas=[-4,-3,-2.5,-2,-1.5,-1,-0.5,-0.2,0.2,0.5,1,1.5,2,2.5,3,4]
    clevs_pr=[-100,-50,-25,-10,-5,-2,2,5,10,25,50,100]
    clevs_zg=[-50,-45,-40,-35,-30,-25,-20,-15,15,20,25,30,35,40,45,50]
 
    ## Estimated clevs, going to change with trial and error
    #clevs_ua=[-25,-20,-15,-10,-5,0,5,10,15,20,25]
    #clevs_ua2=[-25,-20,-15,-10,-5,0,5,10,15,20,25]
    #clevs_rlut=[100,120,140,160,180,200,220,240,260,280,300,320,340]
    #clevs_ts=[-4,-3,-2.5,-2,-1.5,-1,-0.5,-0.2,0.2,0.5,1,1.5,2,2.5,3,4]
    #clevs_va=[-25,-20,-15,-10,-5,0,5,10,15,20,25]
    #clevs_va2=[-25,-20,-15,-10,-5,0,5,10,15,20,25]
    #clevs_uas=[-20,-15,-10,-5,0,5,10,15,20]
    #clevs_vas=[-20,-15,-10,-5,0,5,10,15,20]
    #clevs_psl=[-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
    
    degC = u'\N{DEGREE SIGN}'+'C'
    tas_dict={'name':'tas','plev':'2m','label':'2m Temperature','outname':'2mTemp',
              'clevs':clevs_tas,'clims':(clevs_tas[0],clevs_tas[-1]),'cmap':'ColdHot','units':degC,
              'regions':['Global','NorthAmerica']}
    pr_dict={'name':'pr','plev':'sfc','label':'Total Precipitation','outname':'Precip',
              'clevs':clevs_pr,'clims':(clevs_pr[0],clevs_pr[-1]),'cmap':'DryWet','units':'mm',
              'regions':['Global','NorthAmerica']}
    zg_dict={'name':'zg','plev':'500',
             'label':'500hPa Geopotential Height',
             'outname':'500hPaGeopotentialHeight',
             'clevs':clevs_zg,'clims':(clevs_zg[0],clevs_zg[-1]),
             'cmap':'NegPos','units':'m',
             'regions':['NorthernHemisphere']}
   # ua_dict={'name':'ua','plev':'850','label':'850 hPa Zonal Velocity','outname':'850hPaZonal',
   #           'clevs':clevs_ua,'cmap':'ColdHot','units':'${^oC}$',
   #           'regions':['Global','NorthAmerica']}
   # ua2_dict={'name':'ua','plev':'200','label':'200 hPa Zonal Velocity','outname':'200hPaZonal',
   #           'clevs':clevs_ua250,'cmap':'ColdHot','units':'${^oC}$',
   #           'regions':['Global','NorthAmerica']}
   # rlut_dict={'name':'rlut','plev':'toa','label':'Outgoing Longwave Radiation at Top of Atmosphere','outname':'LongwaveAtToa',
   #           'clevs':clevs_rlut,'cmap':'ColdHot','units':'${^oC}$',
   #           'regions':['Global','NorthAmerica']}
   # ts_dict={'name':'ts','plev':'sfc','label':'Surface Temperature','outname':'SfcTemp',
   #           'clevs':clevs_ts,'cmap':'ColdHot','units':'${^oC}$',
   #           'regions':['Global','NorthAmerica']}
   # va_dict={'name':'va','plev':'850','label':'850 hPa Meridional Velocity','outname':'850hPaMeridional',
   #           'clevs':clevs_va,'cmap':'ColdHot','units':'${^oC}$',
   #           'regions':['Global','NorthAmerica']}
   # va2_dict={'name':'va','plev':'200','label':'200 hPa Meridional Velocity','outname':'200hPaMeridional',
   #           'clevs':clevs_va250,'cmap':'ColdHot','units':'${^oC}$',
   #           'regions':['Global','NorthAmerica']}
   # uas_dict={'name':'uas','plev':'10m','label':'10m Eastward Velocity','outname':'10mEast',
   #           'clevs':clevs_tas,'cmap':'ColdHot','units':'${^oC}$',
   #           'regions':['Global','NorthAmerica']}
   # vas_dict={'name':'vas','plev':'10m','label':'10m Northward Velocity','outname':'10mNorth',
   #           'clevs':clevs_tas,'cmap':'ColdHot','units':'${^oC}$',
   #           'regions':['Global','NorthAmerica']}
   # psl_dict={'name':'psl','plev':'sfc','label':'Mean Sea Level Pressure (MSLP)','outname':'MSLP',
   #           'clevs':clevs_tas,'cmap':'ColdHot','units':'${^oC}$',
   #           'regions':['Global','NorthAmerica']}
    
    var_params_dict=[tas_dict,pr_dict,zg_dict]
    #var_params_dict=[tas_dict, pr_dict, zg_dict, ua_dict, ua2_dict, rlut_dict, ts_dict, va_dict, va2_dict, uas_dict, vas_dict, psl_dict]] contains all new variables
    
    # Dictionary defining parameters for plotting different regions
    
    global_dict={'name':'Global','lons':(0,360),'lats':(-90,90),'clon':0,'mproj':'robin',
                'state_colors':'gray5'}
    na_dict={'name':'NorthAmerica','lons':(190,305),'lats':(15,75),'clon':247.5,'mproj':'pcarree',
            'state_colors':'k'}
    nh_dict={'name':'NorthernHemisphere','lons':(-270,90),'lats':(30,90),'clon':247.5,'mproj':'npstere',
            'state_colors':'gray5','state_colors':'gray5'}
    
    reg_params_dict=[global_dict,na_dict,nh_dict]
    
    return var_params_dict, reg_params_dict

def makeWebImages(ds_subx_fcst,v,unit,vl,clevs,cmap,lonreg,latreg,clon,mproj,statescolor,figname):
   
    # Proplot makes high res pics so need to reduce for web pics
    pplt.rc.savefigdpi = 100
    
    # Get SubX Models List for plot locations
    subxmodels_list,_,_,_,_=initSubxModels()   

    # Loop over all weeks -- one figure per week
    for iweek,week in enumerate(ds_subx_fcst['week'].values):
    
        # Set the plot grid and create the subplot container
        grid = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 0]]
        f, axs = pplt.subplots(grid,
                               proj=mproj,proj_kw={'lon_0': clon},
                               width=11,height=8.5)
 
        wend_str=ds_subx_fcst.attrs['week_end'][iweek].strftime('%b %-d')
        suptitle='SubX Week '+str(week)+' '+ vl+' Anomalies ('+unit+'): Valid week ending '+wend_str
    
        # Loop over all models
        for imodel,subx_model in enumerate(ds_subx_fcst['model'].values):
            
            if (subx_model=='SUBX-MME'):
                iplot=7
            else:
                model_dict=next(item for item in subxmodels_list if item['model'] == subx_model.split('-')[1])
                iplot=model_dict['plot_loc']-1
            
            # Select the model
            ds=ds_subx_fcst.sel(model=subx_model,week=week)
            
            # Get the startdate for this model

            startdate=ds['S'].dt.strftime('%m/%d').values
    
            # Define titles for individual subplot panels
            nens_str=str(ds['nens'].values.astype(int))
            if (subx_model=='SUBX-MME'):
                title=subx_model+' ('+nens_str+' Ensemble Members)'
            else:
                title=subx_model+' (IC: '+str(ds['ic_dates'].values)+'; '+nens_str+' Ens )'
     
            # Define normalization for colorbar centering
            norm = pplt.Norm('diverging', vcenter=0)
           
            # Contour plot for each panel
            if (mproj=='robin'):
                m=axs[iplot].contourf(ds['lon'],ds['lat'],ds[v],levels=clevs,
                                      cmap=cmap,extend='both',norm=norm)
                axs[iplot].format(coast=True,grid=False,borders=True,
                                  borderscolor='gray5',title=title,suptitle=suptitle) 
                
                # Add US state borders    
                axs[iplot].add_feature(cfeature.STATES,edgecolor=statescolor)
            else:
                m=axs[iplot].contourf(ds['lon'],ds['lat'],ds[v],
                                      levels=clevs,cmap=cmap,extend='both',norm=norm)
                axs[iplot].format(coast=True,lonlim=lonreg,latlim=latreg,grid=False,borders=True,
                                  borderscolor='gray5',title=title,suptitle=suptitle) 
                # Add US state borders    
                axs[iplot].add_feature(cfeature.STATES,edgecolor=statescolor)

        # Add colorbar
        f.colorbar(m,loc='b',label=unit, length=0.7) 
    
        # Save Figure
        f.save(figname+'Week'+str(week)+'.png')
        
    # --- Week 3-4 -----------------
    # Create the subplot container

    f, axs = pplt.subplots(grid,
                           proj=mproj,proj_kw={'lon_0': clon},
                           width=11,height=8.5)
    
    wend_str=ds_subx_fcst.attrs['week_end'][3].strftime('%b %-d')
    suptitle='SubX Week 3-4 '+ vl+' Anomalies ('+unit+'): Valid 2 weeks ending '+wend_str
    
    # Loop over all models
    for imodel,subx_model in enumerate(ds_subx_fcst['model'].values):
        
        if (subx_model=='SUBX-MME'):
            iplot=7
        else:
            model_dict=next(item for item in subxmodels_list if item['model'] == subx_model.split('-')[1])
            iplot=model_dict['plot_loc']-1

        # Select the model and take mean or sum for 2-week depending on variable
        if (v=='pr'):
            ds=ds_subx_fcst[v].sel(model=subx_model,week=slice(3,4)).sum(dim='week')
        else:
            ds=ds_subx_fcst[v].sel(model=subx_model,week=slice(3,4)).mean(dim='week')
      
        # Get the startdate for this model
        startdate=ds['S'].dt.strftime('%m/%d').values
    
        # Define titles for individual subplot panels
        nens_str=str(ds['nens'].values.astype(int))
        if (subx_model=='SUBX-MME'):
            title=subx_model+' ('+nens_str+' Ensemble Members)'
        else:
            title=subx_model+' (IC: '+str(ds['ic_dates'].values)+'; '+nens_str+' Ens )'
        
        # Define normalization for colorbar centering
        norm = pplt.Norm('diverging', vcenter=0)

        # Contour plot for each panel
        if (mproj=='robin'):
            m=axs[iplot].contourf(ds['lon'],ds['lat'],ds,levels=clevs,
                                  cmap=cmap,extend='both',norm=norm)
            axs[iplot].format(coast=True,grid=False,borders=True,
                             borderscolor='gray5',title=title,suptitle=suptitle)  
            # Add US state borders
            axs[iplot].add_feature(cfeature.STATES,edgecolor='gray5')

        else:
            m=axs[iplot].contourf(ds['lon'],ds['lat'],ds,levels=clevs,
                                  cmap=cmap,extend='both',norm=norm)
            axs[iplot].format(coast=True,lonlim=lonreg,latlim=latreg,grid=False,borders=True,
                              borderscolor='gray5',title=title,suptitle=suptitle) 
            # Add US state borders
            axs[iplot].add_feature(cfeature.STATES)

    # Add colorbar
    f.colorbar(m,loc='b',label=unit, length=0.7) 
    
    # Save Figure
    f.save(figname+'Weeks3&4.png')
        
        
def makeFCSTWeeks(ds,fcstdate,nweeks):
    
    # Initialize lists
    ds_week_list=[]
    w_start_list=[]
    w_end_list=[]
    
    # Create the date for start of first week 
    # Note: fcstdate is always a thurs b/c that is what I key off of, but valid forecasts are Sat-Fri, 
    # following CPCs product schedule
    
    w_start=fcstdate+pd.Timedelta(days=2)
    
    
    # Loop over the weeks
    for i in range(nweeks):
    
        # End date is today+6-days
        w_end=w_start+pd.Timedelta(days=6)
        
        # Save start and end dates to list
        w_start_list.append(w_start)
        w_end_list.append(w_end)
        
        # Loop over variables and either sum or mean over week depending on how 
        # the variable is handled (e.g. precip is a sum; temperature is a mean)
        
        varnames=list(ds.keys())
        
        ds_var_list=[]
        for v in varnames:
            if (v=='pr'):
                ds_var_list.append((ds[v].sel(lead=slice(w_start,w_end))*86400).sum(dim='lead'))
            else:
                ds_var_list.append(ds[v].sel(lead=slice(w_start,w_end)).mean(dim='lead'))
                
        # Put all the variables back together
        ds_week_list.append(xr.merge(ds_var_list))
            
        # Increment for the start of next week
        w_start=w_end+pd.Timedelta(days=1)
        

    # Put all the weeks together into the same xarray.Dataset
    ds_week=xr.combine_nested(ds_week_list,concat_dim='week')
    ds_week['week']=np.arange(1,nweeks+1)
    ds_week.attrs['week_start']=w_start_list
    ds_week.attrs['week_end']=w_end_list
    
    return ds_week


def setattrs(ds,fcstdate):
    ds.attrs['fcst_date'] = fcstdate
    ds.attrs['title'] = "SubX Weekly Forecast Anomalies" 
    ds.attrs['long_title'] = "SubX Weekly Forecast Anomalies" 
    ds.attrs['comments'] = "SubX project http://weather.ou.edu/~kpegion/subx/" 
    ds.attrs['institution'] = "[IRI,GMU,U.Oklahoma]" 
    ds.attrs['source'] = "SubX IRIDL: http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/"
    ds.attrs['CreationDate'] = date.today().strftime('%Y-%m-%d')
    ds.attrs['CreatedBy'] = os.environ['USER'] 
    ds.attrs['Source'] = os.path.basename(__file__)
    ds.attrs['Reference'] = "DOI: 10.7916/D8PG249H; 10.1175/BAMS-D-18-0270.1"

    return ds

def preprocNRL(ds,startdates):
    
    # Drop any unused dimensions
    ds=ds.squeeze(drop=True)
    
    # Startdates are ensemble members for this lagged ensemble
    # Re-assign coordinates to make this match with S: startdates; M: Ensemble members
    nens=len(ds['S'])
    
    # Create the ic_dates 
    s=pd.to_datetime(ds['S'].values[0]).strftime('%Y%m%d')
    e=pd.to_datetime(ds['S'].values[-1]).strftime('%Y%m%d')
    ds=ds.assign_coords({'ic_dates':s+'-'+e})
    sdate=ds['S'][-1]
   
    ds=ds.rename({'S':'M'})
    ds=ds.assign_coords(S=sdate)
    ds['M']=np.arange(nens)
    
    return ds

def preprocCFSv2(ds):
    
    # Get the start date as YYYYMMDD 
    sdate=ds['S'][-1].values
    
    # Select this start date. There will be 4 for this model 0,6,12,18Z. 
    ds=ds.sel(S=sdate)
    
    # Rename Coordinates
    ds=ds.rename({'M':'E','S':'IC'})
    
    # Stack the ICs and Enembles together into M coordinate
    ds=ds.stack(M=('IC','E'))
    
    # Add the S coordinate back with only single start date
    ds=ds.assign_coords({'S':sdate,'ic_dates':pd.to_datetime(sdate).strftime('%Y%m%d')})
 
    return ds

def subxPlot(ds,path):

    # Get the plotting parameters for variables and regions
    var_params_dict,reg_params_dict=initPlotParams()

    # Loop over variables
    for var_params in var_params_dict:
        print(var_params['name'])    
        # Loop over regions to be plotting for this variable
        for regs in var_params['regions']:
        
            # Find dictionary for this region
            reg_dict=next(item for item in reg_params_dict if item['name'] == regs)
        
            # Output figure partial name
            figname=path+var_params['outname']+reg_dict['name']
       
            # Call plotting with variable and region parameters 
            makeWebImages(ds,var_params['name'],var_params['units'],
                          var_params['label'],var_params['clevs'],var_params['cmap'],
                          reg_dict['lons'],reg_dict['lats'],reg_dict['clon'],
                          reg_dict['mproj'],reg_dict['state_colors'],figname)

def subxIntPlot(ds,varnames,path):
    
    # Get the plotting parameters for variables and regions
    var_params_dict,reg_params_dict=initPlotParams() 
    
    # Get a list of dictionaries for variables to be included in interactive plotting
    var_list=[]
    for var_params in var_params_dict:
        if var_params['name'] in varnames:
            var_list.append(var_params)
            
    # Call Interactive Plotting
    makeInteractiveWebImages(ds,var_list,path)
    
def subxWrite(ds_subx_fcst,fcstdate,emean,filepath):
    
    # Get a dictionary of SubX Models 
    subxmodels_list,_,all_varnames,all_plevstrs,all_units=initSubxModels()
    
    # Loop over all variables to write out
    for v,p,u in zip(all_varnames,all_plevstrs, all_units):

        ds_model_list=[]
    
        # Loop over the SubX Models
        for imodel,subx_model in enumerate(subxmodels_list):
        
            # Get the model, group, variables, and levels from the dictionary 
            varnames=subx_model['varnames']
            plevstrs=subx_model['plevstrs']
            model=subx_model['model']
            group=subx_model['group']

            if (model in ds_subx_fcst['model'].values):
                print(model,'is here')
                
            # Check if this model has this variable and append
            if v in varnames:
                
                # Individual Models
                ds=ds_subx_fcst[v].sel(model=group+'-'+model).to_dataset(name=model)
                ds[model].attrs['long_name']=group+'-'+model+' '+str(ds['ic_dates'].values)
                ds[model].attrs['units']=u
                ds_subx_fcst.attrs['units']=u
                ds_model_list.append(ds.reset_coords(drop=True))
            
                # MME
                ds=ds_subx_fcst[v].sel(model='SUBX-MME').to_dataset(name='MME')
                ds['MME'].attrs['long_name']='SUBX-MME'+' '+str(ds['ic_dates'].values)
                ds['MME'].attrs['units']=u
                
                ds_model_list.append(ds.reset_coords(drop=True))
                
        # Check if list of models for this variable has data
        if (ds_model_list):
       
            # Put all the models together for this variable        
            ds_models=xr.merge(ds_model_list)
           
            # Set time dimension and units for grads readable
            ds_models=ds_models.rename({'week':'time'})
            ds_models['time'].attrs['start_date']= str(ds_subx_fcst.attrs['week_start'])
            ds_models['time'].attrs['end_date']= str(ds_subx_fcst.attrs['week_end'])
            #ds_models['time'].attrs['units']='days since '+str(ds['ic_dates'][0].values)
            
            ds_models['time'].attrs['standard_name']='time'
            ds_models['time'].attrs['long_name']='Time of measurements'
            ds_models.attrs['units']=u
            
            # Write out file
            ofname=filepath+fcstdate.strftime('%Y%m%d')+'/data/fcst_'+fcstdate.strftime('%Y%m%d')+'.anom.'+v+'_'+p+'.'+emean+'.nc'
            #ofname=filepath+'fcst_'+fcstdate.strftime('%Y%m%d')+'.anom.'+v+'_'+p+'.'+emean+'.nc'

            print(ofname)
            ds_models.to_netcdf(ofname)
            
        else: # list is empty, variable does not exist
            print('This variable does not exist in any models. File '+os.path.basename(ofname)+' not written.')

def getFcstDates(date=None):

    if (date):
        print("Using Input Date: ",date)
        currentdate=datetime.strptime(date,'%Y%m%d')
        print("Using Input Date: ",currentdate)
    else:
        currentdate=datetime.today().replace(microsecond=0,second=0,minute=0,hour=0)
        print("Using Current Date: ",currentdate)

    # How far are we from the most recent thurs?
    diffdate=(currentdate.weekday()-3) % 7

    # fcstdate is the most recent Thurs
    fcstdate=currentdate-timedelta(days=diffdate)

    # Identify the previous Friday (6 days earlier) as start of fcst week
    weekdate=fcstdate - timedelta(days=6)

    print("USING FCSTS ICS FROM: ",weekdate.strftime(('%Y%m%d')), " to ",fcstdate.strftime('%Y%m%d'))
    print("Output files and figures will be labeled as: ",fcstdate.strftime('%Y%m%d'))

    # Get the range of dates over the last week from the start of previous fcst week to fcstdate
    fcst_week_dates=pd.date_range(start=weekdate,end=fcstdate,freq='D') 
    
    return fcstdate,fcst_week_dates

def getDataViaIngrid(ds_meta,baseURL):
    
    # Construct the pressure level information for IngridURL if it exists
    ingrid_pvalue=''
    if ('P' in list(ds_meta.coords)):
        ingrid_pvalue='P/%28'+str(int(ds_meta['P'].values))+'%29VALUES'
            
    # Construct the Date Information for the IngridURL
    day='%20'+ds_meta['S'].dt.strftime('%d').values 
    month='%20'+ds_meta['S'].dt.strftime('%b').values
    year='%20'+ds_meta['S'].dt.strftime('%Y').values
    hrs='%20'+ds_meta['S'].dt.strftime('%H').values+'00'
    ingrid_svalue='S/%28'+hrs+day+month+year+'%29VALUES'
   
    # Close the full dataset used to get the metadata
    ds_meta.close()
    
    # Construct the Ingrid URL
    if (ingrid_pvalue==''):
        ingridURL=baseURL+'/'+ingrid_svalue+'/dods/'
    else:
        ingridURL=baseURL+'/'+ingrid_pvalue+'/'+ingrid_svalue+'/dods/'
    
    # Open the subsetted version of the dataset using the Ingrid URL
    ds=xr.open_dataset(ingridURL)
    
    return ds

def combine_models(ds,v):
    
    ds_list=[]
    for m in list(ds.keys()):
        ds_list.append(ds[m])

    ds_all=xr.combine_nested(ds_list,concat_dim='model').to_dataset(name=v)
    ds_all['model']=list(ds.keys())
    print(ds_all)
    
    return ds_all

def makeInteractiveWebImages(ds,var_list,figpath):
    
    # Proplot makes high res pics so need to reduce for web pics
    pplt.rc.savefigdpi = 100
        
    # Make a list of plots for each variable to be plotted
    plot_list=[]
    
    # Loop over the variables to include in interactive plottin
    for v in var_list:
        
        # Get variable attributes for plotting
        vname=v['name']
        clevs=v['clevs']
        units=v['units']
        cmap=v['cmap']
        vlabel=v['label']
        clims=v['clims']
        
        # Get forecast date
        fcstdate=ds.attrs['fcst_date'].strftime(('%Y%m%d'))
        
        # Create plot title
        title='SubX '+vlabel+' Anomalies ('+units+') for Forecast Date: ' +fcstdate

        # Make hvplot image
        tmp_plot=(ds[vname]).hvplot(x='lon',y='lat',groupby=['model','week'],kind='image',
                                    xaxis=False, yaxis=False,  
                                    cmap=cmap, rasterize=True, clim=clims,
                                    geo=True, projection=ccrs.Robinson(), coastline=True, clabel = units,
                                    global_extent=True,
                                    title=title,width=900,height=350)
        # Add geography to hvplot
        tmp_plot=tmp_plot*(gf.states(color=None))
        tmp_plot=tmp_plot*(gf.borders)
        
        # Collect the plot for each variable in a list
        plot_list.append(tmp_plot)
        
    # Make a Column of all the plots on one page and save to html
    pn.Column(plot_list[0],plot_list[1],plot_list[2]).save(filename=figpath+'/Interactive.html', embed=True)