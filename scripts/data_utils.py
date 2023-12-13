import numpy as np
import random
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats, interpolate
from scipy.optimize import minimize
from scipy.special import gamma
from sklearn.linear_model import LinearRegression
from math import factorial
import itertools
from copy import deepcopy
from tqdm import tqdm
import re # for regular expressions

import netCDF4 # module that reads in .nc files (built on top of HDF5 format)
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
import xarray as xr
import rioxarray
import xesmf as xe # for regridding
import pickle # for saving and loading models
from pathlib import Path

# Date and time related libraries
from dateutil.relativedelta import relativedelta
from calendar import monthrange
import datetime

from shapely.geometry import Point, mapping
from shapely.geometry.polygon import Polygon
from pyproj import CRS, Transformer # for transforming projected coordinates to elliptical coordinates
import tensorflow as tf
import tensorflow_probability as tfp
tfd= tfp.distributions

def load_daily_meteorlogical_predictors(var, yrarr, datadir, maxmin_arg= None):

    '''
    Function for loading hourly meteorological predictors and saving them as a single netCDF file at daily resolution

    var: variable of interest
    yr_arr: array of years to load
    datadir: directory where data is stored
    '''
    if var == 'fire_weather_index':
        var_str= 'fwinx'
        var_label= 'FFWI'

        comb_ds= xr.open_mfdataset(f'{datadir}/daily/era5_wus_forecast_daily_%s_*.grib'%var_label, parallel=True)
        comb_ds[var_str].to_netcdf(f'{datadir}/daily/era5_wus_forecast_daily_%s_2002-2020.nc'%var_label)
    else:
        if var == 'total_precipitation':
            const= 1000 # converting from m to mm
            var_str= 'tp'
            var_label = 'Prec'
        elif var == '2m_temperature':
            const= 273.15 # converting Kelvin to Celsius
            var_str= 't2m'
            if maxmin_arg == 'max':
                var_label = 'Tmax'
            elif maxmin_arg == 'min':
                var_label = 'Tmin'
        elif var == '2m_dewpoint_temperature':
            const= 273.15 # converting Kelvin to Celsius
            var_str= 'd2m'
            var_label= 'Tdew'    

        for yr in tqdm(yrarr):
            ds= xr.open_dataset(f'{datadir}hourly/era5_wus_forecast_hourly_%s'%var + f'_%s.grib'%yr, engine='cfgrib')
            if var == 'total_precipitation':
                ds_prev_day= ds[var_str][:-1] # considering total rainfall from 12am to 6am
                ds_rem_day= ds[var_str][1:] # considering total rainfall from 6pm to 12am
                da= (ds.sel(time=ds['time.hour'] == 6).sum('step')[var_str] + ds_prev_day.sel(time=ds_prev_day['time.hour'] == 18, step= ds_prev_day['step'] > np.timedelta64(5, 'h')).sum('step').values + \
                                                                    ds_rem_day.sel(time=ds_rem_day['time.hour'] == 18, step= ds_rem_day['step'] < np.timedelta64(6, 'h')).sum('step').values)*const
            else:
                if maxmin_arg == 'max':
                    da= ds[var_str].resample(time='1D').max() - const
                elif maxmin_arg == 'min':
                    da= ds[var_str].resample(time='1D').min() - const
                else:
                    da= ds[var_str].resample(time='1D').mean() - const
            da.to_netcdf(f'{datadir}daily/era5_wus_forecast_daily_%s'%var_label + f'_%s.nc'%yr)
        comb_ds= xr.open_mfdataset(f'{datadir}/daily/era5_wus_forecast_daily_%s_*.nc'%var_label, parallel=True)
        comb_ds[var_str].to_netcdf(f'{datadir}/daily/era5_wus_forecast_daily_%s_2002-2020.nc'%var_label)

def vapor_pressure(temp):
    return 6.0178*np.exp((17.629*(temp)/(237.3 + (temp)))) #actual vapor pressure in hPa (1 mb = 1 hPa); temp in C

def ds_latlon_subset(ds, area, latname='lat', lonname='lon'):
    
    """
    Function to subset a dataset based on a lat/lon bounding box ("borrowed" from C3S tutorial for seasonal forecasting)
    """
    lon1 = area[1] % 360
    lon2 = area[3] % 360
    if lon2 >= lon1:
        masklon = ( (ds[lonname] % 360 <=lon2) & (ds[lonname] % 360 >=lon1) ) 
    else:
        masklon = ( (ds[lonname] % 360 <=lon2) | (ds[lonname] % 360 >=lon1) ) 
        
    mask = ((ds[latname]<=area[0]) & (ds[latname]>=area[2])) * masklon
    dsout = ds.where(mask,drop=True)
    
    if lon2 < lon1:
        dsout[lonname] = (dsout[lonname] + 180) % 360 - 180
        dsout = dsout.sortby(dsout[lonname])        
    
    return dsout

def coord_transform(coord_a, coord_b, input_crs= 'WGS84', output_crs= 'EPSG:5070'):

    '''
    Function to convert coordinates between different reference systems with a little help from pyproj.Transformer

    coord_a: first coordinate (x or longitude)
    coord_b: second coordinate (y or latitude)
    input_crs: input coordinate reference system
    output_crs: output coordinate reference system
    '''
    #custom crs i/o with https://gis.stackexchange.com/questions/427786/pyproj-and-a-custom-crs
    
    transformer= Transformer.from_crs(input_crs, output_crs, always_xy= True)
        
    # we add another if-else loop to account for differences in input size: for different sizes, we first construct a meshgrid,
    # before transforming coordinates. Thus, the output types will differ depending on the input.
    if len(coord_a) == len(coord_b):
        return transformer.transform(coord_a, coord_b) 
    else:
        coord_grid_a, coord_grid_b= np.meshgrid(coord_a, coord_b)
        return transformer.transform(coord_grid_a, coord_grid_b)
    
def regridding_func(data_ds, subarea, dsout, regrid_scheme, latname= 'lat', lonname= 'lon'):

    '''
    Regridding function for downscaling, regridding, and interpolating dynamical forecasts to match APW's 12km grid
    '''

    tmax_xr= xr.open_dataarray('../data/meteorology/tmax_12km.nc')
    x_fire_grid= xr.DataArray(coord_transform(tmax_xr.X.values, tmax_xr.Y.values, "epsg:5070", "epsg:4326")[0], dims=('Y','X'))
    y_fire_grid= xr.DataArray(coord_transform(tmax_xr.X.values, tmax_xr.Y.values, "epsg:5070", "epsg:4326")[1], dims=('Y','X'))
    maskXY= ~tmax_xr[0].isnull().drop('time')

    data_ds_conus= ds_latlon_subset(data_ds, subarea, latname= latname, lonname= lonname)
    regridder_data= xe.Regridder(data_ds_conus, dsout, method= regrid_scheme)
    data_ds_conus_regridded= regridder_data(data_ds_conus, keep_attrs=True)

    data_ds_conus_XY_regridded= data_ds_conus_regridded.interp({'lat':y_fire_grid, 'lon':x_fire_grid}, method='linear').load()
    data_ds_conus_XY_regridded= data_ds_conus_XY_regridded.assign_coords({'X': (('X'), tmax_xr.X.data, {"units": "meters"}), 'Y': (('Y'), tmax_xr.Y.data, {"units": "meters"})}).drop_vars(['lat','lon'])
    data_ds_conus_XY= data_ds_conus_XY_regridded.where(maskXY)

    return data_ds_conus_XY

def init_fire_counts_raster(start_date= '2002-01-01', end_date= '2021-05-01', sav_flag= False):

    '''
    Function to initialize an empty raster grid for a given time period
    '''
    tmax_da= xr.open_dataarray('../data/12km/Tmax_daily_conus_2002-2020_12km.nc')
    datearr= pd.date_range(start= start_date, end= end_date)
    fc_da= xr.DataArray(np.zeros((len(datearr), 208, 155)), dims= ['time', 'Y', 'X'], \
                                                    coords= {'time': datearr.values, 'Y': tmax_da.Y.values, 'X': tmax_da.X.values})
    maskXY= ~tmax_da[0].isnull().drop('time')
    fc_da= fc_da.where(maskXY)

    if sav_flag:
        fc_da.to_netcdf('../data/12km/fire_raster_2002-2021_12km.nc')
    
    print('Initialized raster for fire counts and burned area!')
    
def init_grid(firedat, res):
    
    '''
    Initializes a raster GeoPandas dataframe with the same extent as the input raster data

    firedat: fire raster data
    res: resolution of the raster data
    
    '''
    
    if res == '12km':
        xmin, xmax= [firedat[:, :, :].X.values.min(), firedat[:, :, :].X.values.max()]
        ymin, ymax= [firedat[:, :, :].Y.values.min(), firedat[:, :, :].Y.values.max()]
        cellwidth= abs(firedat.X[0].values - firedat.X[1].values)
    
    cols= list(np.arange(xmin, xmax + cellwidth, cellwidth))
    rows= list(np.arange(ymax, ymin - cellwidth, -cellwidth))
    
    polygons = []
    for y in rows:
        for x in cols:
            polygons.append(Polygon([(x,y), (x+cellwidth, y), (x+cellwidth, y+cellwidth), (x, y+cellwidth)])) 

    grid= gpd.GeoDataFrame({'geometry': gpd.GeoSeries(polygons)})
    grid['grid_indx']= grid.index.to_numpy()
    grid= grid.set_crs('EPSG:5070')
    
    return grid, rows, cols

def init_fire_alloc_gdf(lcgdf, res= '12km', lctype= 'forest', sav_flag= False): 
    
    '''
    Function to initialize burned area raster grid

    lcgdf: MODIS fire perimeter GeoPandas dataframe 
    '''

    ba_raster= xr.open_dataarray('../data/12km/fire_raster_2002-2021_12km.nc')
    ind_raster= ba_raster.copy(deep= True) # select only X and Y coordinates of ba_raster
    
    grid, rows, cols= init_grid(ba_raster, res)
    cellwidth= int(re.findall(r'\d+', '12km')[0])*1000    
    fire_gdf= lcgdf[['id', 'ig_day', 'event_ig_day', 'dy_ar_km2', 'tot_ar_km2', 'geometry']]
                               
    merged= gpd.overlay(fire_gdf, grid, how= 'intersection')
    merged= merged.sort_values(by= ['id']).reset_index()
    merged= merged.drop('index', axis= 1)

    coord_arr= np.array(list(itertools.product(np.linspace(0, len(rows) - 1, len(rows), dtype= int), np.linspace(0, len(cols) - 1, len(cols), dtype= int))))
    merged['grid_y']= [coord_arr[merged['grid_indx'].loc[[ind]]][0][0] for ind in merged.index]
    merged['grid_x']= [coord_arr[merged['grid_indx'].loc[[ind]]][0][1] for ind in merged.index]
    
    areagroups= merged.groupby('id')
    gridfracarr= np.hstack([((areagroups.get_group(k).area/cellwidth**2)/np.linalg.norm(areagroups.get_group(k).area/cellwidth**2, 1)).to_numpy() \
                                                                                                                 for k in areagroups.groups.keys()])
    merged['cell_frac']= gridfracarr

    for m in tqdm(merged.index.to_numpy()):
        ba_raster[dict(time= merged['event_ig_day'].loc[m], Y= merged['grid_y'].loc[m], X= merged['grid_x'].loc[m])]+= (merged['cell_frac'].loc[m] * merged['tot_ar_km2'].loc[m])
        ind_raster[dict(time= merged['event_ig_day'].loc[m], Y= merged['grid_y'].loc[m], X= merged['grid_x'].loc[m])]= merged['id'].loc[m]

    if sav_flag:
        ba_raster.to_netcdf('../data/12km/%s'%lctype + '_burned_area_2002-2021_12km.nc')
        ind_raster.to_netcdf('../data/12km/%s'%lctype + '_fid_2002-2021_12km.nc')
    
    return ba_raster, ind_raster

def init_fire_clim_df(lctype= 'forest'):

    '''
    Function to initialize a dataframe with climate predictors for each fire ignition point by land cover type

    '''

    #Fire ignition points
    #lc_xarr= xr.open_dataarray('../data/12km/%s'%lctype + '_burned_area_2002-2021_12km.nc')
    lc_fid_xarr= xr.open_dataarray('../data/12km/%s'%lctype + '_fid_2002-2021_12km.nc')
    lc_df= lc_fid_xarr.to_dataframe('fid').reset_index()

    lc_df= lc_df[lc_df.fid > 0]
    lc_df.drop_duplicates(subset=['fid'], inplace= True)
    lc_df.date= pd.to_datetime(lc_df.time)
    lc_df= lc_df[(lc_df.time.dt.month > 3) & (lc_df.time.dt.month < 10) & (lc_df.time >= '2002-08-01') & (lc_df.time <= '2020-09-01')]

    #Climate predictors
    vpd_data= xr.open_dataarray('../data/12km/VPD_daily_conus_2002-2020_12km.nc')
    tmax_data= xr.open_dataarray('../data/12km/tmax_daily_conus_2002-2020_12km.nc')
    tmin_data= xr.open_dataarray('../data/12km/tmin_daily_conus_2002-2020_12km.nc')
    prec_data= xr.open_dataarray('../data/12km/prec_daily_conus_2002-2020_12km.nc').resample(time='1D').first()
    sm_data= xr.open_dataset('../data/12km/casm_conus_12km.nc').CASM_soil_moisture
    sm_data= sm_data.interpolate_na(dim= 'date', fill_value= 'extrapolate')
    csif_data_ur= xr.open_dataarray('../data/12km/csif_clear_inst_conus_12km.nc')
    csif_data= csif_data_ur.interp(time= sm_data.date.values)
    csif_data= csif_data.interpolate_na(dim= 'time', fill_value= 'extrapolate')

    tmp_vpd= vpd_data.sel(time= lc_df.time.values)
    tmp_prec= prec_data.sel(time= lc_df.time.values)
    tmp_tmax= tmax_data.sel(time= lc_df.time.values)
    tmp_tmin= tmin_data.sel(time= lc_df.time.values)

    vpd_data_arr= []
    tmax_data_arr= []
    tmin_data_arr= []
    prec_data_arr= []
    rcsif_data_arr= []
    sm_data_arr= []
    ant_vpd_1_arr= []
    ant_prec_1_arr= []
    ant_sm_1_arr= []
    ant_vpd_2_arr= []
    ant_prec_2_arr= []
    ant_sm_2_arr= []
    ant_vpd_3_arr= []
    ant_prec_3_arr= []
    ant_sm_3_arr= []

    for ind in tqdm(range(len(lc_df))):
        vpd_data_arr.append(tmp_vpd.sel(time= lc_df.iloc[ind].time, Y= lc_df.iloc[ind].Y, X= lc_df.iloc[ind].X).mean().values)
        tmax_data_arr.append(tmp_tmax.sel(time= lc_df.iloc[ind].time, Y= lc_df.iloc[ind].Y, X= lc_df.iloc[ind].X).mean().values)
        tmin_data_arr.append(tmp_tmin.sel(time= lc_df.iloc[ind].time, Y= lc_df.iloc[ind].Y, X= lc_df.iloc[ind].X).mean().values)
        prec_data_arr.append(tmp_prec.sel(time= lc_df.iloc[ind].time, Y= lc_df.iloc[ind].Y, X= lc_df.iloc[ind].X).mean().values)
        sm_data_arr.append(sm_data.sel(date= lc_df.iloc[ind].time, Y= lc_df.iloc[ind].Y, X= lc_df.iloc[ind].X, method= 'nearest').values)
        rcsif_data_arr.append((csif_data.sel(time= lc_df.iloc[ind].time, Y= lc_df.iloc[ind].Y, X= lc_df.iloc[ind].X, method= 'nearest')/csif_data.sel(Y= lc_df.iloc[ind].Y, \
                                                                                                                X= lc_df.iloc[ind].X).max()).values)
        ant_vpd_1_arr.append(vpd_data.sel(time= pd.date_range(end= lc_df.iloc[ind].time, periods= 15, inclusive= 'left'), Y= lc_df.iloc[ind].Y, \
                                                                                                                    X= lc_df.iloc[ind].X).mean(dim= 'time').values)
        ant_prec_1_arr.append(prec_data.sel(time= pd.date_range(end= lc_df.iloc[ind].time, periods= 15, inclusive= 'left'), Y= lc_df.iloc[ind].Y, \
                                                                                                                    X= lc_df.iloc[ind].X).mean(dim= 'time').values)
        ant_sm_1_arr.append(sm_data.sel(date= pd.date_range(end= lc_df.iloc[ind].time, periods= 5, inclusive= 'left'), Y= lc_df.iloc[ind].Y, \
                                                                                                                    X= lc_df.iloc[ind].X, method= 'nearest').mean(dim= 'date').values)
        ant_vpd_2_arr.append(vpd_data.sel(time= pd.date_range(end= lc_df.iloc[ind].time, periods= 30, inclusive= 'left'), Y= lc_df.iloc[ind].Y, \
                                                                                                                    X= lc_df.iloc[ind].X).mean(dim= 'time').values)
        ant_prec_2_arr.append(prec_data.sel(time= pd.date_range(end= lc_df.iloc[ind].time, periods= 30, inclusive= 'left'), Y= lc_df.iloc[ind].Y, \
                                                                                                                    X= lc_df.iloc[ind].X).mean(dim= 'time').values)
        ant_sm_2_arr.append(sm_data.sel(date= pd.date_range(end= lc_df.iloc[ind].time, periods= 10, inclusive= 'left'), Y= lc_df.iloc[ind].Y, \
                                                                                                                    X= lc_df.iloc[ind].X, method= 'nearest').mean(dim= 'date').values)
        ant_vpd_3_arr.append(vpd_data.sel(time= pd.date_range(end= lc_df.iloc[ind].time, periods= 60, inclusive= 'left'), Y= lc_df.iloc[ind].Y, \
                                                                                                                    X= lc_df.iloc[ind].X).mean(dim= 'time').values)
        ant_prec_3_arr.append(prec_data.sel(time= pd.date_range(end= lc_df.iloc[ind].time, periods= 60, inclusive= 'left'), Y= lc_df.iloc[ind].Y, \
                                                                                                                    X= lc_df.iloc[ind].X).mean(dim= 'time').values)
        ant_sm_3_arr.append(sm_data.sel(date= pd.date_range(end= lc_df.iloc[ind].time, periods= 20, inclusive= 'left'), Y= lc_df.iloc[ind].Y, \
                                                                                                                    X= lc_df.iloc[ind].X, method= 'nearest').mean(dim= 'date').values)

    vpd_data_arr= np.array(vpd_data_arr).flatten()
    tmax_data_arr= np.array(tmax_data_arr).flatten()
    tmin_data_arr= np.array(tmin_data_arr).flatten()
    prec_data_arr= np.array(prec_data_arr).flatten()
    rcsif_data_arr= np.nan_to_num(rcsif_data_arr, nan= -999).flatten()
    sm_data_arr= np.nan_to_num(sm_data_arr, nan= -999).flatten()
    ant_vpd_1_arr= np.array(ant_vpd_1_arr).flatten()
    ant_prec_1_arr= np.array(ant_prec_1_arr).flatten()
    ant_sm_1_arr= np.nan_to_num(ant_sm_1_arr, nan= -999).flatten()
    ant_vpd_2_arr= np.array(ant_vpd_2_arr).flatten()
    ant_prec_2_arr= np.array(ant_prec_2_arr).flatten()
    ant_sm_2_arr= np.nan_to_num(ant_sm_2_arr, nan= -999).flatten()
    ant_vpd_3_arr= np.array(ant_vpd_3_arr).flatten()
    ant_prec_3_arr= np.array(ant_prec_3_arr).flatten()
    ant_sm_3_arr= np.nan_to_num(ant_sm_3_arr, nan= -999).flatten()

    lc_fire_clim_df= pd.concat([lc_df.reset_index(), pd.DataFrame({'VPD': vpd_data_arr, 'Tmax': tmax_data_arr, 'Tmin': tmin_data_arr, 'Prec': prec_data_arr, \
                                'SM': sm_data_arr, 'rcSIF': rcsif_data_arr, \
                                'Ant_VPD_15d': ant_vpd_1_arr, 'Ant_Prec_15d': ant_prec_1_arr, 'Ant_SM_15d': ant_sm_1_arr, \
                                'Ant_VPD_1mo': ant_vpd_2_arr, 'Ant_Prec_1mo': ant_prec_2_arr, 'Ant_SM_1mo': ant_sm_2_arr, \
                                'Ant_VPD_2mo': ant_vpd_3_arr, 'Ant_Prec_2mo': ant_prec_3_arr, 'Ant_SM_2mo': ant_sm_3_arr})], axis= 1).drop(columns= ['index'])
    lc_fire_clim_df['fid']= lc_fire_clim_df['fid'].astype(int)

    return lc_fire_clim_df