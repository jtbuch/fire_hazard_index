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
    comb_ds= xr.open_mfdataset(f'{datadir}daily/era5_wus_forecast_daily_%s_*.nc'%var_label, parallel=True)
    comb_ds[var_str].to_netcdf(f'{datadir}daily/era5_wus_forecast_daily_%s_2002-2020.nc'%var_label)

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

