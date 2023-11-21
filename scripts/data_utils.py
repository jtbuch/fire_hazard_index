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