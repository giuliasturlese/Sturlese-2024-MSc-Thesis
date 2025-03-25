import cartopy.crs as ccrs
from eofs.xarray import Eof
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
from scipy.stats import pearsonr
import pingouin as pg # for partial correlations
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Functions to export
__all__ = [
    'plot_map', 'plot_stereographic_map', 'coslat_weights', 'eof_solver', 'standardize_pcs',
    'sig_regression_slope', 'corr_sig', 'pcorr', 'pcorr_df', 'full_pcorr_df'
]

# Map function
def plot_map(darray, min_value='', max_value='', title='', 
             x_step = 25.0, y_step = 20.0, x_rotation = 0, 
             x_offset = 0, y_offset = 5, shave_top=0):
    """ Reads a data array, returns plot with coastlines """
    lons = darray['longitude']
    lats = darray['latitude']
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()-shave_top])
    
    # set minimum and/or maximum value
    if min_value != '' and max_value != '':
        darray.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=min_value, vmax=max_value)
    elif min_value != '': 
        darray.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=min_value)
    elif max_value != '':
        darray.plot(ax=ax, transform=ccrs.PlateCarree(), vmax=max_value)
    else:
        darray.plot(ax=ax, transform=ccrs.PlateCarree())
    
    ax.coastlines()
    
    # add title 
    if title != '':
        ax.set_title(title)
    
    plt.xlabel('') # (longitude, degrees east)')
    plt.ylabel('') # (latitude, degrees north)')
    plt.xticks(np.arange(lons.min()+x_offset, lons.max()+1, x_step), rotation = x_rotation)
    plt.yticks(np.arange(lats.min()+y_offset, lats.max(), y_step))
    
    return ax

# Stereographic map function
def plot_stereographic_map(darray, title='', gridlines='no', min_value='', max_value=''): 
    """ Reads a data array, returns stereographic projection plot of array over a stereographic projection map with coastlines of corresponding latitude-longitude box. If a minimum value is required, set min_value = minimum value (float). If gridlines are wanted, set gridlines='yes'. To set a title, set title='desired title'. """
    lons = darray['longitude']
    lats = darray['latitude']
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], ccrs.PlateCarree())
    
    # set minimum and/or maximum value
    if min_value != '' and max_value != '':
        darray.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=min_value, vmax=max_value)
    elif min_value != '': 
        darray.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=min_value)
    elif max_value != '':
        darray.plot(ax=ax, transform=ccrs.PlateCarree(), vmax=max_value)
    else:
        darray.plot(ax=ax, transform=ccrs.PlateCarree())
    
    ax.coastlines();
    
    # add gridlines
    if gridlines == 'yes':
        ax.gridlines(color='C7', lw=0.6, ls=':', draw_labels=True, rotate_labels=False, ylocs=range(0,91,30), xlocs=range(-180,180,30)) #,ylocs=[60,70,80])
        
    # add title 
    if title != '':
        ax.set_title(title)
        
    return ax


# EOF functions
def coslat_weights(data_array): 
    coslat = np.cos(np.deg2rad(data_array.coords['latitude'].values)).clip(0.,1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    return wgts

def eof_solver(data_array):
    anom = data_array - data_array.mean(dim='time')
    wgts = coslat_weights(data_array)
    solver = Eof(anom, weights=wgts)
    solver = Eof(data_array, weights=wgts)
    return solver

# Standardize pcs function
def standardize_pcs(pcs, mode=0):
    """ Reads pc and mode and returns standardized pc of specified mode. Default mode = 0. """
    pc = pcs.isel(mode=mode)
    mean_pc = pc.mean()
    std_pc = pc.std()
    standardized_pc = (pc - mean_pc) / std_pc
    return standardized_pc

# Regression map function
def sig_regression_slope(x, y, dimension=None, sig = 0.05): 
    """ Returns linear regression slope of DataArray y against DataArray x over dimension 'dimension' (string, optional, default=None). *** Only returns significant values (p-value <= sig), the others are nan. *** """
    y_anom = y - y.mean()
    corr = xs.pearson_r(x, y_anom, dim = dimension) # this library also gives the p-value
    std_x = x.std()
    std_y = y_anom.std()
    slope = corr * (std_y / std_x)
    slope = slope.to_dataset(name='slope')
    p_value = xs.pearson_r_p_value(x, y_anom, dim = dimension)
    p_value = p_value.to_dataset(name='p_value')
    slope_p = xr.merge( [slope, p_value] ) # merge the two datasets
    significant_slope_p = slope_p.where(slope_p.p_value <= sig)
    significant_slope = significant_slope_p['slope']
    return significant_slope


# Correlation table functions

# corr_sig function from here: https://stackoverflow.com/questions/75730027/masking-correlation-matrix-based-on-p-values-and-correlation
def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix
    
# Function to calculate partial correlations
def pcorr(df, x=None, y=None):
    """ Reads pandas dataframe df and two strings: first variable x, second variable y. x and y are columns in the dataframe df. Returns dataframe of partial correlation between x and y, controlling for all the other variables. """
    # create list of all variables except x and y
    ls = [col for col in df.columns.values if (col != x and col != y)]
    # return x-y partial corr with all other variables as covariates
    return pg.partial_corr(data=df, x=x, y=y, covar=ls).round(4)

# Function to create dataframe with all partial correlations 
def pcorr_df(df, x=None): 
    """ Reads pandas dataframe df and a string: variable x (a column in df). Calculates partial correlations of x versus each other variable, controlling for all other variables. Returns dataframe where each row contains the partial correlation between x and one other variable, controlling for all others. """
    # create list of all variables except x
    y_list = [col for col in df.columns.values if col != x]
    # initialize empty list
    df_list = []
    for y in y_list: 
        pcorrel = pcorr(df, x, y)
        pcorrel.insert(0,'Y',y) # insert Y column
        pcorrel.insert(0, 'X', x) # insert X column
        pcorrel = pcorrel.drop(columns=['n','CI95%']) # drop sample size and confidence interval columns
        df_list.append(pcorrel) # add pcorrel to the df_list
    par_corr_df = pd.concat(df_list)
    par_corr_df = par_corr_df.reset_index(drop=True)
    return par_corr_df

# Function to apply pcorr_df to all variables 
def full_pcorr_df(df):
    """ Reads pandas dataframe of time-dependent indexes. 
        Returns dataframe of partial correlations. """
    ls = []
    for x in df.columns.values:
        ls.append(pcorr_df(df,x))
    full_df = pd.concat(ls)
    full_df = full_df.reset_index(drop=True)
    return full_df





