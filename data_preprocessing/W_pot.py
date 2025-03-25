# Calculate the (yearly averages of the) wind potential and 100m wind speed

# Import libraries
import numpy as np
import pandas as pd
import xarray as xr

# Read data
wind_dataset = xr.open_dataset('ERA5_6h_100m_uv_components.nc')
u = wind_dataset['u100']    
v = wind_dataset['v100']

# Calculate hexa-hourly 100m wind speed 
w100m6 = np.sqrt(u**2 + v**2) 

# Function to calculate the wind potential (Jerez et al., 2015)
def w_pot_calculator(wind_darray, cut_in_speed = 3.5, rated_speed = 13, cut_out_speed = 25):
    """ Reads wind speed (V) dataarray, calculates wind potential w_pot: 
            * 0 if V < cut_in_speed 
            * (V^3 - cut_in_speed^3)/(rated_speed^3 - cut_in_speed^3) if cut_in_speed <= V < rated_speed;
            * 1 if rated_speed <= V < cut_out_speed;
            * 0 if V >= cut_out_speed 
        All speeds are in m/s.
        Note: wind_darray has 3 dimensions: 'time', 'latitude', 'longitude'.
        Returns w_pot dataarray. """
    
    # Calculate powers and denominator
    cut_in_3 = pow(cut_in_speed, 3)
    rated_3 = pow(rated_speed, 3)
    denom = (rated_3 - cut_in_3)
    
    # Initialize w_pot DataArray as copy of wind_darray
    w_pot = wind_darray.copy()
    
    # Set w_pot = 0 where wind speed is out of bounds
    w_pot = xr.where(np.logical_or(w_pot < cut_in_speed, w_pot >= cut_out_speed), 0, w_pot)
    
    # Set w_pot = num/denom where wind speed is between cut-in and rated speeds
    w_pot = xr.where(np.logical_and(cut_in_speed <= w_pot, w_pot < rated_speed), ( pow(w_pot, 3) - cut_in_3 ) / denom, w_pot)
    
    # Set w_pot = 1 where wind speed is between rated and cut-out speeds
    w_pot = xr.where(np.logical_and(rated_speed <= w_pot, w_pot < cut_out_speed), 1, w_pot)
    
    return w_pot

# Calculate hexa-hourly w_pot
w_pot6h = w_pot_calculator(w100m6)
w_pot6h = w_pot6h.rename('w_pot')

# Split w_pot6h dataset into smaller datasets to save as netcdf
# (full dataset is too large for CDO)
w_pot6h_1 = w_pot6h.sel(time = (w_pot6h.time.dt.year.isin( range(1981, 1991) ) ) )
w_pot6h_1.to_netcdf('w_pot6h_1.nc')

w_pot6h_2 = w_pot6h.sel(time = (w_pot6h.time.dt.year.isin( range(1991, 2001) ) ) )
w_pot6h_2.to_netcdf('w_pot6h_2.nc')

w_pot6h_3 = w_pot6h.sel(time = (w_pot6h.time.dt.year.isin( range(2001, 2011) ) ) )
w_pot6h_3.to_netcdf('w_pot6h_3.nc')

w_pot6h_4 = w_pot6h.sel(time = (w_pot6h.time.dt.year.isin( range(2011, 2021) ) ) )
w_pot6h_4.to_netcdf('w_pot6h_4.nc')

w_pot6h_5 = w_pot6h.sel(time = (w_pot6h.time.dt.year == 2021) )
w_pot6h_5.to_netcdf('w_pot6h_5.nc')

# Use CDO to compute yearly averages of w_pot:
# cdo yearmean w_pot6h_1.nc w_pot_yearmean_1.nc
# cdo yearmean w_pot6h_2.nc w_pot_yearmean_2.nc
# cdo yearmean w_pot6h_3.nc w_pot_yearmean_3.nc
# cdo yearmean w_pot6h_4.nc w_pot_yearmean_4.nc
# cdo yearmean w_pot6h_5.nc w_pot_yearmean_5.nc

# Read and concatenate into one single dataset
w_pot_1 = xr.open_dataset('w_pot_yearmean_1.nc')['__xarray_dataarray_variable__']
w_pot_2 = xr.open_dataset('w_pot_yearmean_2.nc')['__xarray_dataarray_variable__']
w_pot_3 = xr.open_dataset('w_pot_yearmean_3.nc')['__xarray_dataarray_variable__']
w_pot_4 = xr.open_dataset('w_pot_yearmean_4.nc')['__xarray_dataarray_variable__']
w_pot_5 = xr.open_dataset('w_pot_yearmean_5.nc')['__xarray_dataarray_variable__']
w_pot = xr.concat( [w_pot_1, w_pot_2, w_pot_3, w_pot_4, w_pot_5], dim = 'time' )
w_pot = w_pot.rename('w_pot')

# Save as netcdf
w_pot.to_netcdf('w_pot_yearmean.nc')

# Similarly, split w100m6 into smaller datasets to save as netcdf
w100m6_1 = w100m6.sel(time = (w100m6.time.dt.year.isin( range(1981, 1991) ) ) )
w100m6_1.to_netcdf('wind_speed_1.nc')

w100m6_2 = w100m6.sel(time = (w100m6.time.dt.year.isin( range(1991, 2001) ) ) )
w100m6_2.to_netcdf('wind_speed_2.nc')

w100m6_3 = w100m6.sel(time = (w100m6.time.dt.year.isin( range(2001, 2011) ) ) )
w100m6_3.to_netcdf('wind_speed_3.nc')

w100m6_4 = w100m6.sel(time = (w100m6.time.dt.year.isin( range(2011, 2021) ) ) )
w100m6_4.to_netcdf('wind_speed_4.nc')

w100m6_5 = w100m6.sel(time = (w100m6.time.dt.year == 2021 ) )
w100m6_5.to_netcdf('wind_speed_5.nc') 

# Use CDO to compute yearly averages of w100m: 
# cdo yearmean wind_speed_1.nc w100m_yearmean_1.nc
# cdo yearmean wind_speed_2.nc w100m_yearmean_2.nc
# cdo yearmean wind_speed_3.nc w100m_yearmean_3.nc
# cdo yearmean wind_speed_4.nc w100m_yearmean_4.nc
# cdo yearmean wind_speed_5.nc w100m_yearmean_5.nc

# Then read back into python to create full 41-year dataset
w100m_ymean_1 = xr.open_dataset('w100m_yearmean_1.nc')['__xarray_dataarray_variable__']
w100m_ymean_2 = xr.open_dataset('w100m_yearmean_2.nc')['__xarray_dataarray_variable__']
w100m_ymean_3 = xr.open_dataset('w100m_yearmean_3.nc')['__xarray_dataarray_variable__']
w100m_ymean_4 = xr.open_dataset('w100m_yearmean_4.nc')['__xarray_dataarray_variable__']
w100m_ymean_5 = xr.open_dataset('w100m_yearmean_5.nc')['__xarray_dataarray_variable__']

w100m_ymean_cdo = xr.concat( [w100m_ymean_1, w100m_ymean_2, w100m_ymean_3, w100m_ymean_4, w100m_ymean_5], dim = 'time' )
w100m_ymean_cdo = w100m_ymean_cdo.rename('w100m')

# Finally, save to netcdf
w100m_ymean_cdo.to_netcdf('w100m_yearmean.nc')
