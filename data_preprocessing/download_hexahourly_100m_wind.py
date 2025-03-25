# Download data (u100m, v100m) from Copernicus ERA5 with API script. 
# Years from 1981 to 2021 (included). Hexahourly. 
# ERA5 hourly data on single levels from 1940 to present
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels

import cdsapi
import xarray as xr

c = cdsapi.Client()

for year in range(1981, 2022):
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '100m_u_component_of_wind', '100m_v_component_of_wind',
            ],
            'year': str(year),
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '06:00', '12:00',
                '18:00',
            ],
            'area': [
                73, -25, 25,
                50,
            ],
        },
        'ERA5_6h_100m_uv_components_'+str(year)+'.nc')
    
# Concatenate all years into one dataset 
wind_list = []
for year in range(1981, 2022):
    wind_list.append( xr.open_dataset('ERA5_6h_100m_uv_components_'+str(year)+'.nc') )
  
wind_dataset = xr.concat(wind_list, dim='time')

# Save as netcdf
wind_dataset.to_netcdf('ERA5_6h_100m_uv_components.nc')
