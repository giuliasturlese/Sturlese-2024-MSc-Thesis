# (Jerez et al. 2015, Option 3)
# PV_pot = Perf_ratio * (G / 1000)
# Perf_ratio = 1 - 0.005*(T_cell - T_ref) 
# T_cell = 4.3 + 0.943*T + 0.028*G - 1.528*V
# T_ref = 25 deg Celsius

# Input file has following variables:
# - ssrd = surface solar radiation downwards (J m-2) 
# - u10 = u-component of 10m wind speed (m s-1)
# - v10 = v-component of 10m wind speed (m s-1)
# - t2m = 2m temperature (K)

# Calculate wind_speed and convert t2m to Celsius
cdo aexpr,'wind_speed=sqrt(v10*v10+u10*u10);t2m_celsius=t2m-273.15;' input.nc input_windspeed_celsius.nc

# Convert ssrd to W m-2 by dividing it by the accumulation time of 24 hours expressed in seconds
cdo aexpr,'G = ssrd / 86400;' input_windspeed_celsius.nc input_windspeed_celsius_solarpower.nc
rm input_windspeed_celsius.nc

# Calculate Tcell
cdo expr,'c2T = 0.943 * t2m_celsius; c3G = 0.028 * G; c4V = -1.528 * wind_speed;' input_windspeed_celsius_solarpower.nc termini_Tcell.nc
cdo expr,'T_cell = 4.3 + c2T + c3G + c4V;' termini_Tcell.nc Tcell.nc
rm termini_Tcell.nc 

# Calculate Perform_ratio and create file with input data, Tcell, Perform_ratio
cdo aexpr,'Perform_ratio = 1 - 0.005 * (T_cell - 25);' Tcell.nc Tcell_performance_ratio.nc
rm Tcell.nc
cdo merge input_windspeed_celsius_solarpower.nc Tcell_performance_ratio.nc full_performance_ratio.nc
rm input_windspeed_celsius_solarpower.nc
rm Tcell_performance_ratio.nc

# Calculate PV potential 
cdo aexpr,'PV_pot = Perform_ratio * G / 1000;' full_performance_ratio.nc PV_pot_mon.nc
rm full_performance_ratio.nc

# Calculate yearly averages of all variables in PV_pot
cdo yearmonmean PV_pot_mon.nc PV_pot_yearmonmean.nc
