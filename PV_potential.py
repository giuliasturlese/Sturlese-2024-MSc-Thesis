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
from thesis_functions import plot_map, plot_stereographic_map, coslat_weights, eof_solver, standardize_pcs, sig_regression_slope, corr_sig, pcorr, pcorr_df, full_pcorr_df

# ---------------------------------------------------------------------
# Read data
data1 = xr.open_dataset('PV_pot_yearmonmean.nc')
wind_speed = data1['wind_speed'] # 10m wind speed (m/s)
G = data1['G'] # surface solar radiation downward (W/m^2)
t2m = data1['t2m_celsius'] # 2m temperature 
PV_pot = data1['PV_pot'] # PV potential (yearly mean)

geopotential_dataset = xr.open_dataset('ERA5_geopotential_500hPa_yearmonmean.nc')
geopotential_darray = geopotential_dataset['z']
geopotential_darray = geopotential_darray/9.81 # 500 hPa geopotential height (m) northern hem

tot_cloud_cover_dataset = xr.open_dataset('ERA5_total_cloud_cover_yearmonmean_region.nc')
tot_cloud_cover_darray = tot_cloud_cover_dataset['tcc'] # total cloud cover (0 - 1) 

u_component_dataset = xr.open_dataset('ERA5_u_component_200hPa_yearmonmean.nc')
u_component_darray = u_component_dataset['u'] # 200 hPa u-component of wind (m/s) northern hem

# ---------------------------------------------------------------------
# Climatology of PV_pot, t2m, wind_speed, G, geop, tot_cloud_cover, u

# PV_pot
PV_pot_avg = sum(PV_pot) / len(PV_pot)
PV_pot_avg = PV_pot_avg.rename('PV$_{pot}$')

f = plt.figure()
plot_map(PV_pot_avg, title='Average PV$_{pot}$', x_step=25, y_offset=5, y_step=20, shave_top=3)
f.savefig("average_PVpot_yearmean.pdf", bbox_inches = "tight")
f = plt.figure()
plot_map(PV_pot_avg_yearsum, title='Average PV$_{pot}$', x_step=25, y_offset=5, y_step=20, shave_top=3)
f.savefig("average_PVpot_yearsum.pdf", bbox_inches = "tight")

# temperature
t2m_avg = sum(t2m) / len(t2m)
t2m_avg = t2m_avg.rename('t2m (C)')
f = plt.figure()
plot_map(t2m_avg, title='Average 2m temperature', x_step=25, y_offset=5, y_step=20, shave_top=3)
f.savefig("average_t2m.pdf", bbox_inches = "tight")

# wind speed
wind_speed_avg = sum(wind_speed) / len(wind_speed)
wind_speed_avg = wind_speed_avg.rename('10m wind speed (m/s)')
f = plt.figure()
plot_map(wind_speed_avg, title='Average 10m wind speed', x_step=25, y_offset=5, y_step=20, shave_top=3, max_value=4)
f.savefig("average_10m_wind_speed.pdf", bbox_inches = "tight")

# solar radiation downward
G_avg = sum(G) / len(G)
G_avg = G_avg.rename('G (W/m$^2$)')
f = plt.figure()
plot_map(G_avg, title='Average surface solar radiation downward', x_step=25, y_offset=5, y_step=20, shave_top=3)
f.savefig("average_G.pdf", bbox_inches = "tight")

# geopotential 
geop_north_avg = sum(geopotential_darray) / len(geopotential_darray)
geop_north_avg = geop_north_avg.rename('z (m)')
f = plt.figure()
plot_stereographic_map(geop_north_avg, title = "Average 500hPa geopotential height", gridlines='yes')
f.savefig('average_geop_north.pdf', bbox_inches = 'tight')

# total cloud cover
cloud_avg = sum(tot_cloud_cover_darray) / len(tot_cloud_cover_darray)
cloud_avg = cloud_avg.rename('tot cloud cover (0 - 1)')
f = plt.figure()
plot_map(cloud_avg, title='Average total cloud cover',x_step = 25, y_step = 10, y_offset=5, shave_top=3)
f.savefig('average_tot_cloud_cover.pdf', bbox_inches = 'tight')

# u-component of wind
u_north_avg = sum(u_component_darray) / len(u_component_darray)
u_north_avg = u_north_avg.rename('u-component of wind (m/s)')
f = plt.figure()
plot_stereographic_map(u_north_avg, title = "Average 200hPa u-component of wind", gridlines='yes')
f.savefig('average_u_north.pdf', bbox_inches = 'tight')

# ---------------------------------------------------------------------
# EOF and regression maps for PV_pot
solver = eof_solver(PV_pot)
solver_yearsum = eof_solver(PV_pot_yearsum)
neofs = 5
eofs = solver.eofs(neofs=neofs)
pcs = solver.pcs(npcs=neofs)
# Explained variance
eofvar = solver.varianceFraction(neigs = neofs) * 100
eofvar
# Standardize pcs
standardized_pc0 = standardize_pcs(pcs, 0).rename('PVpot_pc0')
standardized_pc1 = -standardize_pcs(pcs, 1).rename('PVpot_pc1')

# Time series plots
f = plt.figure()
standardized_pc0.plot.line(x='time')
plt.ylabel('')
plt.xlabel('')
plt.title('PC1 of PV$_{pot}$ (26.8%)')
f.savefig("PVpot_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
standardized_pc1.plot.line(x='time')
plt.ylabel('')
plt.xlabel('')
plt.title('PC1 of PV$_{pot}$ (15.3%)')
f.savefig("PVpot_pc1.pdf", bbox_inches = "tight")

# Regression maps 
sig_slope_PVpot_pc0 = sig_regression_slope(standardized_pc0, PV_pot, 'time')
sig_slope_PVpot_pc0 = sig_slope_PVpot_pc0.assign_attrs(long_name = "PV$_{pot}$")

sig_slope_PVpot_pc1 = sig_regression_slope(standardized_pc1, PV_pot, 'time')
sig_slope_PVpot_pc1 = sig_slope_PVpot_pc1.assign_attrs(long_name = "PV$_{pot}$")

f = plt.figure()
plot_map(sig_slope_PVpot_pc0, title='Regression PV$_{pot}$ on PC1', x_step=25, y_offset=5, y_step=20, shave_top=3)
f.savefig("significant_regression_PVpot_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_PVpot_pc1, title='Regression PV$_{pot}$ on PC2', x_step=25, y_offset=5, y_step=20, shave_top=3)
f.savefig("significant_regression_PVpot_pc1.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# EOF and regression maps for the 500hPa geopotential height 
solver_geop = eof_solver(geopotential_darray)
eofs_geop = solver_geop.eofs(neofs = neofs)
pcs_geop = solver_geop.pcs(npcs = neofs)
# Explained variance
eofvar_geop = solver_geop.varianceFraction(neigs = neofs) * 100
eofvar_geop
# Standardize pcs
standardized_pc0_geop = standardize_pcs(pcs_geop, 0).rename('geop_north_pc0')
standardized_pc1_geop = -standardize_pcs(pcs_geop, 1).rename('geop_north_pc1')
# Time series plots
f = plt.figure()
standardized_pc0_geop.plot.line(x="time")
plt.ylabel('') #('geopotential_pc0')
plt.xlabel('')
plt.yticks(np.arange(-2, 3, 1))
plt.title('PC1 of 500hPa geopotential height (26.5%)')
f.savefig("geopotential_north_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
standardized_pc1_geop.plot.line(x="time")
plt.yticks(np.arange(-2, 3, 1))
plt.ylabel('') #('geopotential_pc1')
plt.xlabel('')
plt.title('PC2 of 500hPa geopotential height (19.2%)')
f.savefig("geopotential_north_pc1.pdf", bbox_inches = "tight")

# Regression maps
sig_slope_geop_own_pc0 = sig_regression_slope(standardized_pc0_geop, geopotential_darray, 'time')
sig_slope_geop_own_pc0 = sig_slope_geop_own_pc0.assign_attrs(long_name = "geopotential height (m)")

sig_slope_geop_own_pc1 = sig_regression_slope(standardized_pc1_geop, geopotential_darray, 'time')
sig_slope_geop_own_pc1 = sig_slope_geop_own_pc1.assign_attrs(long_name = "geopotential height (m)")

f = plt.figure()
plot_stereographic_map(sig_slope_geop_own_pc0, title='Regression 500hPa z on PC1', gridlines='yes', min_value=-230)
f.savefig("significant_regression_geopotential_north_pc0_STEREOGRAPHIC.pdf", bbox_inches = "tight")

f = plt.figure()
plot_stereographic_map(sig_slope_geop_own_pc1, title='Regression 500hPa z on PC2', gridlines='yes', min_value=-230)
f.savefig("significant_regression_geopotential_north_pc1_STEREOGRAPHIC.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# EOF for tot_cloud_cover
solver_cloud = eof_solver(tot_cloud_cover_darray)
neofs = 5 
eofs_cloud = solver_cloud.eofs(neofs = neofs)
pcs_cloud = solver_cloud.pcs(npcs = neofs)
# Explained variance
eofvar_cloud = solver_cloud.varianceFraction(neigs = neofs) * 100
eofvar_cloud
# Standardize pcs
standardized_pc0_cloud = standardize_pcs(pcs_cloud, 0).rename('cloud_pc0')
standardized_pc1_cloud = -standardize_pcs(pcs_cloud, 1).rename('cloud_pc1')
# Time series plots 
f = plt.figure()
standardized_pc0_cloud.plot.line(x="time")
plt.yticks(np.arange(-3, 3, 1))
plt.ylabel('') #('tot_cloud_cover_pc0')
plt.title('PC1 of total cloud cover (18.2%)')
f.savefig("cloud_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
standardized_pc1_cloud.plot.line(x="time")
plt.yticks(np.arange(-3, 3, 1))
plt.ylabel('') #('tot_cloud_cover_pc1')
plt.title('PC2 of total cloud cover (12.8%)')
f.savefig("cloud_pc1.pdf", bbox_inches = "tight")

# Regression maps
sig_slope_cloud_own_pc0 = sig_regression_slope(standardized_pc0_cloud, tot_cloud_cover_darray, 'time')
sig_slope_cloud_own_pc0 = sig_slope_cloud_own_pc0.assign_attrs(long_name = "total cloud cover (0-1)")

sig_slope_cloud_own_pc1 = sig_regression_slope(standardized_pc1_cloud, tot_cloud_cover_darray, 'time')
sig_slope_cloud_own_pc1 = sig_slope_cloud_own_pc1.assign_attrs(long_name = "total cloud cover (0-1)")

f = plt.figure()
plot_map(sig_slope_cloud_own_pc0, title='Regression tot cloud cover on PC1', x_step = 25, y_step = 10, y_offset=5, shave_top=3)
f.savefig("significant_regression_tot_cloud_cover_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_cloud_own_pc1, title='Regression tot cloud cover on PC2', x_step = 25, y_step = 10, y_offset=5, shave_top=3)
f.savefig("significant_regression_tot_cloud_cover_pc1.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# EOF for u-component of wind
solver_u = eof_solver(u_component_darray)
eofs_u = solver_u.eofs(neofs = neofs)
pcs_u = solver_u.pcs(npcs = neofs)
# Explained variance
eofvar_u = solver_u.varianceFraction(neigs = neofs) * 100
eofvar_u
# Standardize pcs
standardized_pc0_u = standardize_pcs(pcs_u, 0).rename('u_north_pc0')
standardized_pc1_u = -standardize_pcs(pcs_u, 1).rename('u_north_pc1')
# Time series plots
f = plt.figure()
standardized_pc0_u.plot.line(x="time")
plt.yticks(np.arange(-2, 3.5, 1))
plt.ylabel('') #('u_component_pc0')
plt.title('PC1 of 200hPa u-component of wind (33%)')
f.savefig("u_component_north_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
standardized_pc1_u.plot.line(x="time")
plt.ylabel('') #('u_component_pc1')
plt.title('PC2 of 200hPa u-component of wind (13.6%)')
f.savefig("u_component_north_pc1.pdf", bbox_inches = "tight")

# Regression maps
sig_slope_u_pc0 = sig_regression_slope(standardized_pc0_u, u_component_darray, 'time')
sig_slope_u_pc0 = sig_slope_u_pc0.assign_attrs(long_name = "u component of wind (m/s)")

sig_slope_u_pc1 = sig_regression_slope(standardized_pc1_u, u_component_darray, 'time')
sig_slope_u_pc1 = sig_slope_u_pc1.assign_attrs(long_name = "u component of wind (m/s)")

f = plt.figure()
plot_stereographic_map(sig_slope_u_pc0, title='Regression 200hPa u on PC1', gridlines='yes', min_value=-10.2)
f.savefig("significant_regression_u_component_north_pc0.pdf_STEREOGRAPHIC.pdf", bbox_inches = "tight")

f = plt.figure()
plot_stereographic_map(sig_slope_u_pc1, title = 'Regression 200hPa u on PC2', gridlines='yes', min_value=-10.2)
f.savefig("significant_regression_u_component_north_pc1.pdf_STEREOGRAPHIC.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# EOF for t2m
solver_t2m = eof_solver(t2m)
eofs_t2m = solver_t2m.eofs(neofs = neofs)
pcs_t2m = solver_t2m.pcs(npcs = neofs)
# Explained variance
eofvar_t2m = solver_t2m.varianceFraction(neigs = neofs) * 100
eofvar_t2m
# Standardize pcs
standardized_pc0_t2m = standardize_pcs(pcs_t2m, 0).rename('t2m_pc0')
standardized_pc1_t2m = standardize_pcs(pcs_t2m, 1).rename('t2m_pc1')
# Time series plots
f = plt.figure()
standardized_pc0_t2m.plot.line(x="time")
plt.ylabel('')
plt.title('PC1 of t2m (54.8%)')
f.savefig("t2m_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
standardized_pc1_t2m.plot.line(x="time")
plt.ylabel('')
plt.title('PC2 of t2m (19.8%)')
f.savefig("t2m_pc1.pdf", bbox_inches = "tight")

# Regression maps 
sig_slope_t2m_pc0 = sig_regression_slope(standardized_pc0_t2m, t2m, 'time')
sig_slope_t2m_pc0 = sig_slope_t2m_pc0.assign_attrs(long_name = "t2m (C)")

sig_slope_t2m_pc1 = sig_regression_slope(standardized_pc1_t2m, t2m, 'time')
sig_slope_t2m_pc1 = sig_slope_t2m_pc1.assign_attrs(long_name = "t2m (C)")

f = plt.figure()
plot_map(sig_slope_t2m_pc0, title='Regression t2m on PC1', x_step=25, y_step=20, y_offset=5, shave_top=3, min_value=-7.3)
f.savefig("significant_regression_t2m_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_t2m_pc1, title='Regression t2m on PC2', x_step=25, y_step=20, y_offset=5, shave_top=3, min_value=-7.3)
f.savefig("significant_regression_t2m_pc1.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# EOF for G
solver_G = eof_solver(G)
eofs_G = solver_G.eofs(neofs = neofs)
pcs_G = solver_G.pcs(npcs = neofs)
# Explained variance
eofvar_G = solver_G.varianceFraction(neigs = neofs) * 100
eofvar_G
# Standardize pcs
standardized_pc0_G = standardize_pcs(pcs_G, 0).rename('G_pc0')
standardized_pc1_G = -standardize_pcs(pcs_G, 1).rename('G_pc1')
# Time series plots
f = plt.figure()
standardized_pc0_G.plot.line(x="time")
plt.yticks(np.arange(-3, 4, 1))
plt.ylabel('') #('G_PC1')
plt.title('PC1 of G (29.5%)')
f.savefig("G_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
standardized_pc1_G.plot.line(x="time")
plt.yticks(np.arange(-3, 4, 1))
plt.ylabel('') #('G_PC2')
plt.title('PC2 of G (14.6%)')
f.savefig("G_pc1.pdf", bbox_inches = "tight")

# Regression maps 
sig_slope_G_pc0 = sig_regression_slope(standardized_pc0_G, G, 'time')
sig_slope_G_pc0 = sig_slope_G_pc0.assign_attrs(long_name = "G (W/m$^2$)")

sig_slope_G_pc1 = sig_regression_slope(standardized_pc1_G, G, 'time')
sig_slope_G_pc1 = sig_slope_G_pc1.assign_attrs(long_name = "G (W/m$^2$)")

f = plt.figure()
plot_map(sig_slope_G_pc0, title='Regression G on PC1',x_step=25, y_step=20, y_offset=5, shave_top=3, min_value=-57)
f.savefig("significant_regression_G_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_G_pc1, title='Regression G on PC2',x_step=25, y_step=20, y_offset=5, shave_top=3, min_value=-57)
f.savefig("significant_regression_G_pc1.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# EOF for wind_speed
solver_wind = eof_solver(wind_speed)
eofs_wind = solver_wind.eofs(neofs = neofs)
pcs_wind = solver_wind.pcs(npcs = neofs)
# Explained variance
eofvar_wind = solver_wind.varianceFraction(neigs = neofs) * 100
eofvar_wind
# Standardize pcs
standardized_pc0_wind = standardize_pcs(pcs_wind, 0).rename('wind_speed_pc0')
standardized_pc1_wind = standardize_pcs(pcs_wind, 1).rename('wind_speed_pc1')
# Time series plots
f = plt.figure()
standardized_pc0_wind.plot.line(x="time")
plt.yticks(np.arange(-3, 4, 1))
plt.ylabel('') #('wind speed PC1')
plt.xlabel('')
plt.title('PC1 of 10m wind speed (23.9%)')
f.savefig("wind_speed_10m_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
standardized_pc1_wind.plot.line(x="time")
plt.yticks(np.arange(-3, 4, 1))
plt.xlabel('')
plt.ylabel('')
plt.title('PC2 of 10m wind speed (13.3%)')
f.savefig("wind_speed_10m_pc1.pdf", bbox_inches = "tight")

# Regression maps
sig_slope_wind_pc0 = sig_regression_slope(standardized_pc0_wind, wind_speed, 'time')
sig_slope_wind_pc0 = sig_slope_wind_pc0.assign_attrs(long_name = "wind speed (m/s)")

sig_slope_wind_pc1 = sig_regression_slope(standardized_pc1_wind, wind_speed, 'time')
sig_slope_wind_pc1 = sig_slope_wind_pc1.assign_attrs(long_name = "wind speed (m/s)")

f = plt.figure()
plot_map(sig_slope_wind_pc0, title='Regression 10m wind speed on PC1', x_step=25, y_step=20, y_offset=5, shave_top=3)
f.savefig("sig_regression_10m_wind_speed_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_wind_pc1, title='Regression 10m wind speed on PC2', x_step=25, y_step=20, y_offset=5, shave_top=3)
f.savefig("sig_regression_10m_wind_speed_pc1.pdf", bbox_inches = "tight")



# ---------------------------------------------------------------------
# Correlations table 

# Rename for clarity
standardized_pc0 = standardized_pc0.rename('PV$_{pot}$ PC1')
standardized_pc1 = standardized_pc1.rename('PV$_{pot}$ PC2')
standardized_pc0_G = standardized_pc0_G.rename('G PC1')
standardized_pc1_G = standardized_pc1_G.rename('G PC2')
standardized_pc0_t2m = standardized_pc0_t2m.rename('t2m PC1')
standardized_pc1_t2m = standardized_pc1_t2m.rename('t2m PC2')
standardized_pc0_wind = standardized_pc0_wind.rename('10m wind PC1')
standardized_pc1_wind = standardized_pc1_wind.rename('10m wind PC2')
standardized_pc0_cloud = standardized_pc0_cloud.rename('cloud cover PC1')
standardized_pc1_cloud = standardized_pc1_cloud.rename('cloud cover PC2')
standardized_pc0_geop = standardized_pc0_geop.rename('geopotential PC1')
standardized_pc1_geop = standardized_pc1_geop.rename('geopotential PC2')
standardized_pc0_u = standardized_pc0_u.rename('u-component PC1')
standardized_pc1_u = standardized_pc1_u.rename('u-component PC2')
nao_index = nao_index.rename('NAO index')

# Create pandas dataframe of PCs 
pc_tuple_final = (standardized_pc0, 
                  standardized_pc1, 
                  standardized_pc0_G, 
                  standardized_pc1_G, 
                  standardized_pc0_t2m,
                  standardized_pc1_t2m,
                  standardized_pc0_wind,
                  standardized_pc1_wind,
                  standardized_pc0_cloud,
                  standardized_pc1_cloud,
                  standardized_pc0_geop,
                  standardized_pc1_geop,
                  standardized_pc0_u,
                  standardized_pc1_u, 
                  nao_index)

pc_dict_final = {}
for i in pc_tuple_final: 
    pc_dict_final.update({i.name: i.values.tolist()})

pc_df_final = pd.DataFrame(pc_dict_final)

# Calculate correlations and p-values
corr_final = pc_df_final.corr().round(2)
p_values_final = corr_sig(pc_df_final)

# Lower triangle mask
mask_final1 = np.invert( np.tril( p_values_final <= 0.05, k = -1 ) )

# Plot correlation table
f = plt.figure()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_final, vmin=-1, vmax=1, cmap=cmap, mask=mask_final1, annot=True, annot_kws={"size":6})
plt.title('PC correlation matrix (p$\leq$0.05)')
# Add rectangle
ax = plt.gca()
rect = Rectangle((0.02,0.03),2,14.94,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
f.savefig("PC_correlation_matrix_lower_final.pdf", bbox_inches = "tight")



# ---------------------------------------------------------------------
# Partial correlations table

# Create partial correlations matrix
partial_correlations = full_pcorr_df(pc_df_final)
partial_corrs = partial_correlations.pivot_table(values='r', index=['X'], columns='Y', aggfunc='first')
# Also create p-values matrix in order to apply mask later
partial_pvals = partial_correlations.pivot_table(values='p-val', index=['X'], columns='Y', aggfunc='first')

partial_corrs.index.name = None
partial_corrs.columns.name = None
partial_pvals.index.name = None

# Rearrange rows in the same order as the correlation matrix rows
rows = ['PV$_{pot}$ PC1',
        'PV$_{pot}$ PC2',
        'G PC1',
        'G PC2',
        't2m PC1',
        't2m PC2',
        '10m wind PC1', 
        '10m wind PC2',
        'cloud cover PC1',
        'cloud cover PC2',
        'geopotential PC1',
        'geopotential PC2',
        'u-component PC1',
        'u-component PC2', 
        'NAO index']
partial_corrs = partial_corrs.reindex(rows)
partial_pvals = partial_pvals.reindex(rows)

# Rearrange columns to match rows
cols = rows
partial_corrs = partial_corrs[cols]
partial_corrs.to_html('partial_correlations.html')
partial_pvals = partial_pvals[cols]
partial_pvals.to_html('partial_corr_pvals.html')

# Plot partial correlation matrix as seaborn heatmap
mask_partial1 = np.invert( np.tril( partial_pvals <= 0.05, k = -1 ) )

f = plt.figure()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(partial_corrs, vmin=-1, vmax=1, cmap=cmap, mask = mask_partial1,annot=True, annot_kws={"size":6})
plt.title('PC partial correlation matrix (p$\leq$0.05)')
# Add rectangle
ax = plt.gca()
rect = Rectangle((0.02,0.03),2,14.94,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
f.savefig("PC_partial_correlation_matrix_sig1.pdf", bbox_inches = "tight")



# ---------------------------------------------------------------------
# Environment to circulation (E2C) 
# Regression of geopotential, total_cloud_cover, u_component on pc0 and pc1 of PV_pot 
sig_slope_geop_pc0 = sig_regression_slope(standardized_pc0, geopotential_darray, 'time')
sig_slope_geop_pc0 = sig_slope_geop_pc0.assign_attrs(long_name = "geopotential height (m)")

sig_slope_cloud_pc0 = sig_regression_slope(standardized_pc0, tot_cloud_cover_darray, 'time')
sig_slope_cloud_pc0 = sig_slope_cloud_pc0.assign_attrs(long_name = "total cloud cover (0-1)")

sig_slope_ucomp_pc0 = sig_regression_slope(standardized_pc0, u_component_darray, 'time')
sig_slope_ucomp_pc0 = sig_slope_ucomp_pc0.assign_attrs(long_name = "u-component of wind (m/s)")

sig_slope_geop_pc1 = sig_regression_slope(standardized_pc1, geopotential_darray, 'time')
sig_slope_geop_pc1 = sig_slope_geop_pc1.assign_attrs(long_name = "geopotential height (m)")

sig_slope_cloud_pc1 = sig_regression_slope(standardized_pc1, tot_cloud_cover_darray, 'time')
sig_slope_cloud_pc1 = sig_slope_cloud_pc1.assign_attrs(long_name = "total cloud cover (0-1)")

sig_slope_ucomp_pc1 = sig_regression_slope(standardized_pc1, u_component_darray, 'time')
sig_slope_ucomp_pc1 = sig_slope_ucomp_pc1.assign_attrs(long_name = "u-component of wind (m/s)")

# Plots of significant regression maps
f = plt.figure()
plot_stereographic_map(sig_slope_geop_pc0, title= 'Regression 500hPa z on PC1 of $PV_{pot}$', gridlines='yes', min_value=-180)
f.savefig("significant_regression_geop_PVpot_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_cloud_pc0, title='Regression tot cloud cover on PC1 of $PV_{pot}$', x_step = 25, y_step = 10, y_offset=5, shave_top=3)
f.savefig("significant_regression_tot_cloud_cover_PVpot_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_stereographic_map(sig_slope_ucomp_pc0, title= 'Regression 200hPa u on PC1 of $PV_{pot}$', gridlines='yes')
f.savefig("significant_regression_u_component_PVpot_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_stereographic_map(sig_slope_geop_pc1, title = 'Regression 500hPa z on PC2 of $PV_{pot}$', gridlines='yes', min_value=-180)
f.savefig("significant_regression_geop_PVpot_pc1.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_cloud_pc1, title='Regression tot cloud cover on PC2 of $PV_{pot}$', x_step = 25, y_step = 10, y_offset=5, shave_top=3)
f.savefig("significant_regression_tot_cloud_cover_PVpot_pc1.pdf", bbox_inches = "tight")

f = plt.figure()
plot_stereographic_map(sig_slope_ucomp_pc1, title= 'Regression 200hPa u on PC2 of $PV_{pot}$', gridlines='yes')
f.savefig("significant_regression_u_component_PVpot_pc1.pdf", bbox_inches = "tight")



# ---------------------------------------------------------------------
# Circulation to environment (C2E) 
# Regression of PV_pot on pc0 and pc1 of geop, tot_cloud_cover, u

# PV_pot - geop: 
sig_slope_PVpot_geop_pc0 = sig_regression_slope(standardized_pc0_geop, PV_pot, 'time')
sig_slope_PVpot_geop_pc0 = sig_slope_PVpot_geop_pc0.assign_attrs(long_name = "PV$_{pot}$")
sig_slope_PVpot_geop_pc1 = sig_regression_slope(standardized_pc1_geop, PV_pot, 'time')
sig_slope_PVpot_geop_pc1 = sig_slope_PVpot_geop_pc1.assign_attrs(long_name = "PV$_{pot}$")

# PV_pot - total cloud cover:
sig_slope_PVpot_cloud_pc0 = sig_regression_slope(standardized_pc0_cloud, PV_pot, 'time')
sig_slope_PVpot_cloud_pc0 = sig_slope_PVpot_cloud_pc0.assign_attrs(long_name = "PV$_{pot}$")
sig_slope_PVpot_cloud_pc1 = sig_regression_slope(standardized_pc1_cloud, PV_pot, 'time')
sig_slope_PVpot_cloud_pc1 = sig_slope_PVpot_cloud_pc1.assign_attrs(long_name = "PV$_{pot}$")

# PV_pot - u: 
sig_slope_PVpot_u_pc0 = sig_regression_slope(standardized_pc0_u, PV_pot, 'time')
sig_slope_PVpot_u_pc0 = sig_slope_PVpot_u_pc0.assign_attrs(long_name = "PV$_{pot}$")
sig_slope_PVpot_u_pc1 = sig_regression_slope(standardized_pc1_u, PV_pot, 'time')
sig_slope_PVpot_u_pc1 = sig_slope_PVpot_u_pc1.assign_attrs(long_name = "PV$_{pot}$")

# Plots of regression maps
# PV_pot - geop:
f = plt.figure()
plot_map(sig_slope_PVpot_geop_pc0, title='Regression PV$_{pot}$ on PC1 of 500hPa geop.height', x_step=25, y_offset=5, y_step=20, shave_top=3, min_value=-0.048)
f.savefig("sig_regr_PVpot_vs_geop_north_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_PVpot_geop_pc1, title='Regression PV$_{pot}$ on PC2 of 500hPa geop.height', x_step=25, y_offset=5, y_step=20, shave_top=3, min_value=-0.048)
f.savefig("sig_regr_PVpot_vs_geop_north_pc1.pdf", bbox_inches = "tight")

# PV_pot - total cloud cover: 
f = plt.figure()
plot_map(sig_slope_PVpot_cloud_pc0, title='Regression PV$_{pot}$ on PC1 of cloud cover', x_step=25, y_offset=5, y_step=20, shave_top=3, min_value=-0.048)
f.savefig("sig_regr_PVpot_vs_cloud_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_PVpot_cloud_pc1, title='Regression PV$_{pot}$ on PC2 of cloud cover', x_step=25, y_offset=5, y_step=20, shave_top=3, min_value=-0.048)
f.savefig("sig_regr_PVpot_vs_cloud_pc1.pdf", bbox_inches = "tight")

# PV_pot - u: 
f = plt.figure()
plot_map(sig_slope_PVpot_u_pc0, title='Regression PV$_{pot}$ on PC1 of 200hPa u-component', x_step=25, y_offset=5, y_step=20, shave_top=3, min_value=-0.048)
f.savefig("sig_regr_PVpot_vs_u_north_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_PVpot_u_pc1, title='Regression PV$_{pot}$ on PC2 of 200hPa u-component', x_step=25, y_offset=5, y_step=20, shave_top=3, min_value=-0.048)
f.savefig("sig_regr_PVpot_vs_u_north_pc1.pdf", bbox_inches = "tight")








