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
w_pot = xr.open_dataset('w_pot_yearmean.nc')['w_pot']
w_pot = w_pot.rename('W$_{pot}$')   # wind potential (yearly averaged)

w100m_ymean_cdo = xr.open_dataset('w100m_yearmean.nc')['w100m']
w100m_ymean_cdo = w100m_ymean_cdo.rename('wind speed (m/s)')

geopotential_dataset = xr.open_dataset('ERA5_geopotential_500hPa_yearmonmean.nc')
geop500hPa = geopotential_dataset['z']
geop500hPa = geop500hPa/9.81        # 500hPa geop height (m), Northern Hemisphere

u_component_dataset = xr.open_dataset('ERA5_u_component_200hPa_yearmonmean.nc')
u200hPa = u_component_dataset['u']  # 200 hPa u-component of wind (m/s), Northern Hemisphere

pv_pot = xr.open_dataset('PV_pot_yearmonmean.nc')['PV_pot']
pv_pot = pv_pot.rename('PV$_{pot}$') # PV potential (yearly averaged)

# ---------------------------------------------------------------------
# w_pot climatology
w_pot_avg = sum(w_pot) / len(w_pot) 
w_pot_avg = w_pot_avg.rename('W$_{pot}$')

f = plt.figure()
plot_map(w_pot_avg, title='Average W$_{pot}$', x_step=25, y_offset=5, y_step=20, shave_top=4)
f.savefig("average_Wpot_yearmean.pdf", bbox_inches = "tight")

# w100m climatology
w100m_avg = sum(w100m_ymean_cdo) / len(w100m_ymean_cdo)

f = plt.figure()
plot_map(w100m_avg, title='Average 100m wind speed', x_step=25, y_offset=5, y_step=20, shave_top=4)
f.savefig("average_w100m.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# EOF and regression maps for w_pot
solver = eof_solver(w_pot)
neofs = 5
eofs = solver.eofs(neofs=neofs)
pcs = solver.pcs(npcs=neofs)
# Explained variance
eofvar = solver.varianceFraction(neigs = neofs) * 100
eofvar
# Standardize PCs
standardized_pc0_wpot = standardize_pcs(pcs, 0).rename('Wpot_pc0')
standardized_pc1_wpot = standardize_pcs(pcs, 1).rename('Wpot_pc1')

# Time series plots
f = plt.figure()
standardized_pc0_wpot.plot.line(x="time")
plt.ylabel('') 
plt.xlabel('')
plt.title('PC1 of W$_{pot}$ (26.1%)')
f.savefig("Wpot_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
standardized_pc1_wpot.plot.line(x="time")
plt.ylabel('') 
plt.xlabel('')
plt.title('PC2 of W$_{pot}$ (11.0%)')
f.savefig("Wpot_pc1.pdf", bbox_inches = "tight")

# Regression maps 
sig_slope_Wpot_pc0 = sig_regression_slope(standardized_pc0_wpot, w_pot, 'time')
sig_slope_Wpot_pc0 = sig_slope_Wpot_pc0.assign_attrs(long_name = "W$_{pot}$")

sig_slope_Wpot_pc1 = sig_regression_slope(standardized_pc1_wpot, w_pot, 'time')
sig_slope_Wpot_pc1 = sig_slope_Wpot_pc1.assign_attrs(long_name = "W$_{pot}$")

f = plt.figure()
plot_map(sig_slope_Wpot_pc0, title='Regression W$_{pot}$ on PC1', x_step=25, y_offset=5, y_step=20, shave_top=4)
f.savefig("significant_regression_Wpot_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_Wpot_pc1, title='Regression W$_{pot}$ on PC2', x_step=25, y_offset=5, y_step=20, shave_top=4)
f.savefig("significant_regression_Wpot_pc1.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# EOF and regression maps for w100m
solver = eof_solver(w100m_ymean_cdo)
neofs = 5
eofs = solver.eofs(neofs=neofs)
pcs = solver.pcs(npcs=neofs)
# Explained variance
eofvar = solver.varianceFraction(neigs = neofs) * 100
eofvar
# Standardize PCs
standardized_pc0_w100m = standardize_pcs(pcs, 0).rename('w100m_pc0')
standardized_pc1_w100m = standardize_pcs(pcs, 1).rename('w100m_pc1')

# Time series plots
f = plt.figure()
standardized_pc0_w100m.plot.line(x="time")
plt.ylabel('') 
plt.title('PC1 of 100m wind speed (30.9%)')
f.savefig("w100m_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
standardized_pc1_w100m.plot.line(x="time")
plt.ylabel('') 
plt.title('PC2 of 100m wind speed (11.2%)')
f.savefig("w100m_pc1.pdf", bbox_inches = "tight")

# Regression maps
sig_slope_w100m_pc0 = sig_regression_slope(standardized_pc0_w100m, w100m_ymean_cdo, 'time')
sig_slope_w100m_pc0 = sig_slope_Wpot_pc0.assign_attrs(long_name = "wind speed (m/s)")

sig_slope_w100m_pc1 = sig_regression_slope(standardized_pc1_w100m, w100m_ymean_cdo, 'time')
sig_slope_w100m_pc1 = sig_slope_Wpot_pc1.assign_attrs(long_name = "wind speed (m/s)")

f = plt.figure()
plot_map(sig_slope_w100m_pc0, title='Regression 100m wind speed on PC1', x_step=25, y_offset=4, y_step=20, shave_top=4)
f.savefig("sig_regression_w100m_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_w100m_pc1, title='Regression 100m wind speed on PC2', x_step=25, y_offset=5, y_step=20, shave_top=4)
f.savefig("sig_regression_w100m_pc1.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# Get PCs of other variables (for correlations and C2E regression maps)

# EOF for 500hPa geopotential
solver = eof_solver(geop500hPa)
neofs = 5
pcs = solver.pcs(npcs=neofs)
standardized_pc0_geop = standardize_pcs(pcs, 0).rename('geop_pc0')
standardized_pc1_geop = -standardize_pcs(pcs, 1).rename('geop_pc1')

# EOF for 200hPa u-component of wind
solver = eof_solver(u200hPa)
neofs = 5
pcs = solver.pcs(npcs=neofs)
standardized_pc0_u = standardize_pcs(pcs, 0).rename('u_pc0')
standardized_pc1_u = -standardize_pcs(pcs, 1).rename('u_pc1')

# EOF for PV_pot
solver = eof_solver(pv_pot)
neofs = 5
pcs = solver.pcs(npcs=neofs)
standardized_pc0_PVpot = standardize_pcs(pcs, 0).rename('PV$_{pot}$')
standardized_pc1_PVpot = -standardize_pcs(pcs, 1).rename('PV$_{pot}$')

# ---------------------------------------------------------------------
# E2C: Regression of geop, u on pc0 and pc1 of w_pot
# geop500hPa = a * standardized_pc0_wpot + b
# u200hPa = a * standardized_pc0_wpot + b

# Define new time coordinate
date_int = range(1981, 2022)
date_str = [str(i) for i in date_int]
date = pd.to_datetime(date_str, format="%Y")

# Assign new time coordinate to the variables
geop500hPa = geop500hPa.assign_coords(time = date)
u200hPa = u200hPa.assign_coords(time = date)
w_pot = w_pot.assign_coords(time = date)
standardized_pc0_wpot = standardized_pc0_wpot.assign_coords(time = date)
standardized_pc1_wpot = standardized_pc1_wpot.assign_coords(time = date)

# Get regression slopes 
sig_slope_geop_wpot_pc0 = sig_regression_slope(standardized_pc0_wpot, geop500hPa, 'time').assign_attrs(long_name = "geopotential height (m)")

sig_slope_u200hPa_wpot_pc0 = sig_regression_slope(standardized_pc0_wpot, u200hPa, 'time').assign_attrs(long_name = "u-component of wind (m/s)")

sig_slope_geop_wpot_pc1 = sig_regression_slope(standardized_pc1_wpot, geop500hPa, 'time').assign_attrs(long_name = "geopotential height (m)")

sig_slope_u200hPa_wpot_pc1 = sig_regression_slope(standardized_pc1_wpot, u200hPa, 'time').assign_attrs(long_name = "u-component of wind (m/s)")

# Plot regression maps 
f = plt.figure()
plot_stereographic_map(sig_slope_geop_wpot_pc0, title= 'Regression 500hPa z on PC1 of W$_{pot}$', gridlines='yes')
f.savefig("si_regression_geop_Wpot_pc0_STEREOGRAPHIC.pdf", bbox_inches = "tight")

f = plt.figure()
plot_stereographic_map(sig_slope_u200hPa_wpot_pc0, title= 'Regression 200hPa u on PC1 of W$_{pot}$', gridlines='yes')
f.savefig("sig_regression_u_component_Wpot_pc0_STEREOGRAPHIC.pdf", bbox_inches = "tight")

f = plt.figure()
plot_stereographic_map(sig_slope_geop_wpot_pc1, title = 'Regression 500hPa z on PC2 of W$_{pot}$', gridlines='yes', min_value=-200)
f.savefig("sig_regression_geop_Wpot_pc1_STEREOGRAPHIC.pdf", bbox_inches = "tight")

f = plt.figure()
plot_stereographic_map(sig_slope_u200hPa_wpot_pc1, title= 'Regression 200hPa u on PC2 of W$_{pot}$',gridlines='yes', max_value=9.1)
f.savefig("sig_regression_u_component_Wpot_pc1_STEREOGRAPHIC.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# C2E: Regression of w_pot on pc0 and pc1 of geop, u
# w_pot = a * standardized_pc0_w100m + b
# w_pot = a * standardized_pc0_geop + b
# w_pot = a * standardized_pc0_u + b

# Make dates the same for w100m , geop, u pc0, pc1
# Define new time coordinate
date_int = range(1981, 2022)
date_str = [str(i) for i in date_int]
date = pd.to_datetime(date_str, format="%Y")
# Assign new time coordinate to the variables
standardized_pc0_w100m = standardized_pc0_w100m.assign_coords(time = date)
standardized_pc1_w100m = standardized_pc1_w100m.assign_coords(time = date)
standardized_pc0_geop = standardized_pc0_geop.assign_coords(time = date)
standardized_pc1_geop = standardized_pc1_geop.assign_coords(time = date)
standardized_pc0_u = standardized_pc0_u.assign_coords(time = date)
standardized_pc1_u = standardized_pc1_u.assign_coords(time = date)

# Get regression slopes 
sig_slope_wpot_w100m_pc0 = sig_regression_slope(standardized_pc0_w100m, w_pot, 'time').assign_attrs(long_name = 'W$_{pot}$')

sig_slope_wpot_geop_pc0 = sig_regression_slope(standardized_pc0_geop, w_pot, 'time').assign_attrs(long_name = 'W$_{pot}$')

sig_slope_wpot_u_pc0 = sig_regression_slope(standardized_pc0_u, w_pot, 'time').assign_attrs(long_name = "W$_{pot}$")

sig_slope_wpot_w100m_pc1 = sig_regression_slope(standardized_pc1_w100m, w_pot, 'time').assign_attrs(long_name = 'W$_{pot}$')

sig_slope_wpot_geop_pc1 = sig_regression_slope(standardized_pc1_geop, w_pot, 'time').assign_attrs(long_name = 'W$_{pot}$')

sig_slope_wpot_u_pc1 = sig_regression_slope(standardized_pc1_u, w_pot, 'time').assign_attrs(long_name = "W$_{pot}$")

# Plot regression maps
f = plt.figure()
plot_map(sig_slope_wpot_w100m_pc0, title='Regression W$_{pot}$ on PC1 of 100m wind speed', shave_top=4)
f.savefig("sig_regression_wpot_w100m_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_wpot_geop_pc0, title='Regression W$_{pot}$ on PC1 of 500hPa z', shave_top=4)
f.savefig("sig_regression_wpot_geop_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_wpot_u_pc0, title='Regression W$_{pot}$ on PC1 of 200hPa u', shave_top=4)
f.savefig("sig_regression_wpot_u_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_wpot_w100m_pc1, title='Regression W$_{pot}$ on PC2 of 100m wind speed', shave_top=4)
f.savefig("sig_regression_wpot_w100m_pc1.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_wpot_geop_pc1, title='Regression W$_{pot}$ on PC2 of 500hPa z', shave_top=4)
f.savefig("sig_regression_wpot_geop_pc1.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_wpot_u_pc1, title='Regression W$_{pot}$ on PC2 of 200hPa u',shave_top=4)
f.savefig("sig_regression_wpot_u_pc1.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# PC correlation matrix for w_pot

# Rename for clarity
standardized_pc0_wpot = standardized_pc0_wpot.rename('W$_{pot}$ PC1')
standardized_pc1_wpot = standardized_pc1_wpot.rename('W$_{pot}$ PC2')
standardized_pc0_w100m = standardized_pc0_w100m.rename('w100m PC1')
standardized_pc1_w100m = standardized_pc1_w100m.rename('w100m PC2')
standardized_pc0_geop = standardized_pc0_geop.rename('geop PC1')
standardized_pc1_geop = standardized_pc1_geop.rename('geop PC2')
standardized_pc0_u = standardized_pc0_u.rename('u PC1')
standardized_pc1_u = standardized_pc1_u.rename('u PC2')
nao_index = nao_index.rename('NAO index')

# Create pandas dataframe of PCs 
pc_tuple_final = (standardized_pc0_wpot, 
                  standardized_pc1_wpot,
                  standardized_pc0_w100m,
                  standardized_pc1_w100m,
                  standardized_pc0_geop,
                  standardized_pc1_geop,
                  standardized_pc0_u,
                  standardized_pc1_u, 
                  nao_index)

pc_dict_final = {}
for i in pc_tuple_final: 
    pc_dict_final.update({i.name: i.values.tolist()})

pc_df_final = pd.DataFrame(pc_dict_final)

# Get correlations and p-values
corr_final = pc_df_final.corr().round(2)
p_values_final = corr_sig(pc_df_final)

# Lower triangle mask
mask_final1 = np.invert( np.tril( p_values_final <= 0.05, k = -1 ) )

# Plot correlations table
f = plt.figure()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_final, vmin=-1, vmax=1, cmap=cmap, 
            mask=mask_final1, 
            annot=True, annot_kws={"size":9})
plt.title('W$_{pot}$ PC correlation matrix (p$\leq$0.05)')
# Add rectangle
ax = plt.gca()
rect = Rectangle((0.02,0.03),2,8.945,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
f.savefig("Wpot_PC_correlation_matrix_lower_final.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# PC partial correlation matrix for w_pot
partial_correlations = full_pcorr_df(pc_df_final)
# Create partial correlations matrix
partial_corrs = partial_correlations.pivot_table(values='r', index=['X'], columns='Y', aggfunc='first')
# Also create p-values matrix in order to apply mask later
partial_pvals = partial_correlations.pivot_table(values='p-val', index=['X'], columns='Y', aggfunc='first')
partial_corrs.index.name = None
partial_corrs.columns.name = None
partial_pvals.index.name = None

# Rearrange rows in the same order as the correlation matrix rows
rows = ['W$_{pot}$ PC1',
        'W$_{pot}$ PC2',
        'w100m PC1',
        'w100m PC2',
        'geop PC1',
        'geop PC2',
        'u PC1', 
        'u PC2', 
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
sns.heatmap(partial_corrs, vmin=-1, vmax=1, cmap=cmap, 
            mask = mask_partial1,
            annot=True, annot_kws={"size":9})
plt.title('W$_{pot}$ PC partial correlation matrix (p$\leq$0.05)')
# Add rectangle
ax = plt.gca()
rect = Rectangle((0.02,0.03),2,8.945,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
f.savefig("Wpot_PC_partial_correlation_matrix_sig.pdf", bbox_inches = "tight")
















