###### NOTE: This script should be added at the end of wind_potential.py. 
###### It is stored separately for clarity. 

# ---------------------------------------------------------------------
# Correlations w_pot vs pv_pot

# Rename PCs for clarity
standardized_pc0_wpot = standardized_pc0_wpot.rename('W$_{pot}$ PC1')
standardized_pc1_wpot = standardized_pc1_wpot.rename('W$_{pot}$ PC2')
standardized_pc0_PVpot = standardized_pc0_PVpot.rename('PV$_{pot}$ PC1')
standardized_pc1_PVpot = standardized_pc1_PVpot.rename('PV$_{pot}$ PC2')

# Create pandas dataframe of PCs 
pc_tuple_final = (standardized_pc0_PVpot, 
                  standardized_pc1_PVpot, 
                  standardized_pc0_wpot, 
                  standardized_pc1_wpot
                  )

pc_dict_final = {}
for i in pc_tuple_final: 
    pc_dict_final.update({i.name: i.values.tolist()})

pc_df_final = pd.DataFrame(pc_dict_final)

# Get correlations and p-values
corr_final = pc_df_final.corr().round(2)
p_values_final = corr_sig(pc_df_final)

# Make lower triangle mask
mask_final1 = np.invert( np.tril( p_values_final <= 0.05, k = -1 ) )

# Plot correlations table
f = plt.figure()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_final, vmin=-1, vmax=1, cmap=cmap, 
            mask=mask_final1, 
            annot=True, annot_kws={"size":13})
plt.title('PV$_{pot}-$W$_{pot}$ PC correlation matrix (p$\leq$0.05)')
f.savefig("PVpot_Wpot_PC_correlation_matrix.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# Partial correlations w_pot vs pv_pot 
partial_correlations = full_pcorr_df(pc_df_final)
# Create partial correlations matrix
partial_corrs = partial_correlations.pivot_table(values='r', index=['X'], columns='Y', aggfunc='first')
# Also create p-values matrix in order to apply mask later
partial_pvals = partial_correlations.pivot_table(values='p-val', index=['X'], columns='Y', aggfunc='first')
partial_corrs.index.name = None
partial_corrs.columns.name = None
partial_pvals.index.name = None

# Rearrange rows in the same order as the correlation matrix rows
rows = ['PV$_{pot}$ PC1', 
        'PV$_{pot}$ PC2',
        'W$_{pot}$ PC1',
        'W$_{pot}$ PC2',
        ]
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
            annot=True, annot_kws={"size":13})
plt.title('PV$_{pot}$-W$_{pot}$ PC partial correlation matrix (p$\leq$0.05)')
f.savefig("PVpot_Wpot_PC_partial_correlation_matrix_sig.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# Regression of w_pot on pc0 and pc1 of pv_pot
# w_pot = a * standardized_pc0_PVpot + b

# Make dates the same for PV_pot pc0, pc1
# Define new time coordinate
date_int = range(1981, 2022)
date_str = [str(i) for i in date_int]
date = pd.to_datetime(date_str, format="%Y")
# Assign new time coordinate to the variables
pv_pot = pv_pot.assign_coords(time = date)
w_pot = w_pot.assign_coords(time = date)
standardized_pc0_wpot = standardized_pc0_wpot.assign_coords(time = date)
standardized_pc1_wpot = standardized_pc1_wpot.assign_coords(time = date)
standardized_pc0_PVpot = standardized_pc0_PVpot.assign_coords(time = date)
standardized_pc1_PVpot = standardized_pc1_PVpot.assign_coords(time = date)

# Get regression slopes
sig_slope_wpot_pvpot_pc0 = sig_regression_slope(standardized_pc0_PVpot, w_pot, 'time').assign_attrs(long_name = 'W$_{pot}$')

sig_slope_wpot_pvpot_pc1 = sig_regression_slope(standardized_pc1_PVpot, w_pot, 'time').assign_attrs(long_name = 'W$_{pot}$')

# Plot regression maps
f = plt.figure()
plot_map(sig_slope_wpot_pvpot_pc0, title='Regression W$_{pot}$ on PC1 of PV$_{pot}$', shave_top=4)
f.savefig("sig_regression_wpot_on_PVpot_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_wpot_pvpot_pc1, title='Regression W$_{pot}$ on PC2 of PV$_{pot}$', shave_top=4, max_value = 164)
f.savefig("sig_regression_wpot_on_PVpot_pc1.pdf", bbox_inches = "tight")

# ---------------------------------------------------------------------
# Regression of pv_pot on pc0 and pc1 of w_pot
# PV_pot = a * standardized_pc0_wpot + b

# Make dates the same for PV_pot pc0, pc1
# Define new time coordinate
date_int = range(1981, 2022)
date_str = [str(i) for i in date_int]
date = pd.to_datetime(date_str, format="%Y")
# Assign new time coordinate to the variables
pv_pot = pv_pot.assign_coords(time = date)
w_pot = w_pot.assign_coords(time = date)
standardized_pc0_wpot = standardized_pc0_wpot.assign_coords(time = date)
standardized_pc1_wpot = standardized_pc1_wpot.assign_coords(time = date)
standardized_pc0_PVpot = standardized_pc0_PVpot.assign_coords(time = date)
standardized_pc1_PVpot = standardized_pc1_PVpot.assign_coords(time = date)

# Get regression slopes
sig_slope_pvpot_wpot_pc0 = sig_regression_slope(standardized_pc0_wpot, pv_pot, 'time').assign_attrs(long_name = 'PV$_{pot}$')

sig_slope_pvpot_wpot_pc1 = sig_regression_slope(standardized_pc1_wpot, pv_pot, 'time').assign_attrs(long_name = 'PV$_{pot}$')

# Plots of regression maps
f = plt.figure()
plot_map(sig_slope_pvpot_wpot_pc0, title='Regression PV$_{pot}$ on PC1 of W$_{pot}$', shave_top=4)
f.savefig("sig_regression_PVpot_on_wpot_pc0.pdf", bbox_inches = "tight")

f = plt.figure()
plot_map(sig_slope_pvpot_wpot_pc1, title='Regression PV$_{pot}$ on PC2 of W$_{pot}$', shave_top=4, max_value=0.42)
f.savefig("sig_regression_PVpot_on_wpot_pc1.pdf", bbox_inches = "tight")



