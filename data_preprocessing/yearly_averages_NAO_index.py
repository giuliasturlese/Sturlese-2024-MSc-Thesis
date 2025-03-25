# Yearly averages of NAO index 

# Import libraries 
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt

# Read data
nao_index_full_df = pd.read_table('NAO_index.txt')

# Calculate yearly mean
nao_noyear = nao_index_full_df.drop('Year', axis=1)
nao_index_full_df['yearmean'] = nao_noyear.mean(axis=1)
nao_index_df = nao_index_full_df[['Year', 'yearmean']]

# Convert 'Year' to datetime
nao_index_df['Year'] = nao_index_df['Year'].astype(str)
nao_index_df['Year'] = pd.to_datetime(nao_index_df['Year'])

# Set 'Year' column as index 
nao_index_df = nao_index_df.set_index('Year')

# Convert to xarray DataArray 
nao_index = nao_index_df['yearmean'].to_xarray()

# Plot NAO index
f = plt.figure()
nao_index.plot.line(x="Year")
plt.ylabel('') #('PV$_{pot}$ PC1')
plt.xlabel('') # ('time')
plt.title('NAO index (1981-2021)')
f.savefig("NAO_index.pdf", bbox_inches = "tight")
