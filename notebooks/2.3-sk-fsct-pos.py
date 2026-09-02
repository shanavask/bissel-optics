# %%
import pandas as pd
import plotly.express as px


# %%
prod_df = pd.read_csv('../data/fcst/product_types.csv')
prod_df = prod_df[['Product ID', 'Product Type']]
prod_df.head()
# %%
kpi_df = pd.read_csv('../data/fcst/product_type_kpi_month.csv')
kpi_df['date'] = pd.to_datetime(kpi_df['MONTH_YEAR'], format='%m-%Y')
kpi_df.head()
# %%
pos_df_raw = pd.read_excel('../data/fcst/expected_pos.xlsx', header=[1, 2])
pos_df_raw.columns = ['_'.join(map(str, col)).strip('_') for col in pos_df_raw.columns.values]
pos_df = pos_df_raw[~pos_df_raw['Unnamed: 2_level_0_ASIN'].isna()].copy()

month_names = [
    "JANUARY", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
pos_df = pos_df.drop('December_Total', axis=1)
for month in month_names:
    cols = [u for u in pos_df.columns if month in u]
    pos_df[cols] = pos_df[cols].replace(' $-   ', 0).replace('$-   ', 0).astype(float)
    pos_df[month] = pos_df[cols].sum(axis=1)

pos_df.rename(columns={'Unnamed: 2_level_0_ASIN': 'asin', 'Unnamed: 3_level_0_Model ': 'model'}, inplace=True)
pos_df = pos_df[['asin', 'model']+month_names]

pos_df = pd.merge(pos_df, prod_df, left_on='asin', right_on='Product ID', how='inner')
pos_df = pos_df.groupby('Product Type').agg({month: 'sum' for month in month_names}).reset_index()
pos_df = pos_df.set_index('Product Type')
pos_df.columns = pd.date_range(start='2026-01-01', periods=12, freq='MS').strftime('%Y-%m-%d')
pos_df = pd.melt(pos_df.reset_index(), id_vars='Product Type', var_name='month', value_name='pos')
pos_df

# %%
pdf = kpi_df[kpi_df['PRODUCT_RANGE'] == 'PDC']
pdf
# %%
# %%
a = pos_df[pos_df['Product Type'] == 'PDC']
fig = px.line(a, x='month', y='pos')
fig.add_scatter(x=pdf['date'], y=pdf['TOTAL_SALES'])
fig.show()
# %%
pos_df
# %%
pos_df.to_parquet('../data/fcst/pos_df.parquet')
# %%
