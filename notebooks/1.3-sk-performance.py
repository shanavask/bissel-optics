# %%
import sys
sys.path.append('..')
# %%
import os
import logging

import numpy as np
import pandas as pd
import pandas_gbq as pdq
from tqdm import tqdm
# import plotly_express as px

logging.basicConfig(level=logging.DEBUG)
pd.options.display.max_columns = None

import pipelines.utils.env as env
# %%
sql = f"""
    SELECT Date as date, Creative_ID__Facebook_Ads AS creative_id, Ad_Creative_Image_Hash__Facebook_Ads AS image_hash,
        Ad_Creative_Video_ID__Facebook_Ads AS video_id, Age__Facebook_Ads AS age, Gender__Facebook_Ads AS gender, 
        Impressions AS impressions, Clicks AS clicks, Cost AS cost, Platform_Conversions_USA AS conversions,
        Platform_Revenue_USA AS revenue, Leads AS leads, Complete_video_Views AS complete_video_views
    FROM `{env.CLIENT_PROJECT}.{env.CLIENT_DATASET}.{env.META_TABLE}`
    """
datadf = pdq.read_gbq(sql, project_id=env.CLIENT_PROJECT,)
datadf
# %%
sql = f"""
    SELECT *
    FROM `{env.CLIENT_PROJECT}.{env.OPTICS_DATASET}.meta_video_predictions`
    """
videodf = pdq.read_gbq(sql, project_id=env.CLIENT_PROJECT,)
videodf
# %%
predf = pd.merge(datadf, videodf, on='video_id')
predf
# %%
df = predf[['date', 'video_id', 'video_content_type', 'impressions', 'clicks']]
df = df[df['impressions'] > 0]
df['num_types'] = df['video_content_type'].apply(len)
df = df.explode('video_content_type', ignore_index=True)
df['impressions'] = df['impressions'] / df['num_types']
df['clicks'] = df['clicks'] / df['num_types']
df = df.groupby('video_content_type').agg({'impressions': 'sum', 'clicks': 'sum'}).reset_index()
df['ctr'] = df['clicks'] / df['impressions']
df
# %%
df.to_string()
# %%
from scipy.stats import chi2_contingency
df['non_clicks'] = df['impressions'] - df['clicks']
contingency_table = df[['clicks', 'non_clicks']].round().astype('int').values
contingency_table
# %%
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
chi2, p_value, dof, expected
# %%
predf[predf['video_content_type'].map(len)==0]
# %%
predf['video_content_type'].drop_duplicates()
# %%
predf['video_content_type'].drop_duplicates().to_list()
# %%
