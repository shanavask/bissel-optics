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
import json

logging.basicConfig(level=logging.DEBUG)
pd.options.display.max_columns = None

import pipelines.utils.env as env

from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.advideo import AdVideo
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.ad import Ad

# %%
project = env.PROJECT
client_project = env.CLIENT_PROJECT
client_dataset = env.CLIENT_DATASET
optics_dataset = env.OPTICS_DATASET
meta_table = env.META_TABLE
meta_video_table = env.META_VIDEO_TABLE
creative_type = "video"
channel = "meta"
access_token = env.META_TOKEN
# %%


# %%
#     JOIN `{client_project}.{optics_dataset}.{meta_video_table}` AS meta
    # ON meta.creative_id = Creative_ID__Facebook_Ads AND meta.video_id = Ad_Creative_Video_ID__Facebook_Ads
sql = f"""
    SELECT *
    FROM `{client_project}.{client_dataset}.{meta_table}`
    # JOIN `{client_project}.{optics_dataset}.{meta_video_table}` AS meta
    # ON meta.creative_id = Creative_ID__Facebook_Ads AND meta.video_id = Ad_Creative_Video_ID__Facebook_Ads
    WHERE Ad_Creative_Video_ID__Facebook_Ads IS NOT NULL
    AND Date="2026-01-01"
    # AND Ad_Creative_Video_ID__Facebook_Ads='1013093554203510'
    """
actdf = pdq.read_gbq(sql,
    project_id=project,
    use_bqstorage_api=True,)
actdf
# %%
actdf['Cost'].sum()
# %%
4173.639999/4193.259999
# %%

# %%
errordf = pdq.read_gbq(
            f"SELECT video_id, error, attempt_count, logged_at FROM `{client_project}.{optics_dataset}.{meta_video_table}_errors`",
            project_id=project,
            use_bqstorage_api=True,
        )
errordf
# %%
errordf['attempt_count'] = 1
errordf
# %%
pdq.to_gbq(errordf, f"{client_project}.{optics_dataset}.{meta_video_table}_errors", project_id=project, if_exists="replace")
# %%
FacebookAdsApi.init(access_token=access_token)
api = FacebookAdsApi.get_default_api()
accounts_resp = api.call("GET", ("me", "accounts"), params={"fields": "id,name,access_token"})
page_tokens = {p["id"]: p["access_token"] for p in accounts_resp.json().get("data", [])}
page_tokens
# %%
fields = [
        "id", "title", "description", "source", "length",
        "views", "created_time", "permalink_url",
    ]

video_id = "1013093554203510"
video = AdVideo(video_id)
videodata = video.api_get(fields=fields)
videodata

# %%
actdf[actdf['Ad_Creative_Video_ID__Facebook_Ads']==video_id]['Ad_Creative_Video_Permalink_URL__Facebook_Ads'].iloc[0]
# %%
page_id='38870020038'
page_id='103873948584319'
page_id='215517661811694'
page_token = page_tokens[page_id]
FacebookAdsApi.init(access_token=page_token)
# %%
fields = ['id','source']
video_ids = ['1013093554203510']
for vid in video_ids:
    vdata = AdVideo(vid).api_get(fields=fields)
    print(vid, vdata.get("source"))

# %%
errordf
# %%
