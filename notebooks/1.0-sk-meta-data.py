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
sql = f"""
    SELECT *
    FROM `{client_project}.{client_dataset}.{meta_table}`
    # LEFT JOIN `{client_project}.{optics_dataset}.{meta_video_table}` AS meta
    # ON meta.video_id = Ad_Creative_Video_ID__Facebook_Ads
    # WHERE meta.video_id IS NOT NULL
    # AND Ad_Creative_Video_ID__Facebook_Ads IS NOT NULL
    # AND Ad_Creative_Video_Permalink_URL__Facebook_Ads IS NOT NULL
    LIMIT 100
    """
sql = f"""
    SELECT *
    FROM `{client_project}.{client_dataset}.{meta_table}`
    WHERE Ad_Creative_Video_ID__Facebook_Ads IS NOT NULL
    AND Date>"2026-01-01"
    """
sql = f"""
    SELECT * 
    FROM `jf-tech-investanalytics-prod.funnel_export_bissell.funnel_usa_ad_level_no_ga_meta_creative`
    WHERE Ad_Creative_Video_ID__Facebook_Ads IS NOT NULL
    # AND Date>"2026-01-01"
    LIMIT 100
#    AND Ad_Creative_Video_ID__Facebook_Ads='1034273606045256'
    """
actdf = pdq.read_gbq(sql,
    project_id=project,
    use_bqstorage_api=True,)
actdf
# %%
actdf
# %%
actdf[actdf['Creative_ID__Facebook_Ads'].str.contains('625591193813695')]
# %%
errors_df = pdq.read_gbq(
            f"SELECT video_id, error, attempt_count, logged_at FROM `{client_project}.{optics_dataset}.{meta_video_table}_errors`",
            project_id=project,
            use_bqstorage_api=True,
        )
errors_df
# %%
actdf['Date']
# %%
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.advideo import AdVideo
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.ad import Ad

FacebookAdsApi.init(access_token=access_token)
# %%
fields = ['campaign_id', 'ad_id', 'impressions', 'clicks', 'spend', 'reach']
columns = ['account_id', 'date_start', 'date_stop']+fields

account_id = '2409630422470729'
a_dt = '2026-02-12'
b_dt = '2026-02-13'
params = {
    'time_range': {'since': a_dt,'until': b_dt},
    'level': 'ad',
    'time_increment': 1,
}

async_job = AdAccount(f'act_{account_id}').get_insights(
    fields=fields, params=params, is_async=True)

# %%
status = async_job.api_get()['async_status']
status
# %%
data = async_job.get_result()
datadf = pd.DataFrame([x for x in data])
datadf
# %%
datadf['ad_id'].nunique()
# %%
fields = [
    'id',
    'name',
    'body',
    'effective_object_story_id',
    'object_type',
    'title',
    'image_hash',
    'image_url',
    'link_url',
    'template_url',
    'object_story_spec',
    'call_to_action_type'
]

createlist = []
from time import sleep
# for x, row in datadf.iterrows():
try:
    ad = Ad(fbid=120252386073480577)
    # ad = Ad(fbid=row['ad_id'])
    creatives = ad.get_ad_creatives(fields=fields)
    creatdic = [dict(x) for x in creatives]
    cdf = pd.json_normalize(json.loads(json.dumps(creatdic, default=dict)))
    # cdf['campaign_id'] = row['campaign_id']
    # cdf['ad_id'] = row['ad_id']
    if any(['video_data' in u for u in cdf.columns]): cdf['data_type'] = 'video_data'
    if any(['link_data' in u for u in cdf.columns]): cdf['data_type'] = 'link_data'
    createlist.append(cdf) #+= [cdf]
    a = json.loads(dict(creatives.headers())['x-business-use-case-usage'])
    api_limit = max([max(w['call_count'], w['total_cputime'], w['total_time']) 
                            for u,v in a.items() for w in v])
    logging.info(f'api limit: {api_limit}')
    if api_limit>75: sleep(300)
    else: sleep(1)

except Exception as e:
    print(e)
# %%
creatdic
# %%
# Find the ad_id that actually references video 1728857971690773, then pull
# its creative directly to compare object_story_spec.video_data.video_id
# against the funnel export value for the real access test.
target_video_id = '1728857971690773'
video_ad_row = actdf[actdf['Ad_Creative_Video_ID__Facebook_Ads'] == target_video_id].iloc[0]
video_ad_id = video_ad_row['Ad_ID__Facebook_Ads']
print(f"ad_id {video_ad_id} references video {target_video_id}")

video_ad = Ad(fbid=video_ad_id)
video_creatives = video_ad.get_ad_creatives(fields=fields)
for c in video_creatives:
    spec = c.get('object_story_spec', {})
    graph_video_id = spec.get('video_data', {}).get('video_id')
    print(f"creative {c['id']}: object_story_spec video_id = {graph_video_id}")
# %%
# No video_data on the creative means this is a boosted organic Page post
# (a Reel), not an ad-account-owned video. Fetch the post itself to confirm -
# this needs Page-level permission (pages_read_engagement / page access on
# 38870020038), separate from ads_read on the ad account.
effective_story_id = '38870020038_1654041703429381'
api = FacebookAdsApi.get_default_api()
post_response = api.call(
    'GET',
    (effective_story_id,),
    params={'fields': 'attachments{media_type,media,url,target}'},
)
post_response.json()
# %%
creatdic
# %%
cdf
# %%
createdf = pd.concat(createlist)
createdf
# %%
createdf['object_story_spec.template_data.child_attachments'].iloc[0]
# %%
createdf.columns
# %%
fields = [
        "id", #"title", #"description", "source", "length",
        # "views", "thumbnails", "created_time", "permalink_url"
    ]

video_id = "1728857971690773"
video = AdVideo(video_id)
videodata = video.api_get(fields=fields)
videodata
# %%
for ix, row in actdf.iterrows():
    if row['Ad_Creative_Video_ID__Facebook_Ads'] is not None:
        video_id = row['Ad_Creative_Video_ID__Facebook_Ads']
        video = AdVideo(video_id)
        try:
            videodata = video.api_get(fields=fields)
            print(f"{ix}: {video_id} - {videodata.get('title')}")
        except Exception as e:
            print("error")
        
# %%
row['Ad_Creative_Video_IDs__Facebook_Ads'].split(',')
# %%
# There's no "/act_X/advideos/VIDEO_ID" route, so account+video can't be combined
# into one call. Instead: scan the account's video ids only (cheap, one field),
# confirm the id belongs to this account, then do a single direct fetch for the
# full fields on just that video.
def fetch_video_via_account(target_video_id, account_id, fields):
    target_video_id = str(target_video_id)
    match = next(
        (v for v in AdAccount(account_id).get_ad_videos(fields=['id'], params={'limit': 1000})
         if v['id'] == target_video_id),
        None,
    )
    if match is None:
        print(f"{target_video_id} not found in {account_id}")
        return None
    return AdVideo(target_video_id).api_get(fields=fields)

known_account_id = 'act_2409630422470729'
videodata = fetch_video_via_account(1472665494469856, known_account_id, fields)
videodata
# %%
a = AdAccount('act_2409630422470729').get_ad_videos(fields=['id'], params={'limit': 100})
a
# %%
a
# %%
1472665494469856
1623093879122294
937953705710317
1331531375139249
2156666931782568
# %%
errors_df[errors_df['video_id'].str.contains('1472665494469856')]
# %%
actdf #[actdf['video_id']=='1472665494469856']
# %%
# Sanity check: list ad accounts this token can actually see
from facebook_business.adobjects.user import User

me = User(fbid='me')
accounts = me.get_ad_accounts(fields=['id', 'name', 'account_status'])
acc_list = []
for acct in accounts:
    print(acct['id'], acct.get('name'), acct.get('account_status'))
    acc_list.append({'id': acct['id'], 'name': acct.get('name'), 'status': acct.get('account_status')})
# %%
# Try fetching videos from one of those known-good accounts
known_account_id = 'act_2409630422470729'
account = AdAccount(known_account_id)
account_videos = account.get_ad_videos(fields=['id', 'title'])
for v in account_videos:
    print(v['id'], v.get('title'))
# %%
acc_list
# %%
# Search every accessible ad account for the specific video_id that failed
target_video_id = '1013093554203510'
found_in = []
for acct in acc_list:
    acct_id = acct['id']
    try:
        vids = AdAccount(acct_id).get_ad_videos(fields=['id'], params={'limit': 1000})
        ids = [v['id'] for v in vids]
        if target_video_id in ids:
            found_in.append(acct_id)
            print(f"FOUND in {acct_id} ({acct.get('name')})")
    except Exception as e:
        print(f"Could not list videos for {acct_id} ({acct.get('name')}): {e}")

if not found_in:
    print(f"{target_video_id} was not found in any of the {len(acc_list)} accessible ad accounts.")
# %%

# %%
sql = f"""
    SELECT DISTINCT Creative_ID__Facebook_Ads AS creative_id, Ad_Creative_Video_ID__Facebook_Ads AS video_id,
           Ad_Creative_Video_Permalink_URL__Facebook_Ads AS video_url
    FROM `{env.CLIENT_PROJECT}.{env.CLIENT_DATASET}.{env.META_TABLE}`
    LEFT JOIN `{env.CLIENT_PROJECT}.{env.OPTICS_DATASET}.{env.META_VIDEO_TABLE}` AS meta
    ON meta.video_id = Ad_Creative_Video_ID__Facebook_Ads
    WHERE meta.video_id IS NULL
    AND Ad_Creative_Video_Permalink_URL__Facebook_Ads IS NOT NULL
    """
sql = f"""
        SELECT DISTINCT Creative_ID__Facebook_Ads AS creative_id, Ad_Creative_Image_Hash__Facebook_Ads AS image_hash,
            Ad_Creative_Image_Permalink_URL__Facebook_Ads AS image_url
        FROM `{env.CLIENT_PROJECT}.{env.CLIENT_DATASET}.{env.META_TABLE}`
        LEFT JOIN `{env.CLIENT_PROJECT}.{env.OPTICS_DATASET}.{env.META_IMAGE_TABLE}` AS meta
        ON meta.image_hash = Ad_Creative_Image_Hash__Facebook_Ads
        WHERE meta.image_hash IS NULL
        AND Ad_Creative_Image_Hash__Facebook_Ads IS NOT NULL
        AND Ad_Creative_Image_Permalink_URL__Facebook_Ads IS NOT NULL"""
actdf = pdq.read_gbq(sql,
    project_id=env.CLIENT_PROJECT,
    progress_bar_type='tqdm',
    use_bqstorage_api=True,)
actdf
# %%
df = pd.read_gbq(f'SELECT * FROM `{env.CLIENT_PROJECT}.{env.OPTICS_DATASET}.{env.META_VIDEO_TABLE}`',
                 project_id=env.CLIENT_PROJECT,
                 )
df
# %%
a = actdf[~actdf['video_id'].isna()][['creative_id', 'video_id']].drop_duplicates()
pdf = pd.merge(df, a, on='video_id', how='inner')
# %%
pdf
# %%
actdf['creative_id'].nunique(), actdf['image_hash'].nunique(), actdf['image_url'].nunique()
# %%
df = pd.read_gbq(f'SELECT * FROM `{env.CLIENT_PROJECT}.{env.CLIENT_DATASET}.{env.META_TABLE}` LIMIT 100',
                 project_id=env.CLIENT_PROJECT,)
df
# %%
df[~df['Ad_Creative_Image_Hash__Facebook_Ads'].isna()][['Creative_ID__Facebook_Ads', 'Ad_Creative_Image_Hash__Facebook_Ads']].drop_duplicates().head(10)
# %%
[u for u in df.columns if 'ID' in u]
# %%
import requests
import imghdr
from time import sleep
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from google.cloud import storage

project = env.CLIENT_PROJECT
storage_bucket = env.STORAGE_BUCKET
meta_img_path = 'meta/images'
prefix_url = f'https://storage.cloud.google.com/{storage_bucket}/'

client = storage.Client(project=project)
bucket = client.bucket(storage_bucket)

done_imgs, imagelist = [], []
for i, row in actdf.iterrows():    
    img_url = row['image_url']
    img_hash = row['image_hash']
    if img_hash in done_imgs: continue
    else: sleep(1)
    try:
        # img_data = requests.get(img_url).content
        # file_type = imghdr.what(None, img_data)
        # blob = bucket.blob(f'{meta_img_path}/{img_hash}.{file_type}')
        # if not blob.exists(): blob.upload_from_string(img_data)
        # done_imgs.append(img_hash)

        # image = Image.open(BytesIO(img_data))
        # image.thumbnail(size=(128, 128))
        # buffer = BytesIO()
        # image.save(buffer, format="JPEG")
        img_path = f"{meta_img_path}/thumbnails/{img_hash}.jpeg"
        # oblob = bucket.blob(img_path)
        # if not oblob.exists(): oblob.upload_from_string(buffer.getvalue())

        file_type = 'jpeg'  # Assuming all images are in JPEG format
        logging.info(f'{i}: downloaded: {img_hash}')
        gs_path = f'gs://{storage_bucket}/{meta_img_path}/{img_hash}.{file_type}'
        thumbnail_url = f'{prefix_url}{img_path}'
        imagelist += [{'image_hash':img_hash, 'gs_path':gs_path,'thumbnail_url':thumbnail_url}]
    except Exception as e:
        logging.error(f"Error downloading image {img_hash}: {e}")
        continue

# %%
imgdf = pd.DataFrame(imagelist)
# imgdf = pdq.read_gbq('select * from `rp-tech-bissell-prod.optics.meta_image_data`',)
imgdf
# %%
from google.cloud import storage
project = env.CLIENT_PROJECT
storage_bucket = env.STORAGE_BUCKET
meta_img_path = env.META_IMG_PATH
from google.oauth2.service_account import Credentials
# get credentials from service account json
creds = Credentials.from_service_account_file('../data/jfgp-creative-optics-eabb40c0caac.json')


client = storage.Client(project=project, credentials=creds)
bucket = client.bucket(storage_bucket)
# %%
blob = bucket.blob(f'{meta_img_path}/{img_hash}.{file_type}')
blob.exists()
# %%
# imgs = [u.name for u in bucket.list_blobs(prefix=f'{meta_img_path}') if not u.name.endswith('/')]
blob.name in imgs
# %%
pd.merge(actdf[['creative_id', 'image_hash']].drop_duplicates(), imgdf, on='image_hash', how='inner')
# %%
project='rp-tech-bissell-prod'
database='funnel_export_bissell_us'
table='Creative_Report_Meta'

datadf = pdq.read_gbq(
    f'SELECT * FROM `{project}.{database}.{table}`',
    project_id=project,
    progress_bar_type='tqdm',
    use_bqstorage_api=True,)
# %%
imgdf = datadf[~datadf['Ad_Creative_Image_Hash__Facebook_Ads'].isna()]
# %%
import requests
import imghdr
from time import sleep
from tqdm import tqdm
image_path = '../data/images'
done_imgs = [u.split('.')[0] for u in os.listdir(image_path)]
for i, row in imgdf.iterrows():    
    img_url = row['Ad_Creative_Image_Permalink_URL__Facebook_Ads']
    img_hash = row['Ad_Creative_Image_Hash__Facebook_Ads']
    if img_hash in done_imgs: continue
    else: sleep(1)
    img_data = requests.get(img_url).content
    file_type = imghdr.what(None, img_data)
    img_name = f'../data/{img_hash}.jpg'
    img_file = f'{image_path}/{img_hash}.{file_type}'
    open(img_file, 'wb').write(img_data)
    done_imgs.append(img_hash)
    
# %%
viddf = datadf[~datadf['Ad_Creative_Video_ID__Facebook_Ads'].isna()&~datadf['Ad_Creative_Video_Permalink_URL__Facebook_Ads'].isna()]
# %%
from yt_dlp import YoutubeDL
video_path = '../data/videos'
done_vids = [u.split('.')[0] for u in os.listdir('../data/videos')]
for i, row in viddf.iterrows():
    vid_url = row['Ad_Creative_Video_Permalink_URL__Facebook_Ads']
    vid_hash = row['Ad_Creative_Video_ID__Facebook_Ads']
    if vid_hash in done_vids: continue
    else: sleep(1)
    ydl_opts = {
        'format': 'best',
        'outtmpl': f'{video_path}/{vid_hash}.mp4',
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([vid_url])
        done_vids.append(vid_hash)
    except Exception as e:
        logging.error(f"Error downloading video {vid_hash}: {e}")
        continue
# %%
datadf.head()
# %%
from google.cloud import storage
# from ...utils import env
project = 'rp-tech-bissell-prod'
storage_bucket = 'bkt-bissell-prod-us-creative'
img_path = 'meta/images'
vid_path = 'meta/videos'

client = storage.Client(project=project)
bucket = client.get_bucket(storage_bucket)

# %%
imgs = [u.split('.')[0] for u in os.listdir('../data/images') if u.endswith('.jpg') or u.endswith('.jpeg')]
for img in tqdm(imgs):
    blob = bucket.blob(f'{img_path}/{img}.jpeg')
    if not blob.exists():
        blob.upload_from_filename(f'../data/images/{img}.jpeg')
        logging.info(f'Uploaded {img}.jpg to {storage_bucket}/{img_path}/')
# %%
storage_url = f'https://storage.cloud.google.com/{storage_bucket}/{img_path}/'
# %%
imgdf = pd.DataFrame(imgs, columns=['image_hash'])
imgdf['image_url'] = imgdf['image_hash'].apply(lambda x: f'{storage_url}{x}.jpeg')
imgdf['gs_path'] = imgdf['image_hash'].apply(lambda x: f'gs://{storage_bucket}/{img_path}/{x}.jpeg')
imgdf['image_type'] = 'image/jpeg'
imgdf
# %%
import pandas_gbq as pdq
project='rp-tech-bissell-prod'
database='optics'
table='meta_image_data'

pdq.to_gbq(
    imgdf,
    destination_table=f'{project}.{database}.{table}',
    project_id=project,
    if_exists='replace',
)
# %%
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

videos = [u.split('.')[0] for u in os.listdir('../data/videos') if u.endswith('.mp4')]
for video_id in tqdm(videos):    
    blob = bucket.blob(f'{env.META_VID_PATH}/thumbnails/{video_id}.jpg')
    if not blob.exists():
        blob.upload_from_filename(f'../data/videos/thumbnail/{video_id}.jpg')
        # logging.info(f'Uploaded {video_id}.jpg to {storage_bucket}/{vid_path}/')
    # clip = VideoFileClip(f'../data/videos/{video_id}.mp4')
    # frame = clip.get_frame(1.0)
    # img = Image.fromarray(frame)
    # img.thumbnail(size=(128, 128))
    # img.save(f'../data/videos/thumbnail/{video_id}.jpg', "JPEG")
    # clip.close()
# %%
from datetime import datetime as dt
storage_url = f'https://storage.cloud.google.com/{storage_bucket}/{vid_path}/thumbnails/'
df['video_id'] = df['video_hash']
df['thumbnail_url'] = df['video_id'].apply(lambda x: f'{storage_url}{x}.jpg')
df['downloaded_at'] = dt.now().strftime('%Y-%m-%d %H:%M:%S')
df
# %%
df.iloc[0]['thumbnail_url']
# %%
videodf = pdf[['creative_id', 'video_id', 'gs_path', 'thumbnail_url', 'downloaded_at']].drop_duplicates()
videodf
# %%
viddf = pd.DataFrame(videos, columns=['video_hash'])
viddf['video_url'] = viddf['video_hash'].apply(lambda x: f'{storage_url}{x}.mp4')
viddf['gs_path'] = viddf['video_hash'].apply(lambda x: f'gs://{storage_bucket}/{vid_path}/{x}.mp4')
viddf['video_type'] = 'video/mp4'
viddf
# %%
import pandas_gbq as pdq
project='rp-tech-bissell-prod'
database='optics'
table='meta_video_data'

pdq.to_gbq(
    videodf,
    destination_table=f'{project}.{database}.{table}',
    project_id=project,
    if_exists='replace',
)
# %%
