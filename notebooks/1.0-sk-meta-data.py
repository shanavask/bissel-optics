# %%
import sys
sys.path.append('..')
# %%
import os
import logging

import numpy as np
import pandas as pd
import pandas_gbq as pdq

# import plotly_express as px

logging.basicConfig(level=logging.DEBUG)
pd.options.display.max_columns = None

import pipelines.utils.env as env

# %%
sql = f"""
    SELECT DISTINCT Creative_ID__Facebook_Ads AS creative_id, Ad_Creative_Image_Hash__Facebook_Ads AS image_hash,
           Ad_Creative_Image_Permalink_URL__Facebook_Ads AS image_url
    FROM `{env.CLIENT_PROJECT}.{env.CLIENT_DATASET}.{env.META_TABLE}`
    LEFT JOIN `{env.CLIENT_PROJECT}.{env.OPTICS_DATASET}.{env.META_IMAGE_TABLE}` AS meta
    ON meta.image_hash = Ad_Creative_Image_Hash__Facebook_Ads
    WHERE meta.image_hash IS NULL
    AND Ad_Creative_Image_Hash__Facebook_Ads IS NOT NULL
    AND Ad_Creative_Image_Permalink_URL__Facebook_Ads IS NOT NULL
    """
actdf = pdq.read_gbq(sql,
    project_id=env.CLIENT_PROJECT,
    progress_bar_type='tqdm',
    use_bqstorage_api=True,)
actdf
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
pd.DataFrame([]).empty
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
videos = [u.split('.')[0] for u in os.listdir('../data/videos') if u.endswith('.mp4')]
for video_id in tqdm(videos):
    blob = bucket.blob(f'{vid_path}/{video_id}.mp4')
    if not blob.exists():
        blob.upload_from_filename(f'../data/videos/{video_id}.mp4')
        logging.info(f'Uploaded {video_id}.mp4 to {storage_bucket}/{vid_path}/')
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
    viddf,
    destination_table=f'{project}.{database}.{table}',
    project_id=project,
    if_exists='replace',
)
# %%
