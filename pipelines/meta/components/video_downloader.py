from kfp import dsl

@dsl.component(
        base_image="python:3.12",
        packages_to_install=["google-cloud-logging==3.16.1", "google-cloud-aiplatform==1.160.0",
            "pandas-gbq==0.35.0", "google-cloud-bigquery-storage==2.39.0", "kfp==2.17.0",
            "pillow==11.3.0", "moviepy==2.2.1", "yt-dlp==2026.7.4", "facebook-business==26.0.1",
            "google-cloud-secret-manager==2.30.0",])
def download_videos(
    meta_table: str,
    meta_video_table: str,
    client_project: str,
    client_dataset: str,
    optics_dataset: str,
    storage_bucket: str,
    project: str,
    meta_video_path: str,
) -> str:
    """
    Identifies new videos to download and uploads them to Google Cloud Storage.
    Args:
        meta_table (str): Name of the Meta Funnel export table
        meta_video_table (str): Name of the downloaded video table.
        dataset (str): Name of the dataset.
        storage_bucket (str): Name of the Google Cloud Storage bucket.
        project (str): Google Cloud project ID.
        meta_video_path (str): Path in the storage bucket where videos will be stored.
    """

    import os
    import requests
    import pandas_gbq as pdq
    import pandas as pd
    from PIL import Image
    from io import BytesIO
    from datetime import datetime as dt
    from google.cloud import storage
    from time import sleep
    import logging
    import json

    from google.cloud import secretmanager
    from yt_dlp import YoutubeDL
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.advideo import AdVideo

    logging.basicConfig(level=logging.DEBUG)

    secret_name = "bissel-meta-token"
    # get the API key from secrets
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    key = json.loads(payload)    
    access_token = key['access_token']

    FacebookAdsApi.init(access_token=access_token)

    page_id = "38870020038"  # BISSELL Page
    api = FacebookAdsApi.get_default_api()
    accounts_resp = api.call("GET", ("me", "accounts"), params={"fields": "id,name,access_token"})
    page_tokens = {p["id"]: p["access_token"] for p in accounts_resp.json().get("data", [])}
    page_token = page_tokens[page_id]

    FacebookAdsApi.init(access_token=page_token)

    def download_via_api(video_id, local_path):
        """Fallback for when yt-dlp can't scrape the permalink: pull the direct
        `source` URL from the Marketing API and download that instead."""
        source_url = AdVideo(video_id).api_get(fields=['source']).get('source')
        if not source_url:
            raise ValueError(f"No source url returned for video {video_id}")
        response = requests.get(source_url)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)

    sql = f"""
        SELECT DISTINCT Creative_ID__Facebook_Ads AS creative_id, Ad_Creative_Video_ID__Facebook_Ads AS video_id,
            Ad_Creative_Video_Permalink_URL__Facebook_Ads AS video_url
        FROM `{client_project}.{client_dataset}.{meta_table}`
        LEFT JOIN `{client_project}.{optics_dataset}.{meta_video_table}` AS meta
        ON meta.video_id = Ad_Creative_Video_ID__Facebook_Ads AND meta.creative_id = Creative_ID__Facebook_Ads
        WHERE meta.video_id IS NULL
        AND Ad_Creative_Video_ID__Facebook_Ads IS NOT NULL
        AND Ad_Creative_Video_Permalink_URL__Facebook_Ads IS NOT NULL
        """
    actdf = pdq.read_gbq(sql,
        project_id=project,
        use_bqstorage_api=True,)

    logging.info(f"Found {len(actdf)} videos to download.")

    MAX_DOWNLOAD_ATTEMPTS = 3
    errors_table = f'{client_project}.{optics_dataset}.{meta_video_table}_errors'
    try:
        existing_errors_df = pdq.read_gbq(
            f"SELECT video_id, error, attempt_count, logged_at FROM `{errors_table}`",
            project_id=project,
            use_bqstorage_api=True,
        )
        error_state = existing_errors_df.set_index('video_id')[['error', 'attempt_count', 'logged_at']].to_dict('index')
    except Exception as e:
        logging.info(f"No existing errors table found, starting fresh attempt counts: {e}")
        error_state = {}

    client = storage.Client(project=client_project)
    bucket = client.get_bucket(storage_bucket)
    prefix_url = f'https://storage.cloud.google.com/{storage_bucket}/'

    done_vids, videolist = [], []
    down_vids = [u.name for u in bucket.list_blobs(prefix=f'{meta_video_path}') if not u.name.endswith('/')]
    for i, row in actdf.iterrows():
        video_url = row['video_url']
        video_id = row['video_id']
        if video_id in done_vids: continue
        attempt_count = error_state.get(video_id, {}).get('attempt_count', 0)
        if attempt_count >= MAX_DOWNLOAD_ATTEMPTS:
            logging.info(f"Skipping video {video_id}: already failed {attempt_count} times.")
            continue
        else: sleep(1)
        try:
            ydl_opts = {
                'format': 'best',
                'outtmpl': f'/tmp/{video_id}.mp4',
            }
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                done_vids.append(video_id)
            except Exception as e:
                logging.warning(f"yt-dlp failed for {video_id}, retrying via Marketing API: {e}")
                try:
                    download_via_api(video_id, f'/tmp/{video_id}.mp4')
                    done_vids.append(video_id)
                except Exception as api_e:
                    logging.error(f"Error downloading video {video_id}: {api_e}")
                    error_state[video_id] = {'error': str(api_e), 'attempt_count': attempt_count + 1, 'logged_at': dt.now().strftime('%Y-%m-%d %H:%M:%S')}
                    continue

            blob = bucket.blob(f'{meta_video_path}/{video_id}.mp4')
            if not blob.name in down_vids: blob.upload_from_filename(f'/tmp/{video_id}.mp4')
            done_vids.append(video_id)

            clip = VideoFileClip(f'/tmp/{video_id}.mp4')
            frame = clip.get_frame(1.0)
            img = Image.fromarray(frame)
            img.thumbnail(size=(128, 128))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            img_path = f"{meta_video_path}/thumbnails/{video_id}.jpeg"
            oblob = bucket.blob(img_path)
            oblob.upload_from_string(buffer.getvalue())

            logging.info(f'{i}: downloaded: {video_id}')
            gs_path = f'gs://{storage_bucket}/{meta_video_path}/{video_id}.mp4'
            thumbnail_url = f'{prefix_url}{img_path}'
            videolist += [{'video_id':video_id, 'gs_path':gs_path,'thumbnail_url':thumbnail_url}]
            error_state.pop(video_id, None)
        except Exception as e:
            logging.error(f"Error downloading image {video_id}: {e}")
            error_state[video_id] = {'error': str(e), 'attempt_count': attempt_count + 1, 'logged_at': dt.now().strftime('%Y-%m-%d %H:%M:%S')}
            continue

    videodf = pd.DataFrame(videolist)
    errordf = pd.DataFrame([{'video_id': vid, **info} for vid, info in error_state.items()])
    if not videodf.empty:
        logging.info(f"Uploading {len(videodf)} videos to BigQuery.")
        videodf = pd.merge(actdf[['creative_id', 'video_id']].drop_duplicates(), videodf, on='video_id', how='inner')
        videodf['downloaded_at'] = dt.now().strftime('%Y-%m-%d %H:%M:%S')
        pdq.to_gbq(videodf,
            destination_table=f'{client_project}.{optics_dataset}.{meta_video_table}',
            project_id=project, if_exists='append',
        )

    if not errordf.empty:
        logging.info(f"Replacing video errors table with {len(errordf)} entries.")
        pdq.to_gbq(errordf,
            destination_table=errors_table,
            project_id=project, if_exists='replace',
        )
    return 'success'