from kfp import dsl

@dsl.component(
        base_image="python:3.12", 
        packages_to_install=["google-cloud-logging==3.11.4", "google-cloud-aiplatform==1.100.0",
            "pandas-gbq==0.29.2", "google-cloud-bigquery-storage==2.32.0", "kfp==2.13.0",
            "pillow==11.3.0"])
def download_images(
    meta_table: str,
    meta_image_table: str,
    client_project: str,
    client_dataset: str,
    optics_dataset: str,
    storage_bucket: str,
    project: str,
    meta_img_path: str,
) -> str:
    """
    Identifies new images to download and uploads them to Google Cloud Storage.
    Args:
        meta_table (str): Name of the Meta Funnel export table
        meta_image_table (str): Name of the downloaded image table.
        dataset (str): Name of the dataset.
        storage_bucket (str): Name of the Google Cloud Storage bucket.
        project (str): Google Cloud project ID.
        meta_img_path (str): Path in the storage bucket where images will be stored.
    """
    
    import os
    import pandas_gbq as pdq
    import pandas as pd
    import requests
    import imghdr
    from PIL import Image
    from io import BytesIO
    from datetime import datetime as dt
    from google.cloud import storage
    from time import sleep
    import logging

    logging.basicConfig(level=logging.DEBUG)

    sql = f"""
        SELECT DISTINCT Creative_ID__Facebook_Ads AS creative_id, Ad_Creative_Image_Hash__Facebook_Ads AS image_hash,
            Ad_Creative_Image_Permalink_URL__Facebook_Ads AS image_url
        FROM `{client_project}.{client_dataset}.{meta_table}`
        LEFT JOIN `{client_project}.{optics_dataset}.{meta_image_table}` AS meta
        ON meta.image_hash = Ad_Creative_Image_Hash__Facebook_Ads
        WHERE meta.image_hash IS NULL
        AND Ad_Creative_Image_Hash__Facebook_Ads IS NOT NULL
        AND Ad_Creative_Image_Permalink_URL__Facebook_Ads IS NOT NULL
        """
    actdf = pdq.read_gbq(sql,
        project_id=project,
        use_bqstorage_api=True,)
    
    logging.info(f"Found {len(actdf)} images to download.")

    client = storage.Client(project=client_project)
    bucket = client.get_bucket(storage_bucket)
    prefix_url = f'https://storage.cloud.google.com/{storage_bucket}/'

    done_imgs, imagelist = [], []
    for i, row in actdf.iterrows():    
        img_url = row['image_url']
        img_hash = row['image_hash']
        if img_hash in done_imgs: continue
        else: sleep(1)
        try:
            img_data = requests.get(img_url).content
            file_type = imghdr.what(None, img_data)
            blob = bucket.blob(f'{meta_img_path}/{img_hash}.{file_type}')
            if not blob.exists(): blob.upload_from_string(img_data)
            done_imgs.append(img_hash)

            image = Image.open(BytesIO(img_data))
            image.thumbnail(size=(128, 128))
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            img_path = f"{meta_img_path}/thumbnails/{img_hash}.jpeg"
            oblob = bucket.blob(img_path)
            if not oblob.exists(): oblob.upload_from_string(buffer.getvalue())

            logging.info(f'{i}: downloaded: {img_hash}')
            gs_path = f'gs://{storage_bucket}/{meta_img_path}/{img_hash}.{file_type}'
            thumbnail_url = f'{prefix_url}{img_path}'
            imagelist += [{'image_hash':img_hash, 'gs_path':gs_path,'thumbnail_url':thumbnail_url}]
        except Exception as e:
            logging.error(f"Error downloading image {img_hash}: {e}")
            continue

    imgdf = pd.DataFrame(imagelist)
    if not imgdf.empty:
        logging.info(f"Uploading {len(imgdf)} images to BigQuery.")
        imgdf = pd.merge(actdf[['creative_id', 'image_hash']].drop_duplicates(), imgdf, on='image_hash', how='inner')
        imgdf['downloaded_at'] = dt.now().strftime('%Y-%m-%d %H:%M:%S')
        pdq.to_gbq(imgdf,
            destination_table=f'{client_project}.{optics_dataset}.{meta_image_table}',
            project_id=project, if_exists='replace',
        )

    return 'success'