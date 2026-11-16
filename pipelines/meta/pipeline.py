import kfp
from kfp import compiler
from google.cloud import aiplatform
from pipelines.meta.components.image_downloader import download_images
from pipelines.meta.components.video_downloader import download_videos
from pipelines.meta.components.image_predictor import predict_images
from pipelines.meta.components.video_describer import generate_video_descriptions
from pipelines.meta.components.video_predictor import predict_video_labels
from pipelines.meta.components.video_timer import predict_video_timer
import pipelines.utils.env as env

# Define the workflow of the pipeline.
@kfp.dsl.pipeline(
    name="optics-meta-bissell",
    pipeline_root=f'gs://{env.STAGING_BUCKET}/root',)
def pipeline(
    project_id: str = env.PROJECT,
    storage_bucket: str = env.STORAGE_BUCKET,
    meta_table: str = env.META_TABLE,
    client_project: str = env.CLIENT_PROJECT,
    client_dataset: str = env.CLIENT_DATASET,
    optics_dataset: str = env.OPTICS_DATASET,
    meta_image_table: str = env.META_IMAGE_TABLE,    
    meta_img_path: str = env.META_IMG_PATH,
    meta_video_table: str = env.META_VIDEO_TABLE,
    meta_video_path: str = env.META_VID_PATH,
    model: str = env.MODEL,
    model_location: str = env.MODEL_LOCATION
):
    """
    Pipeline to download images from Meta Ads and upload them to Google Cloud Storage.
    """
    
    download_images_task = download_images(
        meta_table=meta_table,
        meta_image_table=meta_image_table,
        client_project=client_project,
        client_dataset=client_dataset,
        optics_dataset=optics_dataset,
        storage_bucket=storage_bucket,
        project=project_id,
        meta_img_path=meta_img_path,
    )

    download_videos_task = download_videos(
        meta_table=meta_table,
        meta_video_table=meta_video_table,
        client_project=client_project,
        client_dataset=client_dataset,
        optics_dataset=optics_dataset,
        storage_bucket=storage_bucket,
        project=project_id,
        meta_video_path=meta_video_path,
    )

    image_predictor = predict_images(
        creative_type='image',
        client_project=client_project,
        optics_dataset=optics_dataset,
        project=project_id,
        location=model_location,
        model=model,
        storage_bucket=storage_bucket,
    ).set_display_name("image-predictor").after(download_images_task)

    video_descriptor = generate_video_descriptions(
        creative_type='video',
        client_project=client_project,
        optics_dataset=optics_dataset,
        project=project_id,
        location=model_location,
        model=model,
        storage_bucket=storage_bucket,
    ).set_display_name("video-descriptor").after(download_videos_task).after(download_images_task)

    video_predictor = predict_video_labels(
        creative_type='video',
        client_project=client_project,
        optics_dataset=optics_dataset,
        project=project_id,
        location=model_location,
        model=model,
        storage_bucket=storage_bucket,
    ).set_display_name("video-predictor").after(video_descriptor)
    
    video_timer = predict_video_timer(
            creative_type='video',
            client_project=client_project,
            optics_dataset=optics_dataset,
            project=project_id,
            model=model,
            location=model_location,
        ).set_display_name("video-timer").after(video_predictor)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="build/meta_pipeline.json",
        type_check=False,
    )
