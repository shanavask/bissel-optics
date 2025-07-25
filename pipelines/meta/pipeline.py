import kfp
from kfp import compiler
from google.cloud import aiplatform
from pipelines.meta.components.image_downloader import download_images
import pipelines.utils.env as env

# Define the workflow of the pipeline.
@kfp.dsl.pipeline(
    name="optics-meta-bissell",
    pipeline_root=f'gs://{env.STAGING_BUCKET}/root',)
def pipeline(
    project_id: str = env.PROJECT,
    storage_bucket: str = env.STORAGE_BUCKET,
    meta_table: str = env.META_TABLE,
    meta_image_table: str = env.META_IMAGE_TABLE,
    client_project: str = env.CLIENT_PROJECT,
    client_dataset: str = env.CLIENT_DATASET,
    optics_dataset: str = env.OPTICS_DATASET,
    meta_img_path: str = env.META_IMG_PATH,
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
    

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="build/meta_pipeline.json",
        type_check=False,
    )
