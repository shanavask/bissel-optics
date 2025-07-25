import google.cloud.aiplatform as aip
import pipelines.utils.env as env
from typing import Optional, List
import argparse
import logging

def local_run(args: List[str] = None) -> aip.PipelineJob:
    """
    Run the pipeline in a local environment.
    Args:
        args (List[str]): List of arguments to pass to the pipeline.
    Returns:
        aip.PipelineJob: The pipeline job object.
    """

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", help="Platform to run the pipeline on", type=str, default="local")
    parser.add_argument("--enable_caching", type=str, default=None)
    
    # Get commandline args
    args = parser.parse_args(args)

    # If empty value for enable_caching provided on commandline default to None
    if args.enable_caching == "": args.enable_caching = None

    return trigger_pipeline(platform=args.platform,
                            pipeline_template_path=f"build/{args.platform}_pipeline.json",
                            enable_caching=args.enable_caching,)


def trigger_pipeline(
        platform: str,
        pipeline_template_path: str,
        project_id: str = env.PROJECT,
        location: str = env.LOCATION,
        pipeline_root_path: str = f"gs://{env.STAGING_BUCKET}/root",
        service_account: str = env.SERVICE_ACCOUNT,
        enable_caching: bool = False,
) -> aip.PipelineJob:
    """
    Trigger the pipeline from the compiled template.
    Args:
        pipeline_template_path (str): Path to the compiled pipeline (JSON).
        project_id (str): Google Cloud project ID.
        location (str): Google Cloud location.
        pipeline_root_path (str): Root path for the pipeline artifacts.
        enable_caching (bool): Whether to enable caching for the pipeline.
    Returns:
        aip.PipelineJob: The pipeline job object.
    """
    
    aip.init(
        project=project_id,
        location=location,
    )

    # Prepare the pipeline job
    job = aip.PipelineJob(
        display_name=f"optics-{platform}-bissell",
        template_path=pipeline_template_path,
        pipeline_root=pipeline_root_path,
        parameter_values={},
        enable_caching=enable_caching,
    )

    job.submit(service_account=service_account)

    return job