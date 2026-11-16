from kfp import dsl

@dsl.component(
        base_image="python:3.12",
        packages_to_install=["google-cloud-logging==3.16.1", "google-cloud-aiplatform==1.160.0",
            "pandas-gbq==0.35.0", "google-cloud-bigquery-storage==2.39.0", "kfp==2.17.0",
            "pillow==11.3.0"])
def predict_video_timer(
    creative_type: str,
    client_project: str,
    optics_dataset: str,
    project: str,
    model: str,
    channel: str = "meta",
    location: str = "global"
) -> str:
    """
    Predicts time ranges for labels in ad creative video assets using a generative AI model, based on
    previously generated video descriptions.
    Args:
        creative_type (str): Type of creative asset to analyze.
        dataset (str): Name of the dataset.
        storage_bucket (str): Name of the Google Cloud Storage bucket.
        project (str): Google Cloud project ID.
    """

    import pandas_gbq as pdq
    import pandas as pd
    from google.genai import Client, types
    from google.genai.types import Tool, GenerateContentConfig
    from typing import List, Literal, Optional
    from pydantic import BaseModel, Field
    from datetime import datetime as dt
    from google.cloud import storage
    import json

    import logging

    logging.basicConfig(level=logging.DEBUG)
    
    sql = f"""
        SELECT *
        FROM `{client_project}.{optics_dataset}.{channel}_{creative_type}_predictions` AS b
        JOIN `{client_project}.{optics_dataset}.{channel}_{creative_type}_descriptions` AS a ON a.video_id = b.video_id
        LEFT JOIN `{client_project}.{optics_dataset}.{channel}_{creative_type}_predicted_timings` AS c ON a.video_id = c.video_id
        WHERE c.video_id IS NULL
        LIMIT 1000
    """
    actdf = pdq.read_gbq(sql, project_id=project, use_bqstorage_api=True)
    if actdf.empty:
        logging.info(f"No new {creative_type} labels to identify.")
        return f'no new {creative_type}s'

    logging.info(f"Found {len(actdf)} {creative_type} labels to identify.")


    categories = ['value_prop', 'problem_shown', 'solution_shown', 'hook', 'environment',
        'surface_type', 'human_presence', 'human_age', 'pet_presence',
        'cleaning_transformation', 'consumables', 'branding', 'call_to_action']

    # Labels that represent an absence/negative state have nothing to time -
    # they're excluded from the timing prompt output.
    NEGATIVE_LABELS = {"No People", "No Pets", "No CTA", "No Change", "None"}


    class LabelTiming(BaseModel):
        category: Literal[
            'value_prop', 'problem_shown', 'solution_shown', 'hook', 'environment',
            'surface_type', 'human_presence', 'human_age', 'pet_presence',
            'cleaning_transformation', 'consumables', 'branding', 'call_to_action'
        ] = Field(description="The category this label belongs to.")
        label: str = Field(description="The specific label assigned within the category.")
        time_start: float = Field(description="Start time (seconds from the start of the video) of this occurrence of the label.")
        time_end: float = Field(description="End time (seconds from the start of the video) of this occurrence of the label.")


    class LabelTimingOutput(BaseModel):
        timings: List[LabelTiming] = Field(
            default=[],
            description="One entry per contiguous occurrence of an assigned label in the video."
        )
    
    timing_prompt = """
    You are given:
    1. A detailed, scene-by-scene description of a video ad. Each scene begins with a
    Timestamp in the format HH:MM:SS - HH:MM:SS (start - end), followed by details about
    the scene's setting, subjects, actions, mood, products, branding, on-screen text, and audio.
    2. A set of category labels that have already been assigned to this video, grouped by category
    (see "Assigned labels" below).

    Your task is to find, for each assigned label, the time range(s) in the video during which that
    label is actually depicted or evidenced, using the scene timestamps in the description as your
    source of truth.

    Instructions:
    1. For every (category, label) pair listed in "Assigned labels", find the scene(s) in the
    description whose content supports that label.
    2. Convert each scene's timestamp to total elapsed seconds from the start of the video
    (e.g., 00:01:15 -> 75 seconds).
    3. If a label is supported by one contiguous block of scenes, output a single entry with
    time_start equal to the start time of the first supporting scene and time_end equal to the
    end time of the last supporting scene.
    4. If a label is supported by more than one non-contiguous group of scenes (e.g. a surface type
    that appears early in the video and then reappears later), output a separate entry per
    contiguous group. The same (category, label) pair may therefore appear more than once in the
    output.
    5. Do not output an entry for a label that represents an absence or negative state (e.g.
    "No People", "No Pets", "No CTA", "No Change", "None") - there is nothing to time for these.
    6. Only rely on evidence present in the description. If the description gives no scene-level
    support for an assigned label, omit that label from the output entirely rather than guessing
    a time range.
    7. time_start and time_end must be numbers in seconds (not strings, not HH:MM:SS format), and
    time_end must be greater than or equal to time_start.

    Assigned labels for this video (JSON, grouped by category):
    {assigned_labels_json}
    """
    
    def build_assigned_labels(row: pd.Series, categories: list[str]) -> dict:
        """Collect the non-empty, non-negative labels already assigned to a video for each category."""
        assigned = {}
        for cat in categories:
            labels = row.get(cat)
            # BigQuery REPEATED columns come back as numpy arrays (not `list`) via pandas-gbq.
            if pd.api.types.is_list_like(labels):
                labels = [l for l in list(labels) if l not in NEGATIVE_LABELS]
                if labels:
                    assigned[cat] = labels
        return assigned

    llm_client = Client(vertexai=True, project=project, location=location)

    old_timedf = pdq.read_gbq(
        f'SELECT * FROM `{client_project}.{optics_dataset}.{channel}_{creative_type}_predicted_timings`',
        project_id=project,
        use_bqstorage_api=True
    )

    timings = []
    for index, row in actdf.iterrows():
        video_id = row['video_id']
        logging.info(f"Processing video {row['video_id']} ({index+1}/{len(actdf)})")
        try:
            assigned_labels = build_assigned_labels(row, categories)
            prompt = timing_prompt.format(assigned_labels_json=json.dumps(assigned_labels, indent=2))

            response = llm_client.models.generate_content(
                model=model,
                contents=[prompt, row['video_description']],
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LabelTimingOutput
                )
            )
            df = pd.json_normalize(response.to_json_dict()['parsed']['timings'])
            df = df.assign(video_id=row['video_id'], channel=channel, creative_type=creative_type)
            timings.append(df)
        except Exception as e:
            logging.error(f"Error processing {creative_type} {video_id}: {e}")
            continue
        

    if not timings:
        logging.info("No predictions were made.")
        return 'no predictions'

    new_timedf = pd.concat(timings, ignore_index=True)
    new_timedf['first_3sec'] = new_timedf['time_start'].apply(lambda x: x <= 3)
    new_timedf['created_at'] = dt.now().strftime('%Y-%m-%d %H:%M:%S')

    timedf = pd.concat([old_timedf, new_timedf], ignore_index=True)

    pdq.to_gbq(timedf,
        destination_table=f'{client_project}.{optics_dataset}.{channel}_{creative_type}_predicted_timings',
        project_id=project, if_exists='replace',
    )

    return f"processed {len(new_timedf)} {creative_type}"
