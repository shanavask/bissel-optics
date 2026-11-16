# %%
import sys
sys.path.append("..")  # Adjust the path to include the parent directory
# %%
import json
import pandas as pd
import pandas_gbq as pdq
from datetime import datetime as dt

from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from google.genai import Client, types

# from .sub_agents import db_agent
from pipelines.utils import env
# %%
llm_client = Client(vertexai=True, project=env.PROJECT, location='global')
google_search_tool = Tool(google_search = GoogleSearch())
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
    FROM `{client_project}.{optics_dataset}.{channel}_{creative_type}_predictions` AS b
    JOIN `{client_project}.{optics_dataset}.{channel}_{creative_type}_descriptions` AS a ON a.video_id = b.video_id
    LEFT JOIN `{client_project}.{optics_dataset}.{channel}_{creative_type}_predicted_timings` AS c ON a.video_id = c.video_id
    WHERE c.video_id IS NULL
"""
actdf = pdq.read_gbq(sql, project_id=project, use_bqstorage_api=True)
actdf
# %%
actdf['video_id'].nunique()

# %%
categories = ['value_prop', 'problem_shown', 'solution_shown', 'hook', 'environment',
       'surface_type', 'human_presence', 'human_age', 'pet_presence',
       'cleaning_transformation', 'consumables', 'branding', 'call_to_action']
# %%
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
# %%
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
# %%
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
# %%
# timings = []
for index, row in actdf.head(10).iterrows():
    print(f"Processing video {row['video_id']} ({index+1}/{len(actdf)})")
    if timings and row['video_id'] in pd.concat(timings, ignore_index=True)['video_id'].unique():
        print(f"Skipping video {row['video_id']} ({index+1}/{len(actdf)}) - already processed")
        continue
    assigned_labels = build_assigned_labels(row, categories)
    prompt = timing_prompt.format(assigned_labels_json=json.dumps(assigned_labels, indent=2))

    response = llm_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[prompt, row['video_description']],
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LabelTimingOutput
        )
    )
    df = pd.json_normalize(response.to_json_dict()['parsed']['timings'])
    df = df.assign(video_id=row['video_id'], channel=channel, creative_type=creative_type)
    timings.append(df)
# %%
timingdf = pd.concat(timings, ignore_index=True)
timingdf
# %%
timingdf['first_3sec'] = timingdf['time_start'].apply(lambda x: x <= 3)
timingdf['created_at'] = dt.now().strftime('%Y-%m-%d %H:%M:%S')
timingdf
# %%
pdq.to_gbq(timingdf, f"{client_project}.{optics_dataset}.{channel}_{creative_type}_predicted_timings", project_id=project, if_exists="replace")
# %%
f"{client_project}.{optics_dataset}.{channel}_{creative_type}_predicted_timings"
# %%
