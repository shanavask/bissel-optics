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
llm_client = Client(vertexai=True, project=env.PROJECT, location=env.LOCATION)
google_search_tool = Tool(google_search = GoogleSearch())

# %%
prompt = """
Analyze the provided video in detail. Your description should be organized into sections, with each section corresponding to a distinct scene. For each scene, please provide the following:

1.  **Timestamp:** Indicate the start and end times for the scene (e.g., 00:00:15 - 00:00:45).
2.  **Visual Description:**
    * **Setting:** Where and when does this scene take place? (e.g., a sunlit park in the afternoon, a dimly lit office at night).
    * **Main Subject(s):** Who or what is the primary focus of this scene?
    * **Colors and Textures:** Dominant colors, overall color palette, and notable textures visible (e.g., vibrant reds and blues, rough stone, smooth metal).
    * **Actions:** What specific actions are occurring? Describe any movements or events.
3.  **Atmosphere and Emotion:**
    * **Mood/Atmosphere:** What is the overall feeling or vibe of the scene? (e.g., joyful, tense, mysterious, calm).
    * **Emotions Conveyed:** What emotions are the subjects (if any) expressing? How is this conveyed (e.g., facial expressions, body language)?
4. **Products and Branding:**
    * **Products Shown:** Are there any products featured? Describe them in detail 
    * **Branding Elements:** Are there any logos, brand names, or other branding elements visible? Describe their placement and appearance (e.g., a logo on a shirt, a brand name on a product box).

After describing all scenes, please provide the following overall analysis of the video:

1.  **Overall Main Subject:** What is the central topic or focus of the entire video?
2.  **Overall Message/Theme:** What is the overarching message, purpose, or recurring idea the video seems to be communicating?
3.  **Text and Logos:** Detail any text that appears in the video. This includes:
    * Logos (describe them and identify the brand if possible).
    * Signs or posters in the background.
    * On-screen captions, titles, or subtitles (transcribe them).
4.  **Spoken Content:** Transcribe all spoken words, dialogue, or narration present in the video.
5.  **Audio Elements:**
    * **Background Music:** If present, describe its style (e.g., upbeat electronic, melancholic orchestral, acoustic folk).
    * **Music-Video Relationship:** How does the music (if any) contribute to or relate to the mood, pacing, and content of the video?
    * **Sound Effects:** Note any significant sound effects and their impact.

Please be as thorough and precise as possible in your descriptions and transcriptions.     

"""
# %%
project = env.PROJECT
client_project = env.CLIENT_PROJECT
optics_dataset = env.OPTICS_DATASET
creative_type = "video"
channel = "meta"
# %%
sql = f"""
    SELECT DISTINCT a.video_id, a.gs_path
    FROM `{client_project}.{optics_dataset}.{channel}_{creative_type}_data` AS a
    LEFT JOIN `{client_project}.{optics_dataset}.{channel}_{creative_type}_predictions` AS b
    ON a.video_id = b.video_id
    WHERE b.video_id IS NULL
"""
actdf = pdq.read_gbq(sql, project_id=project, use_bqstorage_api=True)
actdf
# %%
video_text = []    
for idx, row in actdf.iterrows():
    video_id = row['video_id']
    print(f"Processing {creative_type} {video_id}...")
    try:
        asset = types.Part.from_uri(file_uri=row['gs_path'])
        response = llm_client.models.generate_content(
                model=env.MODEL, contents=[asset, prompt],
                config=types.GenerateContentConfig(
                )
            )
        response = llm_client.models.generate_content(
            model=env.MODEL, contents=[asset, prompt],
            config=GenerateContentConfig()
        )
        video_text += [(video_id, response.text)]

    except Exception as e:
        print(f"Error processing {creative_type} {video_id}: {e}")
        continue

# %%
new_textdf = pd.DataFrame(video_text, columns=['video_id', 'video_description'])
# %%
new_textdf
# %%
import pandas_gbq as pdq
project='rp-tech-bissell-prod'
database='optics'
table='meta_video_descriptions'

old_textdf = pdq.read_gbq(
    f'SELECT * FROM `{project}.{database}.{table}`',
    project_id=project,
    use_bqstorage_api=True
)
old_textdf
# %%
video_textdf = old_textdf.groupby('video_id').agg({'video_description': 'first'}).reset_index()
video_textdf
# %%
video_textdf = pd.concat([old_textdf, new_textdf], ignore_index=True)
video_textdf
# %%
pdq.to_gbq(
    video_textdf,
    destination_table=f'{project}.{database}.{table}',
    project_id=project,
    if_exists='replace',
)
# %%
# sql = f"""
#     SELECT * FROM `{client_project}.{optics_dataset}.{channel}_{creative_type}_predictions`
# """
# preddf_old = pdq.read_gbq(sql, project_id=project, use_bqstorage_api=True)
# preddf = pd.concat([preddf_old, preddf_new], ignore_index=True)
# # %%
# import os
# from google.genai.types import Part
# video_text = []
# videos = [u.split('.')[0] for u in os.listdir('../data/videos') if u.endswith('.mp4')]
# for video_id in videos[:10]:
#     video_path = f'../data/videos/{video_id}.mp4'
#     video_bytes = Part.from_bytes(
#         data=open(video_path, 'rb').read(),
#         mime_type="video/mp4",
#     )

#     response = llm_client.models.generate_content(
#         model=env.MODEL, contents=[video_bytes, prompt],
#         config=GenerateContentConfig(
#         )
#     )
#     video_text += [(video_id, response.text)]
# # %%
# video_textdf = pd.DataFrame(video_text, columns=['video_id', 'video_description'])
# video_textdf

# %%
class BisselOutput(BaseModel):
    asset_type: List[Literal["Product Only", "Lifestyle", "Infographic", "User-Generated"]] = Field(
        default=[],
        description="Type of asset used in the ad creative.",
        title="Asset Type"
    )
    video_content_type: Optional[List[Literal["Product Demonstration", "How-To / Tutorial", "Testimonial / Review", "Brand Story", "Comparison", "Unboxing"]]] = Field(
        default=None,
        description="Type of video content used in the ad creative.",
        title="Video Content Type"
    )
    product_category: List[Literal["Carpet Cleaners", "Vacuum Cleaners", "Hard Floor Cleaners", "Portable Cleaners", "Air Purifiers", "Cleaning Formulas"]] = Field(
        default=[],
        description="Type of product category used in the ad creative.",
        title="Product Category"
    )
    product_state: List[Literal["In Use", "Posed", "Assembled", "Cleaned"]] = Field(
        default=[],
        description="Type of product state used in the ad creative.",
        title="Product State"
    )
    value_prop: List[Literal["Performance", "Convenience", "Specialty", "Value", "Trust"]] = Field(
        default=[],
        description="Type of value proposition used in the ad creative.",
        title="Value Proposition"
    )
    problem_shown: List[Literal["Pet Messes", "Spills", "Dirt & Dust", "Allergens", "Tough Stains"]] = Field(
        default=[],
        description="Type of problem shown in the ad creative.",
        title="Problem Shown"
    )
    solution_shown: List[Literal["Clean Surface", "Cleaning Action", "Refreshed Room", "Happy Person/Pet"]] = Field(
        default=[],
        description="Type of solution shown in the ad creative.",
        title="Solution Shown"
    )
    environment: List[Literal["Living Room", "Kitchen", "Bedroom", "Office", "Bathroom", "Stairs", "Car interior", "Outdoors/Entryway", "Studio/Abstract"]] = Field(
        default=[],
        description="Type of environment shown in the ad creative.",
        title="Environment"
    )
    surface_type: List[Literal["Carpet", "Hardwood", "Tile/Grout", "Laminate/Vinyl", "Upholstery", "Rug"]] = Field(
        default=[],
        description="Type of surface being cleaned in the ad creative.",
        title="Surface Type Being Cleaned"
    )
    human_presence: List[Literal["No People", "Adult Male", "Adult Female", "Child / Children", "Family", "Hands Only"]] = Field(
        default=[],
        description="Type of human presence shown in the ad creative.",
        title="Human Presence"
    )
    pet_presence: List[Literal["No Pets", "Dog", "Cat", "Multiple Pets"]] = Field(
        default=[],
        description="Type of pet presence shown in the ad creative.",
        title="Pet Presence"
    )
    visual_style: List[Literal["Bright & Airy", "Dark & Moody", "Warm & Cozy", "Clean & Modern", "Authentic & Real Life", "Highly-Styled & Aspirational"]] = Field(
        default=[],
        description="Type of visual style shown in the ad creative.",
        title="Visual Style"
    )
    shot_composition: List[Literal["Wide Shot", "Medium Shot", "Close-up", "Extreme Close-up", "Point-of-View", "Split Screen"]] = Field(
        default=[],
        description="Type of shot composition used in the ad creative.",
        title="Shot Composition"
    )
    text_overlay: List[Literal["No Text", "Headline / Hook", "Benefit Callouts", "Price / Offer", "Brand Logo", "Call-to-Action"]] = Field(
        default=[],
        description="Type of text overlay used in the ad creative.",
        title="Text Overlay"
    )
    branding: List[Literal["Prominent Logo Placement", "Subtle Logo Placement", "Product-Only Branding"]] = Field(
        default=[],
        description="Type of branding used in the ad creative.",
        title="Branding"
    )
    audio: List[Literal["Spoken Voiceover", "On-screen Talent Speaking", "Music Only", "Product Sounds", "Sound Effects"]] = Field(
        default=[],
        description="Type of audio used in the ad creative.",
        title="Audio"
    )
    call_to_action: List[Literal["Visual CTA Button", "Verbal CTA", "Implied CTA", "No CTA"]] = Field(
        min_items=1,
        description="Type of call-to-action used in the ad creative.",
        title="Call-to-Action"
    )
    headline_text: Optional[str] = Field(
        default=None,
        description="Headline text used in the ad creative.",
        title="Headline Text"
    )
    subtitle_text: Optional[str] = Field(
        default=None,
        description="Subtitle text used in the ad creative.",
        title="Subtitle Text"
    )
# %%
prompt = """
Analyze the video_description of the ad and classify the type of asset used in the ad creative.

1. Asset type can be one of the following:
   - Product Only: when shot in a plain background
   - Lifestyle: when shot in a real-world setting with people
   - Infographic: when the ad contains text or graphics to convey information
   - User-Generated: when the ad is created by a user or customer

2. If the asset is video, classify the type of video content:
    - Product Demonstration: showing how the product works
    - How-To / Tutorial: providing instructions on using the product
    - Testimonial / Review: featuring customer feedback
    - Brand Story: telling the brand's story or values
    - Comparison: comparing the product to a competitor or older method
    - Unboxing: showing the product being unboxed

3. Product category can be one of the following:
    - Carpet Cleaners
    - Vacuum Cleaners
    - Hard Floor Cleaners
    - Portable Cleaners
    - Air Purifiers
    - Cleaning Formulas

4. Product state can be one of the following:
    - In Use: when the product is being used in the ad
    - Posed: when the product is posed in the ad
    - Assembled: when the product is assembled in the ad
    - Cleaned: when the product is cleaned in the ad

5. Value proposition can be one of the following:
   - Performance: Deep cleaning, powerful suction, stain removal.
   - Convenience: Cordless, lightweight, easy to use, multi-surface.
   - Specialty: Pet-focused, allergy-friendly, quiet operation.
   - Value: Price point, special offer, bundle deal.
   - Brand Trust: Heritage, Bissell Pet Foundation connection.

6. If one of the following problems is shown in the image, classify it:
    - Pet Messes: hair, urine, mud
    - Spills: Food & Drink wine, coffee, cereal
    - Dirt & Dust: Everyday Dirt & Dust
    - Allergens: pollen, dander
    - Tough Stains: Set-in Stains

7. If one of the following solutions is shown in the image, classify it:
    - Clean Surface: A clean surface after using the product
    - Cleaning Action: The action of cleaning being performed
    - Refreshed Room: A room that looks refreshed after using the product
    - Happy Person/Pet: A person or pet looking happy after using the product

8. Environment can be one of the following:
    - Living Room
    - Kitchen
    - Bedroom
    - Office
    - Bathroom
    - Stairs
    - Car interior
    - Outdoors / Entryway
    - Studio / Abstract

9. If surface type is shown in the image, classify it:
    - Carpet
    - Hardwood
    - Tile / Grout
    - Laminate / Vinyl
    - Upholstery (Sofa, chair)
    - Rug

10. Human Presence:
    - No People
    - Adult Male
    - Adult Female
    - Child / Children
    - Family
    - Hands Only

11. Pet Presence:
    - No Pets
    - Dog
    - Cat
    - Multiple Pets

12. Visual Style:
    - Bright & Airy
    - Dark & Moody
    - Warm & Cozy
    - Clean & Modern
    - Authentic & Real Life
    - Highly-Styled & Aspirational

13. If relevant, classify the shot composition:
    - Wide Shot: showing whole room
    - Medium Shot: person and product
    - Close-up: on the mess or product head
    - Extreme Close-up: on fibers, bristles
    - Point-of-View: from user's perspective
    - Split Screen: Before/After

14. Text Overlay:
    - No Text
    - Headline / Hook
    - Benefit Callouts
    - Price / Offer
    - Brand Logo
    - Call-to-Action

15. Branding:
    - Prominent Logo Placement
    - Subtle Logo Placement
    - Product-Only Branding

16. If ad is a video and audio is present, classify it:
    - Spoken Voiceover
    - On-screen Talent Speaking
    - Music Only
    - Product Sounds
    - Sound Effects

17. Call-to-Action (CTA):
    - Visual CTA Button
    - Verbal CTA
    - Implied CTA (e.g., showing the product on a retail shelf)
    - No CTA

18. If headline or subtitle text is present, extract it.

"""
# %%
project='rp-tech-bissell-prod'
database='optics'
table='meta_video_predictions'

old_preddf = pdq.read_gbq(
    f'SELECT * FROM `{project}.{database}.{table}`',
    project_id=project,
    use_bqstorage_api=True
)
old_preddf
# %%
cols = old_preddf.columns[1:]
preddf = old_preddf.groupby('video_id').agg({col: 'first' for col in cols}).reset_index()
preddf
# %%
predlist = []
for i, row in video_textdf.iterrows():
    video_id = row['video_id']
    if row['video_id'] in old_preddf['video_id'].values: continue
    print(f"Processing {creative_type} {video_id}...")
    video_description = row['video_description']
    response = llm_client.models.generate_content(
        model=env.MODEL, contents=[prompt, video_description],
        config=GenerateContentConfig(
            # tools=[google_search_tool],
            response_mime_type="application/json",
            response_schema=BisselOutput
        )
    )
    predf = pd.json_normalize(response.to_json_dict()['parsed'])
    predlist += [predf.assign(video_id=video_id)]
# %%
new_preddf = pd.concat(predlist, ignore_index=True)
new_preddf['created_at'] = dt.now().strftime('%Y-%m-%d %H:%M:%S')
new_preddf

# %%
preddf = pd.concat([old_preddf, new_preddf], ignore_index=True)
preddf.shape
# %%
pdq.to_gbq(
    preddf,
    destination_table=f'{project}.{database}.{table}',
    project_id=project,
    if_exists='replace',
)
# %%
