from kfp import dsl

@dsl.component(
        base_image="python:3.12", 
        packages_to_install=["google-cloud-logging==3.16.1", "google-cloud-aiplatform==1.160.0",
            "pandas-gbq==0.35.0", "google-cloud-bigquery-storage==2.39.0", "kfp==2.17.0",
            "pillow==11.3.0"])
def predict_images(
    creative_type: str,
    client_project: str,
    optics_dataset: str,
    project: str,
    model: str,
    storage_bucket: str,
    channel: str = "meta",
    location: str = "global"
) -> str:
    """
    Analyzes and classifies ad creative assets using a generative AI model.
    Args:
        creative_type (str): Type of creative asset to analyze.
        dataset (str): Name of the dataset.
        storage_bucket (str): Name of the Google Cloud Storage bucket.
        project (str): Google Cloud project ID.
    """
        
    import pandas_gbq as pdq
    import pandas as pd
    from google.genai import Client, types
    from typing import List, Literal, Optional
    from pydantic import BaseModel, Field
    from datetime import datetime as dt
    from google.cloud import storage
    import json

    import logging

    logging.basicConfig(level=logging.DEBUG)

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
    
    prompt = """
    Analyze the image of the ad and classify the type of asset used in the ad creative.

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

    sql = f"""
        SELECT DISTINCT a.image_hash, a.gs_path
        FROM `{client_project}.{optics_dataset}.{channel}_{creative_type}_data` AS a
        LEFT JOIN `{client_project}.{optics_dataset}.{channel}_{creative_type}_predictions` AS b
        ON a.image_hash = b.image_hash
        WHERE b.image_hash IS NULL
    """
    actdf = pdq.read_gbq(sql, project_id=project, use_bqstorage_api=True)
    if actdf.empty:
        logging.info(f"No new {creative_type}s to predict.")
        return f'no new {creative_type}s'

    logging.info(f"Found {len(actdf)} {creative_type}s to predict.")

    llm_client = Client(vertexai=True, project=project, location=location)
    predlist = []    
    for idx, row in actdf.iterrows():
        image_hash = row['image_hash']
        logging.info(f"{idx}/{len(actdf)}: Processing {creative_type} {image_hash} at {row['gs_path']}")
        try:
            asset = types.Part.from_uri(file_uri=row['gs_path'])
            response = llm_client.models.generate_content(
                    model=model, contents=[asset, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BisselOutput,
                    )
                )
            predf = pd.json_normalize(response.to_json_dict()['parsed'])
            predf = predf.assign(image_hash=image_hash)
            predlist += [predf]

        except Exception as e:
            logging.error(f"Error processing {creative_type} {image_hash}: {e}")
            continue

    if not predlist:
        logging.info("No predictions were made.")
        return 'no predictions'

    preddf_new = pd.concat(predlist, ignore_index=True)

    sql = f"""
    SELECT * FROM `{client_project}.{optics_dataset}.{channel}_{creative_type}_predictions`
    """

    preddf_new['created_at'] = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    preddf_old = pdq.read_gbq(sql, project_id=project, use_bqstorage_api=True)
    preddf = pd.concat([preddf_old, preddf_new], ignore_index=True)
   
    pdq.to_gbq(preddf,
        destination_table=f'{client_project}.{optics_dataset}.{channel}_{creative_type}_predictions',
        project_id=project, if_exists='replace',
    )

    return f"processed {len(preddf_new)} {creative_type}"