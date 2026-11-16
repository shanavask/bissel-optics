from kfp import dsl

@dsl.component(
        base_image="python:3.12",
        packages_to_install=["google-cloud-logging==3.16.1", "google-cloud-aiplatform==1.160.0",
            "pandas-gbq==0.35.0", "google-cloud-bigquery-storage==2.39.0", "kfp==2.17.0",
            "pillow==11.3.0"])
def predict_video_labels(
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
    Classifies ad creative video assets using a generative AI model, based on
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
        product_category: List[Literal["Upright Deep Cleaners", "Vacuum Cleaners", "Hard Floor Cleaners", "Portable Deep Cleaners", "Air Purifiers", "Cleaning Formulas", "Pet Care"]] = Field(
            default=[],
            description="Type of product category used in the ad creative.",
            title="Product Category"
        )
        product_state: List[Literal["In Use", "Posed", "Assembled", "Cleaned", "Tools"]] = Field(
            default=[],
            description="Type of product state used in the ad creative.",
            title="Product State"
        )
        value_prop: List[Literal["Performance", "Convenience", "Specialty", "Promo", "Trust"]] = Field(
            default=[],
            description="Type of value proposition used in the ad creative.",
            title="Value Proposition"
        )
        problem_shown: List[Literal["Pet Urine", "Pet Mud/Dirt", "Pet Stains", "Pet Litter", "Food Messes", "Drink Spills", "Dirt & Dust", "Allergens", "Tough Stains"]] = Field(
            default=[],
            description="Type of problem shown in the ad creative.",
            title="Problem Shown"
        )
        solution_shown: List[Literal["Clean Surface", "Cleaning Action", "Refreshed Room", "Happy Person/Pet"]] = Field(
            default=[],
            description="Type of solution shown in the ad creative.",
            title="Solution Shown"
        )
        hook: List[Literal["Sunday/Home Reset", "POV", "Dirty Water", "Selfie-Mode/Person Talking", "Day in the Life", "Unboxing", "ASMR", "Other",]] = Field(
            default=[],
            description="Type of hook used in the ad creative.",
            title="Hook"
        )
        environment: List[Literal["Living Room", "Kitchen", "Bedroom", "Office", "Bathroom", "Stairs", "Car interior", "Outdoors/Entryway", "Studio/Abstract", "Mudroom/Entry", "Kids Playroom", "Home Office"]] = Field(
            default=[],
            description="Type of environment shown in the ad creative.",
            title="Environment"
        )
        surface_type: List[Literal["Carpet", "Hardwood", "Tile/Grout", "Laminate/Vinyl", "Upholstery", "Rug", "Car Interior", "Pet Bed", "Bed/Mattress", "Baseboards/Windows"]] = Field(
            default=[],
            description="Type of surface being cleaned in the ad creative.",
            title="Surface Type Being Cleaned"
        )
        human_presence: List[Literal["No People", "Adult Male", "Adult Female", "Child / Children", "Family", "Hands Only"]] = Field(
            default=[],
            description="Type of human presence shown in the ad creative.",
            title="Human Presence"
        )
        human_age: List[Literal["Child", "Teen", "Young Adult", "Adult", "Senior"]] = Field(
            default=[],
            description="Type of human age shown in the ad creative.",
            title="Human Age"
        )
        pet_presence: List[Literal["No Pets", "Dog", "Cat", "Multiple Pets"]] = Field(
            default=[],
            description="Type of pet presence shown in the ad creative.",
            title="Pet Presence"
        )
        cleaning_transformation: List[Literal["Complete Clean", "Partial Clean", "No Change"]] = Field(
            default=[],
            description="Type of cleaning transformation shown in the ad creative.",
            title="Cleaning Transformation"
        )
        consumables: List[Literal["BISSELL Formula", "BISSELL Febreze Formula", "Generic/Non-BISSELL Formula", "None"]] = Field(
            default=[],
            description="Type of consumables shown in the ad creative.",
            title="Consumables"
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
Analyze the video_description of the ad creative and classify it according to the criteria below.

1. Asset type can be one of the following:
    - Product Only: when shot in a plain background
    - Lifestyle: when shot in a real-world setting with people
    - Infographic: when the ad contains text or graphics to convey information
    - User-Generated: when the ad is created by a user or customer

2. Video content type can be one of the following:
    - Product Demonstration: showing how the product works
    - How-To / Tutorial: providing instructions on using the product
    - Testimonial / Review: featuring customer feedback
    - Brand Story: telling the brand's story or values
    - Comparison: comparing the product to a competitor or older method
    - Unboxing: showing the product being unboxed

3. Product category can be one of the following:
    - Upright Deep Cleaners
    - Vacuum Cleaners
    - Hard Floor Cleaners
    - Portable Deep Cleaners
    - Air Purifiers
    - Cleaning Formulas
    - Pet Care

4. Product state can be one of the following:
    - In Use: when the product is being used in the ad
    - Posed: when the product is posed in the ad
    - Assembled: when the product is assembled in the ad
    - Cleaned: when the product is cleaned in the ad
    - Tools: when cleaning tools or attachments are shown separately from the main product

5. Value proposition can be one of the following:
    - Performance: Deep cleaning, powerful suction, stain removal.
    - Convenience: Cordless, lightweight, easy to use, multi-surface.
    - Specialty: Pet-focused, allergy-friendly, quiet operation.
    - Promo: Price point, special offer, bundle deal.
    - Trust: Heritage, Bissell Pet Foundation connection.

6. If one of the following problems is shown in the video, classify it:
    - Pet Urine
    - Pet Mud/Dirt
    - Pet Stains
    - Pet Litter
    - Food Messes
    - Drink Spills: wine, coffee, juice
    - Dirt & Dust: Everyday Dirt & Dust
    - Allergens: pollen, dander
    - Tough Stains: Set-in Stains

7. If one of the following solutions is shown in the video, classify it:
    - Clean Surface: A clean surface after using the product
    - Cleaning Action: The action of cleaning being performed
    - Refreshed Room: A room that looks refreshed after using the product
    - Happy Person/Pet: A person or pet looking happy after using the product

8. If the video opens with one of the following hooks, classify it:
    - Sunday/Home Reset: framed as a home reset or cleaning routine
    - POV: point-of-view style opening
    - Dirty Water: opens on dirty extracted water as the hook
    - Selfie-Mode/Person Talking: person talking directly to camera in selfie mode
    - Day in the Life: framed as a day-in-the-life
    - Unboxing: opens with the product being unboxed
    - ASMR: ASMR-style sound or visuals
    - Other: any other type of hook

9. Environment can be one of the following:
    - Living Room
    - Kitchen
    - Bedroom
    - Office
    - Bathroom
    - Stairs
    - Car interior
    - Outdoors/Entryway
    - Studio/Abstract
    - Mudroom/Entry
    - Kids Playroom
    - Home Office

10. If surface type is shown in the video, classify it:
    - Carpet
    - Hardwood
    - Tile/Grout
    - Laminate/Vinyl
    - Upholstery (Sofa, chair)
    - Rug
    - Car Interior
    - Pet Bed
    - Bed/Mattress
    - Baseboards/Windows

11. Human presence can be one of the following:
    - No People
    - Adult Male
    - Adult Female
    - Child / Children
    - Family
    - Hands Only

12. If a person is shown, classify their age group:
    - Child
    - Teen
    - Young Adult
    - Adult
    - Senior

13. Pet presence can be one of the following:
    - No Pets
    - Dog
    - Cat
    - Multiple Pets

14. Cleaning transformation shown can be one of the following:
    - Complete Clean: the mess is fully removed
    - Partial Clean: the mess is only partially removed
    - No Change: no visible before/after transformation

15. If a cleaning consumable is shown, classify it:
    - BISSELL Formula
    - BISSELL Febreze Formula
    - Generic/Non-BISSELL Formula
    - None

16. Text overlay can be one of the following:
    - No Text
    - Headline / Hook
    - Benefit Callouts
    - Price / Offer
    - Brand Logo
    - Call-to-Action

17. Branding can be one of the following:
    - Prominent Logo Placement
    - Subtle Logo Placement
    - Product-Only Branding

18. If audio is present, classify it:
    - Spoken Voiceover
    - On-screen Talent Speaking
    - Music Only
    - Product Sounds
    - Sound Effects

19. Call-to-action (CTA) can be one of the following:
    - Visual CTA Button
    - Verbal CTA
    - Implied CTA (e.g., showing the product on a retail shelf)
    - No CTA

20. If headline or subtitle text is present, extract it.
"""

    sql = f"""
        SELECT DISTINCT a.video_id, a.video_description
        FROM `{client_project}.{optics_dataset}.{channel}_{creative_type}_descriptions` AS a
        LEFT JOIN `{client_project}.{optics_dataset}.{channel}_{creative_type}_predictions` AS b
        ON a.video_id = b.video_id
        WHERE b.video_id IS NULL
        LIMIT 500
    """
    video_textdf = pdq.read_gbq(sql, project_id=project, use_bqstorage_api=True)
    if video_textdf.empty:
        logging.info(f"No new {creative_type} descriptions to classify.")
        return f'no new {creative_type}s'

    logging.info(f"Found {len(video_textdf)} {creative_type} descriptions to classify.")

    llm_client = Client(vertexai=True, project=project, location=location)

    old_preddf = pdq.read_gbq(
        f'SELECT * FROM `{client_project}.{optics_dataset}.{channel}_{creative_type}_predictions`',
        project_id=project,
        use_bqstorage_api=True
    )

    predlist = []
    for i, row in video_textdf.iterrows():
        video_id = row['video_id']
        logging.info(f"{i}/{len(video_textdf)}: Processing labels {creative_type} {video_id}")
        try:
            video_description = row['video_description']
            response = llm_client.models.generate_content(
                model=model, contents=[prompt, video_description],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BisselOutput
                )
            )
            predf = pd.json_normalize(response.to_json_dict()['parsed'])
            predlist += [predf.assign(video_id=video_id)]

        except Exception as e:
            logging.error(f"Error processing {creative_type} {video_id}: {e}")
            continue

    if not predlist:
        raise RuntimeError(f"All {len(video_textdf)} {creative_type} label calls failed; see errors above.")

    new_preddf = pd.concat(predlist, ignore_index=True)
    new_preddf['created_at'] = dt.now().strftime('%Y-%m-%d %H:%M:%S')

    preddf = pd.concat([old_preddf, new_preddf], ignore_index=True)

    pdq.to_gbq(preddf,
        destination_table=f'{client_project}.{optics_dataset}.{channel}_{creative_type}_predictions',
        project_id=project, if_exists='replace',
    )

    return f"processed {len(new_preddf)} {creative_type}"
