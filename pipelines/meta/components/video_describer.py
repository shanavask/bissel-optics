from kfp import dsl

@dsl.component(
        base_image="python:3.12",
        packages_to_install=["google-cloud-logging==3.16.1", "google-cloud-aiplatform==1.160.0",
            "pandas-gbq==0.35.0", "google-cloud-bigquery-storage==2.39.0", "kfp==2.17.0",
            "pillow==11.3.0"])
def generate_video_descriptions(
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
    Generates detailed descriptions of ad creative video assets using a generative AI model.
    Args:
        creative_type (str): Type of creative asset to analyze.
        dataset (str): Name of the dataset.
        storage_bucket (str): Name of the Google Cloud Storage bucket.
        project (str): Google Cloud project ID.
    """

    import pandas_gbq as pdq
    import pandas as pd
    from google.genai import Client, types
    from google.cloud import storage
    import json

    import logging

    logging.basicConfig(level=logging.DEBUG)

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

    sql = f"""
        SELECT DISTINCT a.video_id, a.gs_path
        FROM `{client_project}.{optics_dataset}.{channel}_{creative_type}_data` AS a
        LEFT JOIN `{client_project}.{optics_dataset}.{channel}_{creative_type}_descriptions` AS b
        ON a.video_id = b.video_id
        WHERE b.video_id IS NULL
        LIMIT 500
    """
    actdf = pdq.read_gbq(sql, project_id=project, use_bqstorage_api=True)
    llm_client = Client(vertexai=True, project=project, location=location)
    logging.info(f"Found {len(actdf)} {creative_type} videos to generate descriptions for.")

    video_text = []
    for idx, row in actdf.iterrows():
        video_id = row['video_id']
        print(f"Processing text {idx}/{len(actdf)}:{creative_type} {video_id}...")
        try:
            asset = types.Part.from_uri(file_uri=row['gs_path'])
            response = llm_client.models.generate_content(
                    model=model, contents=[asset, prompt],
                    config=types.GenerateContentConfig()
                )
            video_text += [(video_id, response.text)]

        except Exception as e:
            print(f"Error processing {creative_type} {video_id}: {e}")
            continue


    if len(actdf) and not video_text:
        raise RuntimeError(f"All {len(actdf)} {creative_type} description calls failed; see errors above.")

    new_textdf = pd.DataFrame(video_text, columns=['video_id', 'video_description'])
    if new_textdf.empty:
        logging.info("No descriptions were generated.")
        return 'no descriptions'

    old_textdf = pdq.read_gbq(
        f'SELECT * FROM `{client_project}.{optics_dataset}.{channel}_{creative_type}_descriptions`',
        project_id=project,
        use_bqstorage_api=True
    )
    video_textdf = pd.concat([old_textdf, new_textdf], ignore_index=True)
    pdq.to_gbq(
        video_textdf,
        destination_table=f'{client_project}.{optics_dataset}.{channel}_{creative_type}_descriptions',
        project_id=project,
        if_exists='replace',
    )

    return f"generated {len(new_textdf)} {creative_type} descriptions"
