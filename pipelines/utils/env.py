import os
import dotenv

dotenv.load_dotenv()

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "EMPTY")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = os.getenv("MODEL", "gemini-3.5-flash")
STAGING_BUCKET = os.getenv("STAGING_BUCKET", "")
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", None)
SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT", None)

CLIENT_PROJECT = os.getenv("CLIENT_PROJECT", None)
CLIENT_DATASET = os.getenv("CLIENT_DATASET", None)
META_TABLE = os.getenv("META_TABLE", None)

OPTICS_DATASET = os.getenv("OPTICS_DATASET", 'optics')
META_IMAGE_TABLE = os.getenv("META_IMAGE_TABLE", None)
META_VIDEO_TABLE = os.getenv("META_VIDEO_TABLE", None)
META_IMG_PATH = os.getenv("META_IMG_PATH", None)
META_VID_PATH = os.getenv("META_VID_PATH", None)

META_TOKEN = os.getenv("META_TOKEN", None)