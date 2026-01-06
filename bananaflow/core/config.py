import os

# ---- env ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
ARK_API_KEY = os.getenv("ARK_API_KEY") or "YOUR_ARK_API_KEY_HERE"

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "dayu-ai")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

MODEL_GEMINI = "gemini-3-pro-image-preview"
MODEL_DOUBAO = "doubao-seedream-4.5"
MODEL_AGENT  = "gemini-3-flash-preview"

ARK_VIDEO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
ARK_VIDEO_MODEL_ID = "ep-20251212110333-87frw"
ARK_IMAGE_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
ARK_IMAGE_MODEL_ID = "ep-20251212110122-srwd7"
ARK_VIDEO_MODEL_ID_NEW = "ep-20251225103241-gkccd"
VIDEO_MODEL_1_0 = "Doubao-Seedance-1.0-pro"
VIDEO_MODEL_1_5 = "Doubao-Seedance-1.5-pro"

BASE_DIR = os.getcwd()
LOG_DIR = os.path.join(BASE_DIR, "logs")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_output")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)