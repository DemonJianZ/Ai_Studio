import time
from google import genai
from core.config import API_KEY, PROJECT_ID, LOCATION, MODEL_GEMINI
from core.logging import sys_logger

_client = None

def init_client():
    global _client
    if _client is not None:
        return _client
    try:
        if API_KEY:
            _client = genai.Client(api_key=API_KEY)
        else:
            _client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    except Exception as e:
        sys_logger.critical(f"Client Init Failed: {e}")
        _client = None
    return _client

def get_client():
    return init_client()

def call_genai_retry(contents, config, req_id: str, retries=2):
    client = get_client()
    if client is None:
        raise RuntimeError("AI client not initialized")

    last_err = None
    for i in range(retries):
        try:
            return client.models.generate_content(model=MODEL_GEMINI, contents=contents, config=config)
        except Exception as e:
            last_err = e
            sys_logger.warning(f"[{req_id}] Gemini Retry {i+1}/{retries} failed: {e}")
            time.sleep(1 * (i + 1))
    raise RuntimeError(f"Gemini AI Service Failed: {last_err}")