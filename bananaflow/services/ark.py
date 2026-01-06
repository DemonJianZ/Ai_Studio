import requests
from core.config import ARK_API_KEY, ARK_IMAGE_API_URL, ARK_IMAGE_MODEL_ID
from core.logging import sys_logger
from utils.size import calculate_target_resolution

def call_doubao_image_gen(prompt: str, req_id: str, size_param: str = "1024x1024", aspect_ratio: str = "1:1") -> bytes:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ARK_API_KEY}"}
    valid_size = calculate_target_resolution(size_param, aspect_ratio)

    payload = {
        "model": ARK_IMAGE_MODEL_ID,
        "prompt": prompt,
        "sequential_image_generation": "disabled",
        "response_format": "url",
        "size": valid_size,
        "stream": False,
        "watermark": False
    }

    sys_logger.info(f"[{req_id}] Calling Doubao (Ark): {prompt[:80]}... Size: {valid_size}")
    response = requests.post(ARK_IMAGE_API_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Doubao API Failed: {response.text}")

    res_json = response.json()
    if "data" in res_json and res_json["data"]:
        image_url = res_json["data"][0].get("url")
        if not image_url:
            raise RuntimeError("No image URL in Doubao response")
        img_resp = requests.get(image_url, timeout=30)
        if img_resp.status_code == 200:
            return img_resp.content
        raise RuntimeError("Failed to download generated image from Doubao")

    raise RuntimeError(f"Unexpected response format: {res_json}")