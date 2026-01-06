import base64
from typing import Tuple
from core.logging import sys_logger

def parse_data_url(img_str: str) -> Tuple[str, bytes]:
    if not img_str:
        raise ValueError("Image data is empty")
    mime_type = "image/png"
    b64_str = img_str
    if "base64," in img_str:
        parts = img_str.split("base64,")
        if len(parts) > 1:
            head = parts[0]
            if "image/jpeg" in head:
                mime_type = "image/jpeg"
            elif "image/webp" in head:
                mime_type = "image/webp"
            b64_str = parts[1]
    return mime_type, base64.b64decode(b64_str)

def bytes_to_data_url(data_bytes: bytes, mime_type="image/png") -> str:
    b64 = base64.b64encode(data_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

def get_image_from_response(response):
    if getattr(response, "candidates", None):
        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None):
                return part.inline_data.data
    return None