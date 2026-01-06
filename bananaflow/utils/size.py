import math

def calculate_target_resolution(size_label: str, aspect_ratio: str) -> str:
    if not size_label:
        size_label = "1024x1024"
    if not aspect_ratio:
        aspect_ratio = "1:1"

    base_pixels = 1024 * 1024
    s = size_label.lower()
    if "2k" in s:
        base_pixels = 2048 * 2048
    elif "4k" in s:
        base_pixels = 4096 * 4096
    elif "x" in size_label and "1024" not in size_label:
        return size_label

    try:
        w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
        ratio = w_ratio / h_ratio
    except:
        ratio = 1.0

    height = int(math.sqrt(base_pixels / ratio))
    width = int(height * ratio)
    width = (width // 64) * 64
    height = (height // 64) * 64
    return f"{width}x{height}"

def parse_aspect_ratio(text: str) -> str:
    t = (text or "").lower()
    if "9:16" in t or "竖" in t:
        return "9:16"
    if "16:9" in t or "横" in t:
        return "16:9"
    if "1:1" in t or "方" in t:
        return "1:1"
    return "1:1"