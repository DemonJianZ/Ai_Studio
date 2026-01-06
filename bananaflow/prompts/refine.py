from functools import lru_cache
from google.genai import types
from core.config import MODEL_AGENT
from services.genai_client import get_client

def simple_refine_prompt(user_prompt: str) -> str:
    p = (user_prompt or "").strip()
    p = p.replace("动漫", "anime").replace("二次元", "anime").replace("写实", "photorealistic").replace("更真实", "more photorealistic")
    return (
        f"Edit the input image: {p}. "
        "Keep composition, lighting, camera angle, and background unless explicitly specified. "
        "Do not introduce unrelated objects. Apply only the requested change."
    )

def agent_refine_prompt(user_prompt: str) -> str:
    client = get_client()
    if client is None:
        return simple_refine_prompt(user_prompt)

    SYSTEM = """
You are an image-edit prompt optimizer.
Rewrite the user's short request into a stable English instruction for an image-to-image editing model.

Rules:
- Output ONLY a single English prompt (no JSON, no markdown).
- Default constraints: keep composition/lighting/background unless specified.
- Be explicit about what to change vs what to keep.
"""
    resp = client.models.generate_content(
        model=MODEL_AGENT,
        contents=[types.Part(text=SYSTEM), types.Part(text=f"User request: {user_prompt.strip()}")],
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=220),
    )
    text = resp.candidates[0].content.parts[0].text.strip()
    return text.strip("`").strip()

@lru_cache(maxsize=512)
def cached_refine_prompt(user_prompt: str) -> str:
    return agent_refine_prompt(user_prompt)