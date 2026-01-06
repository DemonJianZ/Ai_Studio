# bananaflow/services/ark_video.py
import json
import time
import requests
from typing import Optional, Dict, Any

from core.config import (
    ARK_VIDEO_MODEL_ID,
    ARK_VIDEO_MODEL_ID_NEW,
    VIDEO_MODEL_1_0,
    VIDEO_MODEL_1_5,
    ARK_VIDEO_API_URL,
    ARK_API_KEY,
)

from core.logging import sys_logger
from utils.images import parse_data_url, bytes_to_data_url


class VideoGenError(RuntimeError):
    pass


def submit_video_task(
    req_id: str,
    image_data_url: str,
    prompt: str,
    duration: int,
    camera_fixed: bool,
    resolution: Optional[str],
    ratio: Optional[str],
    last_frame_data_url: Optional[str] = None,
    model_id: Optional[str] = None,   # ep-xxxx
    seed: Optional[int] = None,
    generate_audio: Optional[bool] = None,  # ✅ 1.5 专用（你前端传 true/false）
) -> str:
    mime_type, img_bytes = parse_data_url(image_data_url)
    img_data_url = bytes_to_data_url(img_bytes, mime_type)

    # ---- 默认值 ----
    resolution = (resolution or "1080p").strip()
    ratio = (ratio or "adaptive").strip()

    # ---- flags（仍保留你原来的拼法）----
    flags = [f"--resolution {resolution}"]
    clamped_duration = max(3, min(12, int(duration)))
    flags.append(f"--duration {clamped_duration}")
    flags.append(f"--camerafixed {'true' if camera_fixed else 'false'}")
    flags.append("--watermark false")
    if ratio and ratio != "adaptive":
        flags.append(f"--ratio {ratio}")

    final_text = (prompt or "").strip() or "make it move naturally"
    full_text_param = f"{final_text} {' '.join(flags)}"

    # ✅ 关键：两张图要用 role 区分
    content_list = [
        {"type": "text", "text": full_text_param},
        {
            "type": "image_url",
            "image_url": {"url": img_data_url},
            "role": "first_frame",
        },
    ]

    if last_frame_data_url:
        tail_mime, tail_bytes = parse_data_url(last_frame_data_url)
        tail_data_url = bytes_to_data_url(tail_bytes, tail_mime)
        content_list.append(
            {
                "type": "image_url",
                "image_url": {"url": tail_data_url},
                "role": "last_frame",
            }
        )

    # ✅ payload：model / content + 可选 seed / generate_audio
    payload: Dict[str, Any] = {
        "model": (model_id or ARK_VIDEO_MODEL_ID),
        "content": content_list,
    }

    if seed is not None:
        payload["seed"] = int(seed)

    # 仅当前端传了才写入（避免影响 1.0）
    if generate_audio is not None:
        payload["generate_audio"] = bool(generate_audio)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ARK_API_KEY}"}

    sys_logger.info(f"[{req_id}] Submitting Video Task to {ARK_VIDEO_API_URL}, model={payload['model']}")
    sys_logger.info(f"[{req_id}] payload_keys={list(payload.keys())}")

    resp = requests.post(ARK_VIDEO_API_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise VideoGenError(f"Task Submission Failed: {resp.status_code} {resp.text}")

    resp_json = resp.json()
    task_id = resp_json.get("id") or (resp_json.get("data") or {}).get("id")
    if not task_id:
        raise VideoGenError(f"No task ID returned: {resp_json}")

    return task_id


def generate_video_from_image(
    req_id: str,
    image_data_url: str,
    prompt: str,
    duration: int,
    camera_fixed: bool,
    resolution: Optional[str],
    ratio: Optional[str],
    last_frame_data_url: Optional[str] = None,
    model: str | None = None,          # VIDEO_MODEL_1_0 / VIDEO_MODEL_1_5
    seed: Optional[int] = None,
    generate_audio: Optional[bool] = None,  # ✅ 给 1.5 用
) -> str:

    endpoint_id = ARK_VIDEO_MODEL_ID  # 默认 1.0
    if model == VIDEO_MODEL_1_5:
        endpoint_id = ARK_VIDEO_MODEL_ID_NEW
    elif model == VIDEO_MODEL_1_0:
        endpoint_id = ARK_VIDEO_MODEL_ID

    task_id = submit_video_task(
        req_id=req_id,
        image_data_url=image_data_url,
        prompt=prompt,
        duration=duration,
        camera_fixed=camera_fixed,
        resolution=resolution,
        ratio=ratio,
        last_frame_data_url=last_frame_data_url,
        model_id=endpoint_id,
        seed=seed,
        # ✅ 只有 1.5 才传 generate_audio；1.0 就别带这个字段
        # generate_audio=(generate_audio if model == VIDEO_MODEL_1_5 else None),
    )

    status_data = poll_video_task(req_id=req_id, task_id=task_id, max_rounds=40, interval_sec=3)
    status = status_data.get("status") or (status_data.get("data") or {}).get("status")

    if status == "failed":
        raise VideoGenError(f"Video generation failed: {json.dumps(status_data, ensure_ascii=False)[:500]}")

    video_url = extract_video_url(status_data)
    if not video_url:
        return json.dumps(status_data, ensure_ascii=False)[:800]

    data_url = download_video_as_data_url(video_url, req_id=req_id)
    return data_url or video_url


def poll_video_task(req_id: str, task_id: str, max_rounds: int = 40, interval_sec: int = 3) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ARK_API_KEY}"}
    status_url = f"{ARK_VIDEO_API_URL}/{task_id}"
    sys_logger.info(f"[{req_id}] Task Created: {task_id}, polling...")

    last_status_data = None
    for _ in range(max_rounds):
        time.sleep(interval_sec)
        r = requests.get(status_url, headers=headers, timeout=10)
        if r.status_code != 200:
            continue
        status_data = r.json()
        last_status_data = status_data
        status = status_data.get("status") or (status_data.get("data") or {}).get("status")

        if status in ("succeeded", "failed"):
            return status_data

    # 超时：把最后一次 status_data 返回给上层做更好的报错
    raise TimeoutError(f"Video generation timed out. last={json.dumps(last_status_data, ensure_ascii=False)[:500]}")


def extract_video_url(status_data: Dict[str, Any]) -> Optional[str]:
    content = status_data.get("content") or (status_data.get("data") or {}).get("content") or {}
    return content.get("video_url")


def download_video_as_data_url(video_url: str, req_id: str) -> Optional[str]:
    try:
        r = requests.get(video_url, timeout=60)
        if r.status_code == 200 and r.content:
            return bytes_to_data_url(r.content, mime_type="video/mp4")
    except Exception as e:
        sys_logger.warning(f"[{req_id}] download video failed: {e}")
    return None


