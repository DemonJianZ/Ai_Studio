import os
import base64
import time
import traceback
import sys
import uuid
import json
import logging
import collections
import re
import requests
import math
import threading
from functools import lru_cache
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, Set, Union, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Google GenAI SDK
from google import genai
from google.genai import types

# ==========================================
# BananaFlow Backend v3.3 (Stable) + Agent v3.1 (Controlled Patch Planner)
# ==========================================

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

MODEL_GEMINI = "gemini-3-pro-image-preview"   # 图像模型（始终用于图像）
MODEL_DOUBAO = "doubao-seedream-4.5"
MODEL_AGENT = "gemini-3-flash-preview"        # Agent Brain（只做规划/改写prompt）

ARK_VIDEO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
ARK_VIDEO_MODEL_ID = "ep-20251212110333-87frw"
ARK_IMAGE_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
ARK_IMAGE_MODEL_ID = "ep-20251212110122-srwd7"

BASE_DIR = os.getcwd()
LOG_DIR = os.path.join(BASE_DIR, "logs")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_output")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# ---- logging ----
sys_logger = logging.getLogger("banana_flow_sys")
sys_logger.setLevel(logging.INFO)
sys_logger.propagate = False
if not sys_logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    sys_logger.addHandler(console_handler)

# ---- agent concurrency guard ----
AGENT_SEM = threading.Semaphore(2)  # 同时最多2个 agent 调用（你可按并发调整）

def run_agent_call(fn):
    if not AGENT_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Agent 并发已满（后端限流保护），请稍后重试。")
    try:
        return fn()
    finally:
        AGENT_SEM.release()

def _new_id(prefix="n") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

# ==========================================
# Prompt Logger & Analyzer
# ==========================================

class PromptLogger:
    def __init__(self, filename="prompts.jsonl"):
        self.filepath = os.path.join(LOG_DIR, filename)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                pass

    def log(
        self,
        req_id: str,
        mode: str,
        inputs: Dict,
        final_prompt: str,
        config: Dict,
        output_meta: Dict,
        latency: float,
        error: str = None,
    ):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": req_id,
            "mode": mode,
            "inputs": self._sanitize(inputs),
            "final_prompt": final_prompt,
            "config": config,
            "output": output_meta,
            "latency_sec": round(latency, 3),
            "error": error,
        }
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            sys_logger.error(f"Failed to log: {e}")

    def _sanitize(self, data: Dict) -> Dict:
        if not isinstance(data, dict):
            return {"_": str(data)}
        clean = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 500:
                clean[k] = "<LONG_TEXT_OR_BASE64>"
            elif isinstance(v, list) and v and isinstance(v[0], str) and len(v[0]) > 500:
                clean[k] = ["<LONG_TEXT_OR_BASE64>" for _ in v]
            else:
                clean[k] = v
        return clean

prompt_logger = PromptLogger()

class LogAnalyzer:
    def __init__(self, log_path):
        self.log_path = log_path

    def _read_logs(self, limit=1000):
        if not os.path.exists(self.log_path):
            return []
        lines = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        lines.append(json.loads(line))
                    except:
                        pass
        except Exception:
            return []
        return lines[-limit:]

    def get_history(self, limit=20):
        logs = self._read_logs(300)
        history = []
        for entry in reversed(logs):
            if entry.get("error"):
                continue
            inputs = entry.get("inputs") or {}
            history.append({
                "id": entry.get("request_id"),
                "time": entry.get("timestamp"),
                "mode": entry.get("mode"),
                "prompt": (inputs.get("prompt") if isinstance(inputs, dict) else "") or "",
                "note": "",
            })
            if len(history) >= limit:
                break
        return history

    def get_stats(self):
        logs = self._read_logs(1000)
        if not logs:
            return {"modes": {}, "keywords": []}

        mode_counter = collections.Counter()
        text_corpus = []

        for entry in logs:
            mode_counter[entry.get("mode", "unknown")] += 1
            inputs = entry.get("inputs") or {}
            if isinstance(inputs, dict):
                p = inputs.get("prompt") or ""
                if p:
                    text_corpus.append(p)

        stop_words = set(["a", "an", "the", "in", "on", "of", "with", "and", "to", "is", "for"])
        words = []
        for text in text_corpus:
            tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            for t in tokens:
                if t not in stop_words:
                    words.append(t)
        top = collections.Counter(words).most_common(10)
        return {"modes": dict(mode_counter), "keywords": [k for k, _ in top]}

analyzer = LogAnalyzer(os.path.join(LOG_DIR, "prompts.jsonl"))

# ==========================================
# GenAI Client
# ==========================================

try:
    if API_KEY:
        client = genai.Client(api_key=API_KEY)
    else:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
except Exception as e:
    sys_logger.critical(f"Client Init Failed: {e}")
    client = None

# ==========================================
# FastAPI init
# ==========================================

app = FastAPI(title="BananaFlow - 电商智能图像工作台", version="3.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request.state.req_id = str(uuid.uuid4())[:8]
    return await call_next(request)

# ==========================================
# DTOs
# ==========================================

class EditRequest(BaseModel):
    image: str
    ref_image: Optional[str] = None
    background_image: Optional[str] = None
    mode: str
    prompt: str
    temperature: Optional[float] = 0.4
    model: Optional[str] = MODEL_GEMINI
    size: Optional[str] = None
    aspect_ratio: Optional[str] = None

class EditResponse(BaseModel):
    image: str

class Text2ImgRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    model: Optional[str] = MODEL_GEMINI
    size: Optional[str] = "1024x1024"
    aspect_ratio: Optional[str] = "1:1"

class Text2ImgResponse(BaseModel):
    images: List[str]

class MultiImageRequest(BaseModel):
    prompt: str
    images: List[str]
    temperature: float = 0.7
    size: Optional[str] = "1024x1024"
    aspect_ratio: Optional[str] = "1:1"

class MultiImageResponse(BaseModel):
    image: str

class Img2VideoRequest(BaseModel):
    image: str
    last_frame_image: Optional[str] = None
    prompt: Optional[str] = ""
    duration: int = Field(default=5, ge=3, le=12)
    fps: Optional[int] = 24
    camera_fixed: Optional[bool] = False
    resolution: Optional[str] = "1080p"
    ratio: Optional[str] = "16:9"

class Img2VideoResponse(BaseModel):
    image: str

class SelectedArtifact(BaseModel):
    url: str
    kind: Optional[str] = "image"
    fromNodeId: Optional[str] = None
    createdAt: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

class AgentRequest(BaseModel):
    prompt: str
    selected_artifact: Optional[SelectedArtifact] = None
    current_nodes: Optional[List[Dict[str, Any]]] = None
    current_connections: Optional[List[Dict[str, Any]]] = None

# ==========================================
# Agent Patch Schema (Pydantic)
# ==========================================

class AddNodeOp(BaseModel):
    op: Literal["add_node"]
    node: Dict[str, Any]

class AddConnOp(BaseModel):
    op: Literal["add_connection"]
    connection: Dict[str, Any]

class UpdateNodeOp(BaseModel):
    op: Literal["update_node"]
    id: str
    data: Dict[str, Any] = Field(default_factory=dict)

class DeleteNodeOp(BaseModel):
    op: Literal["delete_node"]
    id: str

class DeleteConnOp(BaseModel):
    op: Literal["delete_connection"]
    id: str

class SelectNodesOp(BaseModel):
    op: Literal["select_nodes"]
    ids: List[str] = Field(default_factory=list)

class SetViewportOp(BaseModel):
    op: Literal["set_viewport"]
    viewport: Dict[str, float] = Field(default_factory=dict)

PatchOp = Union[
    AddNodeOp, AddConnOp, UpdateNodeOp,
    DeleteNodeOp, DeleteConnOp,
    SelectNodesOp, SetViewportOp
]

class AgentOut(BaseModel):
    patch: List[PatchOp]
    summary: str = ""
    thought: str = ""

# ==========================================
# Image helpers
# ==========================================

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

# ==========================================
# Business prompt builder (edit)
# ==========================================

def build_business_prompt(mode: str, user_prompt: str, has_ref_image: bool) -> str:
    clean_prompt = (user_prompt or "").strip().strip(",").strip()
    style_suffix = f" Additional requirements: {clean_prompt}" if clean_prompt else ""

    if mode == "bg_replace":
        if has_ref_image:
            return f"""Change the background to match the reference image.
CRITICAL: Keep the main product and any hand holding it exactly as is.
Do not change the product or the hand gesture.
Only replace the background environment.{style_suffix}"""
        bg_desc = clean_prompt if clean_prompt else "clean professional studio background"
        return (
            f"Generate a background: '{bg_desc}'. "
            "CRITICAL: Isolate the foreground product and the hand holding it. "
            "Keep the product and hand pixels unchanged. Only replace the background."
        )

    if mode == "gesture_swap":
        if has_ref_image:
            return f"""Change the hand in the main image to match the reference image's hand.
CRITICAL: Keep the product object and the background exactly unchanged.
Only swap the hand and fingers to match the reference hand.{style_suffix}"""
        gesture_desc = clean_prompt if clean_prompt else "a natural hand holding gesture"
        return (
            f"Change the hand gesture to: '{gesture_desc}'. "
            "Constraint: Keep the product/object and the background exactly as is. Only change the hand."
        )

    if mode == "product_swap":
        if has_ref_image:
            return f"""Replace the object held in the hand with the product from the reference image.
CRITICAL: Keep the original hand gesture, skin tone, and background exactly unchanged.
Only swap the held object.{style_suffix}"""
        product_desc = clean_prompt if clean_prompt else "a generic product"
        return (
            f"Replace the held object with: '{product_desc}'. "
            "Constraint: Keep the hand gesture, skin tone, and background exactly as is."
        )

    if mode == "relight":
        return f"""
把整个手的颜色变成黄色调，不要有突兀的颜色不均匀现象，确保手和产品的光影自然融合在一起，像是在同一个环境下拍摄的一样。
保持手部的细节和质感，不要模糊或失真。确保手指的形状和位置与原图一致，不要改变手的姿势。
保持产品的外观和颜色不变，不要影响产品的细节。
确保背景和其他元素保持不变，不要引入新的物体或改变场景的构图。
{style_suffix}
"""

    if mode == "upscale":
        return f"""Upscale this image to high resolution and improve clarity.
Instruction: Denoise, sharpen details, and enhance textures.
{style_suffix}
CRITICAL: Maintain absolute fidelity to the original content.
Do not add new objects or change the subject's features.
The output must look like a high-end commercial photograph."""

    return f"Edit the image. {clean_prompt}"

# ==========================================
# Size helpers
# ==========================================

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

def _parse_aspect_ratio(text: str) -> str:
    t = (text or "").lower()
    if "9:16" in t or "竖" in t:
        return "9:16"
    if "16:9" in t or "横" in t:
        return "16:9"
    if "1:1" in t or "方" in t:
        return "1:1"
    return "1:1"

# ==========================================
# Service wrappers
# ==========================================

def call_genai_retry(contents, config, req_id, retries=2):
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

# ==========================================
# Agent: context slimming + subgraph selection
# ==========================================

def _find_node(nodes: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
    return next((n for n in nodes if n.get("id") == node_id), None)

def _normalize_conns(conns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in conns or []:
        f = c.get("from") or c.get("source")
        t = c.get("to") or c.get("target")
        if not f or not t:
            continue
        out.append({"id": c.get("id") or _new_id("c"), "from": f, "to": t})
    return out

def _collect_subgraph_ids(center_id: str, nodes: List[Dict[str, Any]], conns: List[Dict[str, Any]], depth: int = 2, max_nodes: int = 40) -> Set[str]:
    if not center_id:
        return set()
    conn_norm = _normalize_conns(conns)
    adj = collections.defaultdict(list)
    radj = collections.defaultdict(list)
    for c in conn_norm:
        adj[c["from"]].append(c["to"])
        radj[c["to"]].append(c["from"])

    seen = {center_id}
    frontier = {center_id}
    for _ in range(depth):
        nxt = set()
        for nid in frontier:
            for v in adj.get(nid, []):
                if v not in seen:
                    seen.add(v); nxt.add(v)
            for v in radj.get(nid, []):
                if v not in seen:
                    seen.add(v); nxt.add(v)
        frontier = nxt
        if len(seen) >= max_nodes:
            break

    existing = {n.get("id") for n in nodes or []}
    return {x for x in seen if x in existing}

def _compact_nodes(nodes: List[Dict[str, Any]], keep_ids: Optional[Set[str]] = None, limit: int = 60) -> List[Dict[str, Any]]:
    out = []
    for n in nodes or []:
        nid = n.get("id")
        if keep_ids and nid not in keep_ids:
            continue
        d = n.get("data") or {}
        out.append({
            "id": nid,
            "type": n.get("type"),
            "x": int(n.get("x", 0)),
            "y": int(n.get("y", 0)),
            "data": {
                "label": d.get("label"),  # ✅关键：自然语言指代
                "mode": d.get("mode"),
                "prompt": (d.get("prompt") or "")[:400],
                "text": (d.get("text") or "")[:200],
                "templates": d.get("templates"),
            }
        })
        if len(out) >= limit:
            break
    return out

def _compact_conns(conns: List[Dict[str, Any]], keep_ids: Optional[Set[str]] = None, limit: int = 80) -> List[Dict[str, Any]]:
    out = []
    for c in _normalize_conns(conns):
        if keep_ids and (c["from"] not in keep_ids and c["to"] not in keep_ids):
            continue
        out.append(c)
        if len(out) >= limit:
            break
    return out

# ==========================================
# Agent: prompt refining
# ==========================================

def simple_refine_prompt(user_prompt: str) -> str:
    p = (user_prompt or "").strip()
    p = p.replace("动漫", "anime").replace("二次元", "anime").replace("写实", "photorealistic").replace("更真实", "more photorealistic")
    return (
        f"Edit the input image: {p}. "
        "Keep composition, lighting, camera angle, and background unless explicitly specified. "
        "Do not introduce unrelated objects. Apply only the requested change."
    )

@lru_cache(maxsize=512)
def cached_refine_prompt(user_prompt: str) -> str:
    return agent_refine_prompt(user_prompt)

def agent_refine_prompt(user_prompt: str) -> str:
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
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=220),
    )
    text = resp.candidates[0].content.parts[0].text.strip()
    return text.strip("`").strip()

# ==========================================
# Patch builders (deterministic fallback)
# ==========================================

def _find_upstream_id(conns: List[Dict[str, Any]], to_id: str) -> Optional[str]:
    cn = _normalize_conns(conns)
    hit = next((c for c in cn if c.get("to") == to_id), None)
    return hit.get("from") if hit else None

def build_continue_chain_patch(
    refined_prompt: str,
    current_nodes: Optional[List[Dict[str, Any]]],
    current_connections: Optional[List[Dict[str, Any]]],
    selected_artifact: Dict[str, Any],
    model: str = MODEL_GEMINI,
    templates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    nodes = current_nodes or []
    conns = current_connections or []
    from_node_id = (selected_artifact or {}).get("fromNodeId")

    if not from_node_id:
        raise RuntimeError("selected_artifact.fromNodeId 缺失，无法定位串联锚点")

    from_node = _find_node(nodes, from_node_id)
    if not from_node:
        raise RuntimeError(f"找不到 fromNodeId={from_node_id} 对应节点")

    anchor_id = from_node_id
    if from_node.get("type") == "output":
        upstream = _find_upstream_id(conns, from_node_id)
        if upstream:
            anchor_id = upstream

    anchor_node = _find_node(nodes, anchor_id)
    if not anchor_node:
        raise RuntimeError(f"找不到 anchor 节点：{anchor_id}")

    base_x = int(anchor_node.get("x", 200))
    base_y = int(anchor_node.get("y", 200))

    proc_id = _new_id("proc")
    out_id = _new_id("out")
    tpl = templates or {"size": "1024x1024", "aspect_ratio": "1:1"}

    patch = [
        {
            "op": "add_node",
            "node": {
                "id": proc_id,
                "type": "processor",
                "x": base_x + 350,
                "y": base_y,
                "data": {
                    "mode": "multi_image_generate",
                    "prompt": refined_prompt,
                    "templates": tpl,
                    "batchSize": 1,
                    "status": "idle",
                    "model": model,
                },
            },
        },
        {
            "op": "add_node",
            "node": {
                "id": out_id,
                "type": "output",
                "x": base_x + 700,
                "y": base_y,
                "data": {"images": []},
            },
        },
        {"op": "add_connection", "connection": {"id": _new_id("c"), "from": anchor_id, "to": proc_id}},
        {"op": "add_connection", "connection": {"id": _new_id("c"), "from": proc_id, "to": out_id}},
        {"op": "select_nodes", "ids": [proc_id]},
    ]

    return {
        "patch": patch,
        "summary": "已在上一轮产出节点后追加：图生图 → 输出（并自动填充优化后的提示词）",
        "thought": f"chain-after: {anchor_id} -> {proc_id} -> {out_id}",
    }

def build_iterate_branch_with_new_input_patch(
    refined_prompt: str,
    selected_artifact: Dict[str, Any],
    model: str = MODEL_GEMINI,
    x0: int = 200,
    y0: int = 200,
    templates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    in_id = _new_id("in")
    proc_id = _new_id("proc")
    out_id = _new_id("out")
    tpl = templates or {"size": "1024x1024", "aspect_ratio": "1:1"}

    patch = [
        {"op": "add_node", "node": {"id": in_id, "type": "input", "x": x0, "y": y0, "data": {"images": [selected_artifact["url"]]}}},
        {"op": "add_node", "node": {"id": proc_id, "type": "processor", "x": x0 + 350, "y": y0,
                                   "data": {"mode": "multi_image_generate", "prompt": refined_prompt, "templates": tpl, "batchSize": 1, "status": "idle", "model": model}}},
        {"op": "add_node", "node": {"id": out_id, "type": "output", "x": x0 + 700, "y": y0, "data": {"images": []}}},
        {"op": "add_connection", "connection": {"id": _new_id("c"), "from": in_id, "to": proc_id}},
        {"op": "add_connection", "connection": {"id": _new_id("c"), "from": proc_id, "to": out_id}},
        {"op": "select_nodes", "ids": [proc_id]},
    ]
    return {"patch": patch, "summary": "fromNodeId 缺失，已新建 input→图生图→输出 分支", "thought": "fallback new-input branch"}

def build_from_scratch_patch(user_prompt: str, model: str = MODEL_GEMINI, x0: int = 120, y0: int = 120) -> Dict[str, Any]:
    ar = _parse_aspect_ratio(user_prompt)
    tpl = {"size": "1024x1024", "aspect_ratio": ar}

    text_id = _new_id("text")
    gen_id = _new_id("gen")
    out1_id = _new_id("out")
    edit_id = _new_id("edit")
    out2_id = _new_id("out")

    initial_prompt = (user_prompt or "").strip()
    if not initial_prompt:
        initial_prompt = "Generate a clean commercial product photo, high quality, studio lighting."

    default_edit_prompt = "Refine the image: improve composition and details. Keep style consistent."

    patch = [
        {"op": "add_node", "node": {"id": text_id, "type": "text_input", "x": x0, "y": y0, "data": {"text": initial_prompt}}},
        {"op": "add_node", "node": {"id": gen_id, "type": "processor", "x": x0 + 320, "y": y0,
                                   "data": {"mode": "text2img", "prompt": initial_prompt, "templates": tpl, "batchSize": 1, "status": "idle", "model": model}}},
        {"op": "add_node", "node": {"id": out1_id, "type": "output", "x": x0 + 640, "y": y0, "data": {"images": []}}},
        {"op": "add_node", "node": {"id": edit_id, "type": "processor", "x": x0 + 320, "y": y0 + 220,
                                   "data": {"mode": "multi_image_generate", "prompt": default_edit_prompt, "templates": tpl, "batchSize": 1, "status": "idle", "model": model}}},
        {"op": "add_node", "node": {"id": out2_id, "type": "output", "x": x0 + 640, "y": y0 + 220, "data": {"images": []}}},

        {"op": "add_connection", "connection": {"id": _new_id("c"), "from": text_id, "to": gen_id}},
        {"op": "add_connection", "connection": {"id": _new_id("c"), "from": gen_id, "to": out1_id}},

        {"op": "add_connection", "connection": {"id": _new_id("c"), "from": gen_id, "to": edit_id}},
        {"op": "add_connection", "connection": {"id": _new_id("c"), "from": edit_id, "to": out2_id}},

        {"op": "select_nodes", "ids": [gen_id]},
    ]
    return {"patch": patch, "summary": "已从零搭建：文生图 + 连续图生图编辑链路", "thought": "scratch plan with continuous edit"}

def deterministic_plan_or_patch(
    user_prompt: str,
    selected_artifact: Optional[Dict[str, Any]],
    current_nodes: Optional[List[Dict[str, Any]]],
    current_connections: Optional[List[Dict[str, Any]]],
    fallback_refine: bool = True,
) -> Dict[str, Any]:
    if selected_artifact and selected_artifact.get("url"):
        refined = simple_refine_prompt(user_prompt) if fallback_refine else (user_prompt or "").strip()
        try:
            return build_continue_chain_patch(refined, current_nodes, current_connections, selected_artifact, model=MODEL_GEMINI)
        except Exception:
            return build_iterate_branch_with_new_input_patch(refined, selected_artifact, model=MODEL_GEMINI)

    return build_from_scratch_patch(user_prompt, model=MODEL_GEMINI)

# ==========================================
# Core APIs
# ==========================================

@app.post("/api/text2img", response_model=Text2ImgResponse)
def text_to_image(req: Text2ImgRequest, request: Request):
    req_id = request.state.req_id
    selected_model = req.model or MODEL_GEMINI
    t0 = time.time()

    try:
        img_bytes = None
        if selected_model == MODEL_DOUBAO:
            img_bytes = call_doubao_image_gen(req.prompt, req_id, size_param=req.size, aspect_ratio=req.aspect_ratio)
        else:
            gemini_resolution = "1K"
            s = (req.size or "").lower()
            if "2k" in s:
                gemini_resolution = "2K"
            elif "4k" in s:
                gemini_resolution = "4K"

            gen_config = types.GenerateContentConfig(
                temperature=req.temperature,
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=req.aspect_ratio or "1:1",
                    image_size=gemini_resolution
                )
            )
            response = call_genai_retry(contents=[types.Part(text=req.prompt)], config=gen_config, req_id=req_id)
            img_bytes = get_image_from_response(response)

        if not img_bytes:
            raise RuntimeError("No image returned")

        prompt_logger.log(
            req_id, "text2img", req.model_dump(), req.prompt,
            {"model": selected_model, "temp": req.temperature, "size": req.size, "ar": req.aspect_ratio},
            {"file": "mem"}, time.time() - t0
        )
        return Text2ImgResponse(images=[bytes_to_data_url(img_bytes)])

    except Exception as e:
        sys_logger.error(f"[{req_id}] Text2Img Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/multi_image_generate", response_model=MultiImageResponse)
def multi_image_generate(req: MultiImageRequest, request: Request):
    req_id = request.state.req_id
    t0 = time.time()

    try:
        contents = [types.Part(text=req.prompt)]
        for img_str in req.images:
            m, b = parse_data_url(img_str)
            contents.append(types.Part.from_bytes(data=b, mime_type=m))

        gemini_resolution = "1K"
        s = (req.size or "").lower()
        if "2k" in s:
            gemini_resolution = "2K"
        elif "4k" in s:
            gemini_resolution = "4K"

        gen_config = types.GenerateContentConfig(
            temperature=req.temperature,
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=req.aspect_ratio or "1:1",
                image_size=gemini_resolution
            )
        )

        response = call_genai_retry(contents=contents, config=gen_config, req_id=req_id)
        img_bytes = get_image_from_response(response)
        if not img_bytes:
            raise RuntimeError("No image returned")

        prompt_logger.log(
            req_id, "multi_image_generate", req.model_dump(), req.prompt,
            {"temperature": req.temperature, "ar": req.aspect_ratio},
            {"file": "mem"}, time.time() - t0
        )
        return MultiImageResponse(image=bytes_to_data_url(img_bytes))

    except Exception as e:
        sys_logger.error(f"[{req_id}] MultiImage Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/edit", response_model=EditResponse)
def edit_image(req: EditRequest, request: Request):
    req_id = request.state.req_id
    t0 = time.time()

    final_ref_image = req.ref_image or req.background_image
    has_ref = bool(final_ref_image)
    selected_model = req.model or MODEL_GEMINI
    final_prompt = build_business_prompt(req.mode, req.prompt, has_ref)

    try:
        img_bytes = None

        if selected_model == MODEL_DOUBAO:
            img_bytes = call_doubao_image_gen(
                final_prompt, req_id,
                size_param=req.size or "1024x1024",
                aspect_ratio=req.aspect_ratio or "1:1"
            )
        else:
            fg_mime, fg_bytes = parse_data_url(req.image)
            contents = [types.Part(text=final_prompt), types.Part.from_bytes(data=fg_bytes, mime_type=fg_mime)]

            if has_ref:
                bg_mime, bg_bytes = parse_data_url(final_ref_image)
                contents.append(types.Part.from_bytes(data=bg_bytes, mime_type=bg_mime))

            temp = 0.3 if req.mode in ["relight", "upscale"] else (req.temperature or 0.4)

            response = call_genai_retry(
                contents=contents,
                config=types.GenerateContentConfig(temperature=temp),
                req_id=req_id
            )
            img_bytes = get_image_from_response(response)

        if not img_bytes:
            raise RuntimeError("No image returned")

        prompt_logger.log(
            req_id, req.mode, req.model_dump(), final_prompt,
            {"model": selected_model, "has_ref": has_ref},
            {"file": "mem"}, time.time() - t0
        )
        return EditResponse(image=bytes_to_data_url(img_bytes))

    except Exception as e:
        sys_logger.error(f"[{req_id}] Edit Error ({req.mode}): {e}")
        prompt_logger.log(req_id, req.mode, req.model_dump(), "ERROR", {}, {}, time.time() - t0, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/img2video", response_model=Img2VideoResponse)
def img_to_video(req: Img2VideoRequest, request: Request):
    req_id = request.state.req_id

    try:
        mime_type, img_bytes = parse_data_url(req.image)
        img_base64_str = bytes_to_data_url(img_bytes, mime_type)

        final_text = (req.prompt or "").strip() or "make it move naturally"
        flags = [f"--resolution {req.resolution}"]

        clamped_duration = max(3, min(12, req.duration))
        flags.append(f"--duration {clamped_duration}")
        flags.append(f"--camerafixed {'true' if req.camera_fixed else 'false'}")
        flags.append("--watermark false")

        if req.ratio and req.ratio != "adaptive":
            flags.append(f"--ratio {req.ratio}")

        full_text_param = f"{final_text} {' '.join(flags)}"

        content_list = [
            {"type": "text", "text": full_text_param},
            {"type": "image_url", "image_url": {"url": img_base64_str}}
        ]

        if req.last_frame_image:
            tail_mime, tail_bytes = parse_data_url(req.last_frame_image)
            tail_base64 = bytes_to_data_url(tail_bytes, tail_mime)
            content_list.append({"type": "image_url", "image_url": {"url": tail_base64}})

        payload = {"model": ARK_VIDEO_MODEL_ID, "content": content_list}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ARK_API_KEY}"}

        sys_logger.info(f"[{req_id}] Submitting Video Task to {ARK_VIDEO_API_URL}")
        response = requests.post(ARK_VIDEO_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Task Submission Failed: {response.status_code} {response.text}")

        resp_json = response.json()
        task_id = resp_json.get("id") or resp_json.get("data", {}).get("id")
        if not task_id:
            raise RuntimeError(f"No task ID returned: {resp_json}")

        status_url = f"{ARK_VIDEO_API_URL}/{task_id}"
        sys_logger.info(f"[{req_id}] Task Created: {task_id}, polling...")

        for _ in range(40):
            time.sleep(3)
            status_resp = requests.get(status_url, headers=headers, timeout=10)
            if status_resp.status_code != 200:
                continue
            status_data = status_resp.json()
            status = status_data.get("status") or status_data.get("data", {}).get("status")

            if status == "succeeded":
                content = status_data.get("content") or status_data.get("data", {}).get("content", {})
                video_url = content.get("video_url")
                if not video_url:
                    return Img2VideoResponse(image=json.dumps(status_data, ensure_ascii=False)[:800])

                try:
                    video_content_resp = requests.get(video_url, timeout=60)
                    if video_content_resp.status_code == 200:
                        return Img2VideoResponse(image=bytes_to_data_url(video_content_resp.content, mime_type="video/mp4"))
                except Exception:
                    pass
                return Img2VideoResponse(image=video_url)

            if status == "failed":
                raise RuntimeError("Video generation failed")

        raise HTTPException(status_code=504, detail="Video generation timed out")

    except Exception as e:
        sys_logger.error(f"[{req_id}] VideoGen Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Agent endpoint (flexible + robust)
# ==========================================

def _agent_system_prompt() -> str:
    return f"""
You are a workflow-planning agent for an image editing canvas (FlowStudio).
You must output STRICT JSON ONLY (no markdown).

Your output JSON shape:
{{
  "patch": [ ... ],
  "summary": "...",
  "thought": "..."
}}

Allowed patch ops:
1) {{ "op":"add_node", "node":{{ "id":"...", "type":"text_input|input|processor|post_processor|video_gen|output", "x": int, "y": int, "data": {{...}} }} }}
2) {{ "op":"add_connection", "connection":{{ "id":"...", "from":"nodeId", "to":"nodeId" }} }}
3) {{ "op":"update_node", "id":"nodeId", "data":{{...}} }}
4) {{ "op":"delete_node", "id":"nodeId" }}
5) {{ "op":"delete_connection", "id":"connId" }}
6) {{ "op":"select_nodes", "ids":["nodeId1","nodeId2"] }}
7) {{ "op":"set_viewport", "viewport":{{ "x": float, "y": float, "zoom": float }} }}

Rules:
- NEVER include extra top-level keys besides patch/summary/thought.
- Do NOT output base64 or large blobs.
- Always include x and y for add_node.
- Always include connection.id, connection.from, connection.to for add_connection.
- Keep patch length reasonable.

Node catalog (capabilities):
- text_input: data={{ text: string, label?: string }}
- input: data={{ images: string[], label?: string }}
- output: data={{ images: string[], label?: string }}
- processor/post_processor:
  data must include:
    - mode: one of ["text2img","multi_image_generate","edit","img2video"]
    - prompt: string (English, stable)
    - templates: {{ size:"1024x1024"|"...", aspect_ratio:"1:1"|"16:9"|"9:16" }}
    - model: MUST be "{MODEL_GEMINI}" (fixed)
    - status: "idle"
    - batchSize: 1
  for mode="edit", also include:
    - edit_mode: one of ["bg_replace","gesture_swap","product_swap","relight","upscale"]

Business mapping (natural language -> node plan):
- "从零生成/文生图/生成一张" -> add: text_input -> processor(mode="text2img") -> output
- "继续编辑/基于刚才结果/再来一步" -> chain: processor(mode="multi_image_generate") -> output after anchor
- "换背景/替换背景" -> processor(mode="edit", edit_mode="bg_replace")
- "换手/换手势/用参考手" -> processor(mode="edit", edit_mode="gesture_swap")
- "换商品/替换产品" -> processor(mode="edit", edit_mode="product_swap")
- "补光/调色/relight" -> processor(mode="edit", edit_mode="relight")
- "放大/修复/高清/upscale" -> processor(mode="edit", edit_mode="upscale")

Important business logic:
- If user provided selected_artifact and asks for further edits, prefer to append AFTER the node that produced it (chain it).
- If fromNodeId is missing, fall back to input(selected_artifact.url) -> processor -> output.
- Use node.data.label (if present) to reference "the second node / last output / background node" etc.

Prompt policy:
- Always refine the user request into ONE stable English instruction and put it into processor.data.prompt.
- Default constraint: keep composition/lighting/background unless user explicitly requests changes.

Now produce JSON.
"""

def _safe_json_load(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    s = s.strip("`").strip()
    return json.loads(s)

def normalize_patch(out: Dict[str, Any]) -> Dict[str, Any]:
    # 统一：兼容 remove_*（如果模型偶尔输出）
    patch = out.get("patch") or []
    if not isinstance(patch, list):
        patch = []
    out["patch"] = patch[:80]

    for item in out["patch"]:
        if item.get("op") == "remove_node":
            item["op"] = "delete_node"
        if item.get("op") == "remove_connection":
            item["op"] = "delete_connection"

    seen_node_ids = set()
    seen_conn_ids = set()
    default_tpl = {"size": "1024x1024", "aspect_ratio": "1:1"}

    for item in out["patch"]:
        op = item.get("op")

        if op == "add_node":
            node = item.get("node") or {}
            item["node"] = node

            node.setdefault("id", _new_id("n"))
            if node["id"] in seen_node_ids:
                node["id"] = _new_id("n")
            seen_node_ids.add(node["id"])

            node.setdefault("type", "processor")
            node["x"] = int(node.get("x", 200))
            node["y"] = int(node.get("y", 200))
            data = node.get("data") or {}
            node["data"] = data

            if node["type"] in ("processor", "post_processor"):
                # 强制固定图像模型
                data["model"] = MODEL_GEMINI
                data.setdefault("status", "idle")
                data.setdefault("batchSize", 1)
                data.setdefault("templates", default_tpl)

            if node["type"] == "text_input":
                data.setdefault("text", "")
            if node["type"] == "input":
                data.setdefault("images", [])
            if node["type"] == "output":
                data.setdefault("images", [])

        if op == "add_connection":
            c = item.get("connection") or {}
            item["connection"] = c
            c.setdefault("id", _new_id("c"))
            if c["id"] in seen_conn_ids:
                c["id"] = _new_id("c")
            seen_conn_ids.add(c["id"])

    out["summary"] = str(out.get("summary") or "")
    out["thought"] = str(out.get("thought") or "")
    return out

@app.post("/api/agent/plan", response_model=Dict[str, Any])
def agent_plan(req: AgentRequest, request: Request):
    req_id = request.state.req_id
    user_text = (req.prompt or "").strip()
    selected = req.selected_artifact.model_dump() if req.selected_artifact else None
    nodes = req.current_nodes or []
    conns = req.current_connections or []

    # ---- 如果没 client，直接 deterministic ----
    if client is None:
        out = deterministic_plan_or_patch(user_text, selected, nodes, conns, fallback_refine=True)
        return normalize_patch(out)

    # ---- 画布上下文瘦身（省 token）----
    keep_ids = None
    if selected and selected.get("fromNodeId"):
        keep_ids = _collect_subgraph_ids(selected["fromNodeId"], nodes, conns, depth=2, max_nodes=40)

    compact_nodes = _compact_nodes(nodes, keep_ids=keep_ids, limit=60)
    compact_conns = _compact_conns(conns, keep_ids=keep_ids, limit=80)

    payload = {
        "user_prompt": user_text,
        "selected_artifact": selected,
        "current_nodes": compact_nodes,
        "current_connections": compact_conns,
    }

    def _call():
        resp = client.models.generate_content(
            model=MODEL_AGENT,
            contents=[types.Part(text=_agent_system_prompt()), types.Part(text=json.dumps(payload, ensure_ascii=False))],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1200,
                response_mime_type="application/json",
            ),
        )
        txt = resp.candidates[0].content.parts[0].text
        raw = _safe_json_load(txt)

        # ✅ schema 校验（结构不对直接抛错走兜底）
        parsed = AgentOut.model_validate(raw)
        out = parsed.model_dump()

        # ✅ 自动修补（补字段/强制 model/坐标等）
        out = normalize_patch(out)
        return out

    try:
        out = run_agent_call(_call)
        return out

    except Exception as e:
        msg = str(e)
        sys_logger.error(f"[{req_id}] Agent Plan Error: {msg}")

        # 429/RESOURCE_EXHAUSTED：自动降级
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            out = deterministic_plan_or_patch(user_text, selected, nodes, conns, fallback_refine=True)
            return normalize_patch(out)

        # 其它错误：也降级
        out = deterministic_plan_or_patch(user_text, selected, nodes, conns, fallback_refine=True)
        return normalize_patch(out)

# ==========================================
# Stats/History
# ==========================================

@app.get("/api/stats")
def get_stats():
    return analyzer.get_stats()

@app.get("/api/history")
def get_history():
    return analyzer.get_history()

# ==========================================
# main
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)