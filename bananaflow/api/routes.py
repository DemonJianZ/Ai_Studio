# routes.py
import time
import json
import uuid
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from google.genai import types

from core.config import MODEL_GEMINI, MODEL_DOUBAO
from core.logging import sys_logger

from schemas.api import (
    Text2ImgRequest, Text2ImgResponse,
    MultiImageRequest, MultiImageResponse,
    EditRequest, EditResponse,
    Img2VideoRequest, Img2VideoResponse,
    AgentRequest,
)

from storage.prompt_log import PromptLogger, LogAnalyzer
from services.genai_client import call_genai_retry
from services.ark import call_doubao_image_gen
from services.ark_video import generate_video_from_image, VideoGenError

from utils.images import parse_data_url, bytes_to_data_url, get_image_from_response
from prompts.business import build_business_prompt

# agent
from agent.planner import agent_plan_impl
ALLOWED_VIDEO_MODELS = {"Doubao-Seedance-1.0-pro", "Doubao-Seedance-1.5-pro"}


router = APIRouter()
prompt_logger = PromptLogger()
analyzer = LogAnalyzer("logs/prompts.jsonl")


# =========================================================
# Core image/video endpoints
# =========================================================

@router.post("/api/text2img", response_model=Text2ImgResponse)
def text_to_image(req: Text2ImgRequest, request: Request):
    req_id = request.state.req_id
    selected_model = req.model or MODEL_GEMINI
    t0 = time.time()

    try:
        img_bytes = None

        # ---- Doubao (Ark) ----
        if selected_model == MODEL_DOUBAO:
            img_bytes = call_doubao_image_gen(
                req.prompt,
                req_id,
                size_param=req.size or "1024x1024",
                aspect_ratio=req.aspect_ratio or "1:1",
            )
        # ---- Gemini ----
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
                    image_size=gemini_resolution,
                ),
            )

            response = call_genai_retry(
                contents=[types.Part(text=req.prompt)],
                config=gen_config,
                req_id=req_id,
            )
            img_bytes = get_image_from_response(response)

        if not img_bytes:
            raise RuntimeError("No image returned")

        prompt_logger.log(
            req_id,
            "text2img",
            req.model_dump(),
            req.prompt,
            {"model": selected_model, "temp": req.temperature, "size": req.size, "ar": req.aspect_ratio},
            {"file": "mem"},
            time.time() - t0,
        )

        return Text2ImgResponse(images=[bytes_to_data_url(img_bytes)])

    except Exception as e:
        sys_logger.error(f"[{req_id}] Text2Img Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/multi_image_generate", response_model=MultiImageResponse)
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
                image_size=gemini_resolution,
            ),
        )

        response = call_genai_retry(contents=contents, config=gen_config, req_id=req_id)
        img_bytes = get_image_from_response(response)
        if not img_bytes:
            raise RuntimeError("No image returned")

        prompt_logger.log(
            req_id,
            "multi_image_generate",
            req.model_dump(),
            req.prompt,
            {"temperature": req.temperature, "ar": req.aspect_ratio},
            {"file": "mem"},
            time.time() - t0,
        )
        return MultiImageResponse(image=bytes_to_data_url(img_bytes))

    except Exception as e:
        sys_logger.error(f"[{req_id}] MultiImage Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/edit", response_model=EditResponse)
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
                final_prompt,
                req_id,
                size_param=req.size or "1024x1024",
                aspect_ratio=req.aspect_ratio or "1:1",
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
                req_id=req_id,
            )
            img_bytes = get_image_from_response(response)

        if not img_bytes:
            raise RuntimeError("No image returned")

        prompt_logger.log(
            req_id,
            req.mode,
            req.model_dump(),
            final_prompt,
            {"model": selected_model, "has_ref": has_ref},
            {"file": "mem"},
            time.time() - t0,
        )
        return EditResponse(image=bytes_to_data_url(img_bytes))

    except Exception as e:
        sys_logger.error(f"[{req_id}] Edit Error ({req.mode}): {e}")
        prompt_logger.log(req_id, req.mode, req.model_dump(), "ERROR", {}, {}, time.time() - t0, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/img2video", response_model=Img2VideoResponse)
def img_to_video(req: Img2VideoRequest, request: Request):
    req_id = request.state.req_id
    try:
        selected_model = req.model or "Doubao-Seedance-1.0-pro"
        print("--------选择的模型-------：", selected_model)
        if selected_model not in ALLOWED_VIDEO_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported video model: {selected_model}")
        
        print("--------提示词-------：", req.prompt)
        result = generate_video_from_image(
            req_id=req_id,
            model=req.model,   # ✅ 新增
            image_data_url=req.image,
            last_frame_data_url=req.last_frame_image,
            prompt=req.prompt or "",
            duration=req.duration,
            camera_fixed=req.camera_fixed,
            resolution=req.resolution,
            ratio=req.ratio,
            seed = req.seed,
        )
        return Img2VideoResponse(image=result)

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except VideoGenError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        sys_logger.error(f"[{req_id}] VideoGen Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# Agent endpoints
# =========================================================

@router.post("/api/agent/plan", response_model=Dict[str, Any])
def agent_plan(req: AgentRequest, request: Request, response: Response) -> Dict[str, Any]:
    """
    返回结构必须是：
    {
      "patch": [...],
      "summary": "...",
      "thought": "..."
    }
    """
    req_id = getattr(request.state, "req_id", "noid")

    # 多画布：优先 canvas_id，其次 thread_id，最后兜底
    canvas_id = (getattr(req, "canvas_id", None) or "").strip()
    thread_id = (getattr(req, "thread_id", None) or "").strip()
    if canvas_id:
        thread_id = canvas_id
    elif not thread_id:
        # 没画布概念时兜底（不建议长期用 req_id，因每次请求都不稳定）
        thread_id = f"t_{req_id}"

    # 回传给前端，方便复用
    response.headers["X-Thread-Id"] = thread_id

    try:
        # 透传 thread_id 给 planner / langgraph
        req.thread_id = thread_id

        out = agent_plan_impl(req, request)
        if out is None:
            raise RuntimeError("agent_plan_impl returned None")
        return out

    except HTTPException:
        raise
    except Exception as e:
        sys_logger.error(f"[{req_id}] /api/agent/plan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Threads / Time-travel list
# -----------------------------

def _ensure_graph_ready():
    try:
        from agent.graph import get_graph, get_checkpointer  # 你必须在 graph.py 里提供这俩
        g = get_graph()
        cp = get_checkpointer()
        return g, cp
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"agent.graph.get_graph/get_checkpointer not available: {e}",
        )


def _extract_checkpoint_id(item: Any) -> Optional[str]:
    """
    兼容：CheckpointTuple / dict / tuple
    """
    # CheckpointTuple-like
    if hasattr(item, "checkpoint"):
        ck = getattr(item, "checkpoint") or {}
        if isinstance(ck, dict):
            return str(ck.get("id") or ck.get("checkpoint_id") or ck.get("uuid") or "") or None

    # dict-like
    if isinstance(item, dict):
        if item.get("checkpoint_id") or item.get("id"):
            return str(item.get("checkpoint_id") or item.get("id"))
        ck = item.get("checkpoint") or {}
        if isinstance(ck, dict):
            return str(ck.get("id") or ck.get("checkpoint_id") or "") or None

    # tuple/list-like
    if isinstance(item, (list, tuple)):
        for x in item:
            if isinstance(x, dict) and (x.get("id") or x.get("checkpoint_id")):
                return str(x.get("id") or x.get("checkpoint_id"))
            if isinstance(x, dict) and ("checkpoint" in x):
                ck = x.get("checkpoint") or {}
                if isinstance(ck, dict) and (ck.get("id") or ck.get("checkpoint_id")):
                    return str(ck.get("id") or ck.get("checkpoint_id"))
    return None


def _extract_metadata(item: Any) -> Dict[str, Any]:
    if hasattr(item, "metadata"):
        md = getattr(item, "metadata") or {}
        return md if isinstance(md, dict) else {}
    if isinstance(item, dict):
        md = item.get("metadata") or {}
        return md if isinstance(md, dict) else {}
    if isinstance(item, (list, tuple)):
        for x in item:
            if isinstance(x, dict) and "metadata" in x and isinstance(x["metadata"], dict):
                return x["metadata"]
    return {}


def _extract_values_dict_from_item(item: Any) -> Dict[str, Any]:
    """
    LangGraph saver 常见把状态放在：
      checkpoint["channel_values"] 或 checkpoint["values"] 或 checkpoint["state"]
    """
    ck = None
    if hasattr(item, "checkpoint"):
        ck = getattr(item, "checkpoint")
    elif isinstance(item, dict):
        ck = item.get("checkpoint") or item
    elif isinstance(item, (list, tuple)):
        # tuple 常见：(config, checkpoint, metadata) 或 (checkpoint, metadata)
        for x in item:
            if isinstance(x, dict) and ("channel_values" in x or "values" in x or "state" in x):
                ck = x
                break

    if not isinstance(ck, dict):
        return {}

    for k in ("channel_values", "values", "state"):
        v = ck.get(k)
        if isinstance(v, dict):
            return v

    # 有的 saver 直接把 values 平铺在 checkpoint dict
    return {k: v for k, v in ck.items() if k not in ("id", "checkpoint_id", "metadata")}


def _extract_state_keys(item: Any) -> List[str]:
    values = _extract_values_dict_from_item(item)
    return sorted(list(values.keys())) if isinstance(values, dict) else []


def _extract_errors_tail(item: Any) -> List[str]:
    values = _extract_values_dict_from_item(item)
    if not isinstance(values, dict):
        return []
    errs = values.get("errors") or []
    if isinstance(errs, list):
        return [str(x) for x in errs[-2:]]
    return []

def _extract_step(item: Any) -> str:
    ck = None
    if hasattr(item, "checkpoint"):
        ck = getattr(item, "checkpoint")
    elif isinstance(item, dict):
        ck = item.get("checkpoint") or item
    if not isinstance(ck, dict):
        return ""

    values = ck.get("channel_values") or ck.get("values") or ck.get("state") or {}
    if isinstance(values, dict):
        s = values.get("step")
        if s:
            return str(s)

    # fallback：老数据没有 step 时，用 state_keys 推断
    keys = _extract_state_keys(item)

    # conditional 分支痕迹（你现在 state_keys 里已经有 branch:to:xxx）
    for k in keys:
        if isinstance(k, str) and k.startswith("branch:to:"):
            return k.split("branch:to:", 1)[1]  # normalize/validate/gen...

    # 更粗的推断
    if "raw_text" in keys and "raw_json" not in keys and "parsed_out" not in keys:
        return "gen"
    if "refined_prompt" in keys and "raw_text" not in keys:
        return "refine"
    if "compact_nodes" in keys and "refined_prompt" not in keys:
        return "context"
    if "parsed_out" in keys:
        return "normalize"

    return ""

@router.get("/api/agent/threads/{thread_id}", response_model=Dict[str, Any])
def list_agent_thread(thread_id: str) -> Dict[str, Any]:
    """
    返回该 thread 的 checkpoint 列表（用于“回放/时间旅行”的时间线）。
    """
    _, cp = _ensure_graph_ready()
    if cp is None:
        raise HTTPException(status_code=400, detail="Checkpointer not initialized")

    cfg = {"configurable": {"thread_id": thread_id}}

    if not hasattr(cp, "list"):
        raise HTTPException(status_code=400, detail="Checkpointer does not support list()")

    # 兼容不同 saver.list 的签名
    items = []
    try:
        items = list(cp.list(cfg, limit=50))
    except TypeError:
        items = list(cp.list(cfg))

    out = {"thread_id": thread_id, "checkpoints": []}

    for it in items:
        out["checkpoints"].append({
            "thread_id": thread_id,
            "checkpoint_id": _extract_checkpoint_id(it),
            "metadata": _extract_metadata(it),
            "step": _extract_step(it), 
            "state_keys": _extract_state_keys(it),
            "errors_tail": _extract_errors_tail(it),
        })

    return out


# -----------------------------
# Replay from checkpoint
# -----------------------------

class AgentReplayRequest(BaseModel):
    checkpoint_id: Optional[str] = None
    prompt: Optional[str] = None
    current_nodes: Optional[List[Dict[str, Any]]] = None
    current_connections: Optional[List[Dict[str, Any]]] = None
    selected_artifact: Optional[Dict[str, Any]] = None


def _get_checkpoint_state(cp: Any, thread_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
    cfg = {"configurable": {"thread_id": thread_id}}

    ck = None
    if hasattr(cp, "get"):
        ck = cp.get(cfg, checkpoint_id)
    elif hasattr(cp, "get_tuple"):
        ck = cp.get_tuple(cfg, checkpoint_id)
    else:
        return None

    # dict-like
    if isinstance(ck, dict):
        checkpoint = ck.get("checkpoint") or ck
        if isinstance(checkpoint, dict):
            v = checkpoint.get("channel_values") or checkpoint.get("values") or checkpoint.get("state")
            if isinstance(v, dict):
                return v
            # fallback：有些 saver 直接把 values 平铺
            return checkpoint

    # tuple-like
    if isinstance(ck, (tuple, list)):
        for x in ck:
            if isinstance(x, dict):
                v = x.get("channel_values") or x.get("values") or x.get("state")
                if isinstance(v, dict):
                    return v
        # 兜底
        for x in ck:
            if isinstance(x, dict):
                return x

    return None


@router.post("/api/agent/replay/{thread_id}", response_model=Dict[str, Any])
def replay_agent_thread(thread_id: str, req: AgentReplayRequest, request: Request) -> Dict[str, Any]:
    """
    从某个 checkpoint 继续跑（可覆盖 prompt/nodes/conns/selected_artifact）。
    - 若当前 LangGraph 版本不支持 resume，会降级为普通 invoke（仍可演示“回放/重跑”）
    """
    g, cp = _ensure_graph_ready()
    if g is None or cp is None:
        raise HTTPException(status_code=400, detail="Graph/checkpointer not initialized")

    # 1) 基于 checkpoint state 初始化（如提供）
    init_state: Dict[str, Any]
    if req.checkpoint_id:
        base = _get_checkpoint_state(cp, thread_id, req.checkpoint_id)
        if isinstance(base, dict):
            init_state = dict(base)
        else:
            init_state = {}
    else:
        init_state = {}

    # 2) 如果 checkpoint 读不到，就从“最小 state”起
    if not init_state:
        init_state = {
            "req_id": getattr(request.state, "req_id", "noid"),
            "user_prompt": "",
            "selected_artifact": None,
            "nodes": [],
            "conns": [],
            "errors": [],
            "tried_repair": False,
            "used_fallback": False,
        }

    # 3) 覆盖字段（人类介入后重跑）
    if req.prompt is not None:
        init_state["user_prompt"] = (req.prompt or "").strip()
    if req.selected_artifact is not None:
        init_state["selected_artifact"] = req.selected_artifact
    if req.current_nodes is not None:
        init_state["nodes"] = req.current_nodes
    if req.current_connections is not None:
        init_state["conns"] = req.current_connections

    # 4) invoke：尝试 resume（不支持则降级）
    resume_used = False
    cfg = {"configurable": {"thread_id": thread_id}}
    if req.checkpoint_id:
        cfg["configurable"]["checkpoint_id"] = req.checkpoint_id

    try:
        final = g.invoke(init_state, config=cfg)
        resume_used = bool(req.checkpoint_id)
    except Exception as e:
        sys_logger.warning(f"[replay] resume not supported or failed, fallback invoke: {e}")
        final = g.invoke(init_state, config={"configurable": {"thread_id": thread_id}})
        resume_used = False

    out = final.get("parsed_out") if isinstance(final, dict) else None
    if not isinstance(out, dict):
        out = {"patch": [], "summary": "", "thought": ""}

    errors_tail = []
    if isinstance(final, dict):
        errs = final.get("errors")
        if isinstance(errs, list):
            errors_tail = [str(x) for x in errs[-2:]]

    return {
        "thread_id": thread_id,
        "checkpoint_id": req.checkpoint_id,
        "resume_used": resume_used,
        "result": out,
        "errors_tail": errors_tail,
    }


# =========================================================
# Stats / history endpoints
# =========================================================

@router.get("/api/stats")
def get_stats():
    return analyzer.get_stats()


@router.get("/api/history")
def get_history():
    return analyzer.get_history()