# routes.py
import base64
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
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
# Auth helpers
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
JWT_SECRET = os.getenv("JWT_SECRET", "bananaflow_dev_secret")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", str(BASE_DIR / "auth.db"))

ENTERPRISE_DOMAIN = "dayukeji.com"
PUBLIC_EMAIL_DENYLIST = {
    "gmail.com",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "yahoo.com",
    "icloud.com",
    "qq.com",
    "163.com",
    "126.com",
    "proton.me",
    "yeah.net",
}

db_lock = threading.Lock()
db_conn = sqlite3.connect(AUTH_DB_PATH, check_same_thread=False)
db_conn.row_factory = sqlite3.Row


def _dict_row(row: Optional[sqlite3.Row]):
    return dict(row) if row else None


def extract_domain(email: str) -> str:
    if not email or "@" not in email:
        return ""
    return email.split("@", 1)[1].lower()


def init_auth_db():
    with db_lock:
        cur = db_conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                email_domain TEXT,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_quota (
                user_id INTEGER PRIMARY KEY,
                credits_total INTEGER DEFAULT 1000,
                credits_used INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        cur.execute("PRAGMA table_info(users)")
        columns = {row[1] for row in cur.fetchall()}
        migrations = [
            ("email_domain", "ALTER TABLE users ADD COLUMN email_domain TEXT"),
            ("display_name", "ALTER TABLE users ADD COLUMN display_name TEXT"),
            ("last_login_at", "ALTER TABLE users ADD COLUMN last_login_at TEXT"),
        ]
        for col, ddl in migrations:
            if col not in columns:
                cur.execute(ddl)
        cur.execute(
            "SELECT id, email FROM users WHERE email_domain IS NULL OR email_domain = '' OR display_name IS NULL OR display_name = ''"
        )
        for row in cur.fetchall():
            email = row["email"] or ""
            domain = extract_domain(email)
            display_name = email.split("@", 1)[0] or email
            cur.execute(
                "UPDATE users SET email_domain = ?, display_name = ? WHERE id = ?",
                (domain, display_name, row["id"]),
            )
        db_conn.commit()


def hash_password(pwd: str) -> str:
    salted = f"{pwd}:{JWT_SECRET}".encode("utf-8")
    return hashlib.sha256(salted).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    return hash_password(plain) == hashed


def get_user_by_email(email: str):
    with db_lock:
        cur = db_conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
        return _dict_row(cur.fetchone())


def get_user_by_id(uid: int):
    with db_lock:
        cur = db_conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (uid,))
        return _dict_row(cur.fetchone())


def ensure_quota_record(user_id: int):
    with db_lock:
        cur = db_conn.cursor()
        cur.execute("SELECT 1 FROM user_quota WHERE user_id = ?", (user_id,))
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO user_quota (user_id, credits_total, credits_used, updated_at) VALUES (?, 1000, 0, CURRENT_TIMESTAMP)",
                (user_id,),
            )
            db_conn.commit()


def create_user(email: str, password: str):
    domain = extract_domain(email)
    display_name = email.split("@", 1)[0] or email
    with db_lock:
        cur = db_conn.cursor()
        cur.execute(
            "INSERT INTO users (email, email_domain, password_hash, display_name, status, created_at, last_login_at) VALUES (?, ?, ?, ?, 'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (email.lower(), domain, hash_password(password), display_name),
        )
        db_conn.commit()
        user_id = cur.lastrowid
    ensure_quota_record(user_id)
    return get_user_by_id(user_id)


def update_last_login(user_id: int):
    with db_lock:
        cur = db_conn.cursor()
        cur.execute("UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
        db_conn.commit()


def get_quota(user_id: int):
    with db_lock:
        cur = db_conn.cursor()
        cur.execute("SELECT credits_total, credits_used, updated_at FROM user_quota WHERE user_id = ?", (user_id,))
        return _dict_row(cur.fetchone()) or {"credits_total": 0, "credits_used": 0, "updated_at": None}


def serialize_user(u: Dict[str, Any]):
    if not u:
        return None
    return {
        "id": u["id"],
        "email": u["email"],
        "email_domain": u.get("email_domain"),
        "display_name": u.get("display_name") or u["email"],
        "status": u.get("status"),
        "created_at": u.get("created_at"),
        "last_login_at": u.get("last_login_at"),
    }


def build_user_payload(user: Dict[str, Any]):
    data = serialize_user(user)
    data["quota"] = get_quota(user["id"])
    return data


def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def base64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(sub: int, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": sub, "exp": int(expire.timestamp())}
    header = {"alg": JWT_ALG, "typ": "JWT"}
    header_b64 = base64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = base64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{base64url_encode(signature)}"


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")
        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_sig, base64url_decode(signature_b64)):
            raise ValueError("Invalid signature")
        payload = json.loads(base64url_decode(payload_b64))
        if payload.get("exp") and int(payload["exp"]) < int(time.time()):
            raise ValueError("Token expired")
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_user(request: Request):
    auth = request.headers.get("Authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing credentials")
    token = auth.split(" ", 1)[1].strip()
    payload = decode_access_token(token)
    user = get_user_by_id(int(payload.get("sub")))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if user.get("status") != "active":
        raise HTTPException(status_code=403, detail="User is inactive")
    ensure_quota_record(user["id"])
    return user


def validate_enterprise_email(email: str):
    domain = extract_domain(email)
    if domain in PUBLIC_EMAIL_DENYLIST:
        raise HTTPException(status_code=403, detail="请使用企业邮箱注册")
    if domain != ENTERPRISE_DOMAIN:
        raise HTTPException(status_code=403, detail="仅支持企业邮箱注册")


class AuthRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


# =========================================================
# Auth endpoints
# =========================================================


@router.post("/api/auth/register", response_model=AuthResponse)
def register_user(req: AuthRequest):
    email = (req.email or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="请输入合法邮箱")
    if not req.password or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="密码长度至少6位")
    validate_enterprise_email(email)
    if get_user_by_email(email):
        raise HTTPException(status_code=400, detail="用户已存在，请直接登录")
    user = create_user(email, req.password)
    token = create_access_token(user["id"])
    return {"access_token": token, "token_type": "bearer", "user": build_user_payload(user)}


@router.post("/api/auth/login", response_model=AuthResponse)
def login_user(req: AuthRequest):
    email = (req.email or "").strip().lower()
    user = get_user_by_email(email)
    if not user or not verify_password(req.password, user.get("password_hash")):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    if user.get("status") != "active":
        raise HTTPException(status_code=403, detail="账号不可用")
    update_last_login(user["id"])
    return {"access_token": create_access_token(user["id"]), "token_type": "bearer", "user": build_user_payload(user)}


@router.get("/api/auth/me")
def read_current_user(current_user=Depends(get_current_user)):
    update_last_login(current_user["id"])
    return {"user": build_user_payload(current_user)}


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