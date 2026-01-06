import os
import base64
import time
import json
import hmac
import hashlib
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

BASE_DIR = os.getcwd()
JWT_SECRET = os.getenv("JWT_SECRET", "bananaflow_dev_secret")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", os.path.join(BASE_DIR, "auth.db"))

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
        exists = cur.fetchone()
        if not exists:
            cur.execute(
                "INSERT INTO user_quota (user_id, credits_total, credits_used, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (user_id, 1000, 0),
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


auth_router = APIRouter()


@auth_router.post("/api/auth/register", response_model=AuthResponse)
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


@auth_router.post("/api/auth/login", response_model=AuthResponse)
def login_user(req: AuthRequest):
    email = (req.email or "").strip().lower()
    user = get_user_by_email(email)
    if not user or not verify_password(req.password, user.get("password_hash")):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    if user.get("status") != "active":
        raise HTTPException(status_code=403, detail="账号不可用")
    update_last_login(user["id"])
    return {"access_token": create_access_token(user["id"]), "token_type": "bearer", "user": build_user_payload(user)}


@auth_router.get("/api/auth/me")
def read_current_user(current_user=Depends(get_current_user)):
    update_last_login(current_user["id"])
    return {"user": build_user_payload(current_user)}


__all__ = [
    "auth_router",
    "init_auth_db",
    "get_current_user",
    "AuthRequest",
    "AuthResponse",
]
