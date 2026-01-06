import json, uuid
from typing import Dict, Any
from core.config import MODEL_GEMINI
from core.logging import sys_logger
import re

def new_id(prefix="n") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def safe_json_load(s: str) -> Dict[str, Any]:
    s = (s or "").strip().strip("`").strip()

    # 1) 截取第一个 {...}（避免前后夹杂说明文字）
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]

    # 2) 先严格解析
    try:
        return json.loads(s)
    except Exception:
        pass

    # 3) 轻量修复：去注释、去尾逗号、给裸 key 加双引号、把单引号变双引号（保守）
    s2 = s

    # 去 //... 注释（LLM 偶尔会输出）
    s2 = re.sub(r"//.*?$", "", s2, flags=re.MULTILINE)

    # 去尾逗号：,} 或 ,]
    s2 = re.sub(r",\s*([}\]])", r"\1", s2)

    # 给裸 key 加双引号：{ patch: ... } / , op: ...
    # 仅处理常见标识符 key（不碰已经有引号的 key）
    s2 = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', s2)

    # 把单引号包裹的 key/value 改为双引号（比较常见：'patch':）
    # 注意：这是保守替换，只替换明显的 '...'
    s2 = re.sub(r"\'([^']*)\'", r'"\1"', s2)

    return json.loads(s2)

def normalize_patch(out: Dict[str, Any]) -> Dict[str, Any]:
    patch = out.get("patch") or []
    if not isinstance(patch, list):
        patch = []
    out["patch"] = patch[:80]

    # op 兼容
    for item in out["patch"]:
        if isinstance(item, dict):
            if item.get("op") == "remove_node":
                item["op"] = "delete_node"
            if item.get("op") == "remove_connection":
                item["op"] = "delete_connection"

    seen_node_ids = set()
    seen_conn_ids = set()

    # ✅ 与前端一致的默认模板
    DEFAULT_TPL_PROCESSOR = {"style": "", "vibe": "", "note": "", "size": "1024x1024", "aspect_ratio": "1:1"}
    DEFAULT_TPL_POST = {"style": "", "vibe": "", "direction": "", "note": ""}
    DEFAULT_TPL_VIDEO = {"motion": "", "camera": "", "duration": "5秒", "resolution": "1080p", "ratio": "16:9", "note": ""}

    def _ensure_dict(x):
        return x if isinstance(x, dict) else {}

    for item in out["patch"]:
        if not isinstance(item, dict):
            continue

        op = item.get("op")

        if op == "add_node":
            node = _ensure_dict(item.get("node"))
            item["node"] = node

            node.setdefault("id", new_id("n"))
            if node["id"] in seen_node_ids:
                node["id"] = new_id("n")
            seen_node_ids.add(node["id"])

            node.setdefault("type", "processor")
            node["x"] = int(node.get("x", 200))
            node["y"] = int(node.get("y", 200))

            data = _ensure_dict(node.get("data"))
            node["data"] = data

            ntype = node.get("type")

            # ---- common ----
            data.setdefault("status", "idle")
            data.setdefault("batchSize", 1)

            # ✅ model：只在缺失时补默认（不要覆盖前端选择）
            if ntype in ("processor", "post_processor") and "model" not in data:
                data["model"] = MODEL_GEMINI

            # ---- per type ----
            if ntype == "text_input":
                data.setdefault("text", "")
                data.pop("templates", None)  # text_input 不需要 templates
            elif ntype == "input":
                data.setdefault("images", [])
                data.pop("templates", None)
            elif ntype == "output":
                data.setdefault("images", [])
                data.pop("templates", None)
            elif ntype == "video_gen":
                data.setdefault("mode", "img2video")
                tpl = _ensure_dict(data.get("templates"))
                # 不覆盖已有 key，只补缺失
                for k, v in DEFAULT_TPL_VIDEO.items():
                    tpl.setdefault(k, v)
                data["templates"] = tpl
            elif ntype == "post_processor":
                data.setdefault("mode", "relight")
                tpl = _ensure_dict(data.get("templates"))
                for k, v in DEFAULT_TPL_POST.items():
                    tpl.setdefault(k, v)
                data["templates"] = tpl
            else:  # processor
                data.setdefault("mode", "bg_replace")
                tpl = _ensure_dict(data.get("templates"))

                # 先补通用字段（不覆盖）
                for k, v in DEFAULT_TPL_PROCESSOR.items():
                    tpl.setdefault(k, v)

                # ✅ 只有 text2img / multi_image_generate 强依赖 size/aspect_ratio
                if data.get("mode") not in ("text2img", "multi_image_generate"):
                    # 其他模式允许仍保留 size/aspect_ratio（不删也行），但不要强制覆盖
                    pass

                data["templates"] = tpl

        elif op == "add_connection":
            c = _ensure_dict(item.get("connection"))
            item["connection"] = c

            c.setdefault("id", new_id("c"))
            if c["id"] in seen_conn_ids:
                c["id"] = new_id("c")
            seen_conn_ids.add(c["id"])

    out["summary"] = str(out.get("summary") or "")
    out["thought"] = str(out.get("thought") or "")
    return out