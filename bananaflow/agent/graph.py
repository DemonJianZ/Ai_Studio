# bananaflow/agent/graph.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from google.genai import types

from core.config import MODEL_AGENT, MODEL_GEMINI
from core.logging import sys_logger
from core.rate_limit import run_agent_call
from schemas.agent import AgentOut
from services.genai_client import get_client

from agent.system_prompt import agent_system_prompt
from agent.normalizer import safe_json_load, normalize_patch
from agent.context import collect_subgraph_ids, compact_nodes, compact_conns
from agent.deterministic import deterministic_plan_or_patch
from prompts.refine import cached_refine_prompt, simple_refine_prompt
from agent.checkpointer import create_checkpointer

# --- LangGraph optional import ---
try:
    from langgraph.graph import StateGraph, END  # type: ignore

    _LANGGRAPH_OK = True
except Exception:
    _LANGGRAPH_OK = False

_GRAPH_CLOSER = None
_GRAPH = None
_CHECKPOINTER = None


class PlanState(TypedDict, total=False):
    req_id: str
    step: str
    user_prompt: str
    selected_artifact: Optional[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    conns: List[Dict[str, Any]]

    keep_ids: Optional[List[str]]
    compact_nodes: List[Dict[str, Any]]
    compact_conns: List[Dict[str, Any]]

    refined_prompt: str

    raw_text: str
    raw_json: Dict[str, Any]
    parsed_out: Dict[str, Any]

    errors: List[str]
    tried_repair: bool
    used_fallback: bool

    


def _has_large_blob(obj: Any, threshold: int = 3000) -> bool:
    """粗暴拦截：防止模型吐 base64 / 超长字段塞进 patch。"""
    if isinstance(obj, str):
        return len(obj) > threshold or ("base64," in obj and len(obj) > 500)
    if isinstance(obj, dict):
        return any(_has_large_blob(v, threshold) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_large_blob(v, threshold) for v in obj)
    return False


def _validate_business_rules(out: Dict[str, Any]) -> None:
    # 1) 顶层 key 限制
    allowed_top = {"patch", "summary", "thought"}
    extra = set(out.keys()) - allowed_top
    if extra:
        raise ValueError(f"Extra top-level keys not allowed: {sorted(list(extra))}")

    patch = out.get("patch")
    if not isinstance(patch, list):
        raise ValueError("patch must be a list")
    if len(patch) > 80:
        raise ValueError("patch too long (>80)")

    # 2) 禁止大 blob
    if _has_large_blob(out):
        raise ValueError("patch contains large blob/base64-like content")

    # 3) processor 节点约束
    for op in patch:
        if not isinstance(op, dict):
            raise ValueError("each patch op must be dict")

        if op.get("op") == "add_node":
            node = op.get("node") or {}
            ntype = node.get("type")
            if ntype in ("processor", "post_processor", "video_gen"):
                data = node.get("data") or {}

                # processor/post_processor model 强制
                if ntype in ("processor", "post_processor"):
                    if data.get("model") != MODEL_GEMINI:
                        raise ValueError("processor.data.model must be fixed to MODEL_GEMINI")

                if not data.get("mode"):
                    raise ValueError("processor.data.mode required")

                # prompt：对 text2img / multi_image_generate / edit / img2video 都要求存在
                if not data.get("prompt"):
                    raise ValueError("processor.data.prompt required")

                tpl = data.get("templates") or {}
                if "size" not in tpl or "aspect_ratio" not in tpl:
                    raise ValueError("processor.data.templates.size/aspect_ratio required")

                if data.get("mode") == "edit":
                    if not data.get("edit_mode"):
                        raise ValueError("edit requires data.edit_mode")


def _validate_structural_sanity(state: PlanState, out: Dict[str, Any]) -> None:
    """
    ✅ 结构性 sanity check（解决：模型只加一个 text_input 导致画布不完整）
    - 若当前画布为空（没有任何节点），则 patch 至少要能构成“可运行”的最小流：
      text_input/input + processor/video_gen + output（或者直接 processor + output）
    """
    patch = out.get("patch") or []
    if not isinstance(patch, list):
        return

    # 判断是否空画布：用 compact_nodes 更可靠（因为 init_state 里 nodes 很大，后面会清空）
    is_empty_canvas = len(state.get("compact_nodes") or []) == 0

    if not is_empty_canvas:
        return

    added_types: List[str] = []
    for op in patch:
        if isinstance(op, dict) and op.get("op") == "add_node":
            n = (op.get("node") or {})
            t = n.get("type")
            if isinstance(t, str):
                added_types.append(t)

    # 空画布必须至少新增一个 AI 节点（processor/post_processor/video_gen）
    has_ai = any(t in ("processor", "post_processor", "video_gen") for t in added_types)
    has_output = any(t == "output" for t in added_types)

    if not has_ai:
        raise ValueError("Empty canvas patch must add an AI node (processor/post_processor/video_gen)")
    if not has_output:
        raise ValueError("Empty canvas patch must add an output node")


def _node_build_context(state: PlanState) -> PlanState:
    selected = state.get("selected_artifact")
    nodes = state.get("nodes") or []
    conns = state.get("conns") or []

    keep_ids: Optional[List[str]] = None
    if selected and selected.get("fromNodeId"):
        keep_ids = list(
            collect_subgraph_ids(
                selected["fromNodeId"],
                nodes,
                conns,
                depth=2,
                max_nodes=40,
            )
        )

    cn = compact_nodes(nodes, keep_ids=set(keep_ids) if keep_ids else None, limit=60)
    cc = compact_conns(conns, keep_ids=set(keep_ids) if keep_ids else None, limit=80)

    if keep_ids and len(keep_ids) > 60:
        keep_ids = keep_ids[:60]

    # ✅ 关键：清理大对象，避免 checkpoint/sqlite 爆炸
    return {
        "keep_ids": keep_ids,
        "compact_nodes": cn,
        "compact_conns": cc,
        "nodes": [],
        "conns": [],
        "step": "context",
    }


def _node_refine_prompt(state: PlanState) -> PlanState:
    user_prompt = state.get("user_prompt") or ""
    client = get_client()
    refined = cached_refine_prompt(user_prompt) if client else simple_refine_prompt(user_prompt)
    return {"step": "refine","refined_prompt": refined}


def _node_generate_patch(state: PlanState) -> PlanState:
    client = get_client()
    if client is None:
        raise RuntimeError("AI client not initialized")

    payload = {
        "user_prompt": state.get("user_prompt") or "",
        "selected_artifact": state.get("selected_artifact"),
        "current_nodes": state.get("compact_nodes") or [],
        "current_connections": state.get("compact_conns") or [],
        "refined_prompt": state.get("refined_prompt") or "",
    }

    # ❗不要用 response_schema：Gemini API 会报 additionalProperties 不支持
    def _call() -> str:
        resp = client.models.generate_content(
            model=MODEL_AGENT,
            contents=[
                types.Part(text=agent_system_prompt()),
                types.Part(text=json.dumps(payload, ensure_ascii=False)),
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1400,
                response_mime_type="application/json",
            ),
        )
        return resp.candidates[0].content.parts[0].text

    raw_text = run_agent_call(_call)

    # ✅ 截断保存，避免 checkpoint 里 raw_text 越积越大
    return {"step": "gen", "raw_text": (raw_text or "")[:8000]}


def _node_validate(state: PlanState) -> PlanState:
    raw_text = state.get("raw_text") or ""

    try:
        raw = safe_json_load(raw_text)
        parsed = AgentOut.model_validate(raw).model_dump()

        _validate_business_rules(parsed)
        _validate_structural_sanity(state, parsed)

        return {
            "step": "validate",
            "raw_json": raw,
            "parsed_out": parsed,
            "errors": [],  # ✅ 成功就清空
        }
    except Exception as e:
        errors = list(state.get("errors") or [])
        errors.append(str(e))
        return {"step": "validate", "errors": errors}


def _node_repair_json(state: PlanState) -> PlanState:
    """
    给模型一次机会修复：
    - JSON 语法问题 / schema mismatch
    - 结构性缺陷（例如空画布只加了 text_input）
    """
    client = get_client()
    if client is None:
        return {"tried_repair": True}

    raw_text = state.get("raw_text") or ""
    errors = state.get("errors") or []
    err_text = "\n".join(errors[-6:])

    SYSTEM = f"""
You are a strict JSON repair and completion bot for FlowStudio patches.
Return ONLY a corrected JSON object (no markdown, no comments).

Hard rules:
- Output JSON ONLY. Use double quotes for all keys/strings. No trailing commas.
- Top-level keys must be exactly: patch, summary, thought.
- patch must be a list of patch ops.
- Do NOT include base64 or large blobs.
- If current canvas is empty, ensure patch creates a runnable minimal workflow:
  add at least one AI node (processor/post_processor/video_gen) AND an output node.

Reminder of model requirement:
- processor/post_processor data.model MUST be "{MODEL_GEMINI}".
"""

    def _call() -> str:
        resp = client.models.generate_content(
            model=MODEL_AGENT,
            contents=[
                types.Part(text=SYSTEM),
                types.Part(text=f"Errors:\n{err_text}\n\nBad output:\n{raw_text}"),
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1200,
                response_mime_type="application/json",
            ),
        )
        return resp.candidates[0].content.parts[0].text

    repaired = run_agent_call(_call)
    return {"step": "repair", "raw_text": (repaired or "")[:8000], "tried_repair": True}


def _node_fallback(state: PlanState) -> PlanState:
    out = deterministic_plan_or_patch(
        user_prompt=state.get("user_prompt") or "",
        selected_artifact=state.get("selected_artifact"),
        current_nodes=state.get("compact_nodes") or [],
        current_connections=state.get("compact_conns") or [],
        fallback_refine=True,
    )
    return {"step": "fallback", "parsed_out": out, "used_fallback": True}


def _node_normalize(state: PlanState) -> PlanState:
    out = state.get("parsed_out") or {"patch": [], "summary": "", "thought": ""}
    return {"step": "normalize", "parsed_out": normalize_patch(out)}


def _route_after_validate(state: PlanState) -> str:
    if state.get("parsed_out"):
        return "ok"

    tried = bool(state.get("tried_repair"))
    client = get_client()

    if (not tried) and (client is not None):
        return "repair"
    return "fallback"


_GRAPH_CLOSER = None
_GRAPH = None


def build_graph():
    global _GRAPH_CLOSER, _CHECKPOINTER
    if not _LANGGRAPH_OK:
        return None

    g = StateGraph(PlanState)
    g.add_node("context", _node_build_context)
    g.add_node("refine", _node_refine_prompt)
    g.add_node("gen", _node_generate_patch)
    g.add_node("validate", _node_validate)
    g.add_node("repair", _node_repair_json)
    g.add_node("fallback", _node_fallback)
    g.add_node("normalize", _node_normalize)

    g.set_entry_point("context")
    g.add_edge("context", "refine")
    g.add_edge("refine", "gen")
    g.add_edge("gen", "validate")

    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {
            "ok": "normalize",
            "repair": "repair",
            "fallback": "fallback",
        },
    )

    g.add_edge("repair", "validate")
    g.add_edge("fallback", "normalize")
    g.add_edge("normalize", END)

    cp, closer = create_checkpointer()
    _GRAPH_CLOSER = closer
    _CHECKPOINTER = cp
    return g.compile(checkpointer=cp)


_GRAPH = build_graph()


def plan_with_langgraph(
    req_id: str,
    user_prompt: str,
    selected_artifact: Optional[Dict[str, Any]],
    current_nodes: List[Dict[str, Any]],
    current_connections: List[Dict[str, Any]],
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    if _GRAPH is None:
        raise RuntimeError("LangGraph not available. Please install langgraph.")

    init_state: PlanState = {
        "req_id": req_id,
        "user_prompt": (user_prompt or "").strip(),
        "selected_artifact": selected_artifact,
        "nodes": current_nodes or [],
        "conns": current_connections or [],
        "errors": [],
        "tried_repair": False,
        "used_fallback": False,
    }

    # ✅ 多画布：必须是稳定的 thread_id；为空就兜底
    tid = (thread_id or "").strip() or f"t_{req_id}"
    config = {"configurable": {"thread_id": tid}}

    sys_logger.info(f"[{req_id}] using thread_id={tid}")

    final = _GRAPH.invoke(init_state, config=config)
    out = final.get("parsed_out") or {"patch": [], "summary": "", "thought": ""}

    if final.get("errors"):
        sys_logger.info(f"[{req_id}] langgraph json errors: {final.get('errors')[-2:]}")

    return out


def close_graph_checkpointer():
    """
    可选：在 FastAPI shutdown event 调用，安全关闭 sqlite/资源。
    """
    global _GRAPH_CLOSER
    try:
        if _GRAPH_CLOSER is not None:
            _GRAPH_CLOSER.close()
            _GRAPH_CLOSER = None
    except Exception as e:
        sys_logger.warning(f"close_graph_checkpointer failed: {e}")


# ===== Public APIs for Threads / Replay (add to bananaflow/agent/graph.py) =====
from typing import Iterator, Tuple

def _brief_state(values: Dict[str, Any]) -> Dict[str, Any]:
    """避免把 checkpoint 里的大字段原样吐给前端（比如 compact_nodes 很大）"""
    errors = values.get("errors") or []
    rp = (values.get("refined_prompt") or "")
    parsed_out = values.get("parsed_out") or {}
    patch = (parsed_out.get("patch") or []) if isinstance(parsed_out, dict) else []

    cn = values.get("compact_nodes") or []
    cc = values.get("compact_conns") or []

    # 只给一点点“可演示”的摘要
    return {
        "errors_tail": errors[-3:],
        "refined_prompt_tail": rp[:300],
        "patch_len": len(patch) if isinstance(patch, list) else 0,
        "compact_nodes_len": len(cn) if isinstance(cn, list) else 0,
        "compact_conns_len": len(cc) if isinstance(cc, list) else 0,
        "used_fallback": bool(values.get("used_fallback")),
        "tried_repair": bool(values.get("tried_repair")),
    }


def get_thread_history(thread_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    返回 thread 的每一步 state 摘要（用于对外演示“回放/时间旅行”）
    LangGraph 提供 get_state_history / get_state 等接口。 [oai_citation:1‡aidoczh.com](https://www.aidoczh.com/langgraph/concepts/persistence/)
    """
    if _GRAPH is None:
        raise RuntimeError("LangGraph not available")

    tid = (thread_id or "").strip()
    if not tid:
        raise ValueError("thread_id required")

    config = {"configurable": {"thread_id": tid}}

    # get_state_history 通常返回 iterator[StateSnapshot]
    hist = []
    try:
        it = _GRAPH.get_state_history(config)
        for i, snap in enumerate(it):
            if i >= limit:
                break

            # snap 结构在不同版本可能略有差异，尽量写得鲁棒
            snap_cfg = getattr(snap, "config", None) or {}
            snap_meta = getattr(snap, "metadata", None) or {}
            snap_vals = getattr(snap, "values", None) or {}

            cfg_conf = (snap_cfg.get("configurable") or {}) if isinstance(snap_cfg, dict) else {}
            checkpoint_id = cfg_conf.get("checkpoint_id")

            hist.append({
                "checkpoint_id": checkpoint_id,
                "metadata": snap_meta if isinstance(snap_meta, dict) else {},
                "next": list(getattr(snap, "next", []) or []),
                "state_brief": _brief_state(snap_vals if isinstance(snap_vals, dict) else {}),
            })
    except Exception as e:
        # 兜底：至少别把接口炸了
        return [{"error": f"get_state_history failed: {e}"}]

    return hist


def replay_from_checkpoint(
    thread_id: str,
    checkpoint_id: Optional[str] = None,
    updates: Optional[Dict[str, Any]] = None,
    as_node: Optional[str] = None,
) -> Dict[str, Any]:
    """
    - 纯回放：thread_id(+checkpoint_id) + invoke(None)
    - 分叉重跑：先 update_state(config, updates, as_node=...) 再 invoke(None)
    说明见 LangGraph persistence：invoke(None) 回放、update_state 分叉/编辑状态。 [oai_citation:2‡aidoczh.com](https://www.aidoczh.com/langgraph/concepts/persistence/)
    """
    if _GRAPH is None:
        raise RuntimeError("LangGraph not available")

    tid = (thread_id or "").strip()
    if not tid:
        raise ValueError("thread_id required")

    cfg_conf: Dict[str, Any] = {"thread_id": tid}
    if checkpoint_id:
        cfg_conf["checkpoint_id"] = checkpoint_id
    config = {"configurable": cfg_conf}

    # 可选：先 fork/edit 状态（用于“时间旅行后修改 prompt/nodes 重跑”）
    if updates:
        # 关键：指定 as_node 控制下一步从哪继续
        # - 如果你想“从头重新规划”，通常给 as_node="context"
        # - 如果你想“从 validate 后继续”，就给 as_node="validate" 等
        # 不指定则 LangGraph 会尽量推断上一次更新来自哪个节点。 [oai_citation:3‡aidoczh.com](https://www.aidoczh.com/langgraph/concepts/persistence/)
        try:
            _GRAPH.update_state(config, updates, as_node=as_node)
        except TypeError:
            # 兼容某些版本签名差异
            _GRAPH.update_state(config, updates)

    final_state = _GRAPH.invoke(None, config=config)
    out = (final_state or {}).get("parsed_out") or {"patch": [], "summary": "", "thought": ""}

    # 返回“当前最新 checkpoint”信息（方便前端继续 time-travel）
    latest_snap = None
    try:
        latest_snap = _GRAPH.get_state({"configurable": {"thread_id": tid}})
    except Exception:
        pass

    latest_checkpoint_id = None
    if latest_snap is not None:
        snap_cfg = getattr(latest_snap, "config", None) or {}
        if isinstance(snap_cfg, dict):
            latest_checkpoint_id = (snap_cfg.get("configurable") or {}).get("checkpoint_id")

    return {
        "thread_id": tid,
        "from_checkpoint_id": checkpoint_id,
        "latest_checkpoint_id": latest_checkpoint_id,
        "out": out,
    }

def get_graph():
    return _GRAPH

def get_checkpointer():
    return _CHECKPOINTER