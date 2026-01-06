# bananaflow/agent/deterministic.py
import collections
from typing import List, Dict, Any, Optional

from core.config import MODEL_GEMINI
from agent.normalizer import new_id
from utils.size import parse_aspect_ratio
from prompts.refine import simple_refine_prompt


def _normalize_conns(conns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in conns or []:
        f = c.get("from") or c.get("source")
        t = c.get("to") or c.get("target")
        if not f or not t:
            continue
        out.append({"id": c.get("id") or new_id("c"), "from": f, "to": t})
    return out


def _find_node(nodes: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
    return next((n for n in (nodes or []) if n.get("id") == node_id), None)


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

    # 如果选中的是 output 节点，则从其上游找真正的 anchor
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

    proc_id = new_id("proc")
    out_id = new_id("out")
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
        {"op": "add_connection", "connection": {"id": new_id("c"), "from": anchor_id, "to": proc_id}},
        {"op": "add_connection", "connection": {"id": new_id("c"), "from": proc_id, "to": out_id}},
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
    in_id = new_id("in")
    proc_id = new_id("proc")
    out_id = new_id("out")
    tpl = templates or {"size": "1024x1024", "aspect_ratio": "1:1"}

    patch = [
        {
            "op": "add_node",
            "node": {"id": in_id, "type": "input", "x": x0, "y": y0, "data": {"images": [selected_artifact["url"]]}},
        },
        {
            "op": "add_node",
            "node": {
                "id": proc_id,
                "type": "processor",
                "x": x0 + 350,
                "y": y0,
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
        {"op": "add_node", "node": {"id": out_id, "type": "output", "x": x0 + 700, "y": y0, "data": {"images": []}}},
        {"op": "add_connection", "connection": {"id": new_id("c"), "from": in_id, "to": proc_id}},
        {"op": "add_connection", "connection": {"id": new_id("c"), "from": proc_id, "to": out_id}},
        {"op": "select_nodes", "ids": [proc_id]},
    ]
    return {"patch": patch, "summary": "fromNodeId 缺失，已新建 input→图生图→输出 分支", "thought": "fallback new-input branch"}


def build_from_scratch_patch(
    user_prompt: str,
    model: str = MODEL_GEMINI,
    x0: int = 120,
    y0: int = 120,
) -> Dict[str, Any]:
    ar = parse_aspect_ratio(user_prompt)
    tpl = {"size": "1024x1024", "aspect_ratio": ar}

    text_id = new_id("text")
    gen_id = new_id("gen")
    out1_id = new_id("out")
    edit_id = new_id("edit")
    out2_id = new_id("out")

    initial_prompt = (user_prompt or "").strip() or "Generate a clean commercial product photo, high quality, studio lighting."
    default_edit_prompt = "Refine the image: improve composition and details. Keep style consistent."

    patch = [
        {"op": "add_node", "node": {"id": text_id, "type": "text_input", "x": x0, "y": y0, "data": {"text": initial_prompt}}},
        {
            "op": "add_node",
            "node": {
                "id": gen_id,
                "type": "processor",
                "x": x0 + 320,
                "y": y0,
                "data": {"mode": "text2img", "prompt": initial_prompt, "templates": tpl, "batchSize": 1, "status": "idle", "model": model},
            },
        },
        {"op": "add_node", "node": {"id": out1_id, "type": "output", "x": x0 + 640, "y": y0, "data": {"images": []}}},
        {
            "op": "add_node",
            "node": {
                "id": edit_id,
                "type": "processor",
                "x": x0 + 320,
                "y": y0 + 220,
                "data": {
                    "mode": "multi_image_generate",
                    "prompt": default_edit_prompt,
                    "templates": tpl,
                    "batchSize": 1,
                    "status": "idle",
                    "model": model,
                },
            },
        },
        {"op": "add_node", "node": {"id": out2_id, "type": "output", "x": x0 + 640, "y": y0 + 220, "data": {"images": []}}},
        {"op": "add_connection", "connection": {"id": new_id("c"), "from": text_id, "to": gen_id}},
        {"op": "add_connection", "connection": {"id": new_id("c"), "from": gen_id, "to": out1_id}},
        {"op": "add_connection", "connection": {"id": new_id("c"), "from": gen_id, "to": edit_id}},
        {"op": "add_connection", "connection": {"id": new_id("c"), "from": edit_id, "to": out2_id}},
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
    """
    无模型/模型失败时的确定性规划：
    - 有 selected_artifact：优先续链（根据 fromNodeId 找 anchor）；失败则新建 input 分支
    - 无 selected_artifact：从零搭建（text_input -> text2img -> output + edit链）
    """
    if selected_artifact and selected_artifact.get("url"):
        refined = simple_refine_prompt(user_prompt) if fallback_refine else (user_prompt or "").strip()
        try:
            return build_continue_chain_patch(refined, current_nodes, current_connections, selected_artifact, model=MODEL_GEMINI)
        except Exception:
            return build_iterate_branch_with_new_input_patch(refined, selected_artifact, model=MODEL_GEMINI)

    return build_from_scratch_patch(user_prompt, model=MODEL_GEMINI)