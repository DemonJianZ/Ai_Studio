import json
from fastapi import Request
from google.genai import types

from core.logging import sys_logger
from core.config import MODEL_AGENT
from core.rate_limit import run_agent_call
from services.genai_client import get_client
from schemas.agent import AgentOut
from agent.system_prompt import agent_system_prompt
from agent.normalizer import safe_json_load, normalize_patch
from agent.context import collect_subgraph_ids, compact_nodes, compact_conns
from agent.deterministic import deterministic_plan_or_patch

def agent_plan_legacy_impl(req, request: Request) -> dict:
    req_id = request.state.req_id
    user_text = (req.prompt or "").strip()
    selected = req.selected_artifact.model_dump() if req.selected_artifact else None
    nodes = req.current_nodes or []
    conns = req.current_connections or []

    client = get_client()
    if client is None:
        out = deterministic_plan_or_patch(user_text, selected, nodes, conns, fallback_refine=True)
        return normalize_patch(out)

    keep_ids = None
    if selected and selected.get("fromNodeId"):
        keep_ids = collect_subgraph_ids(selected["fromNodeId"], nodes, conns, depth=2, max_nodes=40)

    payload = {
        "user_prompt": user_text,
        "selected_artifact": selected,
        "current_nodes": compact_nodes(nodes, keep_ids=keep_ids, limit=60),
        "current_connections": compact_conns(conns, keep_ids=keep_ids, limit=80),
    }

    def _call():
        resp = client.models.generate_content(
            model=MODEL_AGENT,
            contents=[types.Part(text=agent_system_prompt()), types.Part(text=json.dumps(payload, ensure_ascii=False))],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1200,
                response_mime_type="application/json",
            ),
        )
        txt = resp.candidates[0].content.parts[0].text
        raw = safe_json_load(txt)
        parsed = AgentOut.model_validate(raw)
        out = parsed.model_dump()
        return normalize_patch(out)

    try:
        return run_agent_call(_call)
    except Exception as e:
        msg = str(e)
        sys_logger.error(f"[{req_id}] Agent Plan Error: {msg}")
        out = deterministic_plan_or_patch(user_text, selected, nodes, conns, fallback_refine=True)
        return normalize_patch(out)