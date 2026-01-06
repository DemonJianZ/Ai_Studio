# bananaflow/agent/context.py
import collections
from typing import List, Dict, Any, Optional, Set

from agent.normalizer import new_id


def _find_node(nodes: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
    return next((n for n in (nodes or []) if n.get("id") == node_id), None)


def _normalize_conns(conns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in conns or []:
        f = c.get("from") or c.get("source")
        t = c.get("to") or c.get("target")
        if not f or not t:
            continue
        out.append({"id": c.get("id") or new_id("c"), "from": f, "to": t})
    return out


def collect_subgraph_ids(
    center_id: str,
    nodes: List[Dict[str, Any]],
    conns: List[Dict[str, Any]],
    depth: int = 2,
    max_nodes: int = 40,
) -> Set[str]:
    """
    以 center_id 为中心，向上下游 BFS 扩展 depth 层，返回需要保留的节点 id 集合（最多 max_nodes）。
    """
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
    for _ in range(max(0, depth)):
        nxt = set()
        for nid in frontier:
            for v in adj.get(nid, []):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
            for v in radj.get(nid, []):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        if len(seen) >= max_nodes:
            break

    existing = {n.get("id") for n in nodes or []}
    return {x for x in seen if x in existing}


def compact_nodes(
    nodes: List[Dict[str, Any]],
    keep_ids: Optional[Set[str]] = None,
    limit: int = 60,
) -> List[Dict[str, Any]]:
    """
    压缩节点字段，减少 token。保留 label/mode/prompt/text/templates 等关键字段。
    """
    out = []
    for n in nodes or []:
        nid = n.get("id")
        if keep_ids and nid not in keep_ids:
            continue

        d = n.get("data") or {}
        out.append(
            {
                "id": nid,
                "type": n.get("type"),
                "x": int(n.get("x", 0)),
                "y": int(n.get("y", 0)),
                "data": {
                    "label": d.get("label"),
                    "mode": d.get("mode"),
                    "prompt": (d.get("prompt") or "")[:400],
                    "text": (d.get("text") or "")[:200],
                    "templates": d.get("templates"),
                },
            }
        )
        if len(out) >= limit:
            break
    return out


def compact_conns(
    conns: List[Dict[str, Any]],
    keep_ids: Optional[Set[str]] = None,
    limit: int = 80,
) -> List[Dict[str, Any]]:
    """
    压缩连线字段，统一成 {id, from, to}。
    keep_ids：只保留涉及该子图的连线。
    """
    out = []
    for c in _normalize_conns(conns):
        if keep_ids and (c["from"] not in keep_ids and c["to"] not in keep_ids):
            continue
        out.append(c)
        if len(out) >= limit:
            break
    return out