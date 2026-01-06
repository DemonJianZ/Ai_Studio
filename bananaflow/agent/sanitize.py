# agent/sanitize.py
from typing import Any

def make_msgpack_safe(x: Any):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): make_msgpack_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [make_msgpack_safe(v) for v in x]
    # callable / object / pydantic / datetime / bytes ... -> string
    return str(x)