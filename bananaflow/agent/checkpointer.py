import os
import contextlib
from typing import Tuple, Any

from core.logging import sys_logger  # 按你项目实际路径


def create_checkpointer() -> Tuple[Any, Any]:
    """
    返回 (checkpointer, closer)
    - checkpointer: 可传给 workflow.compile(checkpointer=...)
    - closer: 用于 shutdown 时释放资源（ExitStack）
    """
    backend = os.getenv("LANGGRAPH_CHECKPOINTER", "sqlite").lower()

    if backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        sys_logger.warning("LangGraph checkpointer = InMemorySaver (non-durable)")
        return InMemorySaver(), None

    db_path = os.getenv("LANGGRAPH_SQLITE_PATH", "./logs/langgraph_checkpoints.sqlite3")

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        stack = contextlib.ExitStack()
        # ⚠️ 你这版返回的是 context manager，需要 enter 才得到真正 saver
        saver = stack.enter_context(SqliteSaver.from_conn_string(db_path))

        sys_logger.info(f"LangGraph checkpointer = SqliteSaver({db_path})")
        return saver, stack

    except Exception as e:
        sys_logger.warning(
            f"SqliteSaver not available ({e}). Fallback to InMemorySaver. "
            f"Try: pip install langgraph-checkpoint-sqlite"
        )
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver(), None