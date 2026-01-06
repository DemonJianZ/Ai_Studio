import threading
from fastapi import HTTPException

AGENT_SEM = threading.Semaphore(2)  # 可调

def run_agent_call(fn):
    if not AGENT_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Agent 并发已满（后端限流保护），请稍后重试。")
    try:
        return fn()
    finally:
        AGENT_SEM.release()