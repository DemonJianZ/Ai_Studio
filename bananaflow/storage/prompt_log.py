import os, json, re, collections
from datetime import datetime
from typing import Dict
from core.config import LOG_DIR
from core.logging import sys_logger

class PromptLogger:
    def __init__(self, filename="prompts.jsonl"):
        self.filepath = os.path.join(LOG_DIR, filename)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                pass

    def log(self, req_id: str, mode: str, inputs: Dict, final_prompt: str, config: Dict,
            output_meta: Dict, latency: float, error: str = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": req_id,
            "mode": mode,
            "inputs": self._sanitize(inputs),
            "final_prompt": final_prompt,
            "config": config,
            "output": output_meta,
            "latency_sec": round(latency, 3),
            "error": error,
        }
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            sys_logger.error(f"Failed to log: {e}")

    def _sanitize(self, data: Dict) -> Dict:
        if not isinstance(data, dict):
            return {"_": str(data)}
        clean = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 500:
                clean[k] = "<LONG_TEXT_OR_BASE64>"
            elif isinstance(v, list) and v and isinstance(v[0], str) and len(v[0]) > 500:
                clean[k] = ["<LONG_TEXT_OR_BASE64>" for _ in v]
            else:
                clean[k] = v
        return clean

class LogAnalyzer:
    def __init__(self, log_path):
        self.log_path = log_path

    def _read_logs(self, limit=1000):
        if not os.path.exists(self.log_path):
            return []
        lines = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        lines.append(json.loads(line))
                    except:
                        pass
        except Exception:
            return []
        return lines[-limit:]

    def get_history(self, limit=20):
        logs = self._read_logs(300)
        history = []
        for entry in reversed(logs):
            if entry.get("error"):
                continue
            inputs = entry.get("inputs") or {}
            history.append({
                "id": entry.get("request_id"),
                "time": entry.get("timestamp"),
                "mode": entry.get("mode"),
                "prompt": (inputs.get("prompt") if isinstance(inputs, dict) else "") or "",
                "note": "",
            })
            if len(history) >= limit:
                break
        return history

    def get_stats(self):
        logs = self._read_logs(1000)
        if not logs:
            return {"modes": {}, "keywords": []}

        mode_counter = collections.Counter()
        text_corpus = []
        for entry in logs:
            mode_counter[entry.get("mode", "unknown")] += 1
            inputs = entry.get("inputs") or {}
            if isinstance(inputs, dict):
                p = inputs.get("prompt") or ""
                if p:
                    text_corpus.append(p)

        stop_words = set(["a", "an", "the", "in", "on", "of", "with", "and", "to", "is", "for"])
        words = []
        for text in text_corpus:
            tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            for t in tokens:
                if t not in stop_words:
                    words.append(t)
        top = collections.Counter(words).most_common(10)
        return {"modes": dict(mode_counter), "keywords": [k for k, _ in top]}