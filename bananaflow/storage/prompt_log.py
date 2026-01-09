import os, json, re, collections
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from core.config import LOG_DIR
from core.logging import sys_logger

class PromptLogger:
    def __init__(self, filename="prompts.jsonl"):
        self.filepath = os.path.join(LOG_DIR, filename)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                pass

    def log(
        self,
        req_id: str,
        mode: str,
        inputs: Dict,
        final_prompt: str,
        config: Dict,
        output_meta: Dict,
        latency: float,
        error: str = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
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
            "success": error is None,
            "user_id": user_id or "unknown",
            "tenant_id": tenant_id or "unknown",
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
                    except Exception:
                        pass
        except Exception:
            return []
        return lines[-limit:]

    def _parse_timestamp(self, value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _percentile(self, values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = max(0, min(len(sorted_vals) - 1, int(round(pct * (len(sorted_vals) - 1))))))
        return float(sorted_vals[idx])

    def _summarize_prompt(self, entry: Dict[str, Any]) -> str:
        inputs = entry.get("inputs") or {}
        prompt = ""
        if isinstance(inputs, dict):
            prompt = inputs.get("prompt") or ""
        if not prompt:
            prompt = entry.get("final_prompt") or ""
        prompt = str(prompt).replace("\n", " ").strip()
        return (prompt[:80] + "…") if len(prompt) > 80 else prompt

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

    def get_recent_requests(self, limit=30) -> List[Dict[str, Any]]:
        logs = self._read_logs(2000)
        items = []
        for entry in reversed(logs):
            config = entry.get("config") or {}
            items.append({
                "time": entry.get("timestamp"),
                "mode": entry.get("mode"),
                "model": config.get("model") or entry.get("model") or "",
                "latency_sec": entry.get("latency_sec") or 0,
                "error": entry.get("error"),
                "success": bool(entry.get("success", entry.get("error") is None)),
                "user_id": entry.get("user_id") or "unknown",
                "tenant_id": entry.get("tenant_id") or "unknown",
                "prompt_summary": self._summarize_prompt(entry),
            })
            if len(items) >= limit:
                break
        return items

    def get_user_stats(self, limit=50) -> List[Dict[str, Any]]:
        logs = self._read_logs(5000)
        buckets = collections.defaultdict(lambda: {
            "total": 0,
            "success": 0,
            "latencies": [],
            "modes": collections.Counter(),
            "tenant_id": "unknown",
        })

        for entry in logs:
            user_id = entry.get("user_id") or "unknown"
            tenant_id = entry.get("tenant_id") or "unknown"
            key = user_id
            record = buckets[key]
            record["tenant_id"] = tenant_id
            record["total"] += 1
            if entry.get("success", entry.get("error") is None):
                record["success"] += 1
            latency = entry.get("latency_sec")
            if isinstance(latency, (int, float)):
                record["latencies"].append(float(latency))
            record["modes"][entry.get("mode") or "unknown"] += 1

        output = []
        for user_id, record in buckets.items():
            latencies = record["latencies"]
            total = record["total"] or 1
            distribution = {
                "le_1s": len([l for l in latencies if l <= 1]),
                "le_3s": len([l for l in latencies if 1 < l <= 3]),
                "le_10s": len([l for l in latencies if 3 < l <= 10]),
                "gt_10s": len([l for l in latencies if l > 10]),
            }
            output.append({
                "user_id": user_id,
                "tenant_id": record["tenant_id"],
                "total": record["total"],
                "success_rate": round(record["success"] / total, 3),
                "p50_latency": round(self._percentile(latencies, 0.5), 3),
                "p95_latency": round(self._percentile(latencies, 0.95), 3),
                "latency_distribution": distribution,
                "modes": dict(record["modes"]),
            })
        output.sort(key=lambda x: x["total"], reverse=True)
        return output[:limit]

    def get_summary(self, days=14) -> Dict[str, Any]:
        logs = self._read_logs(8000)
        if not logs:
            return {
                "total": 0,
                "success_rate": 0,
                "error_rate": 0,
                "p95_latency": 0,
                "top_users": [],
                "top_modes": [],
                "trend": [],
            }

        total = 0
        success = 0
        latencies = []
        mode_counter = collections.Counter()
        user_counter = collections.Counter()
        trend_buckets = collections.defaultdict(list)
        cutoff = datetime.now() - timedelta(days=days)

        for entry in logs:
            total += 1
            is_success = entry.get("success", entry.get("error") is None)
            if is_success:
                success += 1
            latency = entry.get("latency_sec")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
            mode_counter[entry.get("mode") or "unknown"] += 1
            user_key = entry.get("user_id") or "unknown"
            user_counter[user_key] += 1

            ts = self._parse_timestamp(entry.get("timestamp") or "")
            if ts and ts >= cutoff:
                bucket = ts.date().isoformat()
                trend_buckets[bucket].append(entry)

        top_users = []
        for user_id, count in user_counter.most_common(5):
            top_users.append({"user_id": user_id, "total": count})

        top_modes = [{"mode": mode, "total": count} for mode, count in mode_counter.most_common(5)]

        trend = []
        for date_key in sorted(trend_buckets.keys()):
            bucket_entries = trend_buckets[date_key]
            bucket_total = len(bucket_entries)
            bucket_success = len([e for e in bucket_entries if e.get("success", e.get("error") is None)])
            bucket_latencies = [e.get("latency_sec") for e in bucket_entries if isinstance(e.get("latency_sec"), (int, float))]
            trend.append({
                "date": date_key,
                "total": bucket_total,
                "error_rate": round(1 - (bucket_success / bucket_total if bucket_total else 0), 3),
                "p95_latency": round(self._percentile([float(l) for l in bucket_latencies], 0.95), 3),
            })

        return {
            "total": total,
            "success_rate": round(success / total if total else 0, 3),
            "error_rate": round(1 - (success / total if total else 0), 3),
            "p95_latency": round(self._percentile(latencies, 0.95), 3),
            "top_users": top_users,
            "top_modes": top_modes,
            "trend": trend,
        }

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
