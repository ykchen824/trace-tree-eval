"""Thread-safe local JSON disk cache for API responses."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


class DiskCache:
    """Persistent SHA-256 keyed JSON cache to avoid redundant Jev/LLM API spend."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(namespace: str, model: str, payload: Any) -> str:
        raw = json.dumps(
            {"ns": namespace, "model": model, "payload": payload},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        subdir = self.cache_dir / key[:2]
        return subdir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path_for_key(key)
        with self._lock:
            if not path.is_file():
                self.misses += 1
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.hits += 1
                return data
            except (OSError, json.JSONDecodeError):
                self.misses += 1
                return None

    def set(self, key: str, value: Any) -> None:
        path = self._path_for_key(key)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cache_dir": str(self.cache_dir),
                "hits": self.hits,
                "misses": self.misses,
            }
