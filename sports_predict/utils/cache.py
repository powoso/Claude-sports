"""Simple file-based caching for API responses and computed data."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


class DataCache:
    """File-based cache with TTL support for DataFrames and JSON data."""

    def __init__(self, cache_dir: str = ".cache/sports_data", ttl_hours: int = 6):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _key_path(self, key: str, ext: str = ".parquet") -> Path:
        hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:80]
        return self.cache_dir / f"{safe_key}_{hashed}{ext}"

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < self.ttl_seconds

    def get_df(self, key: str) -> pd.DataFrame | None:
        """Retrieve a cached DataFrame, or None if expired/missing."""
        path = self._key_path(key, ".parquet")
        if self._is_fresh(path):
            return pd.read_parquet(path)
        return None

    def set_df(self, key: str, df: pd.DataFrame) -> None:
        """Cache a DataFrame."""
        path = self._key_path(key, ".parquet")
        df.to_parquet(path, index=False)

    def get_json(self, key: str) -> Any | None:
        """Retrieve cached JSON data, or None if expired/missing."""
        path = self._key_path(key, ".json")
        if self._is_fresh(path):
            with open(path) as f:
                return json.load(f)
        return None

    def set_json(self, key: str, data: Any) -> None:
        """Cache JSON-serializable data."""
        path = self._key_path(key, ".json")
        with open(path, "w") as f:
            json.dump(data, f)

    def clear(self) -> int:
        """Remove all cached files. Returns count of files removed."""
        count = 0
        for f in self.cache_dir.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
        return count

    def clear_expired(self) -> int:
        """Remove only expired cache entries. Returns count of files removed."""
        count = 0
        for f in self.cache_dir.iterdir():
            if f.is_file() and not self._is_fresh(f):
                f.unlink()
                count += 1
        return count
