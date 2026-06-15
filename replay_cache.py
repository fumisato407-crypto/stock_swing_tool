from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from historical_data import DATA_CACHE_DIR
from time_utils import now_jst_iso


REPLAY_CACHE_DB_PATH = DATA_CACHE_DIR / "replay_cache.sqlite"
TECHNICAL_SCORE_FEATURE_VERSION = "daily_technical_v1"

_CACHE_STATS = {
    "technical_hits": 0,
    "technical_misses": 0,
    "technical_saved_seconds": 0.0,
}


def _connect(path: Path = REPLAY_CACHE_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def stable_config_hash(config: Any) -> str:
    text = json.dumps(config or {}, ensure_ascii=False, sort_keys=True, default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def technical_score_cache_key(
    symbol: str,
    decision_time: Any,
    preset: str,
    config: Dict[str, Any],
    feature_version: str = TECHNICAL_SCORE_FEATURE_VERSION,
) -> Dict[str, str]:
    decision_text = pd.Timestamp(decision_time).strftime("%Y-%m-%d %H:%M:%S")
    config_hash = stable_config_hash(config)
    raw_key = "|".join([str(symbol), decision_text, str(preset), config_hash, feature_version])
    return {
        "key": hashlib.sha256(raw_key.encode("utf-8")).hexdigest(),
        "symbol": str(symbol),
        "decision_time": decision_text,
        "preset": str(preset),
        "config_hash": config_hash,
        "feature_version": feature_version,
    }


def ensure_replay_cache(path: Path = REPLAY_CACHE_DB_PATH) -> None:
    with closing(_connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS technical_score_cache (
                cache_key TEXT PRIMARY KEY,
                symbol TEXT,
                decision_time TEXT,
                preset TEXT,
                config_hash TEXT,
                feature_version TEXT,
                payload_json TEXT NOT NULL,
                elapsed_seconds REAL DEFAULT 0,
                created_at_jst TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_cached_technical_score(key_info: Dict[str, str], path: Path = REPLAY_CACHE_DB_PATH) -> Optional[Dict[str, Any]]:
    ensure_replay_cache(path)
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT payload_json, elapsed_seconds FROM technical_score_cache WHERE cache_key = ?",
            (key_info["key"],),
        ).fetchone()
    if not row:
        _CACHE_STATS["technical_misses"] += 1
        return None
    _CACHE_STATS["technical_hits"] += 1
    _CACHE_STATS["technical_saved_seconds"] += float(row[1] or 0)
    payload = json.loads(row[0])
    payload["_cache_hit"] = True
    payload["_cache_key"] = key_info["key"]
    return payload


def set_cached_technical_score(
    key_info: Dict[str, str],
    payload: Dict[str, Any],
    elapsed_seconds: float = 0.0,
    path: Path = REPLAY_CACHE_DB_PATH,
) -> None:
    ensure_replay_cache(path)
    payload_json = json.dumps(payload or {}, ensure_ascii=False, default=_json_default)
    with closing(_connect(path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO technical_score_cache (
                cache_key, symbol, decision_time, preset, config_hash, feature_version,
                payload_json, elapsed_seconds, created_at_jst
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key_info["key"],
                key_info["symbol"],
                key_info["decision_time"],
                key_info["preset"],
                key_info["config_hash"],
                key_info["feature_version"],
                payload_json,
                float(elapsed_seconds or 0),
                now_jst_iso(),
            ),
        )
        conn.commit()


def reset_replay_cache_stats() -> None:
    for key in list(_CACHE_STATS):
        _CACHE_STATS[key] = 0.0 if key.endswith("seconds") else 0


def replay_cache_stats(path: Path = REPLAY_CACHE_DB_PATH) -> Dict[str, Any]:
    ensure_replay_cache(path)
    with closing(_connect(path)) as conn:
        technical_count = conn.execute("SELECT COUNT(*) FROM technical_score_cache").fetchone()[0]
    hits = int(_CACHE_STATS.get("technical_hits", 0) or 0)
    misses = int(_CACHE_STATS.get("technical_misses", 0) or 0)
    total = hits + misses
    return {
        "technical_score_cache_count": int(technical_count or 0),
        "technical_hits": hits,
        "technical_misses": misses,
        "technical_hit_rate_pct": round(hits / total * 100, 1) if total else 0.0,
        "technical_saved_seconds": round(float(_CACHE_STATS.get("technical_saved_seconds", 0.0) or 0.0), 3),
        "cache_db_path": str(path),
    }


def clear_replay_cache(path: Path = REPLAY_CACHE_DB_PATH) -> Dict[str, Any]:
    ensure_replay_cache(path)
    with closing(_connect(path)) as conn:
        before = conn.execute("SELECT COUNT(*) FROM technical_score_cache").fetchone()[0]
        conn.execute("DELETE FROM technical_score_cache")
        conn.commit()
    reset_replay_cache_stats()
    return {"deleted_technical_score_rows": int(before or 0), "cache_db_path": str(path)}
