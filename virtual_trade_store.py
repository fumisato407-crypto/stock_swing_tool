from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from config import VIRTUAL_TRADES_DB_PATH


VIRTUAL_TRADE_COLUMNS = [
    "id",
    "timestamp",
    "symbol",
    "name",
    "decision",
    "entry_type",
    "confidence",
    "entry_price",
    "stop_loss",
    "take_profit",
    "max_hold_days",
    "reasons",
    "risk_factors",
    "source_score",
    "market_snapshot_json",
    "status",
    "return_pct",
    "max_profit_pct",
    "max_drawdown_pct",
    "hit_stop_loss",
    "hit_take_profit",
    "outcome",
    "outcome_json",
    "outcome_updated_at",
]


def _connect(path: Path = VIRTUAL_TRADES_DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(path)


def ensure_virtual_trade_store(path: Path = VIRTUAL_TRADES_DB_PATH) -> None:
    with closing(_connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS virtual_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT,
                decision TEXT NOT NULL,
                entry_type TEXT,
                confidence TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                max_hold_days INTEGER,
                reasons TEXT,
                risk_factors TEXT,
                source_score REAL,
                market_snapshot_json TEXT,
                status TEXT,
                return_pct REAL,
                max_profit_pct REAL,
                max_drawdown_pct REAL,
                hit_stop_loss INTEGER DEFAULT 0,
                hit_take_profit INTEGER DEFAULT 0,
                outcome TEXT,
                outcome_json TEXT,
                outcome_updated_at TEXT
            )
            """
        )
        existing = {row[1] for row in conn.execute("PRAGMA table_info(virtual_trades)").fetchall()}
        migrations = {
            "return_pct": "REAL",
            "max_profit_pct": "REAL",
            "max_drawdown_pct": "REAL",
            "hit_stop_loss": "INTEGER DEFAULT 0",
            "hit_take_profit": "INTEGER DEFAULT 0",
            "outcome": "TEXT",
            "outcome_json": "TEXT",
            "outcome_updated_at": "TEXT",
        }
        for column, column_type in migrations.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE virtual_trades ADD COLUMN {column} {column_type}")
        conn.commit()


def _json_text(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False, default=str)


def insert_virtual_trade(record: Dict[str, Any], path: Path = VIRTUAL_TRADES_DB_PATH) -> int:
    ensure_virtual_trade_store(path)
    payload = {
        "timestamp": record.get("timestamp") or datetime.now().isoformat(timespec="seconds"),
        "symbol": str(record.get("symbol", "")),
        "name": str(record.get("name", "")),
        "decision": str(record.get("decision", "")),
        "entry_type": str(record.get("entry_type", "")),
        "confidence": str(record.get("confidence", "")),
        "entry_price": record.get("entry_price"),
        "stop_loss": record.get("stop_loss"),
        "take_profit": record.get("take_profit"),
        "max_hold_days": record.get("max_hold_days"),
        "reasons": _json_text(record.get("reasons")),
        "risk_factors": _json_text(record.get("risk_factors")),
        "source_score": record.get("source_score"),
        "market_snapshot_json": _json_text(record.get("market_snapshot_json")),
        "status": str(record.get("status", "logged")),
    }
    with closing(_connect(path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO virtual_trades (
                timestamp, symbol, name, decision, entry_type, confidence,
                entry_price, stop_loss, take_profit, max_hold_days,
                reasons, risk_factors, source_score, market_snapshot_json, status
            ) VALUES (
                :timestamp, :symbol, :name, :decision, :entry_type, :confidence,
                :entry_price, :stop_loss, :take_profit, :max_hold_days,
                :reasons, :risk_factors, :source_score, :market_snapshot_json, :status
            )
            """,
            payload,
        )
        conn.commit()
        return int(cur.lastrowid)


def load_virtual_trades(
    status: Optional[str] = None,
    limit: int = 500,
    path: Path = VIRTUAL_TRADES_DB_PATH,
) -> pd.DataFrame:
    ensure_virtual_trade_store(path)
    query = "SELECT * FROM virtual_trades"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
    params.append(int(limit))
    with closing(_connect(path)) as conn:
        return pd.read_sql_query(query, conn, params=params)


def find_recent_virtual_trade(
    symbol: str,
    entry_type: str,
    minutes: int = 30,
    path: Path = VIRTUAL_TRADES_DB_PATH,
) -> Optional[Dict[str, Any]]:
    ensure_virtual_trade_store(path)
    cutoff = datetime.now().timestamp() - minutes * 60
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            """
            SELECT * FROM virtual_trades
            WHERE symbol = ? AND entry_type = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT 20
            """,
            (symbol, entry_type),
        ).fetchall()
        columns = [col[1] for col in conn.execute("PRAGMA table_info(virtual_trades)").fetchall()]

    for row in rows:
        record = dict(zip(columns, row))
        try:
            ts = datetime.fromisoformat(str(record.get("timestamp"))).timestamp()
        except ValueError:
            continue
        if ts >= cutoff:
            return record
    return None


def load_open_virtual_trades(path: Path = VIRTUAL_TRADES_DB_PATH) -> List[Dict[str, Any]]:
    df = load_virtual_trades(status="open", limit=1000, path=path)
    return df.to_dict("records")


def update_virtual_trade_outcome(
    trade_id: int,
    updates: Dict[str, Any],
    path: Path = VIRTUAL_TRADES_DB_PATH,
) -> None:
    ensure_virtual_trade_store(path)
    allowed = {
        "status",
        "return_pct",
        "max_profit_pct",
        "max_drawdown_pct",
        "hit_stop_loss",
        "hit_take_profit",
        "outcome",
        "outcome_json",
        "outcome_updated_at",
    }
    payload = {key: value for key, value in updates.items() if key in allowed}
    if not payload:
        return
    assignments = ", ".join(f"{key} = :{key}" for key in payload)
    payload["id"] = trade_id
    with closing(_connect(path)) as conn:
        conn.execute(f"UPDATE virtual_trades SET {assignments} WHERE id = :id", payload)
        conn.commit()


def rows_to_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    display = df.copy()
    for col in ["reasons", "risk_factors"]:
        if col in display.columns:
            display[col] = display[col].map(_compact_json_list)
    return display.fillna("")


def _compact_json_list(value: Any) -> str:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return str(value)
    if isinstance(parsed, list):
        return "、".join(str(item) for item in parsed[:3])
    return str(parsed)
