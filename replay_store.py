from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import REPLAY_TRADES_DB_PATH
from time_utils import now_jst_iso


REPLAY_COLUMNS = [
    "id",
    "replay_run_id",
    "replay_mode",
    "created_at_jst",
    "symbol",
    "name",
    "replay_start_at",
    "replay_end_at",
    "scan_time",
    "selected_rank",
    "scan_score",
    "category",
    "source_watchlist",
    "daily_ok_count",
    "daily_total_count",
    "intraday_ok_count",
    "intraday_total_count",
    "daily_filter_json",
    "intraday_entry_json",
    "multi_timeframe_pass",
    "risk_pass",
    "stop_loss_pct",
    "max_loss_yen",
    "expected_profit_yen",
    "risk_reward_ratio",
    "risk_filter_json",
    "signal_time",
    "entry_price",
    "stop_loss",
    "take_profit",
    "score",
    "entry_type",
    "rule_name",
    "status",
    "outcome",
    "exit_price",
    "shares",
    "required_capital_yen",
    "profit_yen",
    "cumulative_profit_yen",
    "return_pct",
    "max_profit_pct",
    "max_drawdown_pct",
    "hit_stop_loss",
    "hit_take_profit",
    "evaluated_until",
    "exit_time",
    "holding_period",
    "exit_reason",
    "notes_json",
]


def _connect(path: Path = REPLAY_TRADES_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def ensure_replay_store(path: Path = REPLAY_TRADES_DB_PATH) -> None:
    with closing(_connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS replay_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                replay_run_id TEXT NOT NULL,
                replay_mode TEXT,
                created_at_jst TEXT NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT,
                replay_start_at TEXT,
                replay_end_at TEXT,
                scan_time TEXT,
                selected_rank INTEGER,
                scan_score REAL,
                category TEXT,
                source_watchlist TEXT,
                daily_ok_count INTEGER,
                daily_total_count INTEGER,
                intraday_ok_count INTEGER,
                intraday_total_count INTEGER,
                daily_filter_json TEXT,
                intraday_entry_json TEXT,
                multi_timeframe_pass INTEGER DEFAULT 0,
                risk_pass INTEGER DEFAULT 0,
                stop_loss_pct REAL,
                max_loss_yen REAL,
                expected_profit_yen REAL,
                risk_reward_ratio REAL,
                risk_filter_json TEXT,
                signal_time TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                score REAL,
                entry_type TEXT,
                rule_name TEXT,
                status TEXT,
                outcome TEXT,
                exit_price REAL,
                shares INTEGER,
                required_capital_yen REAL,
                profit_yen REAL,
                cumulative_profit_yen REAL,
                return_pct REAL,
                max_profit_pct REAL,
                max_drawdown_pct REAL,
                hit_stop_loss INTEGER DEFAULT 0,
                hit_take_profit INTEGER DEFAULT 0,
                evaluated_until TEXT,
                exit_time TEXT,
                holding_period TEXT,
                exit_reason TEXT,
                notes_json TEXT
            )
            """
        )
        existing = {row[1] for row in conn.execute("PRAGMA table_info(replay_trades)").fetchall()}
        migrations = {
            "holding_period": "TEXT",
            "exit_reason": "TEXT",
            "notes_json": "TEXT",
            "exit_price": "REAL",
            "shares": "INTEGER",
            "required_capital_yen": "REAL",
            "profit_yen": "REAL",
            "cumulative_profit_yen": "REAL",
            "replay_mode": "TEXT",
            "scan_time": "TEXT",
            "selected_rank": "INTEGER",
            "scan_score": "REAL",
            "category": "TEXT",
            "source_watchlist": "TEXT",
            "daily_ok_count": "INTEGER",
            "daily_total_count": "INTEGER",
            "intraday_ok_count": "INTEGER",
            "intraday_total_count": "INTEGER",
            "daily_filter_json": "TEXT",
            "intraday_entry_json": "TEXT",
            "multi_timeframe_pass": "INTEGER DEFAULT 0",
            "risk_pass": "INTEGER DEFAULT 0",
            "stop_loss_pct": "REAL",
            "max_loss_yen": "REAL",
            "expected_profit_yen": "REAL",
            "risk_reward_ratio": "REAL",
            "risk_filter_json": "TEXT",
            "exit_time": "TEXT",
        }
        for column, column_type in migrations.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE replay_trades ADD COLUMN {column} {column_type}")
        conn.commit()


def _json_text(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


def new_replay_run_id() -> str:
    return f"replay-{now_jst_iso()}-{uuid.uuid4().hex[:8]}"


def insert_replay_trade(record: Dict[str, Any], path: Path = REPLAY_TRADES_DB_PATH) -> int:
    ensure_replay_store(path)
    payload = {
        "replay_run_id": str(record.get("replay_run_id", "")),
        "replay_mode": str(record.get("replay_mode", "single_symbol")),
        "created_at_jst": str(record.get("created_at_jst") or now_jst_iso()),
        "symbol": str(record.get("symbol", "")),
        "name": str(record.get("name", "")),
        "replay_start_at": str(record.get("replay_start_at", "")),
        "replay_end_at": str(record.get("replay_end_at", "")),
        "scan_time": str(record.get("scan_time", "")),
        "selected_rank": record.get("selected_rank"),
        "scan_score": record.get("scan_score"),
        "category": str(record.get("category", "")),
        "source_watchlist": str(record.get("source_watchlist", "")),
        "daily_ok_count": record.get("daily_ok_count"),
        "daily_total_count": record.get("daily_total_count"),
        "intraday_ok_count": record.get("intraday_ok_count"),
        "intraday_total_count": record.get("intraday_total_count"),
        "daily_filter_json": _json_text(record.get("daily_filter_json", {})),
        "intraday_entry_json": _json_text(record.get("intraday_entry_json", {})),
        "multi_timeframe_pass": 1 if record.get("multi_timeframe_pass") else 0,
        "risk_pass": 1 if record.get("risk_pass") else 0,
        "stop_loss_pct": record.get("stop_loss_pct"),
        "max_loss_yen": record.get("max_loss_yen"),
        "expected_profit_yen": record.get("expected_profit_yen"),
        "risk_reward_ratio": record.get("risk_reward_ratio"),
        "risk_filter_json": _json_text(record.get("risk_filter_json", {})),
        "signal_time": str(record.get("signal_time", "")),
        "entry_price": record.get("entry_price"),
        "stop_loss": record.get("stop_loss"),
        "take_profit": record.get("take_profit"),
        "score": record.get("score"),
        "entry_type": str(record.get("entry_type", "")),
        "rule_name": str(record.get("rule_name", "")),
        "status": str(record.get("status", "")),
        "outcome": str(record.get("outcome", "")),
        "exit_price": record.get("exit_price"),
        "shares": record.get("shares"),
        "required_capital_yen": record.get("required_capital_yen"),
        "profit_yen": record.get("profit_yen"),
        "cumulative_profit_yen": record.get("cumulative_profit_yen"),
        "return_pct": record.get("return_pct"),
        "max_profit_pct": record.get("max_profit_pct"),
        "max_drawdown_pct": record.get("max_drawdown_pct"),
        "hit_stop_loss": 1 if record.get("hit_stop_loss") else 0,
        "hit_take_profit": 1 if record.get("hit_take_profit") else 0,
        "evaluated_until": str(record.get("evaluated_until", "")),
        "exit_time": str(record.get("exit_time") or record.get("evaluated_until", "")),
        "holding_period": str(record.get("holding_period", "")),
        "exit_reason": str(record.get("exit_reason", "")),
        "notes_json": _json_text(record.get("notes_json", record.get("signal", {}))),
    }
    with closing(_connect(path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO replay_trades (
                replay_run_id, replay_mode, created_at_jst, symbol, name, replay_start_at, replay_end_at,
                scan_time, selected_rank, scan_score, category, source_watchlist,
                daily_ok_count, daily_total_count, intraday_ok_count, intraday_total_count,
                daily_filter_json, intraday_entry_json, multi_timeframe_pass,
                risk_pass, stop_loss_pct, max_loss_yen, expected_profit_yen, risk_reward_ratio, risk_filter_json,
                signal_time, entry_price, stop_loss, take_profit, score, entry_type, rule_name,
                status, outcome, exit_price, shares, required_capital_yen, profit_yen, cumulative_profit_yen,
                return_pct, max_profit_pct, max_drawdown_pct,
                hit_stop_loss, hit_take_profit, evaluated_until, exit_time, holding_period, exit_reason, notes_json
            ) VALUES (
                :replay_run_id, :replay_mode, :created_at_jst, :symbol, :name, :replay_start_at, :replay_end_at,
                :scan_time, :selected_rank, :scan_score, :category, :source_watchlist,
                :daily_ok_count, :daily_total_count, :intraday_ok_count, :intraday_total_count,
                :daily_filter_json, :intraday_entry_json, :multi_timeframe_pass,
                :risk_pass, :stop_loss_pct, :max_loss_yen, :expected_profit_yen, :risk_reward_ratio, :risk_filter_json,
                :signal_time, :entry_price, :stop_loss, :take_profit, :score, :entry_type, :rule_name,
                :status, :outcome, :exit_price, :shares, :required_capital_yen, :profit_yen, :cumulative_profit_yen,
                :return_pct, :max_profit_pct, :max_drawdown_pct,
                :hit_stop_loss, :hit_take_profit, :evaluated_until, :exit_time, :holding_period, :exit_reason, :notes_json
            )
            """,
            payload,
        )
        conn.commit()
        return int(cur.lastrowid)


def insert_replay_trades(
    trades: List[Dict[str, Any]],
    replay_run_id: Optional[str] = None,
    path: Path = REPLAY_TRADES_DB_PATH,
) -> Dict[str, Any]:
    run_id = replay_run_id or new_replay_run_id()
    inserted_ids: List[int] = []
    for trade in trades:
        record = dict(trade)
        record["replay_run_id"] = run_id
        record["created_at_jst"] = record.get("created_at_jst") or now_jst_iso()
        inserted_ids.append(insert_replay_trade(record, path=path))
    return {
        "replay_run_id": run_id,
        "saved_count": len(inserted_ids),
        "inserted_ids": inserted_ids,
        "db_path": str(path),
    }


def load_replay_trades(limit: int = 500, path: Path = REPLAY_TRADES_DB_PATH) -> pd.DataFrame:
    ensure_replay_store(path)
    query = "SELECT * FROM replay_trades ORDER BY created_at_jst DESC, id DESC LIMIT ?"
    with closing(_connect(path)) as conn:
        return pd.read_sql_query(query, conn, params=[int(limit)])


def ensure_replay_runs_store(path: Path = REPLAY_TRADES_DB_PATH) -> None:
    ensure_replay_store(path)
    with closing(_connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS replay_runs (
                replay_run_id TEXT PRIMARY KEY,
                created_at_jst TEXT NOT NULL,
                replay_mode TEXT,
                symbols_count INTEGER,
                period TEXT,
                interval TEXT,
                scan_interval TEXT,
                start_at TEXT,
                end_at TEXT,
                min_score REAL,
                target_mode TEXT,
                max_buys_per_scan INTEGER,
                cooldown TEXT,
                shares INTEGER,
                total_scan_steps INTEGER,
                total_candidates INTEGER,
                total_trades INTEGER,
                total_profit_yen REAL,
                win_rate REAL,
                profit_factor REAL,
                error_count INTEGER,
                summary_json TEXT
            )
            """
        )
        existing = {row[1] for row in conn.execute("PRAGMA table_info(replay_runs)").fetchall()}
        migrations = {
            "summary_json": "TEXT",
        }
        for column, column_type in migrations.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE replay_runs ADD COLUMN {column} {column_type}")
        conn.commit()


def insert_replay_run(record: Dict[str, Any], path: Path = REPLAY_TRADES_DB_PATH) -> None:
    ensure_replay_runs_store(path)
    payload = {
        "replay_run_id": str(record.get("replay_run_id", "")),
        "created_at_jst": str(record.get("created_at_jst") or now_jst_iso()),
        "replay_mode": str(record.get("replay_mode", "")),
        "symbols_count": record.get("symbols_count"),
        "period": str(record.get("period", "")),
        "interval": str(record.get("interval", "")),
        "scan_interval": str(record.get("scan_interval", "")),
        "start_at": str(record.get("start_at", "")),
        "end_at": str(record.get("end_at", "")),
        "min_score": record.get("min_score"),
        "target_mode": str(record.get("target_mode", "")),
        "max_buys_per_scan": record.get("max_buys_per_scan"),
        "cooldown": str(record.get("cooldown", "")),
        "shares": record.get("shares"),
        "total_scan_steps": record.get("total_scan_steps"),
        "total_candidates": record.get("total_candidates"),
        "total_trades": record.get("total_trades"),
        "total_profit_yen": record.get("total_profit_yen"),
        "win_rate": record.get("win_rate"),
        "profit_factor": record.get("profit_factor"),
        "error_count": record.get("error_count"),
        "summary_json": _json_text(record.get("summary_json", record)),
    }
    with closing(_connect(path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO replay_runs (
                replay_run_id, created_at_jst, replay_mode, symbols_count, period, interval,
                scan_interval, start_at, end_at, min_score, target_mode, max_buys_per_scan,
                cooldown, shares, total_scan_steps, total_candidates, total_trades,
                total_profit_yen, win_rate, profit_factor, error_count, summary_json
            ) VALUES (
                :replay_run_id, :created_at_jst, :replay_mode, :symbols_count, :period, :interval,
                :scan_interval, :start_at, :end_at, :min_score, :target_mode, :max_buys_per_scan,
                :cooldown, :shares, :total_scan_steps, :total_candidates, :total_trades,
                :total_profit_yen, :win_rate, :profit_factor, :error_count, :summary_json
            )
            """,
            payload,
        )
        conn.commit()
