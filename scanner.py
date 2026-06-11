from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from ai_judge import generate_hyena_comment
from config import TRADES_PATH, WATCHLIST_PATH
from data_fetcher import fetch_price_data
from indicators import add_indicators
from notifier import build_notification_text
from scoring import score_stock


WATCHLIST_COLUMNS = ["code", "name", "theme", "market"]
TABLE_COLUMNS = [
    "code",
    "name",
    "price",
    "score",
    "category",
    "entry_type",
    "wait_condition",
    "entry_zone",
    "stop_loss",
    "target_1",
    "target_2",
    "expected_value_label",
    "confidence",
    "risk_reward",
    "risk_reward_label",
    "monitoring_reason",
    "entry_trigger",
    "invalidation_condition",
    "comment",
]
TRADES_COLUMNS = [
    "date",
    "code",
    "name",
    "score",
    "category",
    "entry_type",
    "entry_low",
    "entry_high",
    "stop_loss",
    "target_1",
    "target_2",
    "close_price",
    "result_1d",
    "result_3d",
    "result_5d",
    "memo",
]


def load_watchlist(path: Path = WATCHLIST_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str}).fillna("")
    for col in WATCHLIST_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["code"] = df["code"].astype(str).str.strip()
    return df[WATCHLIST_COLUMNS]


def scan_watchlist(records: Iterable[Dict[str, Any]], period: str = "9mo") -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    for record in records:
        code = str(record.get("code", "")).strip()
        if not code:
            continue
        fetched = fetch_price_data(code, period=period)
        if fetched.error:
            signals.append(
                {
                    "code": code,
                    "name": record.get("name", ""),
                    "theme": record.get("theme", ""),
                    "market": record.get("market", ""),
                    "price": None,
                    "score": 0,
                    "category": "取得失敗",
                    "entry_type": "-",
                    "wait_condition": "-",
                    "entry_zone": "-",
                    "stop_loss": None,
                    "target_1": None,
                    "target_2": None,
                    "expected_value_label": "-",
                    "confidence": "-",
                    "risk_reward": "-",
                    "risk_reward_label": "-",
                    "comment": fetched.error,
                    "notification_text": f"{code} {record.get('name', '')}: {fetched.error}",
                    "history": fetched.data,
                    "score_breakdown": {
                        "market_score": 0,
                        "trend_score": 0,
                        "pullback_score": 0,
                        "volume_score": 0,
                        "theme_score": 0,
                        "risk_penalty": 0,
                        "total_score": 0,
                    },
                    "positive_reasons": [],
                    "negative_reasons": [fetched.error],
                    "wait_reasons": [],
                    "risk_notes": [fetched.error],
                    "reasons": [],
                    "buy_condition": "-",
                    "entry_trigger": "-",
                    "invalid_conditions": fetched.error,
                    "invalidation_condition": fetched.error,
                    "no_buy_conditions": fetched.error,
                    "monitoring_reason": fetched.error,
                    "last_date": "-",
                }
            )
            continue

        analyzed = add_indicators(fetched.data)
        signal = score_stock(analyzed, record)
        signal["comment"] = generate_hyena_comment(signal)
        signal["notification_text"] = build_notification_text(signal)
        signal["history"] = analyzed.tail(160)
        signals.append(signal)

    return sorted(signals, key=lambda item: item.get("score", -1), reverse=True)


def build_signal_table(signals: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        row = {}
        for col in TABLE_COLUMNS:
            value = signal.get(col, "")
            if col == "risk_reward":
                value = _format_table_risk_reward(value)
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _format_table_risk_reward(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value or "-")


def ensure_trades_file(path: Path = TRADES_PATH) -> None:
    if not path.exists():
        pd.DataFrame(columns=TRADES_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def append_trade_candidate(signal: Dict[str, Any], memo: str = "", path: Path = TRADES_PATH) -> None:
    ensure_trades_file(path)
    row = {
        "date": date.today().isoformat(),
        "code": signal.get("code", ""),
        "name": signal.get("name", ""),
        "score": signal.get("score", ""),
        "category": signal.get("category", ""),
        "entry_type": signal.get("entry_type", ""),
        "entry_low": signal.get("entry_zone_low", ""),
        "entry_high": signal.get("entry_zone_high", ""),
        "stop_loss": signal.get("stop_loss", ""),
        "target_1": signal.get("target_1", ""),
        "target_2": signal.get("target_2", ""),
        "close_price": signal.get("price", ""),
        "result_1d": "",
        "result_3d": "",
        "result_5d": "",
        "memo": memo,
    }
    pd.DataFrame([row], columns=TRADES_COLUMNS).to_csv(
        path,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig",
    )
