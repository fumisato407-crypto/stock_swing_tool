from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from ai_judge import generate_hyena_comment
from config import TRADES_PATH, WATCHLIST_PATH
from data_fetcher import fetch_price_data, normalize_jp_symbol
from indicators import add_indicators
from notifier import build_notification_text
from scoring import score_stock


WATCHLIST_COLUMNS = ["code", "name", "theme", "market", "raw_code", "normalized_symbol"]
CODE_COLUMN_CANDIDATES = ["code", "ticker", "銘柄コード", "コード"]
OPTIONAL_COLUMN_ALIASES = {
    "name": ["name", "銘柄名", "名称"],
    "theme": ["theme", "テーマ"],
    "market": ["market", "市場"],
}
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
    "normalized_symbol",
    "error_type",
    "error_message",
    "fetched_rows",
    "last_attempt_at",
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
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)
    except Exception as exc:
        empty = pd.DataFrame(columns=WATCHLIST_COLUMNS)
        empty.attrs["load_error"] = f"watchlist.csvを読み込めませんでした: {exc}"
        return empty

    df.columns = [str(col).strip().replace("\ufeff", "") for col in df.columns]
    code_col = next((col for col in CODE_COLUMN_CANDIDATES if col in df.columns), "")
    if not code_col:
        empty = pd.DataFrame(columns=WATCHLIST_COLUMNS)
        empty.attrs["load_error"] = "銘柄コード列が見つかりません。code / ticker / 銘柄コード / コード のいずれかを用意してください。"
        return empty

    normalized = df[code_col].map(normalize_jp_symbol)
    df["raw_code"] = df[code_col].astype(str).str.strip()
    df["normalized_symbol"] = normalized
    df["code"] = normalized.str.replace(r"\.T$", "", regex=True)

    for target_col, aliases in OPTIONAL_COLUMN_ALIASES.items():
        if target_col not in df.columns:
            source_col = next((col for col in aliases if col in df.columns), "")
            df[target_col] = df[source_col] if source_col else ""

    for col in WATCHLIST_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    valid_df = df[df["normalized_symbol"].astype(str).str.len() > 0].copy()
    result = valid_df[WATCHLIST_COLUMNS]
    if result.empty and not df.empty:
        result.attrs["load_error"] = "有効な銘柄コードが見つかりません。コードの空欄、NaN、形式崩れを確認してください。"
    return result


def scan_watchlist(records: Iterable[Dict[str, Any]], period: str = "6mo") -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    for record in records:
        raw_code = record.get("raw_code") or record.get("code", "")
        normalized_symbol = normalize_jp_symbol(record.get("normalized_symbol") or raw_code)
        if not normalized_symbol:
            continue
        code = normalized_symbol.replace(".T", "")
        fetched = fetch_price_data(code, period=period)
        if fetched.error:
            error_message = fetched.error_message or fetched.error or "株価取得に失敗しました。"
            signals.append(
                {
                    "code": code,
                    "name": record.get("name", ""),
                    "theme": record.get("theme", ""),
                    "market": record.get("market", ""),
                    "raw_code": raw_code,
                    "normalized_symbol": fetched.ticker,
                    "error_type": fetched.error_type,
                    "error_message": error_message,
                    "fetched_rows": fetched.fetched_rows,
                    "last_attempt_at": fetched.last_attempt_at,
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
                    "comment": error_message,
                    "notification_text": f"{code} {record.get('name', '')}: {error_message}",
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
                    "negative_reasons": [error_message],
                    "wait_reasons": [],
                    "risk_notes": [error_message],
                    "reasons": [],
                    "buy_condition": "-",
                    "entry_trigger": "-",
                    "invalid_conditions": error_message,
                    "invalidation_condition": error_message,
                    "no_buy_conditions": error_message,
                    "monitoring_reason": error_message,
                    "last_date": "-",
                }
            )
            continue

        analyzed = add_indicators(fetched.data)
        scoring_record = dict(record)
        scoring_record["code"] = code
        scoring_record["normalized_symbol"] = fetched.ticker
        signal = score_stock(analyzed, scoring_record)
        signal["comment"] = generate_hyena_comment(signal)
        signal["notification_text"] = build_notification_text(signal)
        signal["history"] = analyzed.tail(160)
        signal["normalized_symbol"] = fetched.ticker
        signal["error_type"] = ""
        signal["error_message"] = ""
        signal["fetched_rows"] = fetched.fetched_rows
        signal["last_attempt_at"] = fetched.last_attempt_at
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
