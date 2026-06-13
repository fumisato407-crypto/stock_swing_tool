from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd

from virtual_trade_store import load_virtual_trades


def reliability_label(sample_count: int) -> str:
    if sample_count < 30:
        return "仮説"
    if sample_count < 50:
        return "参考"
    if sample_count < 100:
        return "信頼度 中"
    return "信頼度 高"


def _market_bucket(snapshot_text: Any) -> str:
    try:
        snapshot = json.loads(snapshot_text) if isinstance(snapshot_text, str) else snapshot_text
    except json.JSONDecodeError:
        return "不明"
    score = 0
    if isinstance(snapshot, dict):
        score = int(float(snapshot.get("score", 0) or 0))
    if score >= 70:
        return "強い"
    if score >= 60:
        return "中立"
    return "弱い"


def _stats_for_group(group: pd.DataFrame) -> Dict[str, Any]:
    returns = pd.to_numeric(group["return_pct"], errors="coerce").dropna()
    sample_count = int(len(returns))
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    avg_profit = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    win_rate = float((returns > 0).mean() * 100) if sample_count else 0.0
    expected_value = float(returns.mean()) if sample_count else 0.0
    return {
        "sample_count": sample_count,
        "win_rate_pct": round(win_rate, 2),
        "avg_profit_pct": round(avg_profit, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "expected_value_pct": round(expected_value, 2),
        "reliability": reliability_label(sample_count),
    }


def calculate_pattern_stats() -> Dict[str, pd.DataFrame]:
    df = load_virtual_trades(limit=5000)
    if df.empty or "return_pct" not in df.columns:
        empty = pd.DataFrame()
        return {"by_entry_type": empty, "by_symbol": empty, "by_market": empty}

    evaluated = df[pd.to_numeric(df["return_pct"], errors="coerce").notna()].copy()
    if evaluated.empty:
        empty = pd.DataFrame()
        return {"by_entry_type": empty, "by_symbol": empty, "by_market": empty}

    evaluated["market_context"] = evaluated["market_snapshot_json"].map(_market_bucket)
    return {
        "by_entry_type": _group_stats(evaluated, "entry_type"),
        "by_symbol": _group_stats(evaluated, "symbol"),
        "by_market": _group_stats(evaluated, "market_context"),
    }


def _group_stats(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(column, dropna=False):
        row = {column: value or "-"}
        row.update(_stats_for_group(group))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["sample_count", "expected_value_pct"], ascending=[False, False])
