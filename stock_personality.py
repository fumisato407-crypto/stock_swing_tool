from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd

from pattern_stats import reliability_label
from virtual_trade_store import load_virtual_trades


def _compact_patterns(df: pd.DataFrame, ascending: bool = False) -> List[str]:
    if df.empty:
        return []
    grouped = (
        df.groupby("entry_type")
        .agg(
            sample_count=("return_pct", "count"),
            expected_value_pct=("return_pct", "mean"),
            max_drawdown_pct=("max_drawdown_pct", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values("expected_value_pct", ascending=ascending)
    labels = []
    for row in grouped.head(3).to_dict("records"):
        labels.append(
            f"{row['entry_type']}（期待値 {row['expected_value_pct']:.2f}% / 平均逆行 {row['max_drawdown_pct']:.2f}%）"
        )
    return labels


def _best_hold_hint(group: pd.DataFrame) -> str:
    candidates: Dict[str, List[float]] = {"1h": [], "close": [], "1d": [], "3d": [], "5d": []}
    for value in group.get("outcome_json", pd.Series(dtype=str)).dropna():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        checkpoints = parsed.get("checkpoints", {}) if isinstance(parsed, dict) else {}
        for key in candidates:
            if checkpoints.get(key) is not None:
                candidates[key].append(float(checkpoints[key]))
    averages = {key: sum(values) / len(values) for key, values in candidates.items() if values}
    if not averages:
        return "検証サンプル不足"
    best = max(averages, key=averages.get)
    return f"{best}目安（平均 {averages[best]:.2f}%）"


def generate_stock_personalities() -> pd.DataFrame:
    df = load_virtual_trades(limit=5000)
    if df.empty or "return_pct" not in df.columns:
        return pd.DataFrame()

    evaluated = df[pd.to_numeric(df["return_pct"], errors="coerce").notna()].copy()
    if evaluated.empty:
        return pd.DataFrame()
    evaluated["return_pct"] = pd.to_numeric(evaluated["return_pct"], errors="coerce")
    evaluated["max_drawdown_pct"] = pd.to_numeric(evaluated["max_drawdown_pct"], errors="coerce").fillna(0)

    rows = []
    for symbol, group in evaluated.groupby("symbol"):
        sample_count = int(len(group))
        win_rate = float((group["return_pct"] > 0).mean() * 100)
        expected_value = float(group["return_pct"].mean())
        avg_drawdown = float(group["max_drawdown_pct"].mean())
        good_patterns = _compact_patterns(group, ascending=False)
        weak_patterns = _compact_patterns(group, ascending=True)
        caution = "逆行が大きめ。エントリー位置と損切り幅を厳しめに確認。"
        if expected_value > 0 and avg_drawdown > -2:
            caution = "期待値はプラス寄り。過熱時の飛び乗りだけ注意。"
        elif expected_value <= 0:
            caution = "現時点の期待値は弱い。条件を絞るか仮説扱い。"
        rows.append(
            {
                "symbol": symbol,
                "name": group["name"].dropna().iloc[0] if not group["name"].dropna().empty else "",
                "sample_count": sample_count,
                "reliability": reliability_label(sample_count),
                "win_rate_pct": round(win_rate, 2),
                "expected_value_pct": round(expected_value, 2),
                "avg_drawdown_pct": round(avg_drawdown, 2),
                "good_patterns": "、".join(good_patterns) if good_patterns else "-",
                "weak_patterns": "、".join(weak_patterns) if weak_patterns else "-",
                "best_hold_period": _best_hold_hint(group),
                "caution": caution,
            }
        )
    return pd.DataFrame(rows).sort_values(["sample_count", "expected_value_pct"], ascending=[False, False])
