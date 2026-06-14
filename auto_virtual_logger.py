from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from config import VIRTUAL_TRADES_DB_PATH
from market_hours import market_status_label, now_jst
from paper_trader import process_virtual_trade_signals


def _score(signal: Dict[str, Any]) -> int:
    try:
        return int(float(signal.get("score", signal.get("intraday_score", 0)) or 0))
    except (TypeError, ValueError):
        return 0


def select_rule_buy_candidates(
    signals: List[Dict[str, Any]],
    target_mode: str,
    min_score: int,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    """Select rule-based paper-trading candidates. This never calls OpenAI."""
    normalized_mode = str(target_mode or "buy").strip().lower()
    allowed_categories = {"買い候補"}
    if normalized_mode in {"buy_watch", "buy+watch", "watch", "買い候補＋監視"}:
        allowed_categories.add("監視")

    selected = [
        signal
        for signal in list(signals or [])
        if signal.get("category") in allowed_categories and _score(signal) >= int(min_score)
    ]
    return sorted(selected, key=lambda item: _score(item), reverse=True)[: int(max_candidates)]


def _result_counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "saved_count": sum(1 for result in results if result.get("saved")),
        "duplicate_count": sum(1 for result in results if result.get("reason") == "duplicate_recent"),
        "failed_count": sum(
            1
            for result in results
            if not result.get("saved") and result.get("reason") != "duplicate_recent"
        ),
    }


def run_rule_based_virtual_logging(
    signals: List[Dict[str, Any]],
    target_mode: str,
    min_score: int,
    max_candidates: int,
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Save rule-based virtual buy logs. This is paper trading only and never calls OpenAI."""
    target_db_path = Path(db_path) if db_path else VIRTUAL_TRADES_DB_PATH
    timestamp = now_jst().strftime("%Y-%m-%d %H:%M:%S JST")
    selected = select_rule_buy_candidates(signals, target_mode, min_score, max_candidates)
    summary: Dict[str, Any] = {
        "timestamp_jst": timestamp,
        "market_status": market_status_label(),
        "input_signals_count": len(signals or []),
        "selected_count": len(selected),
        "saved_count": 0,
        "duplicate_count": 0,
        "failed_count": 0,
        "results": [],
        "error": "",
        "db_path": str(target_db_path),
        "openai_api_enabled": False,
        "use_openai": False,
    }

    try:
        results = process_virtual_trade_signals(selected, use_openai=False, db_path=target_db_path)
        summary["results"] = results
        summary.update(_result_counts(results))
    except Exception as exc:
        summary["error"] = f"{exc.__class__.__name__}: {exc}"

    return summary
