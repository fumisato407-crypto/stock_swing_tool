from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ai_judge import build_market_snapshot, judge_virtual_trade
from config import VIRTUAL_TRADES_DB_PATH
from data_fetcher import normalize_jp_symbol
from time_utils import now_jst_iso
from virtual_trade_store import find_recent_virtual_trade, insert_virtual_trade


DUPLICATE_COOLDOWN_MINUTES = 30


def _source_score(signal: Dict[str, Any]) -> float:
    try:
        return float(signal.get("score", signal.get("intraday_score", 0)) or 0)
    except (TypeError, ValueError):
        return 0.0


def _symbol(signal: Dict[str, Any]) -> str:
    return normalize_jp_symbol(signal.get("normalized_symbol") or signal.get("code"))


def _is_ai_generated(decision: Dict[str, Any]) -> bool:
    value = decision.get("is_ai_generated", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "gpt"}
    return bool(value)


def _judge_source(decision: Dict[str, Any]) -> str:
    model_used = str(decision.get("model_used", "") or "")
    if _is_ai_generated(decision):
        return "gpt"
    if model_used == "rule_based_fallback":
        return "fallback"
    if model_used:
        return "fallback" if "fallback" in model_used.lower() else "unknown"
    return "unknown"


def _model_used(decision: Dict[str, Any]) -> str:
    source = _judge_source(decision)
    if source == "fallback":
        return "rule_based_fallback"
    return str(decision.get("model_used", "") or "unknown")


def build_virtual_trade_record(
    signal: Dict[str, Any],
    decision: Dict[str, Any],
    market_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    timestamp_jst = now_jst_iso()
    is_ai_generated = _is_ai_generated(decision)
    judge_source = _judge_source(decision)
    model_used = _model_used(decision)
    return {
        "timestamp": timestamp_jst,
        "timestamp_jst": timestamp_jst,
        "created_at_jst": timestamp_jst,
        "symbol": _symbol(signal),
        "name": str(signal.get("name", "")),
        "decision": decision.get("decision", "virtual_watch"),
        "entry_type": decision.get("entry_type", signal.get("entry_type", signal.get("signal_type", ""))),
        "confidence": decision.get("confidence", ""),
        "entry_price": decision.get("entry_price"),
        "stop_loss": decision.get("stop_loss"),
        "take_profit": decision.get("take_profit"),
        "max_hold_days": decision.get("max_hold_days", 5),
        "reasons": decision.get("reasons", []),
        "risk_factors": decision.get("risk_factors", []),
        "source_score": _source_score(signal),
        "market_snapshot_json": market_snapshot,
        "status": "open" if decision.get("decision") == "virtual_buy" else "logged",
        "model_used": model_used,
        "is_ai_generated": 1 if is_ai_generated else 0,
        "judge_source": judge_source,
        "fallback_reason": decision.get("fallback_reason") or "",
        "fallback_error_type": decision.get("fallback_error_type") or "",
        "fallback_error_message": decision.get("fallback_error_message") or "",
    }


def process_virtual_trade_signal(
    signal: Dict[str, Any],
    use_openai: bool = False,
    db_path: Path | None = None,
) -> Dict[str, Any]:
    snapshot = build_market_snapshot(signal)
    decision = judge_virtual_trade(signal, use_openai=use_openai)
    record = build_virtual_trade_record(signal, decision, snapshot)
    target_db_path = db_path or VIRTUAL_TRADES_DB_PATH
    db_path_text = str(target_db_path)

    if not record["symbol"]:
        return {
            "saved": False,
            "reason": "symbol_missing",
            "decision": decision,
            "record": record,
            "db_path": db_path_text,
        }

    duplicate = find_recent_virtual_trade(
        symbol=record["symbol"],
        entry_type=record["entry_type"],
        judge_source=record["judge_source"],
        minutes=DUPLICATE_COOLDOWN_MINUTES,
        path=target_db_path,
    )
    if duplicate:
        return {
            "saved": False,
            "reason": "duplicate_recent",
            "duplicate_id": duplicate.get("id"),
            "decision": decision,
            "record": record,
            "db_path": db_path_text,
        }

    trade_id = insert_virtual_trade(record, path=target_db_path)
    return {
        "saved": True,
        "reason": "saved",
        "trade_id": trade_id,
        "db_path": db_path_text,
        "decision": decision,
        "record": record,
    }


def process_virtual_trade_signals(
    signals: List[Dict[str, Any]],
    use_openai: bool = False,
    db_path: Path | None = None,
) -> List[Dict[str, Any]]:
    results = []
    for signal in signals:
        try:
            results.append(process_virtual_trade_signal(signal, use_openai=use_openai, db_path=db_path))
        except Exception as exc:
            results.append(
                {
                    "saved": False,
                    "reason": f"error:{exc.__class__.__name__}",
                    "error_message": str(exc),
                    "decision": {},
                    "record": {"code": signal.get("code"), "name": signal.get("name")},
                    "db_path": str(db_path or VIRTUAL_TRADES_DB_PATH),
                }
            )
    return results
