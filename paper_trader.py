from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from ai_judge import build_market_snapshot, judge_virtual_trade
from data_fetcher import normalize_jp_symbol
from virtual_trade_store import find_recent_virtual_trade, insert_virtual_trade


DUPLICATE_COOLDOWN_MINUTES = 30


def _source_score(signal: Dict[str, Any]) -> float:
    try:
        return float(signal.get("score", signal.get("intraday_score", 0)) or 0)
    except (TypeError, ValueError):
        return 0.0


def _symbol(signal: Dict[str, Any]) -> str:
    return normalize_jp_symbol(signal.get("normalized_symbol") or signal.get("code"))


def build_virtual_trade_record(
    signal: Dict[str, Any],
    decision: Dict[str, Any],
    market_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
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
    }


def process_virtual_trade_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = build_market_snapshot(signal)
    decision = judge_virtual_trade(signal)
    record = build_virtual_trade_record(signal, decision, snapshot)

    if not record["symbol"]:
        return {"saved": False, "reason": "symbol_missing", "decision": decision, "record": record}

    duplicate = find_recent_virtual_trade(
        symbol=record["symbol"],
        entry_type=record["entry_type"],
        minutes=DUPLICATE_COOLDOWN_MINUTES,
    )
    if duplicate:
        return {"saved": False, "reason": "duplicate_recent", "decision": decision, "record": record}

    trade_id = insert_virtual_trade(record)
    return {"saved": True, "reason": "saved", "trade_id": trade_id, "decision": decision, "record": record}


def process_virtual_trade_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for signal in signals:
        try:
            results.append(process_virtual_trade_signal(signal))
        except Exception as exc:
            results.append(
                {
                    "saved": False,
                    "reason": f"error:{exc.__class__.__name__}",
                    "decision": {},
                    "record": {"code": signal.get("code"), "name": signal.get("name")},
                }
            )
    return results
