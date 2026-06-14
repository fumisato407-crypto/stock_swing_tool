from __future__ import annotations

from typing import Any, Dict, Iterable, List


def daily_score_from_filter(daily_filter: Dict[str, Any]) -> int:
    try:
        ok_count = int(daily_filter.get("daily_ok_count", 0) or 0)
        total_count = int(daily_filter.get("daily_total_count", 4) or 4)
    except (TypeError, ValueError):
        return 0
    if total_count <= 0:
        return 0
    return int(round(ok_count / total_count * 100))


def rank_daily_candidates(
    candidates: Iterable[Dict[str, Any]],
    top_n: int = 10,
    min_daily_score: int = 70,
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        item = dict(candidate)
        daily_filter = item.get("daily_filter") or item.get("daily_filter_json") or {}
        if not isinstance(daily_filter, dict):
            daily_filter = {}
        if item.get("daily_score") in (None, ""):
            item["daily_score"] = daily_score_from_filter(daily_filter)
        if item.get("daily_ok_count") in (None, ""):
            item["daily_ok_count"] = daily_filter.get("daily_ok_count", 0)
        ranked.append(item)

    ranked.sort(
        key=lambda item: (
            float(item.get("daily_score") or 0),
            int(item.get("daily_ok_count") or 0),
            float(item.get("score") or item.get("intraday_score") or 0),
        ),
        reverse=True,
    )
    total = len(ranked)
    for index, item in enumerate(ranked, start=1):
        score = float(item.get("daily_score") or 0)
        item["daily_rank_at_scan"] = index
        item["daily_rank_total"] = total
        item["daily_top_n"] = int(top_n or 0)
        if enabled:
            item["daily_top_n_pass"] = bool(index <= int(top_n or 0) and score >= float(min_daily_score or 0))
        else:
            item["daily_top_n_pass"] = True
    return ranked


def build_buy_condition_json(signal: Dict[str, Any], settings: Dict[str, Any] | None = None) -> Dict[str, Any]:
    settings = dict(settings or {})
    daily_total = signal.get("daily_total_count", (signal.get("daily_filter") or {}).get("daily_total_count", "-"))
    intraday_total = signal.get("intraday_total_count", (signal.get("intraday_entry") or {}).get("intraday_total_count", "-"))
    return {
        "daily_rank_at_scan": signal.get("daily_rank_at_scan"),
        "daily_rank_total": signal.get("daily_rank_total"),
        "daily_score": signal.get("daily_score"),
        "daily_ok_count": signal.get("daily_ok_count"),
        "daily_total_count": daily_total,
        "intraday_ok_count": signal.get("intraday_ok_count"),
        "intraday_total_count": intraday_total,
        "risk_pass": signal.get("risk_pass"),
        "min_score": settings.get("min_score"),
        "use_daily_top_n": settings.get("use_daily_top_n"),
        "daily_top_n": settings.get("daily_top_n", signal.get("daily_top_n")),
        "min_daily_score": settings.get("min_daily_score"),
        "daily_top_n_pass": signal.get("daily_top_n_pass"),
        "max_stop_loss_pct": settings.get("max_stop_loss_pct"),
        "max_loss_yen_limit": settings.get("max_loss_yen_limit"),
        "min_risk_reward": settings.get("min_risk_reward"),
        "daily_filter": signal.get("daily_filter") or signal.get("daily_filter_json") or {},
        "intraday_entry": signal.get("intraday_entry") or signal.get("intraday_entry_json") or {},
        "risk_filter": signal.get("risk_filter") or signal.get("risk_filter_json") or {},
    }
