from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from daily_top_n_filter import build_buy_condition_json, daily_score_from_filter, rank_daily_candidates
from data_fetcher import normalize_jp_symbol
from daily_technical import get_replay_technical_config
from historical_data import fetch_historical_data, load_or_fetch_historical_data, validate_historical_ohlcv
from multi_timeframe_rules import build_replay_daily_context, evaluate_daily_filter
from replay_engine import (
    CANDIDATE_MODE_EXISTING,
    CANDIDATE_MODE_TECHNICAL_ONLY,
    ReplayRuleConfig,
    apply_replay_money_metrics,
    create_replay_trade,
    empty_replay_stage_counts,
    evaluate_replay_step,
    evaluate_replay_trade_outcome,
    leak_check_for_trade,
    replay_zero_trade_reasons,
    summarize_replay_money,
    summarize_replay_results,
    update_replay_stage_counts,
)
from replay_batch import build_config_hash, build_run_summary_export, build_trade_detail_export
from replay_cache import TECHNICAL_SCORE_FEATURE_VERSION
from time_utils import now_jst_iso


ProgressCallback = Callable[[str, int, int, str], None]
WATCHLIST_REPLAY_MODE = "watchlist_replay"
WATCHLIST_REPLAY_APP_VERSION = "stock_swing_tool_watchlist_replay_v1"


@dataclass
class HistoricalScanReplayConfig:
    candidate_generation_mode: str = CANDIDATE_MODE_EXISTING
    period: str = "1mo"
    interval: str = "5m"
    start_at: Any = None
    end_at: Any = None
    scan_interval: str = "5m"
    min_score: int = 70
    target_mode: str = "buy_only"
    max_buys_per_scan: int = 3
    cooldown: str = "30m"
    shares: int = 100
    max_symbols: Optional[int] = 30
    source_watchlist: str = "watchlist.csv"
    use_multi_timeframe: bool = True
    daily_filter_required: bool = True
    daily_min_ok: int = 3
    intraday_min_ok: int = 2
    use_vwap: bool = True
    use_volume_spike: bool = True
    use_risk_filter: bool = True
    max_stop_loss_pct: float = 3.0
    max_loss_yen_limit: float = 20000.0
    min_risk_reward: float = 1.2
    use_daily_top_n: bool = False
    daily_top_n: int = 10
    min_daily_score: int = 70
    use_technical_score: bool = False
    technical_preset: str = "standard_swing"
    technical_min_score: int = 0
    technical_min_confidence: int = 0
    technical_show_breakdown: bool = True
    technical_config: Dict[str, Any] = None
    use_replay_cache: bool = True
    one_position_per_symbol: bool = False
    parallel: bool = True
    max_workers: int = 2


def _timestamp_text(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value or "")


def _score(signal: Dict[str, Any]) -> int:
    try:
        return int(float(signal.get("intraday_score", signal.get("score", 0)) or 0))
    except (TypeError, ValueError):
        return 0


def _category(signal: Dict[str, Any]) -> str:
    judgement = str(signal.get("judgement", ""))
    if judgement == "買い検討OK":
        return "買い候補"
    if judgement == "監視強化":
        return "監視"
    return "見送り"


def _target_allows(category: str, target_mode: str) -> bool:
    normalized = str(target_mode or "buy_only")
    if normalized in {"buy_watch", "買い候補＋監視"}:
        return category in {"買い候補", "監視"}
    return category == "買い候補"


def _scan_interval_minutes(value: str) -> Optional[int]:
    mapping = {
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "60m": 60,
        "1h": 60,
    }
    return mapping.get(str(value))


def _filter_scan_times(all_times: List[pd.Timestamp], scan_interval: str) -> List[pd.Timestamp]:
    unique = sorted(set(pd.Timestamp(ts) for ts in all_times))
    if scan_interval == "1d":
        by_date: Dict[Any, pd.Timestamp] = {}
        for ts in unique:
            by_date[ts.date()] = ts
        return list(by_date.values())

    minutes = _scan_interval_minutes(scan_interval)
    if not minutes:
        return unique
    filtered = []
    for ts in unique:
        minutes_from_midnight = ts.hour * 60 + ts.minute
        if minutes_from_midnight % minutes == 0:
            filtered.append(ts)
    return filtered


def _cooldown_allows(last_time: Optional[pd.Timestamp], current_time: pd.Timestamp, cooldown: str) -> bool:
    if last_time is None:
        return True
    if cooldown == "day":
        return last_time.date() != current_time.date()
    minutes = {
        "30m": 30,
        "1h": 60,
        "3h": 180,
    }.get(cooldown, 30)
    return (current_time - last_time).total_seconds() >= minutes * 60


def _records_from_watchlist(records: List[Dict[str, Any]], max_symbols: Optional[int]) -> List[Dict[str, Any]]:
    prepared = []
    for record in records:
        normalized = normalize_jp_symbol(record.get("normalized_symbol") or record.get("code") or record.get("raw_code"))
        if not normalized:
            continue
        item = dict(record)
        item["normalized_symbol"] = normalized
        item["code"] = normalized.replace(".T", "")
        prepared.append(item)
    if max_symbols:
        return prepared[: int(max_symbols)]
    return prepared


def _fetch_one_watchlist_record(record: Dict[str, Any], config: HistoricalScanReplayConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    symbol = str(record.get("normalized_symbol", ""))
    name = str(record.get("name", ""))
    fetcher = load_or_fetch_historical_data if config.use_replay_cache else fetch_historical_data
    fetched = fetcher(symbol, config.period, config.interval)
    validation = validate_historical_ohlcv(fetched.data)
    cleaned = validation.get("cleaned_data", pd.DataFrame())
    daily_cleaned = pd.DataFrame()
    daily_error = ""
    daily_from_cache = False
    if config.use_multi_timeframe:
        daily_fetched = fetcher(symbol, "6mo", "1d")
        daily_validation = validate_historical_ohlcv(daily_fetched.data)
        daily_cleaned = daily_validation.get("cleaned_data", pd.DataFrame())
        daily_error = daily_fetched.error_message or ""
        daily_from_cache = bool(daily_fetched.from_cache)
    meta = {
        "symbol": symbol,
        "name": name,
        "fetched_rows": fetched.fetched_rows,
        "cleaned_rows": len(cleaned),
        "daily_rows": len(daily_cleaned),
        "from_cache": fetched.from_cache,
        "daily_from_cache": daily_from_cache,
        "error_type": fetched.error_type,
        "error_message": fetched.error_message,
        "daily_error": daily_error,
        "excluded_count": validation.get("excluded_count", 0),
        "first_timestamp": fetched.first_timestamp,
        "last_timestamp": fetched.last_timestamp,
        "processing_seconds": round(time.perf_counter() - started, 3),
    }
    if fetched.error_message or cleaned.empty:
        return {"success": False, "meta": meta}
    return {
        "success": True,
        "symbol": symbol,
        "data": {
            "record": record,
            "name": name,
            "data": cleaned,
            "daily_data": daily_cleaned,
            "meta": meta,
            "validation": {key: value for key, value in validation.items() if key != "cleaned_data"},
        },
        "meta": meta,
    }


def _fetch_watchlist_data(
    records: List[Dict[str, Any]],
    config: HistoricalScanReplayConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    data_by_symbol: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    total = len(records)
    if config.parallel and total > 1:
        workers = max(1, min(int(config.max_workers or 1), 8))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_one_watchlist_record, record, config): record for record in records}
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    result = future.result()
                except Exception as exc:
                    record = futures[future]
                    result = {
                        "success": False,
                        "meta": {
                            "symbol": str(record.get("normalized_symbol", "")),
                            "name": str(record.get("name", "")),
                            "error_type": "fetch_exception",
                            "error_message": f"{exc.__class__.__name__}: {exc}",
                            "processing_seconds": 0,
                        },
                    }
                meta = result.get("meta", {})
                if progress_callback:
                    progress_callback("データ取得中", index, total, f"{meta.get('symbol', '')} {meta.get('name', '')}".strip())
                if result.get("success"):
                    data_by_symbol[str(result.get("symbol", ""))] = result["data"]
                else:
                    failures.append(meta)
    else:
        for index, record in enumerate(records, start=1):
            result = _fetch_one_watchlist_record(record, config)
            meta = result.get("meta", {})
            if progress_callback:
                progress_callback("データ取得中", index, total, f"{meta.get('symbol', '')} {meta.get('name', '')}".strip())
            if result.get("success"):
                data_by_symbol[str(result.get("symbol", ""))] = result["data"]
            else:
                failures.append(meta)
    return data_by_symbol, failures


def _build_scan_times(data_by_symbol: Dict[str, Dict[str, Any]], config: HistoricalScanReplayConfig) -> List[pd.Timestamp]:
    times: List[pd.Timestamp] = []
    start = pd.Timestamp(config.start_at) if config.start_at is not None else None
    end = pd.Timestamp(config.end_at) if config.end_at is not None else None
    for item in data_by_symbol.values():
        df = item["data"]
        scoped = df
        if start is not None:
            scoped = scoped[scoped.index >= start]
        if end is not None:
            scoped = scoped[scoped.index <= end]
        times.extend([pd.Timestamp(ts) for ts in scoped.index])
    return _filter_scan_times(times, config.scan_interval)


def _symbol_trade_summary(
    records: List[Dict[str, Any]],
    data_by_symbol: Dict[str, Dict[str, Any]],
    failures: List[Dict[str, Any]],
    shares: int,
) -> pd.DataFrame:
    rows = []
    failures_by_symbol = {item.get("symbol"): item for item in failures}
    symbols = sorted(set(data_by_symbol.keys()) | set(failures_by_symbol.keys()))
    for symbol in symbols:
        trades = [trade for trade in records if trade.get("symbol") == symbol]
        money = summarize_replay_money(trades, shares)
        perf = summarize_replay_results(trades)
        data_rows = 0
        name = ""
        error = ""
        if symbol in data_by_symbol:
            data_rows = int(len(data_by_symbol[symbol]["data"]))
            name = str(data_by_symbol[symbol].get("name", ""))
        if symbol in failures_by_symbol:
            error = failures_by_symbol[symbol].get("error_message", "")
            name = failures_by_symbol[symbol].get("name", name)
        rows.append(
            {
                "symbol": symbol,
                "name": name,
                "data_rows": data_rows,
                "trade_count": len(trades),
                "win_rate_pct": perf.get("win_rate_pct", 0),
                "net_profit_yen": money.get("net_profit_yen", 0),
                "gross_profit_yen": money.get("gross_profit_yen", 0),
                "gross_loss_yen": money.get("gross_loss_yen", 0),
                "profit_loss_ratio": money.get("profit_loss_ratio"),
                "avg_return_pct": perf.get("avg_return_pct", 0),
                "max_profit_yen": money.get("max_profit_yen"),
                "max_loss_yen": money.get("max_loss_yen"),
                "max_win_streak": money.get("max_win_streak", 0),
                "max_loss_streak": money.get("max_loss_streak", 0),
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def _daily_top_n_settings(config: HistoricalScanReplayConfig) -> Dict[str, Any]:
    return {
        "use_daily_top_n": bool(config.use_daily_top_n),
        "daily_top_n": int(config.daily_top_n),
        "min_daily_score": int(config.min_daily_score),
        "min_score": int(config.min_score),
        "max_stop_loss_pct": config.max_stop_loss_pct,
        "max_loss_yen_limit": config.max_loss_yen_limit,
        "min_risk_reward": config.min_risk_reward,
    }


def _daily_rank_info_for_scan_time(
    data_by_symbol: Dict[str, Dict[str, Any]],
    scan_time: pd.Timestamp,
    config: HistoricalScanReplayConfig,
) -> Dict[str, Dict[str, Any]]:
    daily_candidates: List[Dict[str, Any]] = []
    mtf_config = {
        "daily_min_ok": config.daily_min_ok if config.daily_filter_required else 0,
        "intraday_min_ok": config.intraday_min_ok,
        "use_vwap": config.use_vwap,
        "use_volume_spike": config.use_volume_spike,
    }
    for symbol, item in data_by_symbol.items():
        df: pd.DataFrame = item["data"]
        if scan_time not in df.index:
            continue
        history = df.loc[df.index <= scan_time]
        if history.empty:
            continue
        daily_context = build_replay_daily_context(history, scan_time, prior_daily_df=item.get("daily_data"))
        daily_filter = evaluate_daily_filter(daily_context, current_time=None, config=mtf_config)
        daily_score = int(daily_filter.get("daily_score", daily_score_from_filter(daily_filter)) or 0)
        daily_candidates.append(
            {
                "symbol": symbol,
                "daily_filter": daily_filter,
                "daily_filter_json": daily_filter,
                "daily_ok_count": daily_filter.get("daily_ok_count", 0),
                "daily_total_count": daily_filter.get("daily_total_count", 4),
                "daily_score": daily_score,
            }
        )

    ranked = rank_daily_candidates(
        daily_candidates,
        top_n=config.daily_top_n,
        min_daily_score=config.min_daily_score,
        enabled=config.use_daily_top_n,
    )
    return {str(item["symbol"]): item for item in ranked}


def _watchlist_context_from_summary(summary: Dict[str, Any], trades: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    technical_config = summary.get("technical_config") if isinstance(summary.get("technical_config"), dict) else {}
    target_symbols = summary.get("target_symbols")
    if not target_symbols and trades:
        target_symbols = ",".join(sorted({str(trade.get("symbol", "")) for trade in trades if str(trade.get("symbol", ""))}))
    watchlist_filter = summary.get("watchlist_filter") if isinstance(summary.get("watchlist_filter"), dict) else {}
    excluded_symbols = summary.get("excluded_symbols") or ",".join(watchlist_filter.get("excluded_symbols", []) or [])
    return {
        "run_id": summary.get("replay_run_id", ""),
        "run_datetime": summary.get("run_datetime") or summary.get("created_at_jst") or now_jst_iso(),
        "mode": WATCHLIST_REPLAY_MODE,
        "target_symbols": target_symbols or "",
        "excluded_symbols": excluded_symbols,
        "watchlist_limit": summary.get("watchlist_limit", ""),
        "period": summary.get("period", ""),
        "interval": summary.get("interval", ""),
        "start_date": str(summary.get("start_at", "")).split(" ")[0] if summary.get("start_at") else "",
        "end_date": str(summary.get("end_at", "")).split(" ")[0] if summary.get("end_at") else "",
        "start_time": str(summary.get("start_at", "")).split(" ")[-1] if summary.get("start_at") else "",
        "end_time": str(summary.get("end_at", "")).split(" ")[-1] if summary.get("end_at") else "",
        "assumed_shares": summary.get("shares", ""),
        "max_trades_per_symbol": summary.get("max_buys_per_scan", summary.get("max_virtual_buys_per_scan", "")),
        "cooldown_bars_or_minutes": summary.get("cooldown", ""),
        "one_position_per_symbol": summary.get("one_position_per_symbol", False),
        "cache_enabled": summary.get("use_replay_cache", False),
        "parallel_enabled": summary.get("parallel", False),
        "workers": summary.get("max_workers", 1),
        "use_risk_condition": summary.get("use_risk_filter", False),
        "use_vwap_condition": summary.get("use_vwap", False),
        "use_volume_spike_condition": summary.get("use_volume_spike", False),
        "use_multi_timeframe": summary.get("use_multi_timeframe", False),
        "use_daily_filter": summary.get("daily_filter_required", False),
        "use_daily_top_n": summary.get("use_daily_top_n", False),
        "daily_score_min": summary.get("min_daily_score", ""),
        "feature_version": TECHNICAL_SCORE_FEATURE_VERSION,
        "app_version": WATCHLIST_REPLAY_APP_VERSION,
        "processing_seconds": summary.get("total_seconds", 0),
        "cache_hit_count": summary.get("cache_hit_count", 0),
        "cache_miss_count": summary.get("cache_miss_count", 0),
        "scan_count": summary.get("scan_count", summary.get("total_scan_steps", 0)),
        "existing_logic_checked_count": summary.get("existing_logic_checked_count", 0),
        "existing_logic_pass_count": summary.get("existing_logic_pass_count", 0),
        "existing_logic_reject_count": summary.get("existing_logic_reject_count", 0),
        "technical_score_checked_count": summary.get("technical_score_checked_count", 0),
        "technical_score_pass_count": summary.get("technical_score_pass_count", 0),
        "technical_score_reject_count": summary.get("technical_score_reject_count", 0),
        "hard_filter_reject_count": summary.get("hard_filter_reject_count", 0),
        "use_technical_score": summary.get("use_technical_score", False),
        "use_penalty": technical_config.get("use_penalty", ""),
        "use_hard_filter": technical_config.get("use_hard_filter", ""),
        "hard_filter_mode": "hard" if technical_config.get("use_hard_filter", False) else "OFF",
    }


def _watchlist_variant_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    technical_config = summary.get("technical_config") if isinstance(summary.get("technical_config"), dict) else {}
    return {
        "candidate_mode": summary.get("candidate_generation_mode", ""),
        "preset": summary.get("technical_preset", ""),
        "min_technical_score": summary.get("technical_min_score", ""),
        "use_penalty": technical_config.get("use_penalty", ""),
        "use_hard_filter": technical_config.get("use_hard_filter", ""),
    }


def attach_watchlist_replay_exports(result: Dict[str, Any]) -> Dict[str, Any]:
    summary = result.setdefault("summary", {})
    trades = result.get("trades", []) or []
    context = _watchlist_context_from_summary(summary, trades)
    variant = _watchlist_variant_from_summary(summary)
    config_hash = build_config_hash(context, variant)
    summary["config_hash"] = config_hash
    summary["run_datetime"] = context.get("run_datetime", "")
    stage_counts = summary.get("stage_counts") if isinstance(summary.get("stage_counts"), dict) else {}
    summary["scan_count"] = int(stage_counts.get("scan_count", stage_counts.get("scan_total_count", summary.get("scan_count", 0))) or 0)
    summary["existing_logic_checked_count"] = int(stage_counts.get("existing_logic_checked_count", summary.get("existing_logic_checked_count", 0)) or 0)
    summary["existing_logic_pass_count"] = int(stage_counts.get("existing_logic_pass_count", summary.get("existing_logic_pass_count", 0)) or 0)
    summary["existing_logic_reject_count"] = int(stage_counts.get("existing_logic_reject_count", summary.get("existing_logic_reject_count", 0)) or 0)
    summary["technical_score_checked_count"] = int(
        stage_counts.get("technical_score_checked_count", stage_counts.get("technical_score_attempt_count", summary.get("technical_score_checked_count", 0))) or 0
    )
    summary["technical_score_pass_count"] = int(
        stage_counts.get("technical_score_pass_count", stage_counts.get("technical_min_score_pass_count", summary.get("technical_score_pass_count", 0))) or 0
    )
    summary["technical_score_reject_count"] = int(stage_counts.get("technical_score_reject_count", summary.get("technical_score_reject_count", 0)) or 0)
    summary["hard_filter_reject_count"] = int(
        stage_counts.get("hard_filter_reject_count", stage_counts.get("technical_hard_filter_excluded_count", summary.get("hard_filter_reject_count", 0))) or 0
    )
    summary["raw_signal_count"] = int(stage_counts.get("raw_signal_count", summary.get("raw_signal_count", 0)) or 0)
    summary["cooldown_filtered_count"] = int(stage_counts.get("cooldown_filtered_count", summary.get("cooldown_filtered_count", 0)) or 0)
    summary["position_filtered_count"] = int(
        stage_counts.get("position_filtered_count", stage_counts.get("position_open_excluded_count", summary.get("position_filtered_count", 0))) or 0
    )
    summary["max_trade_filtered_count"] = int(
        stage_counts.get("max_trade_filtered_count", stage_counts.get("max_trades_excluded_count", summary.get("max_trade_filtered_count", 0))) or 0
    )
    summary["final_trade_count"] = len(trades)
    summary["settled_trade_count"] = int(
        stage_counts.get(
            "settled_trade_count",
            sum(1 for trade in trades if trade.get("status") == "closed" and trade.get("exit_price") is not None),
        )
        or 0
    )
    summary["csv_export_count"] = len(trades)
    summary["db_saved_count"] = int(summary.get("db_saved_count", 0) or 0)
    summary["mode"] = WATCHLIST_REPLAY_MODE
    for key in (
        "scan_count",
        "existing_logic_checked_count",
        "existing_logic_pass_count",
        "existing_logic_reject_count",
        "technical_score_checked_count",
        "technical_score_pass_count",
        "technical_score_reject_count",
        "hard_filter_reject_count",
        "raw_signal_count",
        "cooldown_filtered_count",
        "position_filtered_count",
        "max_trade_filtered_count",
        "final_trade_count",
        "settled_trade_count",
        "db_saved_count",
        "csv_export_count",
    ):
        context[key] = summary.get(key, 0)

    for trade in trades:
        trade["run_id"] = context.get("run_id", "")
        trade["run_datetime"] = context.get("run_datetime", "")
        trade["mode"] = WATCHLIST_REPLAY_MODE
        trade["candidate_mode"] = summary.get("candidate_generation_mode", "")
        trade["preset"] = summary.get("technical_preset", "")
        trade["min_technical_score"] = summary.get("technical_min_score", "")
        trade["use_technical_score"] = summary.get("use_technical_score", False)
        trade["use_penalty"] = context.get("use_penalty", "")
        trade["use_hard_filter"] = context.get("use_hard_filter", "")
        trade["hard_filter_mode"] = context.get("hard_filter_mode", "")
        trade["config_hash"] = config_hash

    summary_row = {
        "condition_name": WATCHLIST_REPLAY_MODE,
        "candidate_mode": summary.get("candidate_generation_mode", ""),
        "preset": summary.get("technical_preset", ""),
        "min_technical_score": summary.get("technical_min_score", ""),
        "use_technical_score": summary.get("use_technical_score", False),
        "use_penalty": context.get("use_penalty", ""),
        "use_hard_filter": context.get("use_hard_filter", ""),
        "hard_filter_mode": context.get("hard_filter_mode", ""),
        "scan_count": summary.get("scan_count", 0),
        "existing_logic_checked_count": summary.get("existing_logic_checked_count", 0),
        "existing_logic_pass_count": summary.get("existing_logic_pass_count", 0),
        "existing_logic_reject_count": summary.get("existing_logic_reject_count", 0),
        "technical_score_checked_count": summary.get("technical_score_checked_count", 0),
        "technical_score_pass_count": summary.get("technical_score_pass_count", 0),
        "technical_score_reject_count": summary.get("technical_score_reject_count", 0),
        "hard_filter_reject_count": summary.get("hard_filter_reject_count", 0),
        "raw_signal_count": summary.get("raw_signal_count", 0),
        "cooldown_filtered_count": summary.get("cooldown_filtered_count", 0),
        "position_filtered_count": summary.get("position_filtered_count", 0),
        "max_trade_filtered_count": summary.get("max_trade_filtered_count", 0),
        "final_trade_count": summary.get("final_trade_count", 0),
        "settled_trade_count": summary.get("settled_trade_count", 0),
        "db_saved_count": summary.get("db_saved_count", 0),
        "csv_export_count": summary.get("csv_export_count", 0),
        "win_rate": summary.get("win_rate_pct", 0),
        "total_profit": summary.get("gross_profit_yen", 0),
        "total_loss": summary.get("gross_loss_yen", 0),
        "net_profit": summary.get("net_profit_yen", 0),
        "average_profit": summary.get("average_profit_yen", 0),
        "average_loss": summary.get("average_loss_yen", 0),
        "profit_factor": summary.get("profit_factor", ""),
        "max_win": summary.get("max_profit_yen", 0),
        "max_loss": summary.get("max_loss_yen", 0),
        "max_win_streak": summary.get("max_win_streak", 0),
        "max_loss_streak": summary.get("max_loss_streak", 0),
        "processing_seconds": summary.get("total_seconds", 0),
        "cache_hit_count": summary.get("cache_hit_count", 0),
        "cache_miss_count": summary.get("cache_miss_count", 0),
        "leak_check_ok_count": stage_counts.get("leak_check_ok_count", 0),
        "leak_check_ng_count": stage_counts.get("leak_check_ng_count", 0),
        "config_hash": config_hash,
    }
    result["run_context"] = context
    result["trade_detail_df"] = build_trade_detail_export(trades, context)
    result["run_summary_df"] = build_run_summary_export(pd.DataFrame([summary_row]), context, summary)
    return result


def run_historical_scan_replay(
    watchlist_records: List[Dict[str, Any]],
    config: HistoricalScanReplayConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    records = _records_from_watchlist(watchlist_records, config.max_symbols)
    total_started = time.perf_counter()
    fetch_started = time.perf_counter()
    data_by_symbol, failures = _fetch_watchlist_data(records, config, progress_callback)
    fetch_seconds = round(time.perf_counter() - fetch_started, 3)
    scan_times = _build_scan_times(data_by_symbol, config)
    total_scan_steps = len(scan_times)
    trades: List[Dict[str, Any]] = []
    last_signal_by_symbol: Dict[str, pd.Timestamp] = {}
    open_until_by_symbol: Dict[str, pd.Timestamp] = {}
    total_candidates = 0
    selected_by_scan_time: List[Dict[str, Any]] = []
    stage_counts = empty_replay_stage_counts(len(records))
    stage_counts["scan_count"] = total_scan_steps
    stage_counts["scan_total_count"] = total_scan_steps

    replay_config = ReplayRuleConfig(
        candidate_generation_mode=config.candidate_generation_mode,
        min_score=config.min_score,
        target_rule="すべて",
        max_trades=config.max_buys_per_scan,
        interval=config.interval,
        debug=True,
        use_multi_timeframe=False if config.candidate_generation_mode == CANDIDATE_MODE_TECHNICAL_ONLY else config.use_multi_timeframe,
        daily_min_ok=config.daily_min_ok if config.daily_filter_required else 0,
        intraday_min_ok=config.intraday_min_ok,
        use_vwap=config.use_vwap,
        use_volume_spike=config.use_volume_spike,
        shares=config.shares,
        use_risk_filter=False if config.candidate_generation_mode == CANDIDATE_MODE_TECHNICAL_ONLY else config.use_risk_filter,
        max_stop_loss_pct=config.max_stop_loss_pct,
        max_loss_yen_limit=config.max_loss_yen_limit,
        min_risk_reward=config.min_risk_reward,
        use_technical_score=(
            config.use_technical_score
            or config.candidate_generation_mode != CANDIDATE_MODE_EXISTING
        ),
        technical_preset=config.technical_preset,
        technical_min_score=config.technical_min_score,
        technical_min_confidence=config.technical_min_confidence,
        technical_show_breakdown=config.technical_show_breakdown,
        technical_config=config.technical_config or get_replay_technical_config(),
        use_replay_cache=config.use_replay_cache,
        one_position_per_symbol=config.one_position_per_symbol,
    )

    scan_started = time.perf_counter()
    for step_index, scan_time in enumerate(scan_times, start=1):
        if progress_callback:
            progress_callback("スキャン再現中", step_index, max(1, total_scan_steps), _timestamp_text(scan_time))
        candidates: List[Dict[str, Any]] = []
        daily_rank_info = (
            {}
            if config.candidate_generation_mode == CANDIDATE_MODE_TECHNICAL_ONLY
            else _daily_rank_info_for_scan_time(data_by_symbol, scan_time, config)
        )
        top_n_pool_count = sum(1 for item in daily_rank_info.values() if item.get("daily_top_n_pass"))
        for symbol, item in data_by_symbol.items():
            df: pd.DataFrame = item["data"]
            if scan_time not in df.index:
                continue
            daily_meta = daily_rank_info.get(symbol, {})
            if (
                config.candidate_generation_mode != CANDIDATE_MODE_TECHNICAL_ONLY
                and config.use_daily_top_n
                and not daily_meta.get("daily_top_n_pass")
            ):
                continue
            history = df.loc[df.index <= scan_time]
            if len(history) < 25:
                continue
            signal = evaluate_replay_step(
                history,
                scan_time,
                replay_config,
                symbol=symbol,
                name=str(item.get("name", "")),
                daily_df=item.get("daily_data"),
            )
            update_replay_stage_counts(stage_counts, signal, replay_config)
            if signal.get("replay_skip_reason"):
                continue
            signal.update(
                {
                    key: value
                    for key, value in daily_meta.items()
                    if key
                    in {
                        "daily_score",
                        "daily_rank_at_scan",
                        "daily_rank_total",
                        "daily_top_n_pass",
                        "daily_top_n",
                    }
                }
            )
            signal["buy_condition_json"] = build_buy_condition_json(signal, _daily_top_n_settings(config))
            category = _category(signal)
            if config.candidate_generation_mode == CANDIDATE_MODE_TECHNICAL_ONLY:
                category = "買い候補"
                signal["category"] = category
            elif not _target_allows(category, config.target_mode):
                reasons = stage_counts.setdefault("skip_reasons", {})
                reasons["target_mode外"] = int(reasons.get("target_mode外", 0)) + 1
                continue
            stage_counts["raw_signal_count"] = int(stage_counts.get("raw_signal_count", 0)) + 1
            if not _cooldown_allows(last_signal_by_symbol.get(symbol), scan_time, config.cooldown):
                stage_counts["cooldown_filtered_count"] = int(stage_counts.get("cooldown_filtered_count", 0)) + 1
                reasons = stage_counts.setdefault("skip_reasons", {})
                reasons["cooldown_recent"] = int(reasons.get("cooldown_recent", 0)) + 1
                continue
            if config.one_position_per_symbol and symbol in open_until_by_symbol and scan_time <= open_until_by_symbol[symbol]:
                stage_counts["position_open_excluded_count"] = int(stage_counts.get("position_open_excluded_count", 0)) + 1
                reasons = stage_counts.setdefault("skip_reasons", {})
                reasons["position_open"] = int(reasons.get("position_open", 0)) + 1
                continue
            signal["category"] = category
            signal["symbol"] = symbol
            signal["name"] = str(item.get("name", ""))
            signal["scan_time"] = _timestamp_text(scan_time)
            signal["source_watchlist"] = config.source_watchlist
            candidates.append(signal)

        total_candidates += len(candidates)
        selected = sorted(candidates, key=lambda signal: _score(signal), reverse=True)[: int(config.max_buys_per_scan)]
        if len(candidates) > len(selected):
            stage_counts["max_trades_excluded_count"] = int(stage_counts.get("max_trades_excluded_count", 0)) + (
                len(candidates) - len(selected)
            )
        selected_by_scan_time.append(
            {
                "scan_time": _timestamp_text(scan_time),
                "selected_count": len(selected),
                "daily_top_n_pool_count": top_n_pool_count,
                "daily_rank_total": len(daily_rank_info),
            }
        )
        for rank, signal in enumerate(selected, start=1):
            symbol = str(signal["symbol"])
            item = data_by_symbol[symbol]
            df = item["data"]
            current_bar = df.loc[scan_time]
            trade = create_replay_trade(signal, current_bar, scan_time)
            future = df.loc[df.index > scan_time]
            outcome_start = _timestamp_text(future.index[0]) if not future.empty else ""
            trade["outcome_start_timestamp"] = outcome_start
            trade.update(leak_check_for_trade(trade, outcome_start))
            if trade.get("leak_check_result") == "NG":
                stage_counts["leak_check_ng_count"] = int(stage_counts.get("leak_check_ng_count", 0)) + 1
                reasons = stage_counts.setdefault("skip_reasons", {})
                reasons["leak_check_ng"] = int(reasons.get("leak_check_ng", 0)) + 1
                continue
            stage_counts["leak_check_ok_count"] = int(stage_counts.get("leak_check_ok_count", 0)) + 1
            trade.update(
                evaluate_replay_trade_outcome(
                    trade,
                    future,
                    interval=config.interval,
                    debug=True,
                    symbol=symbol,
                )
            )
            trade.update(
                {
                    "replay_mode": "watchlist_scan",
                    "scan_time": _timestamp_text(scan_time),
                    "selected_rank": rank,
                    "scan_score": _score(signal),
                    "category": signal.get("category", ""),
                    "source_watchlist": config.source_watchlist,
                    "symbol": symbol,
                    "name": str(item.get("name", "")),
                    "replay_start_at": _timestamp_text(config.start_at or df.index[0]),
                    "replay_end_at": _timestamp_text(config.end_at or df.index[-1]),
                    "notes_json": {
                        "scan_time": _timestamp_text(scan_time),
                        "lookahead_guard": "buy decision used bars at or before scan_time only",
                        "signal": {key: value for key, value in signal.items() if key != "data"},
                    },
                }
            )
            trades.append(trade)
            stage_counts["final_virtual_buy_count"] = len(trades)
            exit_reference = trade.get("exit_time") or trade.get("evaluated_until")
            if config.one_position_per_symbol and exit_reference:
                try:
                    open_until_by_symbol[symbol] = pd.Timestamp(exit_reference)
                except Exception:
                    pass
            last_signal_by_symbol[symbol] = scan_time
    scan_seconds = round(time.perf_counter() - scan_started, 3)

    if progress_callback:
        progress_callback("結果集計中", 1, 1, "100株想定損益を集計しています")

    summary_started = time.perf_counter()
    trades = apply_replay_money_metrics(trades, config.shares)
    performance = summarize_replay_results(trades)
    money = summarize_replay_money(trades, config.shares)
    symbol_summary = _symbol_trade_summary(trades, data_by_symbol, failures, config.shares)
    settled_trade_count = sum(
        1
        for trade in trades
        if trade.get("status") == "closed" and trade.get("exit_price") is not None
    )
    stage_counts["final_virtual_buy_count"] = len(trades)
    stage_counts["settled_trade_count"] = settled_trade_count
    summary_seconds = round(time.perf_counter() - summary_started, 3)
    total_seconds = round(time.perf_counter() - total_started, 3)
    fetch_meta_rows = [item["meta"] for item in data_by_symbol.values()] + failures
    fetch_cache_hit_count = sum(1 for item in fetch_meta_rows if item.get("from_cache"))
    fetch_cache_miss_count = max(0, len(fetch_meta_rows) - fetch_cache_hit_count)
    profit_factor = money.get("profit_loss_ratio")
    run_datetime = now_jst_iso()
    replay_run_id = f"watchlist-scan-{run_datetime}"
    summary = {
        "replay_run_id": replay_run_id,
        "run_datetime": run_datetime,
        "mode": WATCHLIST_REPLAY_MODE,
        "replay_mode": "watchlist_scan",
        "candidate_generation_mode": config.candidate_generation_mode,
        "symbols_count": len(records),
        "fetch_success_count": len(data_by_symbol),
        "fetch_failed_count": len(failures),
        "period": config.period,
        "interval": config.interval,
        "scan_interval": config.scan_interval,
        "start_at": _timestamp_text(config.start_at),
        "end_at": _timestamp_text(config.end_at),
        "min_score": config.min_score,
        "target_mode": config.target_mode,
        "max_buys_per_scan": config.max_buys_per_scan,
        "cooldown": config.cooldown,
        "shares": config.shares,
        "use_multi_timeframe": config.use_multi_timeframe,
        "daily_min_ok": config.daily_min_ok,
        "intraday_min_ok": config.intraday_min_ok,
        "use_vwap": config.use_vwap,
        "use_volume_spike": config.use_volume_spike,
        "use_risk_filter": config.use_risk_filter,
        "max_stop_loss_pct": config.max_stop_loss_pct,
        "max_loss_yen_limit": config.max_loss_yen_limit,
        "min_risk_reward": config.min_risk_reward,
        "use_daily_top_n": config.use_daily_top_n,
        "daily_top_n": config.daily_top_n,
        "min_daily_score": config.min_daily_score,
        "use_technical_score": config.use_technical_score,
        "technical_preset": config.technical_preset,
        "technical_min_score": config.technical_min_score,
        "technical_min_confidence": config.technical_min_confidence,
        "technical_config": config.technical_config or get_replay_technical_config(),
        "use_replay_cache": config.use_replay_cache,
        "parallel": config.parallel,
        "max_workers": config.max_workers,
        "one_position_per_symbol": config.one_position_per_symbol,
        "watchlist_limit": config.max_symbols,
        "max_virtual_buys_per_scan": config.max_buys_per_scan,
        "risk_filter_settings": {
            "use_risk_filter": config.use_risk_filter,
            "max_stop_loss_pct": config.max_stop_loss_pct,
            "max_loss_yen_limit": config.max_loss_yen_limit,
            "min_risk_reward": config.min_risk_reward,
        },
        "multi_timeframe_settings": {
            "use_multi_timeframe": config.use_multi_timeframe,
            "daily_filter_required": config.daily_filter_required,
            "daily_min_ok": config.daily_min_ok,
            "intraday_min_ok": config.intraday_min_ok,
            "use_vwap": config.use_vwap,
            "use_volume_spike": config.use_volume_spike,
        },
        "total_scan_steps": total_scan_steps,
        "scan_count": stage_counts.get("scan_count", total_scan_steps),
        "total_candidates": total_candidates,
        "total_trades": len(trades),
        "existing_logic_checked_count": stage_counts.get("existing_logic_checked_count", 0),
        "existing_logic_pass_count": stage_counts.get("existing_logic_pass_count", 0),
        "existing_logic_reject_count": stage_counts.get("existing_logic_reject_count", 0),
        "technical_score_checked_count": stage_counts.get("technical_score_checked_count", 0),
        "technical_score_pass_count": stage_counts.get("technical_score_pass_count", 0),
        "technical_score_reject_count": stage_counts.get("technical_score_reject_count", 0),
        "hard_filter_reject_count": stage_counts.get("hard_filter_reject_count", 0),
        "raw_signal_count": stage_counts.get("raw_signal_count", 0),
        "cooldown_filtered_count": stage_counts.get("cooldown_filtered_count", 0),
        "position_filtered_count": stage_counts.get("position_open_excluded_count", 0),
        "max_trade_filtered_count": stage_counts.get("max_trades_excluded_count", 0),
        "final_trade_count": len(trades),
        "settled_trade_count": settled_trade_count,
        "csv_export_count": len(trades),
        "db_saved_count": 0,
        "stage_counts": stage_counts,
        "zero_trade_reasons": replay_zero_trade_reasons(stage_counts, config.candidate_generation_mode),
        "fetch_seconds": fetch_seconds,
        "feature_seconds": 0.0,
        "scan_seconds": scan_seconds,
        "summary_seconds": summary_seconds,
        "total_seconds": total_seconds,
        "cache_hit_count": int(stage_counts.get("cache_hit_count", 0) or 0) + fetch_cache_hit_count,
        "cache_miss_count": int(stage_counts.get("cache_miss_count", 0) or 0) + fetch_cache_miss_count,
        "technical_cache_hit_count": stage_counts.get("cache_hit_count", 0),
        "technical_cache_miss_count": stage_counts.get("cache_miss_count", 0),
        "data_cache_hit_count": fetch_cache_hit_count,
        "data_cache_miss_count": fetch_cache_miss_count,
        "avg_seconds_per_scan": round(scan_seconds / max(1, total_scan_steps), 4),
        "win_rate_pct": performance.get("win_rate_pct", 0),
        "profit_factor": profit_factor,
        "error_count": len(failures),
        **money,
    }
    for trade in trades:
        trade["replay_run_id"] = replay_run_id

    result = {
        "summary": summary,
        "performance_summary": performance,
        "money_summary": money,
        "trades": trades,
        "symbol_summary": symbol_summary,
        "fetch_failures": failures,
        "fetch_meta": [item["meta"] for item in data_by_symbol.values()],
        "selected_by_scan_time": selected_by_scan_time,
        "lookahead_note": "各スキャン時刻では、その時刻以前のOHLCVだけを買い判定に使っています。未来データは結果検証だけに使用します。",
    }
    return attach_watchlist_replay_exports(result)
