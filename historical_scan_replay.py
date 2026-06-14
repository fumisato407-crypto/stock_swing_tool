from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from daily_top_n_filter import build_buy_condition_json, daily_score_from_filter, rank_daily_candidates
from data_fetcher import normalize_jp_symbol
from historical_data import load_or_fetch_historical_data, validate_historical_ohlcv
from multi_timeframe_rules import build_replay_daily_context, evaluate_daily_filter
from replay_engine import (
    ReplayRuleConfig,
    apply_replay_money_metrics,
    create_replay_trade,
    evaluate_replay_step,
    evaluate_replay_trade_outcome,
    summarize_replay_money,
    summarize_replay_results,
)
from time_utils import now_jst_iso


ProgressCallback = Callable[[str, int, int, str], None]


@dataclass
class HistoricalScanReplayConfig:
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


def _fetch_watchlist_data(
    records: List[Dict[str, Any]],
    config: HistoricalScanReplayConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    data_by_symbol: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        symbol = str(record.get("normalized_symbol", ""))
        name = str(record.get("name", ""))
        if progress_callback:
            progress_callback("データ取得中", index, total, f"{symbol} {name}".strip())
        fetched = load_or_fetch_historical_data(symbol, config.period, config.interval)
        validation = validate_historical_ohlcv(fetched.data)
        cleaned = validation.get("cleaned_data", pd.DataFrame())
        daily_cleaned = pd.DataFrame()
        daily_error = ""
        if config.use_multi_timeframe:
            daily_fetched = load_or_fetch_historical_data(symbol, "6mo", "1d")
            daily_validation = validate_historical_ohlcv(daily_fetched.data)
            daily_cleaned = daily_validation.get("cleaned_data", pd.DataFrame())
            daily_error = daily_fetched.error_message or ""
        meta = {
            "symbol": symbol,
            "name": name,
            "fetched_rows": fetched.fetched_rows,
            "cleaned_rows": len(cleaned),
            "daily_rows": len(daily_cleaned),
            "from_cache": fetched.from_cache,
            "error_type": fetched.error_type,
            "error_message": fetched.error_message,
            "daily_error": daily_error,
            "excluded_count": validation.get("excluded_count", 0),
            "first_timestamp": fetched.first_timestamp,
            "last_timestamp": fetched.last_timestamp,
        }
        if fetched.error_message or cleaned.empty:
            failures.append(meta)
            continue
        data_by_symbol[symbol] = {
            "record": record,
            "name": name,
            "data": cleaned,
            "daily_data": daily_cleaned,
            "meta": meta,
            "validation": {key: value for key, value in validation.items() if key != "cleaned_data"},
        }
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


def run_historical_scan_replay(
    watchlist_records: List[Dict[str, Any]],
    config: HistoricalScanReplayConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    records = _records_from_watchlist(watchlist_records, config.max_symbols)
    data_by_symbol, failures = _fetch_watchlist_data(records, config, progress_callback)
    scan_times = _build_scan_times(data_by_symbol, config)
    total_scan_steps = len(scan_times)
    trades: List[Dict[str, Any]] = []
    last_signal_by_symbol: Dict[str, pd.Timestamp] = {}
    total_candidates = 0
    selected_by_scan_time: List[Dict[str, Any]] = []

    replay_config = ReplayRuleConfig(
        min_score=config.min_score,
        target_rule="すべて",
        max_trades=config.max_buys_per_scan,
        interval=config.interval,
        debug=True,
        use_multi_timeframe=config.use_multi_timeframe,
        daily_min_ok=config.daily_min_ok if config.daily_filter_required else 0,
        intraday_min_ok=config.intraday_min_ok,
        use_vwap=config.use_vwap,
        use_volume_spike=config.use_volume_spike,
        shares=config.shares,
        use_risk_filter=config.use_risk_filter,
        max_stop_loss_pct=config.max_stop_loss_pct,
        max_loss_yen_limit=config.max_loss_yen_limit,
        min_risk_reward=config.min_risk_reward,
    )

    for step_index, scan_time in enumerate(scan_times, start=1):
        if progress_callback:
            progress_callback("スキャン再現中", step_index, max(1, total_scan_steps), _timestamp_text(scan_time))
        candidates: List[Dict[str, Any]] = []
        daily_rank_info = _daily_rank_info_for_scan_time(data_by_symbol, scan_time, config)
        top_n_pool_count = sum(1 for item in daily_rank_info.values() if item.get("daily_top_n_pass"))
        for symbol, item in data_by_symbol.items():
            df: pd.DataFrame = item["data"]
            if scan_time not in df.index:
                continue
            daily_meta = daily_rank_info.get(symbol, {})
            if config.use_daily_top_n and not daily_meta.get("daily_top_n_pass"):
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
            if not _target_allows(category, config.target_mode):
                continue
            if not _cooldown_allows(last_signal_by_symbol.get(symbol), scan_time, config.cooldown):
                continue
            signal["category"] = category
            signal["symbol"] = symbol
            signal["name"] = str(item.get("name", ""))
            signal["scan_time"] = _timestamp_text(scan_time)
            signal["source_watchlist"] = config.source_watchlist
            candidates.append(signal)

        total_candidates += len(candidates)
        selected = sorted(candidates, key=lambda signal: _score(signal), reverse=True)[: int(config.max_buys_per_scan)]
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
            last_signal_by_symbol[symbol] = scan_time

    if progress_callback:
        progress_callback("結果集計中", 1, 1, "100株想定損益を集計しています")

    trades = apply_replay_money_metrics(trades, config.shares)
    performance = summarize_replay_results(trades)
    money = summarize_replay_money(trades, config.shares)
    symbol_summary = _symbol_trade_summary(trades, data_by_symbol, failures, config.shares)
    profit_factor = money.get("profit_loss_ratio")
    replay_run_id = f"watchlist-scan-{now_jst_iso()}"
    summary = {
        "replay_run_id": replay_run_id,
        "replay_mode": "watchlist_scan",
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
        "total_candidates": total_candidates,
        "total_trades": len(trades),
        "win_rate_pct": performance.get("win_rate_pct", 0),
        "profit_factor": profit_factor,
        "error_count": len(failures),
        **money,
    }
    for trade in trades:
        trade["replay_run_id"] = replay_run_id

    return {
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
