from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from daily_technical import get_technical_config_for_preset
from data_fetcher import normalize_jp_symbol
from historical_data import load_or_fetch_historical_data, validate_historical_ohlcv
from replay_cache import TECHNICAL_SCORE_FEATURE_VERSION, replay_cache_stats, reset_replay_cache_stats, stable_config_hash
from replay_engine import (
    CANDIDATE_MODE_EXISTING,
    CANDIDATE_MODE_TECHNICAL_ONLY,
    apply_replay_money_metrics,
    run_replay,
    summarize_replay_money,
    summarize_replay_results,
)
from time_utils import now_jst_iso


ProgressCallback = Callable[[str, int, int, str], None]
APP_VERSION = "stock_swing_tool_replay_v1"
COUNT_KEYS = [
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
]


@dataclass
class ConditionVariant:
    name: str
    candidate_generation_mode: str = CANDIDATE_MODE_TECHNICAL_ONLY
    technical_preset: str = "standard_swing"
    technical_min_score: int = 32
    use_penalty: bool = True
    use_hard_filter: bool = True

    def technical_config(self) -> Dict[str, Any]:
        config = get_technical_config_for_preset(self.technical_preset)
        config["use_penalty"] = bool(self.use_penalty)
        config["use_hard_filter"] = bool(self.use_hard_filter)
        return config

    def config_hash(self) -> str:
        return stable_config_hash(
            {
                "candidate_generation_mode": self.candidate_generation_mode,
                "technical_preset": self.technical_preset,
                "technical_min_score": self.technical_min_score,
                "use_penalty": self.use_penalty,
                "use_hard_filter": self.use_hard_filter,
            }
        )


@dataclass
class BatchReplayConfig:
    period: str = "1mo"
    interval: str = "5m"
    start_at: Any = None
    end_at: Any = None
    max_trades: int = 50
    cooldown_bars: int = 12
    shares: int = 100
    use_replay_cache: bool = True
    parallel: bool = False
    max_workers: int = 3
    one_position_per_symbol: bool = False
    variants: List[ConditionVariant] = field(default_factory=list)
    target_symbols: str = ""
    excluded_symbols: str = ""
    watchlist_limit: Any = ""
    start_date: str = ""
    end_date: str = ""
    start_time: str = ""
    end_time: str = ""
    use_risk_condition: bool = False
    use_vwap_condition: bool = False
    use_volume_spike_condition: bool = False
    use_multi_timeframe: bool = False
    use_daily_filter: bool = False
    use_daily_top_n: bool = False
    daily_score_min: Any = ""
    feature_version: str = TECHNICAL_SCORE_FEATURE_VERSION
    app_version: str = APP_VERSION


def build_condition_variants(
    modes: List[str],
    presets: List[str],
    min_scores: List[int],
    hard_filter_options: List[bool],
    use_penalty: bool = True,
) -> List[ConditionVariant]:
    variants: List[ConditionVariant] = []
    for mode in modes:
        for preset in presets:
            for min_score in min_scores:
                for use_hard_filter in hard_filter_options:
                    hard_label = "強制ON" if use_hard_filter else "強制OFF"
                    variants.append(
                        ConditionVariant(
                            name=f"{mode}|{preset}|{min_score}点|{hard_label}",
                            candidate_generation_mode=mode,
                            technical_preset=preset,
                            technical_min_score=int(min_score),
                            use_penalty=bool(use_penalty),
                            use_hard_filter=bool(use_hard_filter),
                        )
                    )
    return variants


def _record_symbol(record: Dict[str, Any]) -> Dict[str, str]:
    symbol = normalize_jp_symbol(record.get("code") or record.get("symbol") or record.get("ticker"))
    return {
        "symbol": symbol or str(record.get("code") or ""),
        "name": str(record.get("name") or record.get("銘柄名") or ""),
    }


def _pct_count(trades: List[Dict[str, Any]], key: str) -> float:
    if not trades:
        return 0.0
    return round(sum(1 for trade in trades if trade.get(key)) / len(trades) * 100, 1)


def _avg(values: List[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _most_common(values: List[Any]) -> str:
    counts: Dict[str, int] = {}
    for value in values:
        text = str(value or "")
        if text:
            counts[text] = counts.get(text, 0) + 1
    if not counts:
        return "-"
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[0][0]


def _closed_trade_count(trades: List[Dict[str, Any]]) -> int:
    return sum(1 for trade in trades if trade.get("status") == "closed" and trade.get("exit_price") is not None)


def _count_fields(trades: List[Dict[str, Any]], stage_counts: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    stage = dict(stage_counts or {})
    final_count = len(trades)
    settled_count = _closed_trade_count(trades)
    return {
        "scan_count": int(stage.get("scan_count", stage.get("scan_total_count", 0)) or 0),
        "existing_logic_checked_count": int(stage.get("existing_logic_checked_count", 0) or 0),
        "existing_logic_pass_count": int(stage.get("existing_logic_pass_count", 0) or 0),
        "existing_logic_reject_count": int(stage.get("existing_logic_reject_count", 0) or 0),
        "technical_score_checked_count": int(stage.get("technical_score_checked_count", stage.get("technical_score_attempt_count", 0)) or 0),
        "technical_score_pass_count": int(stage.get("technical_score_pass_count", stage.get("technical_min_score_pass_count", 0)) or 0),
        "technical_score_reject_count": int(stage.get("technical_score_reject_count", 0) or 0),
        "hard_filter_reject_count": int(stage.get("hard_filter_reject_count", stage.get("technical_hard_filter_excluded_count", 0)) or 0),
        "raw_signal_count": int(stage.get("raw_signal_count", final_count) or 0),
        "cooldown_filtered_count": int(stage.get("cooldown_filtered_count", 0) or 0),
        "position_filtered_count": int(stage.get("position_filtered_count", stage.get("position_open_excluded_count", 0)) or 0),
        "max_trade_filtered_count": int(stage.get("max_trade_filtered_count", stage.get("max_trades_excluded_count", 0)) or 0),
        "final_trade_count": final_count,
        "settled_trade_count": int(stage.get("settled_trade_count", settled_count) or 0),
        "db_saved_count": int(stage.get("db_saved_count", 0) or 0),
        "csv_export_count": final_count,
        "cache_hit_count": int(stage.get("cache_hit_count", 0) or 0),
        "cache_miss_count": int(stage.get("cache_miss_count", 0) or 0),
        "leak_check_ok_count": int(stage.get("leak_check_ok_count", sum(1 for trade in trades if trade.get("leak_check_result") == "OK")) or 0),
        "leak_check_ng_count": int(stage.get("leak_check_ng_count", sum(1 for trade in trades if trade.get("leak_check_result") == "NG")) or 0),
    }


def _sum_count_fields(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    output: Dict[str, int] = {}
    for key in COUNT_KEYS + ["cache_hit_count", "cache_miss_count", "leak_check_ok_count", "leak_check_ng_count"]:
        output[key] = sum(int(row.get(key, 0) or 0) for row in rows)
    return output


def _summary_row(
    symbol: str,
    name: str,
    condition_name: str,
    trades: List[Dict[str, Any]],
    elapsed: float,
    stage_counts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    performance = summarize_replay_results(trades)
    money = summarize_replay_money(trades, int(trades[0].get("shares", 100)) if trades else 100)
    technical_scores = [
        float(trade.get("technical_score_final"))
        for trade in trades
        if trade.get("technical_score_final") not in (None, "")
    ]
    counts = _count_fields(trades, stage_counts)
    profit_factor = money.get("profit_loss_ratio")
    return {
        "code": symbol,
        "name": name,
        "condition_name": condition_name,
        **counts,
        "win_rate": performance.get("win_rate_pct", 0),
        "total_profit": money.get("gross_profit_yen", 0),
        "total_loss": money.get("gross_loss_yen", 0),
        "net_profit": money.get("net_profit_yen", 0),
        "average_profit": money.get("average_profit_yen", 0),
        "average_loss": money.get("average_loss_yen", 0),
        "profit_factor": profit_factor,
        "max_win": money.get("max_profit_yen", 0),
        "max_loss": money.get("max_loss_yen", 0),
        "max_win_streak": money.get("max_win_streak", 0),
        "max_loss_streak": money.get("max_loss_streak", 0),
        "processing_seconds": round(elapsed, 3),
        "仮想買い件数": len(trades),
        "勝率": performance.get("win_rate_pct", 0),
        "純損益": money.get("net_profit_yen", 0),
        "平均損益": money.get("average_profit_yen", 0),
        "最大利益": money.get("max_profit_yen", 0),
        "最大損失": money.get("max_loss_yen", 0),
        "最大連勝": money.get("max_win_streak", 0),
        "最大連敗": money.get("max_loss_streak", 0),
        "利確到達率": _pct_count(trades, "hit_take_profit"),
        "損切り到達率": _pct_count(trades, "hit_stop_loss"),
        "期限到達率": round(sum(1 for trade in trades if trade.get("outcome") == "timeout") / len(trades) * 100, 1) if trades else 0.0,
        "平均最大上昇率": performance.get("avg_max_profit_pct", 0),
        "平均最大下落率": performance.get("avg_max_drawdown_pct", 0),
        "平均technical_score_final": _avg(technical_scores),
        "最頻technical_judgement": _most_common([trade.get("technical_judgement") for trade in trades]),
        "leak_check_ok": sum(1 for trade in trades if trade.get("leak_check_result") == "OK"),
        "leak_check_ng": sum(1 for trade in trades if trade.get("leak_check_result") == "NG"),
        "処理秒数": round(elapsed, 3),
    }


def _run_symbol(record: Dict[str, Any], config: BatchReplayConfig) -> Dict[str, Any]:
    symbol_record = _record_symbol(record)
    symbol = symbol_record["symbol"]
    name = symbol_record["name"]
    started = time.perf_counter()
    try:
        intraday_result = load_or_fetch_historical_data(symbol, config.period, config.interval)
        daily_result = load_or_fetch_historical_data(symbol, "6mo", "1d")
        intraday_validation = validate_historical_ohlcv(intraday_result.data)
        daily_validation = validate_historical_ohlcv(daily_result.data)
        intraday = intraday_validation.get("cleaned_data", intraday_result.data)
        daily = daily_validation.get("cleaned_data", daily_result.data)
        if intraday is None or intraday.empty:
            return {
                "symbol": symbol,
                "name": name,
                "success": False,
                "error": intraday_result.error_message or "5分足データが空です",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "trades": [],
                "symbol_rows": [],
            }
        symbol_rows: List[Dict[str, Any]] = []
        all_trades: List[Dict[str, Any]] = []
        variants = config.variants or [ConditionVariant(name="default")]
        for variant in variants:
            variant_started = time.perf_counter()
            variant_uses_technical = variant.candidate_generation_mode != CANDIDATE_MODE_EXISTING
            result = run_replay(
                symbol=symbol,
                name=name,
                df=intraday,
                daily_df=daily,
                start_at=config.start_at,
                end_at=config.end_at,
                rule_config={
                    "candidate_generation_mode": variant.candidate_generation_mode,
                    "use_technical_score": variant_uses_technical,
                    "technical_preset": variant.technical_preset,
                    "technical_min_score": variant.technical_min_score,
                    "technical_config": variant.technical_config(),
                    "max_trades": config.max_trades,
                    "cooldown_bars": config.cooldown_bars,
                    "interval": config.interval,
                    "shares": config.shares,
                    "use_replay_cache": config.use_replay_cache,
                    "one_position_per_symbol": config.one_position_per_symbol,
                    "debug": False,
                },
            )
            trades = apply_replay_money_metrics(result.get("trades", []), config.shares)
            variant_hash = variant.config_hash()
            stage_counts = (result.get("summary") or {}).get("stage_counts", {})
            for trade in trades:
                trade["condition_name"] = variant.name
                trade["config_hash"] = variant_hash
                trade["candidate_mode"] = variant.candidate_generation_mode
                trade["preset"] = variant.technical_preset
                trade["min_technical_score"] = variant.technical_min_score
                trade["use_technical_score"] = variant_uses_technical
                trade["use_penalty"] = bool(variant.use_penalty)
                trade["use_hard_filter"] = bool(variant.use_hard_filter)
                trade["hard_filter_mode"] = "hard" if variant.use_hard_filter else "OFF"
                trade["symbol"] = symbol
                trade["name"] = name
            all_trades.extend(trades)
            row = _summary_row(
                symbol,
                name,
                variant.name,
                trades,
                time.perf_counter() - variant_started,
                stage_counts=stage_counts,
            )
            row.update(
                {
                    "candidate_mode": variant.candidate_generation_mode,
                    "preset": variant.technical_preset,
                    "min_technical_score": variant.technical_min_score,
                    "use_technical_score": variant_uses_technical,
                    "use_penalty": variant.use_penalty,
                    "use_hard_filter": variant.use_hard_filter,
                    "hard_filter_mode": "hard" if variant.use_hard_filter else "OFF",
                    "config_hash": variant_hash,
                }
            )
            symbol_rows.append(row)
        return {
            "symbol": symbol,
            "name": name,
            "success": True,
            "error": "",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "from_cache": bool(intraday_result.from_cache),
            "intraday_rows": len(intraday),
            "daily_rows": len(daily) if daily is not None else 0,
            "trades": all_trades,
            "symbol_rows": symbol_rows,
        }
    except Exception as exc:
        return {
            "symbol": symbol,
            "name": name,
            "success": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "trades": [],
            "symbol_rows": [],
        }


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def build_run_context(records: List[Dict[str, Any]], config: BatchReplayConfig) -> Dict[str, Any]:
    symbols = [_record_symbol(record).get("symbol", "") for record in records]
    run_datetime = now_jst_iso()
    return {
        "run_id": f"batch-{run_datetime}-{uuid.uuid4().hex[:8]}",
        "run_datetime": run_datetime,
        "mode": "batch_replay",
        "target_symbols": config.target_symbols or ",".join(symbol for symbol in symbols if symbol),
        "excluded_symbols": config.excluded_symbols,
        "watchlist_limit": config.watchlist_limit,
        "period": config.period,
        "interval": config.interval,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "start_time": config.start_time,
        "end_time": config.end_time,
        "assumed_shares": config.shares,
        "max_trades_per_symbol": config.max_trades,
        "cooldown_bars_or_minutes": config.cooldown_bars,
        "one_position_per_symbol": config.one_position_per_symbol,
        "cache_enabled": config.use_replay_cache,
        "parallel_enabled": config.parallel,
        "workers": config.max_workers,
        "use_risk_condition": config.use_risk_condition,
        "use_vwap_condition": config.use_vwap_condition,
        "use_volume_spike_condition": config.use_volume_spike_condition,
        "use_multi_timeframe": config.use_multi_timeframe,
        "use_daily_filter": config.use_daily_filter,
        "use_daily_top_n": config.use_daily_top_n,
        "daily_score_min": config.daily_score_min,
        "feature_version": config.feature_version,
        "app_version": config.app_version,
    }


def build_config_hash(context: Dict[str, Any], variant: ConditionVariant | Dict[str, Any]) -> str:
    if isinstance(variant, ConditionVariant):
        variant_values = {
            "candidate_mode": variant.candidate_generation_mode,
            "preset": variant.technical_preset,
            "min_technical_score": variant.technical_min_score,
            "use_penalty": variant.use_penalty,
            "use_hard_filter": variant.use_hard_filter,
        }
    else:
        variant_values = {
            "candidate_mode": variant.get("candidate_mode") or variant.get("candidate_generation_mode"),
            "preset": variant.get("preset") or variant.get("technical_preset"),
            "min_technical_score": variant.get("min_technical_score") or variant.get("technical_min_score"),
            "use_penalty": variant.get("use_penalty"),
            "use_hard_filter": variant.get("use_hard_filter"),
        }
    hash_context = {
        key: value
        for key, value in context.items()
        if key not in {"run_id", "run_datetime"}
    }
    hash_context.update(variant_values)
    return stable_config_hash(hash_context)


def _trade_export_row(trade: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": context.get("run_id", ""),
        "run_datetime": context.get("run_datetime", ""),
        "mode": context.get("mode", ""),
        "candidate_mode": trade.get("candidate_mode") or trade.get("candidate_generation_mode", ""),
        "preset": trade.get("preset") or trade.get("technical_preset", ""),
        "min_technical_score": trade.get("min_technical_score", ""),
        "use_technical_score": trade.get("use_technical_score", True),
        "use_penalty": trade.get("use_penalty", ""),
        "use_hard_filter": trade.get("use_hard_filter", ""),
        "hard_filter_mode": trade.get("hard_filter_mode", ""),
        "use_risk_condition": context.get("use_risk_condition", ""),
        "use_vwap_condition": context.get("use_vwap_condition", ""),
        "use_volume_spike_condition": context.get("use_volume_spike_condition", ""),
        "use_multi_timeframe": context.get("use_multi_timeframe", ""),
        "use_daily_filter": context.get("use_daily_filter", ""),
        "use_daily_top_n": context.get("use_daily_top_n", ""),
        "daily_score_min": context.get("daily_score_min", ""),
        "target_symbols": context.get("target_symbols", ""),
        "excluded_symbols": context.get("excluded_symbols", ""),
        "watchlist_limit": context.get("watchlist_limit", ""),
        "period": context.get("period", ""),
        "interval": context.get("interval", ""),
        "start_date": context.get("start_date", ""),
        "end_date": context.get("end_date", ""),
        "start_time": context.get("start_time", ""),
        "end_time": context.get("end_time", ""),
        "assumed_shares": context.get("assumed_shares", ""),
        "max_trades_per_symbol": context.get("max_trades_per_symbol", ""),
        "cooldown_bars_or_minutes": context.get("cooldown_bars_or_minutes", ""),
        "one_position_per_symbol": context.get("one_position_per_symbol", ""),
        "cache_enabled": context.get("cache_enabled", ""),
        "parallel_enabled": context.get("parallel_enabled", ""),
        "workers": context.get("workers", ""),
        "cache_hit_count": context.get("cache_hit_count", ""),
        "cache_miss_count": context.get("cache_miss_count", ""),
        "processing_seconds": context.get("processing_seconds", ""),
        "existing_logic_checked_count": context.get("existing_logic_checked_count", ""),
        "existing_logic_pass_count": context.get("existing_logic_pass_count", ""),
        "existing_logic_reject_count": context.get("existing_logic_reject_count", ""),
        "technical_score_checked_count": context.get("technical_score_checked_count", ""),
        "technical_score_pass_count": context.get("technical_score_pass_count", ""),
        "technical_score_reject_count": context.get("technical_score_reject_count", ""),
        "hard_filter_reject_count": context.get("hard_filter_reject_count", ""),
        "config_hash": trade.get("config_hash", ""),
        "feature_version": context.get("feature_version", ""),
        "app_version": context.get("app_version", ""),
        "symbol": trade.get("symbol", ""),
        "name": trade.get("name", ""),
        "condition_name": trade.get("condition_name", ""),
        "signal_time": trade.get("signal_time", ""),
        "entry_type": trade.get("entry_type", ""),
        "entry_price": trade.get("entry_price"),
        "stop_loss": trade.get("stop_loss"),
        "take_profit": trade.get("take_profit"),
        "status": trade.get("status", ""),
        "outcome": trade.get("outcome", ""),
        "exit_time": trade.get("exit_time", ""),
        "exit_price": trade.get("exit_price"),
        "return_pct": trade.get("return_pct"),
        "max_profit_pct": trade.get("max_profit_pct"),
        "max_drawdown_pct": trade.get("max_drawdown_pct"),
        "shares": trade.get("shares"),
        "required_capital_yen": trade.get("required_capital_yen"),
        "profit_yen": trade.get("profit_yen"),
        "cumulative_profit_yen": trade.get("cumulative_profit_yen"),
        "technical_score_final": trade.get("technical_score_final"),
        "technical_score_raw": trade.get("technical_score_raw"),
        "technical_penalty_score": trade.get("technical_penalty_score"),
        "technical_confidence": trade.get("technical_confidence"),
        "leak_check_result": trade.get("leak_check_result", ""),
    }


TRADE_DETAIL_EXPORT_COLUMNS = list(_trade_export_row({}, {}).keys())


def build_trade_detail_export(trades: List[Dict[str, Any]], context: Dict[str, Any]) -> pd.DataFrame:
    rows = [_trade_export_row(trade, context) for trade in trades]
    return pd.DataFrame(rows, columns=TRADE_DETAIL_EXPORT_COLUMNS)


def build_run_summary_export(
    condition_rows: pd.DataFrame,
    context: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    source_rows = condition_rows.to_dict("records") if isinstance(condition_rows, pd.DataFrame) and not condition_rows.empty else [summary or {}]
    for row in source_rows:
        rows.append(
            {
                "run_id": context.get("run_id", ""),
                "run_datetime": context.get("run_datetime", ""),
                "mode": context.get("mode", ""),
                "candidate_mode": row.get("candidate_mode", ""),
                "preset": row.get("preset", ""),
                "min_technical_score": row.get("min_technical_score", ""),
                "use_penalty": row.get("use_penalty", ""),
                "use_hard_filter": row.get("use_hard_filter", ""),
                "one_position_per_symbol": context.get("one_position_per_symbol", ""),
                "cache_enabled": context.get("cache_enabled", ""),
                "parallel_enabled": context.get("parallel_enabled", ""),
                "workers": context.get("workers", ""),
                "period": context.get("period", ""),
                "interval": context.get("interval", ""),
                "target_symbols": context.get("target_symbols", ""),
                "condition_name": row.get("condition_name", ""),
                "scan_count": row.get("scan_count", 0),
                "existing_logic_checked_count": row.get("existing_logic_checked_count", 0),
                "existing_logic_pass_count": row.get("existing_logic_pass_count", 0),
                "existing_logic_reject_count": row.get("existing_logic_reject_count", 0),
                "technical_score_checked_count": row.get("technical_score_checked_count", 0),
                "technical_score_pass_count": row.get("technical_score_pass_count", 0),
                "technical_score_reject_count": row.get("technical_score_reject_count", 0),
                "hard_filter_reject_count": row.get("hard_filter_reject_count", 0),
                "raw_signal_count": row.get("raw_signal_count", 0),
                "cooldown_filtered_count": row.get("cooldown_filtered_count", 0),
                "position_filtered_count": row.get("position_filtered_count", 0),
                "max_trade_filtered_count": row.get("max_trade_filtered_count", 0),
                "final_trade_count": row.get("final_trade_count", 0),
                "settled_trade_count": row.get("settled_trade_count", 0),
                "db_saved_count": row.get("db_saved_count", 0),
                "csv_export_count": row.get("csv_export_count", 0),
                "win_rate": row.get("win_rate", row.get("win_rate_pct", 0)),
                "total_profit": row.get("total_profit", 0),
                "total_loss": row.get("total_loss", 0),
                "net_profit": row.get("net_profit", 0),
                "average_profit": row.get("average_profit", 0),
                "average_loss": row.get("average_loss", 0),
                "profit_factor": row.get("profit_factor", ""),
                "max_win": row.get("max_win", 0),
                "max_loss": row.get("max_loss", 0),
                "max_win_streak": row.get("max_win_streak", 0),
                "max_loss_streak": row.get("max_loss_streak", 0),
                "processing_seconds": row.get("processing_seconds", context.get("processing_seconds", 0)),
                "cache_hit_count": row.get("cache_hit_count", 0),
                "cache_miss_count": row.get("cache_miss_count", 0),
                "leak_check_ok_count": row.get("leak_check_ok_count", row.get("leak_check_ok", 0)),
                "leak_check_ng_count": row.get("leak_check_ng_count", row.get("leak_check_ng", 0)),
                "config_hash": row.get("config_hash", ""),
                "feature_version": context.get("feature_version", ""),
                "app_version": context.get("app_version", ""),
            }
        )
    return pd.DataFrame(rows)


def run_multi_symbol_replay_comparison(
    records: List[Dict[str, Any]],
    config: BatchReplayConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    reset_replay_cache_stats()
    started = time.perf_counter()
    run_context = build_run_context(records, config)
    results: List[Dict[str, Any]] = []
    total = len(records)
    if config.parallel and total > 1:
        with ThreadPoolExecutor(max_workers=max(1, min(int(config.max_workers or 1), 8))) as executor:
            futures = {executor.submit(_run_symbol, record, config): record for record in records}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if progress_callback:
                    progress_callback("複数銘柄比較中", index, total, result.get("symbol", ""))
    else:
        for index, record in enumerate(records, start=1):
            result = _run_symbol(record, config)
            results.append(result)
            if progress_callback:
                progress_callback("複数銘柄比較中", index, total, result.get("symbol", ""))

    all_trades = [trade for result in results for trade in result.get("trades", [])]
    symbol_rows = [row for result in results for row in result.get("symbol_rows", [])]
    variants = config.variants or [ConditionVariant(name="default")]
    variant_by_name = {variant.name: variant for variant in variants}
    for trade in all_trades:
        variant = variant_by_name.get(str(trade.get("condition_name", "")))
        trade["run_id"] = run_context["run_id"]
        trade["run_datetime"] = run_context["run_datetime"]
        if variant:
            trade["config_hash"] = build_config_hash(run_context, variant)
        else:
            trade["config_hash"] = build_config_hash(run_context, trade)
    condition_rows: List[Dict[str, Any]] = []
    for condition_name in sorted({row.get("condition_name") for row in symbol_rows if row.get("condition_name")}):
        scoped_trades = [trade for trade in all_trades if trade.get("condition_name") == condition_name]
        scoped_symbol_rows = [row for row in symbol_rows if row.get("condition_name") == condition_name]
        aggregate_counts = _sum_count_fields(scoped_symbol_rows)
        row = _summary_row("ALL", "", str(condition_name), scoped_trades, 0.0, stage_counts=aggregate_counts)
        variant = variant_by_name.get(str(condition_name))
        if variant:
            row.update(
                {
                    "candidate_mode": variant.candidate_generation_mode,
                    "preset": variant.technical_preset,
                    "min_technical_score": variant.technical_min_score,
                    "use_technical_score": True,
                    "use_penalty": variant.use_penalty,
                    "use_hard_filter": variant.use_hard_filter,
                    "hard_filter_mode": "hard" if variant.use_hard_filter else "OFF",
                    "config_hash": build_config_hash(run_context, variant),
                }
            )
        condition_rows.append(row)

    elapsed = round(time.perf_counter() - started, 3)
    cache = replay_cache_stats()
    aggregate_counts = _sum_count_fields(symbol_rows)
    aggregate_counts["final_trade_count"] = len(all_trades)
    aggregate_counts["csv_export_count"] = len(all_trades)
    aggregate_counts["settled_trade_count"] = _closed_trade_count(all_trades)
    summary = {
        "symbols_count": total,
        "success_count": sum(1 for item in results if item.get("success")),
        "error_count": sum(1 for item in results if not item.get("success")),
        "total_trades": len(all_trades),
        "elapsed_seconds": elapsed,
        "cache": cache,
        "one_position_per_symbol": config.one_position_per_symbol,
        "parallel": config.parallel,
        "max_workers": config.max_workers,
        **aggregate_counts,
    }
    run_context["processing_seconds"] = elapsed
    run_context.update(aggregate_counts)
    condition_df = pd.DataFrame(condition_rows)
    detail_df = build_trade_detail_export(all_trades, run_context)
    summary["csv_export_count"] = len(detail_df)
    run_summary_df = build_run_summary_export(condition_df, run_context, summary)
    return {
        "summary": summary,
        "run_context": run_context,
        "symbol_rows": pd.DataFrame(symbol_rows),
        "condition_rows": condition_df,
        "trade_detail_df": detail_df,
        "run_summary_df": run_summary_df,
        "trades": all_trades,
        "errors": [item for item in results if not item.get("success")],
        "raw_results": results,
    }
