from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from daily_technical import get_replay_technical_config, score_technical_item
from entry_rules import evaluate_intraday_entry
from multi_timeframe_rules import build_replay_daily_context, evaluate_multi_timeframe_signal, evaluate_risk_filter
from replay_cache import (
    TECHNICAL_SCORE_FEATURE_VERSION,
    get_cached_technical_score,
    set_cached_technical_score,
    technical_score_cache_key,
)


INTERVAL_MAX_HOLD_BARS = {
    "5m": 390,
    "15m": 130,
    "60m": 30,
    "1h": 30,
    "1d": 5,
}
DEFAULT_REPLAY_SHARES = 100
CANDIDATE_MODE_EXISTING = "existing"
CANDIDATE_MODE_TECHNICAL_ONLY = "technical_only"
CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL = "existing_plus_technical"
CANDIDATE_MODE_LABELS = {
    CANDIDATE_MODE_EXISTING: "既存ロジック",
    CANDIDATE_MODE_TECHNICAL_ONLY: "テクニカルのみ",
    CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL: "既存ロジック＋テクニカル",
}
_CANDIDATE_MODE_ALIASES = {
    "既存ロジック": CANDIDATE_MODE_EXISTING,
    "テクニカルのみ": CANDIDATE_MODE_TECHNICAL_ONLY,
    "既存ロジック＋テクニカル": CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL,
    "existing": CANDIDATE_MODE_EXISTING,
    "technical_only": CANDIDATE_MODE_TECHNICAL_ONLY,
    "existing_plus_technical": CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL,
}


@dataclass
class ReplayRuleConfig:
    candidate_generation_mode: str = CANDIDATE_MODE_EXISTING
    min_score: int = 70
    target_rule: str = "すべて"
    max_trades: int = 10
    interval: str = "5m"
    cooldown_bars: int = 12
    debug: bool = False
    use_multi_timeframe: bool = True
    daily_min_ok: int = 3
    intraday_min_ok: int = 2
    use_vwap: bool = True
    use_volume_spike: bool = True
    shares: int = DEFAULT_REPLAY_SHARES
    use_risk_filter: bool = True
    max_stop_loss_pct: float = 3.0
    max_loss_yen_limit: float = 20000.0
    min_risk_reward: float = 1.2
    use_technical_score: bool = False
    technical_preset: str = "standard_swing"
    technical_min_score: int = 0
    technical_min_confidence: int = 0
    technical_show_breakdown: bool = True
    technical_config: Dict[str, Any] = field(default_factory=get_replay_technical_config)
    use_replay_cache: bool = True
    feature_version: str = TECHNICAL_SCORE_FEATURE_VERSION
    one_position_per_symbol: bool = False


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(price: Optional[float], entry_price: float) -> Optional[float]:
    if price is None or not entry_price:
        return None
    return round((float(price) / entry_price - 1) * 100, 2)


def _timestamp_text(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value or "")


def _normalize_rule_config(rule_config: Optional[Dict[str, Any] | ReplayRuleConfig]) -> ReplayRuleConfig:
    if isinstance(rule_config, ReplayRuleConfig):
        return rule_config
    values = dict(rule_config or {})
    def _int_setting(key: str, default: int) -> int:
        return default if values.get(key) is None else int(values.get(key))

    return ReplayRuleConfig(
        candidate_generation_mode=_normalize_candidate_generation_mode(
            values.get("candidate_generation_mode", CANDIDATE_MODE_EXISTING)
        ),
        min_score=int(values.get("min_score", 70) or 70),
        target_rule=str(values.get("target_rule", "すべて") or "すべて"),
        max_trades=int(values.get("max_trades", 10) or 10),
        interval=str(values.get("interval", "5m") or "5m"),
        cooldown_bars=int(values.get("cooldown_bars", 12) or 12),
        debug=bool(values.get("debug", False)),
        use_multi_timeframe=bool(values.get("use_multi_timeframe", True)),
        daily_min_ok=_int_setting("daily_min_ok", 3),
        intraday_min_ok=_int_setting("intraday_min_ok", 2),
        use_vwap=bool(values.get("use_vwap", True)),
        use_volume_spike=bool(values.get("use_volume_spike", True)),
        shares=_int_setting("shares", DEFAULT_REPLAY_SHARES),
        use_risk_filter=bool(values.get("use_risk_filter", True)),
        max_stop_loss_pct=float(values.get("max_stop_loss_pct", 3.0) or 3.0),
        max_loss_yen_limit=float(values.get("max_loss_yen_limit", 20000.0) or 20000.0),
        min_risk_reward=float(values.get("min_risk_reward", 1.2) or 1.2),
        use_technical_score=bool(values.get("use_technical_score", False)),
        technical_preset=str(values.get("technical_preset", "standard_swing") or "standard_swing"),
        technical_min_score=int(values.get("technical_min_score", 0) or 0),
        technical_min_confidence=int(values.get("technical_min_confidence", 0) or 0),
        technical_show_breakdown=bool(values.get("technical_show_breakdown", True)),
        technical_config=dict(values.get("technical_config") or get_replay_technical_config()),
        use_replay_cache=bool(values.get("use_replay_cache", True)),
        feature_version=str(values.get("feature_version", TECHNICAL_SCORE_FEATURE_VERSION) or TECHNICAL_SCORE_FEATURE_VERSION),
        one_position_per_symbol=bool(values.get("one_position_per_symbol", False)),
    )


def _normalize_candidate_generation_mode(value: Any) -> str:
    return _CANDIDATE_MODE_ALIASES.get(str(value or "").strip(), CANDIDATE_MODE_EXISTING)


def _multi_timeframe_config(config: ReplayRuleConfig) -> Dict[str, Any]:
    return {
        "daily_min_ok": config.daily_min_ok,
        "intraday_min_ok": config.intraday_min_ok,
        "use_vwap": config.use_vwap,
        "use_volume_spike": config.use_volume_spike,
    }


def _risk_filter_config(config: ReplayRuleConfig) -> Dict[str, Any]:
    return {
        "enabled": config.use_risk_filter,
        "max_stop_loss_pct": config.max_stop_loss_pct,
        "max_loss_yen_limit": config.max_loss_yen_limit,
        "min_risk_reward": config.min_risk_reward,
    }


def _max_hold_bars(interval: str) -> int:
    return INTERVAL_MAX_HOLD_BARS.get(str(interval), 30)


def _previous_session_low(history_df: pd.DataFrame, current_time: pd.Timestamp) -> Optional[float]:
    if history_df.empty:
        return None
    dates = pd.Series(history_df.index.date, index=history_df.index)
    current_date = current_time.date()
    previous_dates = [date for date in pd.unique(dates) if date < current_date]
    if not previous_dates:
        return None
    previous = history_df[dates == previous_dates[-1]]
    if previous.empty:
        return None
    return float(previous["Low"].min())


def _technical_config(config: ReplayRuleConfig) -> Dict[str, Any]:
    technical_config = dict(config.technical_config or get_replay_technical_config())
    technical_config["preset_name"] = config.technical_preset or technical_config.get("preset_name", "standard_swing")
    return technical_config


def _score_technical_for_history(
    safe_history: pd.DataFrame,
    current_ts: pd.Timestamp,
    config: ReplayRuleConfig,
    symbol: str,
    name: str,
    daily_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    technical_config = _technical_config(config)
    key_info = technical_score_cache_key(
        symbol=symbol,
        decision_time=current_ts,
        preset=config.technical_preset,
        config=technical_config,
        feature_version=config.feature_version,
    )
    if config.use_replay_cache:
        cached = get_cached_technical_score(key_info)
        if cached is not None:
            return cached
    started = time.perf_counter()
    try:
        technical_daily = build_replay_daily_context(safe_history, current_ts, prior_daily_df=daily_df)
        feature_max_timestamp = _timestamp_text(technical_daily.index.max()) if not technical_daily.empty else ""
        technical = score_technical_item(
            technical_daily,
            config=technical_config,
            code=str(symbol).replace(".T", ""),
            name=name,
            normalized_symbol=symbol,
        )
        technical["_feature_max_timestamp"] = feature_max_timestamp
        technical["_cache_hit"] = False
        technical["_cache_key"] = key_info["key"]
        technical["_config_hash"] = key_info["config_hash"]
    except Exception as exc:
        technical = {
            "technical_preset": config.technical_preset,
            "technical_score_raw": 0,
            "penalty_score": 0,
            "technical_score_final": 0,
            "technical_judgement": "判定不可",
            "confidence": 0,
            "score_breakdown": {},
            "penalty_reasons": [],
            "hard_filter_reason": ["テクニカル採点失敗"],
            "missing_data": ["technical_score_error"],
            "technical_error": f"{exc.__class__.__name__}: {exc}",
            "_feature_max_timestamp": "",
            "_cache_hit": False,
            "_cache_key": key_info["key"],
            "_config_hash": key_info["config_hash"],
        }
    if config.use_replay_cache:
        set_cached_technical_score(key_info, technical, elapsed_seconds=time.perf_counter() - started)
    return technical


def _attach_technical_fields(signal: Dict[str, Any], technical: Dict[str, Any], config: ReplayRuleConfig) -> None:
    signal["technical"] = technical
    signal["technical_preset"] = technical.get("technical_preset", config.technical_preset)
    signal["technical_score_raw"] = technical.get("technical_score_raw")
    signal["technical_penalty_score"] = technical.get("penalty_score")
    signal["technical_score_final"] = technical.get("technical_score_final")
    signal["technical_judgement"] = technical.get("technical_judgement")
    signal["technical_confidence"] = technical.get("confidence")
    signal["technical_score_breakdown"] = technical.get("score_breakdown", {})
    signal["technical_penalty_reasons"] = technical.get("penalty_reasons", [])
    signal["technical_hard_filter_reason"] = technical.get("hard_filter_reason", [])
    signal["technical_missing_data"] = technical.get("missing_data", [])
    signal["technical_stop_loss_candidate"] = technical.get("stop_loss_candidate")
    signal["technical_target_price_candidate"] = technical.get("target_price_candidate")
    signal["technical_risk_reward"] = technical.get("risk_reward")
    signal["technical_hold_days_hint"] = technical.get("hold_days_hint")
    signal["technical_buy_timing_hint"] = technical.get("buy_timing_hint")
    if technical.get("technical_error"):
        signal["technical_error"] = technical.get("technical_error")
    signal["technical_cache_hit"] = bool(technical.get("_cache_hit"))
    signal["technical_cache_key"] = technical.get("_cache_key")
    signal["technical_config_hash"] = technical.get("_config_hash")
    if technical.get("_feature_max_timestamp"):
        signal["feature_max_timestamp"] = technical.get("_feature_max_timestamp")


def _technical_reasons(technical: Dict[str, Any]) -> List[str]:
    breakdown = technical.get("score_breakdown", {})
    if not isinstance(breakdown, dict):
        return []
    reasons: List[str] = []
    for section in ("trend", "entry_position", "volume", "candle", "breakout", "momentum", "risk_reward"):
        values = breakdown.get(section, [])
        if isinstance(values, list):
            reasons.extend(str(item) for item in values[:2])
    return reasons[:8]


def _fallback_stop_loss(safe_history: pd.DataFrame, entry_price: float) -> Optional[float]:
    technical_low = _num(safe_history.tail(12)["Low"].min(), 0)
    if entry_price <= 0:
        return None
    if technical_low and technical_low < entry_price:
        return float(round(technical_low))
    return float(round(entry_price * 0.97))


def _fallback_take_profit(entry_price: float, stop_loss: Optional[float]) -> Optional[float]:
    if entry_price <= 0:
        return None
    if stop_loss and entry_price > stop_loss:
        return float(round(entry_price + (entry_price - stop_loss) * 1.5))
    return float(round(entry_price * 1.03))


def _technical_only_signal(
    safe_history: pd.DataFrame,
    current_ts: pd.Timestamp,
    config: ReplayRuleConfig,
    symbol: str,
    name: str,
    daily_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    current_bar = safe_history.iloc[-1]
    entry_price = _num(current_bar.get("Close"))
    technical = _score_technical_for_history(safe_history, current_ts, config, symbol, name, daily_df)
    stop_loss = technical.get("stop_loss_candidate") or _fallback_stop_loss(safe_history, entry_price)
    take_profit = technical.get("target_price_candidate") or _fallback_take_profit(entry_price, stop_loss)
    signal = {
        "code": symbol,
        "name": name,
        "current_price": entry_price,
        "judgement": "買い検討OK",
        "signal_type": "テクニカルのみ",
        "intraday_score": int(technical.get("technical_score_final", 0) or 0),
        "score": int(technical.get("technical_score_final", 0) or 0),
        "stop_loss": stop_loss,
        "take_profit_1": take_profit,
        "take_profit_2": _fallback_take_profit(entry_price, stop_loss),
        "reasons": _technical_reasons(technical),
        "candidate_generation_mode": CANDIDATE_MODE_TECHNICAL_ONLY,
        "multi_timeframe_enabled": False,
        "multi_timeframe_pass": None,
        "daily_ok_count": None,
        "daily_total_count": None,
        "intraday_ok_count": None,
        "intraday_total_count": None,
        "risk_pass": None,
        "buy_condition_json": {
            "candidate_generation_mode": CANDIDATE_MODE_TECHNICAL_ONLY,
            "technical_min_score": config.technical_min_score,
            "technical_score_final": technical.get("technical_score_final"),
            "technical_judgement": technical.get("technical_judgement"),
            "technical_confidence": technical.get("confidence"),
            "existing_logic_disabled": True,
            "risk_filter_disabled": True,
            "multi_timeframe_disabled": True,
        },
    }
    _attach_technical_fields(signal, technical, config)
    if int(technical.get("technical_score_final", 0) or 0) < int(config.technical_min_score):
        signal["replay_skip_reason"] = "technical_score未満"
        return signal
    technical_config = _technical_config(config)
    if technical_config.get("use_hard_filter", True) and technical.get("hard_filter_reason"):
        signal["replay_skip_reason"] = "technical_hard_filter"
        return signal
    if int(technical.get("confidence", 0) or 0) < int(config.technical_min_confidence):
        signal["replay_skip_reason"] = "technical_confidence未満"
        return signal
    return signal


def evaluate_replay_step(
    history_df: pd.DataFrame,
    current_time: Any,
    rule_config: Dict[str, Any] | ReplayRuleConfig,
    symbol: str = "",
    name: str = "",
    daily_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    config = _normalize_rule_config(rule_config)
    if history_df.empty or len(history_df) < 25:
        return {"judgement": "見送り", "intraday_score": 0, "reason": "判定に必要な履歴足が不足"}

    current_ts = pd.Timestamp(current_time)
    safe_history = history_df.loc[history_df.index <= current_ts].copy()
    if len(safe_history) < 25:
        return {"judgement": "見送り", "intraday_score": 0, "reason": "判定に必要な履歴足が不足"}

    if config.candidate_generation_mode == CANDIDATE_MODE_TECHNICAL_ONLY:
        return _technical_only_signal(safe_history, current_ts, config, symbol, name, daily_df)

    signal = evaluate_intraday_entry(
        code=symbol,
        name=name,
        intraday_data=safe_history.tail(160),
        previous_low=_previous_session_low(safe_history, current_ts),
    )
    if config.use_multi_timeframe:
        daily_context = build_replay_daily_context(safe_history, current_ts, prior_daily_df=daily_df)
        mtf = evaluate_multi_timeframe_signal(
            daily_context,
            safe_history.tail(160),
            current_time=current_ts,
            config=_multi_timeframe_config(config),
        )
        decision = str(mtf.get("decision_category", "avoid"))
        if decision == "buy":
            signal["judgement"] = "買い検討OK"
        elif decision == "watch":
            signal["judgement"] = "監視強化"
        else:
            signal["judgement"] = "見送り"
        signal["intraday_score"] = int(mtf.get("score", signal.get("intraday_score", 0)) or 0)
        signal["signal_type"] = str(mtf.get("entry_type") or signal.get("signal_type", "見送り"))
        signal["multi_timeframe_enabled"] = True
        signal["multi_timeframe_pass"] = bool(mtf.get("multi_timeframe_pass"))
        signal["multi_timeframe_score"] = mtf.get("score")
        signal["multi_timeframe_decision"] = decision
        signal["daily_filter"] = mtf.get("daily_filter", {})
        signal["intraday_entry"] = mtf.get("intraday_entry", {})
        signal["daily_ok_count"] = signal["daily_filter"].get("daily_ok_count", 0)
        signal["daily_total_count"] = signal["daily_filter"].get("daily_total_count", 4)
        signal["intraday_ok_count"] = signal["intraday_entry"].get("intraday_ok_count", 0)
        signal["intraday_total_count"] = signal["intraday_entry"].get("intraday_total_count", 4)
        signal["daily_filter_json"] = signal["daily_filter"]
        signal["intraday_entry_json"] = signal["intraday_entry"]
        signal["multi_timeframe_detail"] = mtf.get("detail_json", {})
        signal["reasons"] = list(dict.fromkeys(list(signal.get("reasons", [])) + list(mtf.get("reasons", []))))

    risk_filter = evaluate_risk_filter(signal, shares=config.shares, config=_risk_filter_config(config))
    signal["risk_filter"] = risk_filter
    signal["risk_filter_json"] = risk_filter.get("risk_filter_json", risk_filter)
    signal["risk_pass"] = risk_filter.get("risk_pass")
    signal["stop_loss_pct"] = risk_filter.get("stop_loss_pct")
    signal["max_loss_yen"] = risk_filter.get("max_loss_yen")
    signal["expected_profit_yen"] = risk_filter.get("expected_profit_yen")
    signal["risk_reward_ratio"] = risk_filter.get("risk_reward_ratio")
    signal["risk_reasons"] = risk_filter.get("risk_reasons", [])
    if config.use_risk_filter and not risk_filter.get("risk_pass"):
        signal["judgement"] = "監視強化"
        signal["replay_skip_reason"] = "risk_filter_ng"
        signal["risk_exclusion_reason"] = "、".join(risk_filter.get("risk_reasons", []))
        return signal
    score = int(signal.get("intraday_score", 0) or 0)
    signal_type = str(signal.get("signal_type", ""))
    if score < config.min_score:
        signal["replay_skip_reason"] = "min_score未満"
        return signal
    if config.target_rule != "すべて" and signal_type != config.target_rule:
        signal["replay_skip_reason"] = "対象ルール外"
        return signal
    if signal.get("judgement") not in {"買い検討OK", "監視強化"}:
        signal["replay_skip_reason"] = "買い判定未達"
        return signal
    signal["existing_logic_pass"] = True
    if config.candidate_generation_mode == CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL:
        technical_config = _technical_config(config)
        technical = _score_technical_for_history(safe_history, current_ts, config, symbol, name, daily_df)
        _attach_technical_fields(signal, technical, config)
        if int(technical.get("technical_score_final", 0) or 0) < int(config.technical_min_score):
            signal["replay_skip_reason"] = "technical_score_below_min"
            return signal
        if technical_config.get("use_hard_filter", True) and technical.get("hard_filter_reason"):
            signal["replay_skip_reason"] = "technical_hard_filter"
            return signal
        if int(technical.get("confidence", 0) or 0) < int(config.technical_min_confidence):
            signal["replay_skip_reason"] = "technical_confidence_below_min"
            return signal
    return signal


def create_replay_trade(signal: Dict[str, Any], current_bar: pd.Series, current_time: Any) -> Dict[str, Any]:
    entry_price = _num(signal.get("current_price"), _num(current_bar.get("Close")))
    return {
        "signal_time": _timestamp_text(current_time),
        "entry_price": entry_price,
        "stop_loss": signal.get("stop_loss"),
        "take_profit": signal.get("take_profit_1"),
        "score": int(signal.get("intraday_score", 0) or 0),
        "entry_type": signal.get("signal_type", "-"),
        "rule_name": (
            "technical_only_replay_rule"
            if signal.get("candidate_generation_mode") == CANDIDATE_MODE_TECHNICAL_ONLY
            else "intraday_replay_rule"
        ),
        "candidate_generation_mode": signal.get("candidate_generation_mode"),
        "decision_time": _timestamp_text(current_time),
        "feature_max_timestamp": signal.get("feature_max_timestamp") or _timestamp_text(current_time),
        "technical_cache_hit": signal.get("technical_cache_hit"),
        "technical_config_hash": signal.get("technical_config_hash"),
        "daily_ok_count": signal.get("daily_ok_count"),
        "daily_total_count": signal.get("daily_total_count"),
        "daily_score": signal.get("daily_score"),
        "daily_rank_at_scan": signal.get("daily_rank_at_scan"),
        "daily_rank_total": signal.get("daily_rank_total"),
        "daily_top_n_pass": signal.get("daily_top_n_pass"),
        "daily_top_n": signal.get("daily_top_n"),
        "intraday_ok_count": signal.get("intraday_ok_count"),
        "intraday_total_count": signal.get("intraday_total_count"),
        "daily_filter_json": signal.get("daily_filter_json", signal.get("daily_filter", {})),
        "intraday_entry_json": signal.get("intraday_entry_json", signal.get("intraday_entry", {})),
        "multi_timeframe_pass": signal.get("multi_timeframe_pass"),
        "risk_pass": signal.get("risk_pass"),
        "stop_loss_pct": signal.get("stop_loss_pct"),
        "max_loss_yen": signal.get("max_loss_yen"),
        "expected_profit_yen": signal.get("expected_profit_yen"),
        "risk_reward_ratio": signal.get("risk_reward_ratio"),
        "risk_filter_json": signal.get("risk_filter_json", signal.get("risk_filter", {})),
        "risk_reasons": signal.get("risk_reasons", []),
        "buy_condition_json": signal.get("buy_condition_json", {}),
        "technical_preset": signal.get("technical_preset"),
        "technical_score_raw": signal.get("technical_score_raw"),
        "technical_penalty_score": signal.get("technical_penalty_score"),
        "technical_score_final": signal.get("technical_score_final"),
        "technical_judgement": signal.get("technical_judgement"),
        "technical_confidence": signal.get("technical_confidence"),
        "technical_score_breakdown": signal.get("technical_score_breakdown", {}),
        "technical_penalty_reasons": signal.get("technical_penalty_reasons", []),
        "technical_hard_filter_reason": signal.get("technical_hard_filter_reason", []),
        "technical_missing_data": signal.get("technical_missing_data", []),
        "technical_stop_loss_candidate": signal.get("technical_stop_loss_candidate"),
        "technical_target_price_candidate": signal.get("technical_target_price_candidate"),
        "technical_risk_reward": signal.get("technical_risk_reward"),
        "technical_hold_days_hint": signal.get("technical_hold_days_hint"),
        "technical_buy_timing_hint": signal.get("technical_buy_timing_hint"),
        "signal": {
            key: value
            for key, value in signal.items()
            if key not in {"data"}
        },
    }


def evaluate_replay_trade_outcome(
    trade: Dict[str, Any],
    future_df: pd.DataFrame,
    interval: str = "5m",
    debug: bool = False,
    symbol: str = "",
) -> Dict[str, Any]:
    entry_price = _num(trade.get("entry_price"))
    stop_loss = _num(trade.get("stop_loss"), 0)
    take_profit = _num(trade.get("take_profit"), 0)
    max_bars = _max_hold_bars(interval)
    horizon = pd.DataFrame() if future_df is None else future_df.copy().sort_index().head(max_bars)
    future_len = 0 if future_df is None else len(future_df)
    signal_time = trade.get("signal_time")

    if horizon.empty or not entry_price:
        result = {
            "status": "open",
            "outcome": "open",
            "return_pct": None,
            "max_profit_pct": None,
            "max_drawdown_pct": None,
            "hit_stop_loss": False,
            "hit_take_profit": False,
            "evaluated_until": "",
            "exit_time": "",
            "exit_price": None,
            "holding_period": "-",
            "exit_reason": "検証中",
        }
        if debug:
            result["debug"] = {
                "symbol": symbol,
                "signal_time": signal_time,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "future_df_rows": 0,
                "outcome": "open",
            }
        return result

    return_pct = _pct(float(horizon["Close"].iloc[-1]), entry_price)
    outcome = "timeout"
    status = "closed"
    hit_stop = False
    hit_target = False
    evaluated_until = horizon.index[-1]
    exit_price = float(horizon["Close"].iloc[-1])
    exit_reason = "timeout"

    for ts, row in horizon.iterrows():
        low = _num(row.get("Low"))
        high = _num(row.get("High"))
        # Conservative assumption: if one candle touches both stop and target,
        # the stop-loss is treated as hit first.
        if stop_loss and low <= stop_loss:
            hit_stop = True
            outcome = "hit_stop_loss"
            return_pct = _pct(stop_loss, entry_price)
            evaluated_until = ts
            exit_price = stop_loss
            exit_reason = "hit_stop_loss"
            break
        if take_profit and high >= take_profit:
            hit_target = True
            outcome = "hit_take_profit"
            return_pct = _pct(take_profit, entry_price)
            evaluated_until = ts
            exit_price = take_profit
            exit_reason = "hit_take_profit"
            break

    if future_len < max_bars and outcome == "timeout":
        status = "open"
        outcome = "open"
        exit_reason = "検証中"

    outcome_window = horizon.loc[horizon.index <= evaluated_until].copy()
    if outcome_window.empty:
        outcome_window = horizon.head(1).copy()

    min_low = float(outcome_window["Low"].min())
    max_high = float(outcome_window["High"].max())
    min_low_time = outcome_window["Low"].idxmin()
    max_high_time = outcome_window["High"].idxmax()
    max_profit_pct = _pct(max_high, entry_price)
    max_drawdown_pct = _pct(min_low, entry_price)

    signal_time = pd.Timestamp(trade.get("signal_time"))
    holding_delta = pd.Timestamp(evaluated_until) - signal_time
    holding_period = f"{holding_delta.days}日 {holding_delta.seconds // 3600}時間"

    anomaly_rows = outcome_window[
        (outcome_window["High"] >= entry_price * 1.2)
        | (outcome_window["Low"] <= entry_price * 0.8)
    ]

    result = {
        "status": status,
        "outcome": outcome,
        "return_pct": return_pct,
        "max_profit_pct": max_profit_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "hit_stop_loss": hit_stop,
        "hit_take_profit": hit_target,
        "evaluated_until": _timestamp_text(evaluated_until),
        "exit_time": _timestamp_text(evaluated_until),
        "exit_price": exit_price,
        "holding_period": holding_period,
        "exit_reason": exit_reason,
    }
    if debug:
        result["debug"] = {
            "symbol": symbol,
            "signal_time": _timestamp_text(signal_time),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "future_df_start": _timestamp_text(horizon.index[0]),
            "future_df_end": _timestamp_text(horizon.index[-1]),
            "future_df_rows": len(horizon),
            "outcome_window_start": _timestamp_text(outcome_window.index[0]),
            "outcome_window_end": _timestamp_text(outcome_window.index[-1]),
            "outcome_window_rows": len(outcome_window),
            "min_low": min_low,
            "min_low_time": _timestamp_text(min_low_time),
            "max_high": max_high,
            "max_high_time": _timestamp_text(max_high_time),
            "exit_time": _timestamp_text(evaluated_until),
            "exit_price": exit_price,
            "outcome": outcome,
            "return_pct": return_pct,
            "max_profit_pct": max_profit_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "entry_relative_anomaly_count": len(anomaly_rows),
        }
    return result


def empty_replay_stage_counts(symbols_count: int = 1) -> Dict[str, Any]:
    return {
        "scan_count": 0,
        "scan_total_count": 0,
        "target_symbols_count": int(symbols_count or 0),
        "existing_logic_checked_count": 0,
        "existing_logic_pass_count": 0,
        "existing_logic_reject_count": 0,
        "technical_score_checked_count": 0,
        "technical_score_pass_count": 0,
        "technical_score_reject_count": 0,
        "hard_filter_reject_count": 0,
        "technical_score_attempt_count": 0,
        "technical_score_success_count": 0,
        "technical_min_score_pass_count": 0,
        "technical_hard_filter_excluded_count": 0,
        "technical_confidence_excluded_count": 0,
        "raw_signal_count": 0,
        "cooldown_filtered_count": 0,
        "max_trades_excluded_count": 0,
        "position_open_excluded_count": 0,
        "final_virtual_buy_count": 0,
        "settled_trade_count": 0,
        "existing_min_score_pass_count": 0,
        "target_filter_pass_count": 0,
        "intraday_entry_pass_count": 0,
        "use_conditions_pass_count": 0,
        "risk_filter_pass_count": 0,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "leak_check_ok_count": 0,
        "leak_check_ng_count": 0,
        "skip_reasons": {},
    }


def update_replay_stage_counts(
    stage_counts: Dict[str, Any],
    signal: Dict[str, Any],
    config: Dict[str, Any] | ReplayRuleConfig,
) -> None:
    normalized = _normalize_rule_config(config)
    skip_reason = signal.get("replay_skip_reason")
    if skip_reason:
        reasons = stage_counts.setdefault("skip_reasons", {})
        reasons[str(skip_reason)] = int(reasons.get(str(skip_reason), 0)) + 1

    mode = normalized.candidate_generation_mode
    has_technical = signal.get("technical_score_final") not in (None, "") or bool(signal.get("technical_error"))
    if mode != CANDIDATE_MODE_TECHNICAL_ONLY:
        stage_counts["existing_logic_checked_count"] = int(stage_counts.get("existing_logic_checked_count", 0)) + 1
        if bool(signal.get("existing_logic_pass")):
            stage_counts["existing_logic_pass_count"] = int(stage_counts.get("existing_logic_pass_count", 0)) + 1
        else:
            stage_counts["existing_logic_reject_count"] = int(stage_counts.get("existing_logic_reject_count", 0)) + 1

    if has_technical:
        stage_counts["technical_score_checked_count"] = int(stage_counts.get("technical_score_checked_count", 0)) + 1
        stage_counts["technical_score_attempt_count"] = int(stage_counts.get("technical_score_attempt_count", 0)) + 1
        if signal.get("technical_cache_hit"):
            stage_counts["cache_hit_count"] = int(stage_counts.get("cache_hit_count", 0)) + 1
        else:
            stage_counts["cache_miss_count"] = int(stage_counts.get("cache_miss_count", 0)) + 1
        if not signal.get("technical_error"):
            stage_counts["technical_score_success_count"] = int(stage_counts.get("technical_score_success_count", 0)) + 1
        if int(signal.get("technical_score_final", 0) or 0) >= int(normalized.technical_min_score):
            stage_counts["technical_score_pass_count"] = int(stage_counts.get("technical_score_pass_count", 0)) + 1
            stage_counts["technical_min_score_pass_count"] = int(stage_counts.get("technical_min_score_pass_count", 0)) + 1
        elif not signal.get("technical_error"):
            stage_counts["technical_score_reject_count"] = int(stage_counts.get("technical_score_reject_count", 0)) + 1
        if skip_reason == "technical_hard_filter":
            stage_counts["hard_filter_reject_count"] = int(stage_counts.get("hard_filter_reject_count", 0)) + 1
            stage_counts["technical_hard_filter_excluded_count"] = int(stage_counts.get("technical_hard_filter_excluded_count", 0)) + 1
        if skip_reason in {"technical_confidence未満", "technical_confidence_below_min"}:
            stage_counts["technical_score_reject_count"] = int(stage_counts.get("technical_score_reject_count", 0)) + 1
            stage_counts["technical_confidence_excluded_count"] = int(stage_counts.get("technical_confidence_excluded_count", 0)) + 1

    if mode != CANDIDATE_MODE_TECHNICAL_ONLY:
        if int(signal.get("intraday_score", 0) or 0) >= int(normalized.min_score):
            stage_counts["existing_min_score_pass_count"] = int(stage_counts.get("existing_min_score_pass_count", 0)) + 1
        judgement = signal.get("judgement")
        if judgement in {"買い検討OK", "監視強化"}:
            stage_counts["target_filter_pass_count"] = int(stage_counts.get("target_filter_pass_count", 0)) + 1
        if int(signal.get("intraday_ok_count", 0) or 0) >= int(normalized.intraday_min_ok):
            stage_counts["intraday_entry_pass_count"] = int(stage_counts.get("intraday_entry_pass_count", 0)) + 1
        if bool(signal.get("multi_timeframe_pass")) or not normalized.use_multi_timeframe:
            stage_counts["use_conditions_pass_count"] = int(stage_counts.get("use_conditions_pass_count", 0)) + 1
        if bool(signal.get("risk_pass")) or not normalized.use_risk_filter:
            stage_counts["risk_filter_pass_count"] = int(stage_counts.get("risk_filter_pass_count", 0)) + 1


def leak_check_for_trade(trade: Dict[str, Any], outcome_start_timestamp: Any = None) -> Dict[str, str]:
    decision = trade.get("decision_time") or trade.get("signal_time")
    feature_max = trade.get("feature_max_timestamp") or decision
    outcome_start = outcome_start_timestamp or trade.get("outcome_start_timestamp")
    reasons: List[str] = []
    ok = True
    try:
        decision_ts = pd.Timestamp(decision)
        feature_ts = pd.Timestamp(feature_max)
        if feature_ts > decision_ts:
            ok = False
            reasons.append("feature_max_timestampがdecision_timeより後です")
    except Exception:
        ok = False
        reasons.append("decision_timeまたはfeature_max_timestampを解釈できません")
    if outcome_start:
        try:
            outcome_ts = pd.Timestamp(outcome_start)
            decision_ts = pd.Timestamp(decision)
            if outcome_ts <= decision_ts:
                ok = False
                reasons.append("outcome_start_timestampがdecision_time以下です")
        except Exception:
            ok = False
            reasons.append("outcome_start_timestampを解釈できません")
    return {
        "leak_check_result": "OK" if ok else "NG",
        "leak_check_reason": " / ".join(reasons) if reasons else "feature<=decision and outcome>decision",
    }


def replay_zero_trade_reasons(stage_counts: Dict[str, Any], mode: str) -> List[str]:
    if int(stage_counts.get("final_virtual_buy_count", 0) or 0) > 0:
        return []
    reasons: List[str] = []
    normalized_mode = _normalize_candidate_generation_mode(mode)
    if normalized_mode in {CANDIDATE_MODE_TECHNICAL_ONLY, CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL}:
        attempts = int(stage_counts.get("technical_score_attempt_count", 0) or 0)
        if attempts <= 0:
            reasons.append("テクニカル採点自体が実行されていません。")
        elif int(stage_counts.get("technical_min_score_pass_count", 0) or 0) <= 0:
            reasons.append("テクニカル採点は実行されましたが、最小テクニカルスコアを1件も通過しませんでした。")
        if int(stage_counts.get("technical_hard_filter_excluded_count", 0) or 0) >= max(1, attempts):
            reasons.append("テクニカル強制見送り条件で全件除外されました。")
    if normalized_mode != CANDIDATE_MODE_TECHNICAL_ONLY:
        if int(stage_counts.get("existing_logic_checked_count", 0) or 0) <= 0:
            reasons.append("既存ロジックの判定自体が実行されていません。")
        elif int(stage_counts.get("existing_logic_pass_count", 0) or 0) <= 0:
            reasons.append("既存ロジックを通過した候補が1件もありませんでした。")
        if int(stage_counts.get("existing_min_score_pass_count", 0) or 0) <= 0:
            reasons.append("既存の最小スコア条件で全件除外された可能性があります。")
        if int(stage_counts.get("intraday_entry_pass_count", 0) or 0) <= 0:
            reasons.append("既存の5分足エントリー判定で全件除外された可能性があります。")
        if int(stage_counts.get("risk_filter_pass_count", 0) or 0) <= 0:
            reasons.append("リスク条件で全件除外された可能性があります。")
    if not reasons:
        reasons.append("候補は出ましたが、連続シグナル抑制または最大仮想買い件数制限で最終0件になった可能性があります。")
    return reasons


def run_replay(
    symbol: str,
    df: pd.DataFrame,
    start_at: Any = None,
    end_at: Any = None,
    rule_config: Optional[Dict[str, Any] | ReplayRuleConfig] = None,
    name: str = "",
    daily_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    config = _normalize_rule_config(rule_config)
    if df is None or df.empty:
        stage_counts = empty_replay_stage_counts(1)
        return {
            "trades": [],
            "summary": summarize_replay_results([]) | {
                "evaluated_steps": 0,
                "error": "過去データが空です。",
                "candidate_generation_mode": config.candidate_generation_mode,
                "stage_counts": stage_counts,
                "zero_trade_reasons": ["過去データが空のため、候補判定を実行できませんでした。"],
            },
        }

    data = df.copy().sort_index()
    if start_at:
        data = data[data.index >= pd.Timestamp(start_at)]
    if end_at:
        data = data[data.index <= pd.Timestamp(end_at)]

    trades: List[Dict[str, Any]] = []
    evaluated_steps = 0
    last_trade_index = -10**9
    all_data = df.copy().sort_index()
    stage_counts = empty_replay_stage_counts(1)
    position_open_until: Optional[pd.Timestamp] = None

    for position, (current_time, current_bar) in enumerate(data.iterrows()):
        global_position = all_data.index.get_indexer([current_time])[0]
        if global_position < 24:
            continue
        history = all_data.loc[all_data.index <= current_time]
        signal = evaluate_replay_step(history, current_time, config, symbol=symbol, name=name, daily_df=daily_df)
        evaluated_steps += 1
        stage_counts["scan_count"] = evaluated_steps
        stage_counts["scan_total_count"] = evaluated_steps
        update_replay_stage_counts(stage_counts, signal, config)
        if signal.get("replay_skip_reason") or signal.get("judgement") not in {"買い検討OK", "監視強化"}:
            continue

        stage_counts["raw_signal_count"] = int(stage_counts.get("raw_signal_count", 0)) + 1
        if position - last_trade_index < config.cooldown_bars:
            stage_counts["cooldown_filtered_count"] = int(stage_counts.get("cooldown_filtered_count", 0)) + 1
            reasons = stage_counts.setdefault("skip_reasons", {})
            reasons["cooldown_recent"] = int(reasons.get("cooldown_recent", 0)) + 1
            continue
        if config.one_position_per_symbol and position_open_until is not None and pd.Timestamp(current_time) <= position_open_until:
            stage_counts["position_open_excluded_count"] = int(stage_counts.get("position_open_excluded_count", 0)) + 1
            reasons = stage_counts.setdefault("skip_reasons", {})
            reasons["position_open"] = int(reasons.get("position_open", 0)) + 1
            continue
        if len(trades) >= config.max_trades:
            stage_counts["max_trades_excluded_count"] = int(stage_counts.get("max_trades_excluded_count", 0)) + 1
            reasons = stage_counts.setdefault("skip_reasons", {})
            reasons["max_trades"] = int(reasons.get("max_trades", 0)) + 1
            continue

        trade = create_replay_trade(signal, current_bar, current_time)
        future = all_data.loc[all_data.index > current_time]
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
                debug=config.debug,
                symbol=symbol,
            )
        )
        trade.update(
            {
                "symbol": symbol,
                "name": name,
                "replay_start_at": _timestamp_text(data.index[0]) if not data.empty else "",
                "replay_end_at": _timestamp_text(data.index[-1]) if not data.empty else "",
            }
        )
        trades.append(trade)
        stage_counts["final_virtual_buy_count"] = len(trades)
        exit_reference = trade.get("exit_time") or trade.get("evaluated_until")
        if exit_reference:
            try:
                position_open_until = pd.Timestamp(exit_reference)
            except Exception:
                position_open_until = None
        last_trade_index = position

    summary = summarize_replay_results(trades)
    settled_count = sum(
        1
        for trade in trades
        if trade.get("status") == "closed" and trade.get("exit_price") is not None
    )
    stage_counts["final_virtual_buy_count"] = len(trades)
    stage_counts["settled_trade_count"] = settled_count
    summary["evaluated_steps"] = evaluated_steps
    summary["data_rows"] = len(data)
    summary["first_timestamp"] = _timestamp_text(data.index[0]) if not data.empty else "-"
    summary["last_timestamp"] = _timestamp_text(data.index[-1]) if not data.empty else "-"
    summary["candidate_generation_mode"] = config.candidate_generation_mode
    summary["raw_signal_count"] = stage_counts.get("raw_signal_count", 0)
    summary["cooldown_filtered_count"] = stage_counts.get("cooldown_filtered_count", 0)
    summary["position_filtered_count"] = stage_counts.get("position_open_excluded_count", 0)
    summary["max_trade_filtered_count"] = stage_counts.get("max_trades_excluded_count", 0)
    summary["final_trade_count"] = len(trades)
    summary["settled_trade_count"] = settled_count
    summary["csv_export_count"] = len(trades)
    summary["stage_counts"] = stage_counts
    summary["zero_trade_reasons"] = replay_zero_trade_reasons(stage_counts, config.candidate_generation_mode)
    return {"trades": trades, "summary": summary}


def summarize_replay_results(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(trades)
    closed = [trade for trade in trades if trade.get("status") == "closed"]
    returns = [_num(trade.get("return_pct")) for trade in closed if trade.get("return_pct") is not None]
    wins = [trade for trade in closed if _num(trade.get("return_pct")) > 0]
    avg_return = round(sum(returns) / len(returns), 2) if returns else 0.0
    avg_profit = round(
        sum(_num(trade.get("max_profit_pct")) for trade in trades if trade.get("max_profit_pct") is not None)
        / max(1, len([trade for trade in trades if trade.get("max_profit_pct") is not None])),
        2,
    )
    avg_drawdown = round(
        sum(_num(trade.get("max_drawdown_pct")) for trade in trades if trade.get("max_drawdown_pct") is not None)
        / max(1, len([trade for trade in trades if trade.get("max_drawdown_pct") is not None])),
        2,
    )
    return {
        "trade_count": total,
        "closed_count": len(closed),
        "open_count": total - len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
        "avg_return_pct": avg_return,
        "avg_max_profit_pct": avg_profit,
        "avg_max_drawdown_pct": avg_drawdown,
        "hit_stop_loss_count": sum(1 for trade in trades if trade.get("hit_stop_loss")),
        "hit_take_profit_count": sum(1 for trade in trades if trade.get("hit_take_profit")),
    }


def _trade_sort_key(trade: Dict[str, Any]) -> pd.Timestamp:
    try:
        return pd.Timestamp(trade.get("signal_time"))
    except Exception:
        return pd.Timestamp.max


def _is_closed_money_trade(trade: Dict[str, Any]) -> bool:
    if str(trade.get("status", "")) != "closed":
        return False
    return trade.get("entry_price") is not None and trade.get("exit_price") is not None


def apply_replay_money_metrics(
    trades: List[Dict[str, Any]],
    shares: int = DEFAULT_REPLAY_SHARES,
) -> List[Dict[str, Any]]:
    share_count = int(shares or DEFAULT_REPLAY_SHARES)
    cumulative = 0.0
    rows: List[Dict[str, Any]] = []
    for trade in sorted([dict(item) for item in trades], key=_trade_sort_key):
        entry_price = _num(trade.get("entry_price"), 0)
        exit_price = _num(trade.get("exit_price"), 0)
        required_capital = entry_price * share_count if entry_price else None
        profit_yen = None
        cumulative_profit = None
        if _is_closed_money_trade(trade) and entry_price and exit_price:
            profit_yen = round((exit_price - entry_price) * share_count)
            cumulative += profit_yen
            cumulative_profit = round(cumulative)
        trade["shares"] = share_count
        trade["required_capital_yen"] = round(required_capital) if required_capital is not None else None
        trade["profit_yen"] = profit_yen
        trade["cumulative_profit_yen"] = cumulative_profit
        rows.append(trade)
    return rows


def _max_streak(values: List[float], positive: bool) -> int:
    best = 0
    current = 0
    for value in values:
        is_hit = value > 0 if positive else value < 0
        if is_hit:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize_replay_money(
    trades: List[Dict[str, Any]],
    shares: int = DEFAULT_REPLAY_SHARES,
) -> Dict[str, Any]:
    prepared = apply_replay_money_metrics(trades, shares)
    closed = [trade for trade in prepared if trade.get("profit_yen") is not None]
    profits = [float(trade["profit_yen"]) for trade in closed]
    wins = [value for value in profits if value > 0]
    losses = [value for value in profits if value < 0]
    required_capitals = [
        float(trade["required_capital_yen"])
        for trade in closed
        if trade.get("required_capital_yen") is not None
    ]
    avg_profit = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    profit_loss_ratio = None
    if avg_profit is not None and avg_loss not in (None, 0):
        profit_loss_ratio = round(avg_profit / abs(avg_loss), 2)

    return {
        "shares": int(shares or DEFAULT_REPLAY_SHARES),
        "closed_trade_count": len(closed),
        "average_required_capital_yen": round(sum(required_capitals) / len(required_capitals)) if required_capitals else None,
        "gross_profit_yen": round(sum(wins)) if wins else 0,
        "gross_loss_yen": round(sum(losses)) if losses else 0,
        "net_profit_yen": round(sum(profits)) if profits else 0,
        "profit_loss_ratio": profit_loss_ratio,
        "average_profit_yen": round(avg_profit) if avg_profit is not None else None,
        "average_loss_yen": round(avg_loss) if avg_loss is not None else None,
        "max_profit_yen": round(max(profits)) if profits else None,
        "max_loss_yen": round(min(profits)) if profits else None,
        "max_win_streak": _max_streak(profits, positive=True),
        "max_loss_streak": _max_streak(profits, positive=False),
    }
