from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class MultiTimeframeConfig:
    daily_min_ok: int = 3
    intraday_min_ok: int = 2
    near_5day_high_pct: float = 0.98
    daily_volume_multiplier: float = 1.2
    breakout_lookback_bars: int = 12
    intraday_volume_multiplier: float = 1.5
    use_vwap: bool = True
    use_volume_spike: bool = True


@dataclass
class RiskFilterConfig:
    enabled: bool = True
    max_stop_loss_pct: float = 3.0
    max_loss_yen_limit: float = 20000.0
    min_risk_reward: float = 1.2


def normalize_config(config: Optional[Dict[str, Any] | MultiTimeframeConfig]) -> MultiTimeframeConfig:
    if isinstance(config, MultiTimeframeConfig):
        return config
    values = dict(config or {})
    def _int_setting(key: str, default: int) -> int:
        return default if values.get(key) is None else int(values.get(key))

    return MultiTimeframeConfig(
        daily_min_ok=_int_setting("daily_min_ok", 3),
        intraday_min_ok=_int_setting("intraday_min_ok", 2),
        near_5day_high_pct=float(values.get("near_5day_high_pct", 0.98) or 0.98),
        daily_volume_multiplier=float(values.get("daily_volume_multiplier", 1.2) or 1.2),
        breakout_lookback_bars=int(values.get("breakout_lookback_bars", 12) or 12),
        intraday_volume_multiplier=float(values.get("intraday_volume_multiplier", 1.5) or 1.5),
        use_vwap=bool(values.get("use_vwap", True)),
        use_volume_spike=bool(values.get("use_volume_spike", True)),
    )


def normalize_risk_config(config: Optional[Dict[str, Any] | RiskFilterConfig]) -> RiskFilterConfig:
    if isinstance(config, RiskFilterConfig):
        return config
    values = dict(config or {})
    return RiskFilterConfig(
        enabled=bool(values.get("enabled", True)),
        max_stop_loss_pct=float(values.get("max_stop_loss_pct", 3.0) or 3.0),
        max_loss_yen_limit=float(values.get("max_loss_yen_limit", 20000.0) or 20000.0),
        min_risk_reward=float(values.get("min_risk_reward", 1.2) or 1.2),
    )


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _scoped(df: pd.DataFrame, current_time: Any = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    data = df.copy().sort_index()
    if current_time is not None:
        current_ts = pd.Timestamp(current_time)
        data = data.loc[data.index <= current_ts]
    return data


def _daily_from_intraday(intraday_df: pd.DataFrame) -> pd.DataFrame:
    if intraday_df is None or intraday_df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    data = intraday_df.copy().sort_index()
    return data.resample("1D").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna(subset=["Open", "High", "Low", "Close"])


def build_replay_daily_context(
    intraday_df: pd.DataFrame,
    current_time: Any,
    prior_daily_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build daily bars from bars visible at current_time only.

    This intentionally avoids yfinance daily bars for the current day, because
    those would include future close/volume during historical replay.
    """
    current_ts = pd.Timestamp(current_time)
    history = _scoped(intraday_df, current_ts)
    partial_daily = _daily_from_intraday(history)
    if prior_daily_df is None or prior_daily_df.empty:
        return partial_daily

    confirmed_daily = prior_daily_df.copy().sort_index()
    confirmed_daily = confirmed_daily.loc[confirmed_daily.index < current_ts.normalize()]
    combined = pd.concat([confirmed_daily, partial_daily])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def evaluate_daily_filter(
    daily_df: pd.DataFrame,
    current_time: Any = None,
    config: Optional[Dict[str, Any] | MultiTimeframeConfig] = None,
) -> Dict[str, Any]:
    cfg = normalize_config(config)
    data = _scoped(daily_df, current_time)
    if len(data) < 25:
        return {
            "ma25_ok": False,
            "above_prev_close_ok": False,
            "near_5day_high_ok": False,
            "daily_volume_increase_ok": False,
            "daily_ok_count": 0,
            "daily_total_count": 4,
            "daily_score": 0,
            "daily_pass": False,
            "reason": "日足25本未満のため判定不可",
        }

    close = _num(data["Close"].iloc[-1])
    previous_close = _num(data["Close"].iloc[-2]) if len(data) >= 2 else 0.0
    ma25 = _num(data["Close"].rolling(25, min_periods=25).mean().iloc[-1])
    recent_5day_high = _num(data["High"].tail(5).max())
    current_volume = _num(data["Volume"].iloc[-1])
    avg_volume_5d = _num(data["Volume"].shift(1).tail(5).mean())

    ma25_ok = bool(ma25 and close > ma25)
    above_prev_close_ok = bool(previous_close and close > previous_close)
    near_5day_high_ok = bool(recent_5day_high and close >= recent_5day_high * cfg.near_5day_high_pct)
    daily_volume_increase_ok = bool(avg_volume_5d and current_volume >= avg_volume_5d * cfg.daily_volume_multiplier)
    flags = [ma25_ok, above_prev_close_ok, near_5day_high_ok, daily_volume_increase_ok]
    ok_count = int(sum(flags))
    daily_score = int(round(ok_count / 4 * 100))
    reasons = []
    if ma25_ok:
        reasons.append("25日線より上")
    if above_prev_close_ok:
        reasons.append("前日終値より上")
    if near_5day_high_ok:
        reasons.append("直近5日高値圏")
    if daily_volume_increase_ok:
        reasons.append("日足出来高増加")

    return {
        "ma25_ok": ma25_ok,
        "above_prev_close_ok": above_prev_close_ok,
        "near_5day_high_ok": near_5day_high_ok,
        "daily_volume_increase_ok": daily_volume_increase_ok,
        "daily_ok_count": ok_count,
        "daily_total_count": 4,
        "daily_score": daily_score,
        "daily_pass": bool(ok_count >= cfg.daily_min_ok),
        "close": close,
        "ma25": ma25,
        "previous_close": previous_close,
        "recent_5day_high": recent_5day_high,
        "current_volume": current_volume,
        "avg_volume_5d": avg_volume_5d,
        "reasons": reasons,
    }


def _with_vwap(intraday_df: pd.DataFrame) -> pd.DataFrame:
    data = intraday_df.copy().sort_index()
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    volume = data["Volume"].fillna(0).clip(lower=0)
    cumulative_volume = volume.cumsum()
    cumulative_value = (typical_price * volume).cumsum()
    fallback = typical_price.expanding(min_periods=1).mean()
    data["vwap"] = (cumulative_value / cumulative_volume.replace(0, pd.NA)).fillna(fallback)
    return data


def evaluate_intraday_entry(
    intraday_df: pd.DataFrame,
    current_time: Any = None,
    config: Optional[Dict[str, Any] | MultiTimeframeConfig] = None,
) -> Dict[str, Any]:
    cfg = normalize_config(config)
    data = _scoped(intraday_df, current_time)
    if data.empty:
        return {
            "vwap_ok": None,
            "breakout_ok": None,
            "pullback_rebound_ok": None,
            "intraday_volume_spike_ok": None,
            "intraday_ok_count": 0,
            "intraday_total_count": 4,
            "intraday_pass": None,
            "intraday_available": False,
            "intraday_data_status": "データなし",
            "current_close": None,
            "previous_close": None,
            "vwap": None,
            "past_n_bars_high": None,
            "current_volume": None,
            "avg_volume_12bars": None,
            "reason": "5分足データが空です",
            "reasons": [],
        }
    if len(data) < max(3, cfg.breakout_lookback_bars + 1):
        return {
            "vwap_ok": None,
            "breakout_ok": None,
            "pullback_rebound_ok": None,
            "intraday_volume_spike_ok": None,
            "intraday_ok_count": 0,
            "intraday_total_count": 4,
            "intraday_pass": None,
            "intraday_available": False,
            "intraday_data_status": "データ不足",
            "current_close": None,
            "previous_close": None,
            "vwap": None,
            "past_n_bars_high": None,
            "current_volume": None,
            "avg_volume_12bars": None,
            "reason": f"5分足履歴不足: {len(data)}本",
            "reasons": [],
        }

    data = _with_vwap(data)
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    current_close = _num(latest.get("Close"))
    previous_close = _num(previous.get("Close"))
    current_low = _num(latest.get("Low"))
    current_volume = _num(latest.get("Volume"))
    vwap = _num(latest.get("vwap"))

    lookback = int(cfg.breakout_lookback_bars)
    past_high = _num(data["High"].iloc[-lookback - 1 : -1].max())
    avg_volume = _num(data["Volume"].shift(1).tail(lookback).mean())
    vwap_ok = bool(current_close > vwap) if cfg.use_vwap else True
    breakout_ok = bool(past_high and current_close > past_high)
    pullback_rebound_ok = bool(current_low <= vwap * 1.003 and current_close > vwap and current_close > previous_close)
    volume_spike_ok = bool(avg_volume and current_volume >= avg_volume * cfg.intraday_volume_multiplier)
    if not cfg.use_volume_spike:
        volume_spike_ok = True

    flags = [vwap_ok, breakout_ok, pullback_rebound_ok, volume_spike_ok]
    ok_count = int(sum(flags))
    reasons = []
    if vwap_ok:
        reasons.append("VWAP上")
    if breakout_ok:
        reasons.append("直近高値突破")
    if pullback_rebound_ok:
        reasons.append("押し目反発")
    if volume_spike_ok:
        reasons.append("出来高急増")

    entry_type = "見送り"
    if breakout_ok:
        entry_type = "ブレイク狙い"
    elif pullback_rebound_ok:
        entry_type = "押し目反発"
    elif vwap_ok:
        entry_type = "VWAP上監視"

    return {
        "vwap_ok": vwap_ok,
        "breakout_ok": breakout_ok,
        "pullback_rebound_ok": pullback_rebound_ok,
        "intraday_volume_spike_ok": volume_spike_ok,
        "intraday_ok_count": ok_count,
        "intraday_total_count": 4,
        "intraday_pass": bool(ok_count >= cfg.intraday_min_ok),
        "intraday_available": True,
        "intraday_data_status": "取得成功",
        "entry_type": entry_type,
        "current_close": current_close,
        "previous_close": previous_close,
        "vwap": vwap,
        "past_n_bars_high": past_high,
        "current_volume": current_volume,
        "avg_volume_12bars": avg_volume,
        "reasons": reasons,
    }


def evaluate_risk_filter(
    signal: Dict[str, Any],
    shares: int = 100,
    config: Optional[Dict[str, Any] | RiskFilterConfig] = None,
) -> Dict[str, Any]:
    cfg = normalize_risk_config(config)
    share_count = int(shares or 100)
    entry_price = _num(
        signal.get("entry_price")
        or signal.get("current_price")
        or signal.get("price")
        or signal.get("close")
    )
    stop_loss = _num(signal.get("stop_loss"))
    take_profit = _num(signal.get("take_profit") or signal.get("take_profit_1") or signal.get("target_1"))

    risk_reasons = []
    if not entry_price:
        risk_reasons.append("entry_priceが取得できない")
    if not stop_loss or (entry_price and stop_loss >= entry_price):
        risk_reasons.append("損切り価格が不正、または買値以上")
    if not take_profit or (entry_price and take_profit <= entry_price):
        risk_reasons.append("利確目標が不正、または買値以下")

    stop_loss_pct = ((entry_price - stop_loss) / entry_price * 100) if entry_price and stop_loss else None
    max_loss_yen = ((entry_price - stop_loss) * share_count) if entry_price and stop_loss else None
    expected_profit_yen = ((take_profit - entry_price) * share_count) if entry_price and take_profit else None
    risk_reward_ratio = (
        expected_profit_yen / max_loss_yen
        if expected_profit_yen is not None and max_loss_yen and max_loss_yen > 0
        else None
    )

    stop_loss_pct_ok = bool(stop_loss_pct is not None and 0 < stop_loss_pct <= cfg.max_stop_loss_pct)
    max_loss_yen_ok = bool(max_loss_yen is not None and 0 < max_loss_yen <= cfg.max_loss_yen_limit)
    risk_reward_ok = bool(risk_reward_ratio is not None and risk_reward_ratio >= cfg.min_risk_reward)

    if not stop_loss_pct_ok:
        risk_reasons.append(f"損切り幅が{cfg.max_stop_loss_pct:.1f}%超")
    if not max_loss_yen_ok:
        risk_reasons.append(f"想定損失が{cfg.max_loss_yen_limit:,.0f}円超")
    if not risk_reward_ok:
        risk_reasons.append(f"損益比が{cfg.min_risk_reward:.2f}未満")

    risk_pass = bool(stop_loss_pct_ok and max_loss_yen_ok and risk_reward_ok)
    if not cfg.enabled:
        risk_pass = True
        risk_reasons = ["リスク条件OFF"]

    result = {
        "risk_pass": risk_pass,
        "risk_filter_enabled": cfg.enabled,
        "risk_entry_price": entry_price,
        "risk_stop_loss": stop_loss,
        "risk_take_profit": take_profit,
        "risk_shares": share_count,
        "stop_loss_pct": round(stop_loss_pct, 2) if stop_loss_pct is not None else None,
        "max_loss_yen": round(max_loss_yen) if max_loss_yen is not None else None,
        "expected_profit_yen": round(expected_profit_yen) if expected_profit_yen is not None else None,
        "risk_reward_ratio": round(risk_reward_ratio, 2) if risk_reward_ratio is not None else None,
        "max_stop_loss_pct": cfg.max_stop_loss_pct,
        "max_loss_yen_limit": cfg.max_loss_yen_limit,
        "min_risk_reward": cfg.min_risk_reward,
        "stop_loss_pct_ok": stop_loss_pct_ok,
        "max_loss_yen_ok": max_loss_yen_ok,
        "risk_reward_ok": risk_reward_ok,
        "risk_reasons": list(dict.fromkeys(risk_reasons)),
    }
    result["risk_filter_json"] = dict(result)
    return result


def evaluate_multi_timeframe_signal(
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame,
    current_time: Any = None,
    config: Optional[Dict[str, Any] | MultiTimeframeConfig] = None,
) -> Dict[str, Any]:
    cfg = normalize_config(config)
    daily_filter = evaluate_daily_filter(daily_df, current_time=None, config=cfg)
    intraday_entry = evaluate_intraday_entry(intraday_df, current_time=current_time, config=cfg)
    daily_pass = bool(daily_filter.get("daily_pass"))
    intraday_available = bool(intraday_entry.get("intraday_available", True))
    intraday_pass = bool(intraday_entry.get("intraday_pass")) if intraday_available else False

    if daily_pass and intraday_pass:
        decision_category = "buy"
    elif daily_pass:
        decision_category = "unavailable" if not intraday_available else "watch"
    else:
        decision_category = "avoid"

    intraday_ok_count = int(intraday_entry.get("intraday_ok_count") or 0)
    intraday_total_count = int(intraday_entry.get("intraday_total_count") or 4)
    raw_score = int(
        round(
            daily_filter.get("daily_ok_count", 0) / max(1, daily_filter.get("daily_total_count", 4)) * 50
            + intraday_ok_count
            / max(1, intraday_total_count)
            * 50
        )
    )
    if decision_category == "buy":
        score = max(70, raw_score)
    elif decision_category == "watch":
        score = min(69, max(60, raw_score))
    elif decision_category == "unavailable":
        score = min(69, max(60, raw_score))
    else:
        score = min(59, raw_score)
    reasons = list(daily_filter.get("reasons", [])) + list(intraday_entry.get("reasons", []))
    entry_type = str(intraday_entry.get("entry_type") or "見送り")
    return {
        "daily_filter": daily_filter,
        "intraday_entry": intraday_entry,
        "decision_category": decision_category,
        "entry_type": entry_type,
        "score": score,
        "reasons": reasons,
        "detail_json": {
            "daily_filter": daily_filter,
            "intraday_entry": intraday_entry,
            "config": cfg.__dict__,
        },
        "multi_timeframe_pass": bool(decision_category == "buy"),
    }
