from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


INTRADAY_BUY_THRESHOLD = 75
INTRADAY_WATCH_THRESHOLD = 65


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_price(value: Optional[float]) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    if value < 100:
        return round(float(value), 1)
    return float(round(float(value)))


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)
    return rsi.fillna(50)


def add_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_index()
    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3
    volume = out["Volume"].fillna(0).clip(lower=0)
    cumulative_volume = volume.cumsum()
    cumulative_value = (typical_price * volume).cumsum()
    fallback_vwap = typical_price.expanding(min_periods=1).mean()
    out["vwap"] = (cumulative_value / cumulative_volume.replace(0, np.nan)).fillna(fallback_vwap)
    out["ma_short"] = out["Close"].rolling(5, min_periods=1).mean()
    out["ma_mid"] = out["Close"].rolling(10, min_periods=1).mean()
    out["ma_long"] = out["Close"].rolling(20, min_periods=1).mean()
    out["volume_avg_20"] = out["Volume"].shift(1).rolling(20, min_periods=3).mean()
    out["rsi_14"] = calculate_rsi(out["Close"], 14)
    out["recent_high_12"] = out["High"].rolling(12, min_periods=1).max()
    out["recent_low_12"] = out["Low"].rolling(12, min_periods=1).min()
    return out


def price_step(price: float) -> int:
    if price < 1000:
        return 10
    if price < 3000:
        return 25
    if price < 10000:
        return 50
    return 100


def support_level(price: float) -> float:
    step = price_step(price)
    return float(np.floor(price / step) * step)


def next_level(price: float, multiplier: int = 1) -> float:
    step = price_step(price)
    return float(np.floor(price / step) * step + step * multiplier)


def _higher_low(df: pd.DataFrame) -> bool:
    if len(df) < 12:
        return False
    prior_low = _num(df["Low"].iloc[-12:-6].min())
    recent_low = _num(df["Low"].iloc[-6:].min())
    current = _num(df["Close"].iloc[-1])
    return bool(recent_low > prior_low * 1.001 and current > recent_low * 1.002)


def _recovered_short_ma(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    return bool(_num(previous["Close"]) <= _num(previous["ma_short"]) and _num(latest["Close"]) > _num(latest["ma_short"]))


def _build_trade_levels(current: float, recent_low: float, level: float, pattern: str) -> Dict[str, Any]:
    step = price_step(current)
    if pattern in {"後場V字回復", "節目ブレイク"} and current > level:
        stop_loss = level
    else:
        stop_loss = min(recent_low, current - step * 0.4)

    if stop_loss >= current:
        stop_loss = min(recent_low, current - step * 0.4)

    target_1 = next_level(current, 1)
    target_2 = next_level(current, 2)
    if target_1 <= current:
        target_1 = current + step
        target_2 = current + step * 2

    entry_low = max(stop_loss + step * 0.1, current - step * 0.4)
    entry_high = current + step * 0.3

    return {
        "buy_zone_low": _round_price(entry_low),
        "buy_zone_high": _round_price(entry_high),
        "stop_loss": _round_price(stop_loss),
        "take_profit_1": _round_price(target_1),
        "take_profit_2": _round_price(target_2),
    }


def _score_signal(
    *,
    above_mas: bool,
    higher_low: bool,
    level_breakout: bool,
    level_hold: bool,
    volume_ratio: float,
    risk_pct: float,
    reward_risk: float,
    rebound_pct: float,
    below_previous_low: bool,
    updating_day_low: bool,
    extreme_volume_low: bool,
    rsi: float,
    target_too_near: bool,
    stop_too_far: bool,
) -> tuple[int, List[str], List[str], List[str]]:
    score = 0
    reasons: List[str] = []
    risks: List[str] = []
    no_buy: List[str] = []

    if above_mas:
        score += 20
        reasons.append("現在値が短期線・中期線・長期線を上回っています")
    else:
        score -= 20
        risks.append("移動平均線の下で推移しています")
        no_buy.append("短期線・中期線・長期線を回復するまで買わない")

    if higher_low:
        score += 20
        reasons.append("直近安値を切り上げています")
    else:
        risks.append("直近安値の切り上げが弱いです")

    if level_breakout or level_hold:
        score += 20
        reasons.append("節目価格を突破または維持しています")
    else:
        risks.append("節目突破が未確認です")

    if volume_ratio >= 1.2:
        score += 15
        reasons.append("出来高が直近平均より増えています")
    elif volume_ratio < 0.7:
        score -= 15
        risks.append("出来高が不足しています")
        no_buy.append("出来高が直近平均を回復するまで買わない")

    if 0 < risk_pct <= 1.2:
        score += 15
        reasons.append("損切りを近く置けます")
    elif stop_too_far:
        score -= 20
        risks.append("損切りが遠すぎます")
        no_buy.append("損切り幅が広すぎる位置では買わない")

    if reward_risk >= 1.0:
        score += 10
        reasons.append("第1利確までの値幅が損切り幅以上あります")
    elif target_too_near:
        score -= 20
        risks.append("第1利確までの距離が近すぎます")
        no_buy.append("利確目安まで近すぎる位置では買わない")

    if rebound_pct >= 1.0:
        score += 10
        reasons.append("当日安値から反発しています")

    if below_previous_low:
        score -= 30
        risks.append("前日安値を割っています")
        no_buy.append("前日安値割れでは買わない")
    if updating_day_low:
        score -= 30
        risks.append("当日安値を更新中です")
        no_buy.append("当日安値更新中は買わない")
    if extreme_volume_low:
        risks.append("出来高が極端に少ないです")
        no_buy.append("出来高が極端に少ない場合は買わない")
    if rsi >= 75:
        score -= 10
        risks.append("RSIが高く過熱感があります")
        no_buy.append("RSI過熱時の飛び乗りは避ける")

    return int(max(0, min(100, score))), reasons, risks, no_buy


def evaluate_intraday_entry(
    *,
    code: str,
    name: str,
    intraday_data: pd.DataFrame,
    previous_low: Optional[float] = None,
) -> Dict[str, Any]:
    if intraday_data.empty:
        return {
            "code": code,
            "name": name,
            "judgement": "取得失敗",
            "intraday_score": 0,
            "error": "場中データが空でした。",
        }
    if len(intraday_data) < 25:
        return {
            "code": code,
            "name": name,
            "judgement": "取得失敗",
            "intraday_score": 0,
            "error": "5分足データが不足しています。",
        }

    df = add_intraday_indicators(intraday_data)
    latest = df.iloc[-1]
    current = _num(latest["Close"])
    day_high = _num(df["High"].max())
    day_low = _num(df["Low"].min())
    recent_high = _num(df["High"].tail(12).max())
    recent_low = _num(df["Low"].tail(12).min())
    ma_short = _num(latest["ma_short"])
    ma_mid = _num(latest["ma_mid"])
    ma_long = _num(latest["ma_long"])
    current_volume = _num(latest["Volume"])
    volume_avg = _num(latest["volume_avg_20"], _num(df["Volume"].tail(20).mean(), 1))
    volume_ratio = current_volume / max(volume_avg, 1)
    vwap = _num(latest.get("vwap"))
    rsi = _num(latest["rsi_14"], 50)
    rebound_pct = (current / day_low - 1) * 100 if day_low else 0
    level = support_level(current)
    next_1 = next_level(current, 1)

    above_mas = bool(current > ma_short and current > ma_mid and current > ma_long)
    higher_low = _higher_low(df)
    level_breakout = bool(current > level and df["Close"].tail(3).min() > level)
    level_hold = bool(current >= level * 1.001 and df["Low"].tail(3).min() >= level * 0.998)
    previous_low_value = _num(previous_low, 0)
    below_previous_low = bool(previous_low_value and current < previous_low_value)
    updating_day_low = bool(_num(latest["Low"]) <= day_low * 1.001 and current <= day_low * 1.004)
    extreme_volume_low = bool(volume_ratio < 0.4 or current_volume <= 0)
    recovered_short_ma = _recovered_short_ma(df)
    near_recent_high = bool(current >= recent_high * 0.995)

    if rebound_pct >= 1.2 and higher_low and above_mas and (level_breakout or level_hold):
        signal_type = "後場V字回復"
    elif level_breakout and near_recent_high and above_mas:
        signal_type = "節目ブレイク"
    elif higher_low and recovered_short_ma and current > ma_mid and current > ma_long:
        signal_type = "押し目再反発"
    else:
        signal_type = "見送り"

    levels = _build_trade_levels(current, recent_low, level, signal_type)
    stop_loss = _num(levels["stop_loss"], current)
    take_profit_1 = _num(levels["take_profit_1"], next_1)
    risk = current - stop_loss
    reward = take_profit_1 - current
    risk_pct = risk / current * 100 if current else 0
    upside_pct = reward / current * 100 if current else 0
    reward_risk = reward / risk if risk > 0 else 0
    stop_too_far = bool(risk_pct > 2.0 or risk <= 0)
    target_too_near = bool(upside_pct < 0.3 or reward_risk < 0.8)

    score, reasons, risks, no_buy = _score_signal(
        above_mas=above_mas,
        higher_low=higher_low,
        level_breakout=level_breakout,
        level_hold=level_hold,
        volume_ratio=volume_ratio,
        risk_pct=risk_pct,
        reward_risk=reward_risk,
        rebound_pct=rebound_pct,
        below_previous_low=below_previous_low,
        updating_day_low=updating_day_low,
        extreme_volume_low=extreme_volume_low,
        rsi=rsi,
        target_too_near=target_too_near,
        stop_too_far=stop_too_far,
    )

    hard_blockers = [
        below_previous_low,
        updating_day_low,
        extreme_volume_low,
        stop_too_far,
        target_too_near,
        not above_mas,
    ]
    if any(hard_blockers):
        judgement = "見送り"
    elif score >= INTRADAY_BUY_THRESHOLD:
        judgement = "買い検討OK"
    elif score >= INTRADAY_WATCH_THRESHOLD:
        judgement = "監視強化"
    else:
        judgement = "見送り"

    if signal_type == "見送り" and judgement != "見送り":
        judgement = "監視強化"

    if not no_buy:
        no_buy.append("節目割れ、出来高不足、利確目安手前での失速")

    return {
        "code": str(code),
        "name": str(name),
        "current_price": _round_price(current),
        "last_time": df.index[-1].strftime("%Y-%m-%d %H:%M") if hasattr(df.index[-1], "strftime") else str(df.index[-1]),
        "day_high": _round_price(day_high),
        "day_low": _round_price(day_low),
        "recent_high": _round_price(recent_high),
        "recent_low": _round_price(recent_low),
        "ma_short": _round_price(ma_short),
        "ma_mid": _round_price(ma_mid),
        "ma_long": _round_price(ma_long),
        "current_volume": int(current_volume),
        "volume_avg": int(volume_avg),
        "volume_ratio": round(volume_ratio, 2),
        "vwap": _round_price(vwap),
        "close_above_vwap": bool(current > vwap) if vwap else False,
        "rsi": round(rsi, 1),
        "rebound_from_day_low_pct": round(rebound_pct, 2),
        "higher_low": higher_low,
        "level_price": _round_price(level),
        "next_level": _round_price(next_1),
        "level_breakout": level_breakout,
        "level_hold": level_hold,
        "above_mas": above_mas,
        "below_previous_low": below_previous_low,
        "updating_day_low": updating_day_low,
        "signal_type": signal_type,
        "judgement": judgement,
        "intraday_score": score,
        "buy_zone": f"{levels['buy_zone_low']:,.0f}〜{levels['buy_zone_high']:,.0f}円"
        if levels["buy_zone_low"] is not None and levels["buy_zone_high"] is not None
        else "-",
        "buy_zone_low": levels["buy_zone_low"],
        "buy_zone_high": levels["buy_zone_high"],
        "stop_loss": levels["stop_loss"],
        "take_profit_1": levels["take_profit_1"],
        "take_profit_2": levels["take_profit_2"],
        "risk_reward": round(reward_risk, 2),
        "risk_pct": round(risk_pct, 2),
        "upside_pct": round(upside_pct, 2),
        "reasons": reasons,
        "risk_notes": risks,
        "no_buy_conditions": no_buy,
        "timeframe": "短期ブレイク〜1日スイング",
        "data": df.tail(80),
        "error": None,
    }
