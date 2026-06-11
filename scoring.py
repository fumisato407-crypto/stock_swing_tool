from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

THEME_KEYWORDS = ("AI", "防衛", "データセンター", "半導体", "電力", "宇宙", "電線")
BUY_ENTRY_TYPES = ("押し目反発", "ブレイク狙い", "急落リバ")
WATCH_ENTRY_TYPES = ("押し目待ち", "反発待ち", "ブレイク待ち", "地合い待ち", "出来高待ち")


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


def _pct_distance(value: float, base: float) -> float:
    if not base:
        return 999.0
    return abs(value / base - 1) * 100


def _entry_zone_text(low: Optional[float], high: Optional[float]) -> str:
    if low is None or high is None:
        return "-"
    return f"{low:,.0f}〜{high:,.0f}円"


def _score_market_context(row: pd.Series) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    close = _num(row.get("Close"))
    sma_25 = _num(row.get("sma_25"))
    sma_75 = _num(row.get("sma_75"))
    return_20d = _num(row.get("return_20d_pct"))

    if close > sma_75:
        score += 8
        reasons.append("終値が75日線を上回り、中期の崩れは限定的")
    if sma_25 > sma_75:
        score += 6
        reasons.append("25日線が75日線を上回り、基調は上向き")
    if close > sma_25 or return_20d > 0:
        score += 6
        reasons.append("個別トレンドで仮地合いを加点")
    return score, reasons


def _score_trend(df: pd.DataFrame) -> Tuple[int, List[str]]:
    row = df.iloc[-1]
    score = 0
    reasons: List[str] = []
    close = _num(row.get("Close"))
    sma_5 = _num(row.get("sma_5"))
    sma_25 = _num(row.get("sma_25"))
    sma_25_before = _num(df["sma_25"].iloc[-6] if len(df) >= 6 else np.nan, sma_25)

    if close > sma_25:
        score += 8
        reasons.append("終値が25日線より上")
    if sma_5 > sma_25:
        score += 6
        reasons.append("5日線が25日線より上")
    if sma_25 > sma_25_before:
        score += 6
        reasons.append("25日線が上向き")
    return score, reasons


def _score_pullback(row: pd.Series) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    close = _num(row.get("Close"))
    sma_25 = _num(row.get("sma_25"))
    rsi = _num(row.get("rsi_14"), 50)
    drawdown = _num(row.get("drawdown_from_20d_high_pct"))

    if -8 <= drawdown <= -3:
        score += 10
        reasons.append("直近20日高値から3〜8%の押し目")
    if 35 <= rsi <= 55:
        score += 5
        reasons.append("RSIが35〜55の反発待ちゾーン")
    if _pct_distance(close, sma_25) <= 2.5:
        score += 5
        reasons.append("終値が25日線付近")
    return score, reasons


def _score_rebound(row: pd.Series) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    volume = _num(row.get("Volume"))
    volume_ma = _num(row.get("volume_ma_20"))
    lower_wick = _num(row.get("lower_wick_ratio"))
    low = _num(row.get("Low"))
    close = _num(row.get("Close"))
    prev_low = _num(row.get("prev_low"), low)
    candle_range = max(_num(row.get("High")) - low, 1e-9)

    if volume_ma and volume >= volume_ma:
        score += 5
        reasons.append("出来高が20日平均以上")
    if lower_wick >= 0.3:
        score += 5
        reasons.append("下ヒゲがあり、安値圏で買い戻しの跡")
    if low >= prev_low * 0.985 and (close - low) / candle_range >= 0.45:
        score += 5
        reasons.append("前日安値を大きく割らずに戻している")
    return score, reasons


def _score_theme(row: pd.Series, theme: str) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    drawdown = _num(row.get("drawdown_from_20d_high_pct"))
    return_5d = _num(row.get("return_5d_pct"))

    if theme.strip():
        score += 5
        reasons.append("監視リストにテーマが設定済み")
    if any(keyword in theme for keyword in THEME_KEYWORDS):
        score += 5
        reasons.append("短期資金が入りやすいテーマを含む")
    if drawdown > -15 and return_5d > -12:
        score += 5
        reasons.append("直近で極端に売られすぎていない")
    return score, reasons


def _risk_penalty(row: pd.Series) -> Tuple[int, List[str]]:
    penalty = 0
    notes: List[str] = []
    close = _num(row.get("Close"))
    sma_25 = _num(row.get("sma_25"))
    rsi = _num(row.get("rsi_14"), 50)
    return_5d = _num(row.get("return_5d_pct"))
    upper_wick = _num(row.get("upper_wick_ratio"))
    volume = _num(row.get("Volume"))
    volume_ma = _num(row.get("volume_ma_20"))

    if rsi >= 70:
        penalty -= 10
        notes.append("RSI70以上で短期過熱")
    if return_5d >= 15:
        penalty -= 10
        notes.append("直近5日で15%以上上昇")
    if upper_wick >= 0.45:
        penalty -= 5
        notes.append("上ヒゲが大きく、上値で売りが出ている")
    if sma_25 and close < sma_25 * 0.95:
        penalty -= 15
        notes.append("終値が25日線を大きく下回る")
    if volume_ma and volume < volume_ma * 0.5:
        penalty -= 5
        notes.append("出来高が極端に少ない")
    return penalty, notes


def classify_score(score: int) -> str:
    if score >= 70:
        return "買い候補"
    if score >= 60:
        return "監視"
    return "触らない"


def _strict_buy_type(row: pd.Series) -> Optional[str]:
    close = _num(row.get("Close"))
    sma_5 = _num(row.get("sma_5"))
    sma_25 = _num(row.get("sma_25"))
    rsi = _num(row.get("rsi_14"), 50)
    drawdown = _num(row.get("drawdown_from_20d_high_pct"))
    lower_wick = _num(row.get("lower_wick_ratio"))
    volume = _num(row.get("Volume"))
    volume_ma = _num(row.get("volume_ma_20"))
    return_3d = _num(row.get("return_3d_pct"))
    high_20_prev = _num(row.get("recent_20_high_prev"), _num(row.get("recent_20_high")))

    volume_increased = bool(volume_ma and volume >= volume_ma * 1.1)
    has_lower_wick = lower_wick >= 0.25
    near_20_high = bool(high_20_prev and close >= high_20_prev * 0.98)

    if close > sma_25 and -8 <= drawdown <= -3 and 35 <= rsi <= 55 and has_lower_wick:
        return "押し目反発"
    if near_20_high and volume_increased and sma_5 > sma_25:
        return "ブレイク狙い"
    if return_3d <= -4 and volume_increased and has_lower_wick:
        return "急落リバ"
    return None


def _fallback_buy_type(row: pd.Series) -> str:
    close = _num(row.get("Close"))
    sma_5 = _num(row.get("sma_5"))
    sma_25 = _num(row.get("sma_25"))
    lower_wick = _num(row.get("lower_wick_ratio"))
    return_3d = _num(row.get("return_3d_pct"))
    high_20_prev = _num(row.get("recent_20_high_prev"), _num(row.get("recent_20_high")))

    if return_3d <= -3 and lower_wick >= 0.15:
        return "急落リバ"
    if high_20_prev and close >= high_20_prev * 0.965 and sma_5 > sma_25:
        return "ブレイク狙い"
    return "押し目反発"


def _watch_type(row: pd.Series, score_breakdown: Dict[str, int]) -> str:
    close = _num(row.get("Close"))
    sma_25 = _num(row.get("sma_25"))
    sma_75 = _num(row.get("sma_75"))
    drawdown = _num(row.get("drawdown_from_20d_high_pct"))
    volume = _num(row.get("Volume"))
    volume_ma = _num(row.get("volume_ma_20"))
    high_20_prev = _num(row.get("recent_20_high_prev"), _num(row.get("recent_20_high")))
    volume_ratio = volume / max(volume_ma, 1)

    if close < sma_25 * 0.97 or close < sma_75 or score_breakdown["market_score"] < 12:
        return "地合い待ち"
    if high_20_prev and (close >= high_20_prev * 0.96 or drawdown > -2):
        return "ブレイク待ち"
    if -8 <= drawdown <= -2.5 and close >= sma_25 * 0.97:
        return "反発待ち"
    if volume_ratio < 0.9:
        return "出来高待ち"
    return "押し目待ち"


def determine_entry_type(row: pd.Series, category: str, score_breakdown: Dict[str, int]) -> str:
    if category == "買い候補":
        return _strict_buy_type(row) or _fallback_buy_type(row)
    if category == "監視":
        return _watch_type(row, score_breakdown)
    return "見送り"


def _buy_plan(row: pd.Series, entry_type: str) -> Dict[str, Any]:
    close = _num(row.get("Close"))
    sma_25 = _num(row.get("sma_25"), close)
    recent_5_high = _num(row.get("recent_5_high"), close * 1.02)
    recent_20_high = _num(row.get("recent_20_high"), close * 1.04)
    recent_5_low = _num(row.get("recent_5_low"), close * 0.98)
    high_20_prev = _num(row.get("recent_20_high_prev"), recent_20_high)
    prev_high = _num(row.get("prev_high"), close * 1.005)
    prev_low = _num(row.get("prev_low"), close * 0.98)

    if entry_type == "押し目反発":
        entry_low = min(sma_25, close) * 0.995
        entry_high = max(sma_25, close) * 1.005
        stop_loss = min(recent_5_low, sma_25 * 0.98)
        target_1 = max(recent_5_high, entry_high * 1.02)
        target_2 = max(recent_20_high, entry_high * 1.04)
        trigger = "前日高値突破、5日線回復、または25日線付近で反発確認"
        invalid = "前日安値割れ、25日線の明確な下抜け、出来高不足"
        holding = "1〜3営業日"
    elif entry_type == "ブレイク狙い":
        breakout = max(high_20_prev, recent_5_high)
        entry_low = breakout
        entry_high = breakout * 1.01
        stop_loss = breakout * 0.985
        entry_mid = (entry_low + entry_high) / 2
        target_1 = entry_mid * 1.03
        target_2 = entry_mid * 1.05
        trigger = "直近高値突破後に出来高を伴う"
        invalid = "ブレイクライン割れ、上ヒゲ急増、出来高失速"
        holding = "1〜5営業日"
    else:
        entry_low = prev_high
        entry_high = prev_high * 1.01
        stop_loss = prev_low * 0.995
        entry_mid = (entry_low + entry_high) / 2
        target_1 = entry_mid * 1.02
        target_2 = entry_mid * 1.04
        trigger = "前日高値突破、または寄り後に前日安値を割らない"
        invalid = "前日安値割れ、寄り後の戻り失敗、出来高不足"
        holding = "1〜3営業日"

    return _plan_payload(
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        watch_condition=f"{entry_type}の条件がそろっているため、トリガー確認後に検討",
        entry_trigger=trigger,
        invalidation_condition=invalid,
        expected_holding_days=holding,
        no_buy_conditions=f"{invalid}。条件未達の飛び乗りは禁止。",
        is_provisional=False,
    )


def _watch_plan(row: pd.Series, entry_type: str) -> Dict[str, Any]:
    close = _num(row.get("Close"))
    sma_5 = _num(row.get("sma_5"), close)
    sma_25 = _num(row.get("sma_25"), close)
    recent_5_high = _num(row.get("recent_5_high"), close * 1.02)
    recent_20_high = _num(row.get("recent_20_high"), close * 1.04)
    recent_5_low = _num(row.get("recent_5_low"), close * 0.98)
    recent_20_low = _num(row.get("recent_20_low"), close * 0.95)
    high_20_prev = _num(row.get("recent_20_high_prev"), recent_20_high)
    prev_high = _num(row.get("prev_high"), close * 1.005)
    prev_low = _num(row.get("prev_low"), close * 0.98)

    if entry_type == "押し目待ち":
        entry_low = sma_25 * 0.995
        entry_high = max(sma_25 * 1.01, min(close, recent_5_high) * 0.995)
        stop_loss = min(recent_5_low, sma_25 * 0.98)
        target_1 = max(recent_5_high, entry_high * 1.02)
        target_2 = max(recent_20_high, entry_high * 1.04)
        watch_condition = "25日線付近まで押したら反発狙い"
        trigger = "前日高値突破、または5日線回復"
        invalid = "25日線を明確に割る、出来高が増えない、地合い悪化"
    elif entry_type == "反発待ち":
        entry_low = min(close, sma_25) * 0.995
        entry_high = max(close, sma_5, sma_25) * 1.005
        stop_loss = min(prev_low, recent_5_low, sma_25 * 0.98)
        target_1 = max(prev_high, recent_5_high)
        target_2 = max(recent_20_high, entry_high * 1.04)
        watch_condition = "下げ止まりを確認できたら反発狙い"
        trigger = "前日高値突破、5日線回復、下ヒゲ陽線"
        invalid = "前日安値割れ、25日線割れ、戻りの出来高不足"
    elif entry_type == "ブレイク待ち":
        breakout = max(high_20_prev, recent_5_high)
        entry_low = breakout
        entry_high = breakout * 1.01
        stop_loss = breakout * 0.985
        entry_mid = (entry_low + entry_high) / 2
        target_1 = entry_mid * 1.03
        target_2 = entry_mid * 1.05
        watch_condition = "直近高値を出来高つきで抜くのを待つ"
        trigger = "直近20日高値突破、かつ出来高20日平均以上"
        invalid = "高値突破失敗、上ヒゲで失速、ブレイクライン割れ"
    elif entry_type == "地合い待ち":
        entry_low = max(sma_25 * 0.995, close * 0.995)
        entry_high = max(sma_25, close) * 1.01
        stop_loss = min(recent_20_low, sma_25 * 0.97)
        target_1 = max(recent_5_high, entry_high * 1.02)
        target_2 = max(recent_20_high, entry_high * 1.04)
        watch_condition = "25日線回復や地合い改善を待つ"
        trigger = "終値で25日線回復、または5日線が上向きに転換"
        invalid = "25日線からさらに下放れ、75日線割れ、指数の悪化"
    else:
        entry_low = close * 0.995
        entry_high = close * 1.01
        stop_loss = min(prev_low, recent_5_low)
        target_1 = max(recent_5_high, entry_high * 1.02)
        target_2 = max(recent_20_high, entry_high * 1.04)
        watch_condition = "出来高が戻るまで待つ"
        trigger = "出来高が20日平均以上に増え、前日高値を突破"
        invalid = "出来高が増えない、反発が弱い、前日安値割れ"

    return _plan_payload(
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        watch_condition=watch_condition,
        entry_trigger=trigger,
        invalidation_condition=invalid,
        expected_holding_days="1〜5営業日",
        no_buy_conditions=f"{trigger}が出ていない状態での先回り買い。{invalid}。",
        is_provisional=True,
    )


def _avoid_plan() -> Dict[str, Any]:
    return _plan_payload(
        entry_low=None,
        entry_high=None,
        stop_loss=None,
        target_1=None,
        target_2=None,
        watch_condition="買いの優位性が足りないため見送り",
        entry_trigger="-",
        invalidation_condition="スコア不足、トレンド不足、出来高不足、またはリスク過多",
        expected_holding_days="-",
        no_buy_conditions="反発確認なし、出来高不足、25日線を大きく下回る、高値掴みになりやすい位置。",
        is_provisional=False,
    )


def _plan_payload(
    *,
    entry_low: Optional[float],
    entry_high: Optional[float],
    stop_loss: Optional[float],
    target_1: Optional[float],
    target_2: Optional[float],
    watch_condition: str,
    entry_trigger: str,
    invalidation_condition: str,
    expected_holding_days: str,
    no_buy_conditions: str,
    is_provisional: bool,
) -> Dict[str, Any]:
    low = _round_price(entry_low)
    high = _round_price(entry_high)
    stop = _round_price(stop_loss)
    t1 = _round_price(target_1)
    t2 = _round_price(target_2)
    zone = _entry_zone_text(low, high)
    return {
        "entry_zone_low": low,
        "entry_zone_high": high,
        "entry_zone": zone,
        "stop_loss": stop,
        "target_1": t1,
        "target_2": t2,
        "expected_holding_days": expected_holding_days,
        "watch_condition": watch_condition,
        "wait_condition": watch_condition,
        "entry_trigger": entry_trigger,
        "buy_condition": entry_trigger,
        "invalidation_condition": invalidation_condition,
        "invalid_conditions": invalidation_condition,
        "no_buy_conditions": no_buy_conditions,
        "provisional_entry_zone": zone if is_provisional else "-",
        "provisional_stop_loss": stop if is_provisional else None,
        "provisional_target_1": t1 if is_provisional else None,
        "provisional_target_2": t2 if is_provisional else None,
        "is_provisional": is_provisional,
    }


def _build_trade_plan(row: pd.Series, category: str, entry_type: str) -> Dict[str, Any]:
    if category == "買い候補":
        return _buy_plan(row, entry_type)
    if category == "監視":
        return _watch_plan(row, entry_type)
    return _avoid_plan()


def _calculate_risk_reward(signal: Dict[str, Any]) -> Dict[str, Any]:
    if signal["category"] == "触らない":
        return {
            "risk_reward": "-",
            "risk_reward_label": "算出対象外",
            "estimated_upside_percent": 0.0,
            "estimated_downside_percent": 0.0,
            "expected_value_label": "低",
            "confidence": "低",
        }

    low = signal.get("entry_zone_low")
    high = signal.get("entry_zone_high")
    entry_mid = (low + high) / 2 if low is not None and high is not None else signal["price"]
    stop_loss = signal.get("stop_loss")
    target_1 = signal.get("target_1")

    risk_reward = 0.0
    upside_pct = 0.0
    downside_pct = 0.0
    if stop_loss and target_1 and entry_mid:
        risk = entry_mid - stop_loss
        reward = target_1 - entry_mid
        if risk > 0 and reward > 0:
            risk_reward = reward / risk
            upside_pct = reward / entry_mid * 100
            downside_pct = risk / entry_mid * 100

    rr = round(risk_reward, 2)
    score = signal["score"]
    if signal["category"] == "買い候補" and rr >= 2.0 and score >= 75:
        expected = "高"
    elif rr >= 1.5 and score >= 70:
        expected = "中"
    elif signal["category"] == "監視" and rr >= 1.5 and score >= 65:
        expected = "中"
    else:
        expected = "低"

    if signal["category"] == "買い候補" and score >= 75 and rr >= 1.5:
        confidence = "高"
    elif signal["category"] in {"買い候補", "監視"} and score >= 65:
        confidence = "中"
    else:
        confidence = "低"

    label = f"参考 {rr}" if signal["category"] == "監視" else str(rr)
    return {
        "risk_reward": rr,
        "risk_reward_label": label,
        "estimated_upside_percent": round(upside_pct, 1),
        "estimated_downside_percent": round(downside_pct, 1),
        "expected_value_label": expected,
        "confidence": confidence,
    }


def _negative_reasons(row: pd.Series, risk_notes: List[str], category: str) -> List[str]:
    close = _num(row.get("Close"))
    sma_5 = _num(row.get("sma_5"))
    sma_25 = _num(row.get("sma_25"))
    sma_75 = _num(row.get("sma_75"))
    volume = _num(row.get("Volume"))
    volume_ma = _num(row.get("volume_ma_20"))
    drawdown = _num(row.get("drawdown_from_20d_high_pct"))

    reasons = list(risk_notes)
    if close <= sma_25:
        reasons.append("終値が25日線を回復していない")
    if sma_5 <= sma_25:
        reasons.append("5日線が25日線を下回っている")
    if close < sma_75:
        reasons.append("終値が75日線を下回り地合いが弱い")
    if volume_ma and volume < volume_ma:
        reasons.append("出来高が20日平均未満")
    if drawdown > -2:
        reasons.append("押し目が浅く、高値掴みリスクがある")
    if category != "買い候補":
        reasons.append("エントリー条件未達")
    return _dedupe(reasons)


def _wait_reasons(entry_type: str) -> List[str]:
    mapping = {
        "押し目反発": ["前日高値突破確認", "5日線回復確認", "前日安値を割らない確認"],
        "ブレイク狙い": ["直近高値突破確認", "出来高増加確認", "ブレイクライン維持確認"],
        "急落リバ": ["前日高値突破確認", "前日安値を割らない確認", "戻りの出来高確認"],
        "押し目待ち": ["25日線付近まで押すのを待つ", "RSIが35〜55に入るのを待つ", "下ヒゲまたは反発足待ち"],
        "反発待ち": ["前日高値突破待ち", "5日線回復待ち", "下げ止まり確認待ち"],
        "ブレイク待ち": ["直近20日高値突破待ち", "出来高増加待ち", "上ヒゲで終わらない確認待ち"],
        "地合い待ち": ["25日線回復待ち", "5日線の上向き転換待ち", "地合い悪化停止待ち"],
        "出来高待ち": ["出来高20日平均回復待ち", "前日高値突破待ち", "買い戻しの強さ確認待ち"],
        "見送り": ["条件が整うまで触らない"],
    }
    return mapping.get(entry_type, ["条件待ち"])


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def score_stock(df: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, Any]:
    row = df.iloc[-1]
    theme = str(meta.get("theme", "") or "")

    market_score, market_reasons = _score_market_context(row)
    trend_score, trend_reasons = _score_trend(df)
    pullback_score, pullback_reasons = _score_pullback(row)
    volume_score, volume_reasons = _score_rebound(row)
    theme_score, theme_reasons = _score_theme(row, theme)
    risk_penalty, risk_notes = _risk_penalty(row)

    raw_score = market_score + trend_score + pullback_score + volume_score + theme_score + risk_penalty
    total_score = int(max(0, min(100, round(raw_score))))
    category = classify_score(total_score)
    score_breakdown = {
        "market_score": market_score,
        "trend_score": trend_score,
        "pullback_score": pullback_score,
        "volume_score": volume_score,
        "theme_score": theme_score,
        "risk_penalty": risk_penalty,
        "total_score": total_score,
    }
    entry_type = determine_entry_type(row, category, score_breakdown)
    plan = _build_trade_plan(row, category, entry_type)
    positive_reasons = _dedupe(market_reasons + trend_reasons + pullback_reasons + volume_reasons + theme_reasons)
    negative_reasons = _negative_reasons(row, risk_notes, category)
    wait_reasons = _wait_reasons(entry_type)

    signal: Dict[str, Any] = {
        "code": str(meta.get("code", "")),
        "name": str(meta.get("name", "")),
        "theme": theme,
        "market": str(meta.get("market", "")),
        "last_date": row.name.strftime("%Y-%m-%d") if hasattr(row.name, "strftime") else str(row.name),
        "price": _round_price(_num(row.get("Close"))),
        "score": total_score,
        "category": category,
        "entry_type": entry_type,
        "score_breakdown": score_breakdown,
        "positive_reasons": positive_reasons,
        "negative_reasons": negative_reasons,
        "wait_reasons": wait_reasons,
        "reasons": positive_reasons,
        "risk_notes": risk_notes,
        "raw_metrics": {
            "rsi_14": round(_num(row.get("rsi_14")), 1),
            "volume_ratio": round(_num(row.get("Volume")) / max(_num(row.get("volume_ma_20")), 1), 2),
            "drawdown_from_20d_high_pct": round(_num(row.get("drawdown_from_20d_high_pct")), 1),
            "return_3d_pct": round(_num(row.get("return_3d_pct")), 1),
            "return_5d_pct": round(_num(row.get("return_5d_pct")), 1),
        },
    }
    signal.update(plan)
    if category == "監視":
        signal["monitoring_reason"] = signal["watch_condition"]
    elif category == "買い候補":
        signal["monitoring_reason"] = f"{entry_type}の買い条件が近い"
    else:
        signal["monitoring_reason"] = "買いの優位性が足りない"
    signal.update(_calculate_risk_reward(signal))
    return signal
