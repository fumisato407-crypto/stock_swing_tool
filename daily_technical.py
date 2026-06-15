from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from data_fetcher import fetch_price_data, normalize_jp_symbol
from time_utils import now_jst_display


MIN_DAILY_HISTORY_ROWS = 120
DEFAULT_DAILY_FETCH_PERIOD = "1y"
TECHNICAL_PRESET_LABELS = {
    "standard_swing": "標準スイング",
    "breakout": "ブレイク重視",
    "pullback": "押し目重視",
    "volume": "出来高重視",
    "candle": "ローソク足重視",
    "lightweight": "軽量モード",
}


@dataclass(frozen=True)
class TechnicalScoreConfig:
    preset_name: str = "standard_swing"
    use_trend: bool = True
    use_entry_position: bool = True
    use_volume: bool = True
    use_candle: bool = True
    use_breakout: bool = True
    use_momentum: bool = True
    use_risk_reward: bool = True
    use_penalty: bool = True
    use_hard_filter: bool = True


def get_default_technical_config() -> Dict[str, Any]:
    return asdict(TechnicalScoreConfig())


def get_replay_technical_config() -> Dict[str, Any]:
    config = TechnicalScoreConfig(preset_name="standard_swing")
    return asdict(config)


def get_intraday_filter_config() -> Dict[str, Any]:
    config = TechnicalScoreConfig(
        preset_name="lightweight",
        use_entry_position=False,
        use_candle=False,
        use_breakout=False,
        use_risk_reward=False,
        use_penalty=False,
    )
    return asdict(config)


def get_technical_preset_options() -> Dict[str, str]:
    return dict(TECHNICAL_PRESET_LABELS)


def get_technical_config_for_preset(preset_name: str) -> Dict[str, Any]:
    key = str(preset_name or "standard_swing")
    base = TechnicalScoreConfig(preset_name=key)
    presets = {
        "standard_swing": base,
        "breakout": replace(base, use_entry_position=False, use_risk_reward=False),
        "pullback": base,
        "volume": replace(base, use_trend=False, use_entry_position=False, use_momentum=False, use_risk_reward=False),
        "candle": replace(base, use_trend=False, use_entry_position=False, use_volume=False, use_breakout=False, use_momentum=False, use_risk_reward=False),
        "lightweight": replace(base, use_entry_position=False, use_candle=False, use_breakout=False, use_risk_reward=False, use_penalty=False),
    }
    return asdict(presets.get(key, base))


def normalize_technical_config(config: Any = None) -> TechnicalScoreConfig:
    if isinstance(config, TechnicalScoreConfig):
        return config
    if config is None:
        return TechnicalScoreConfig()
    values = dict(config or {})
    preset_name = str(values.get("preset_name") or "standard_swing")
    merged = get_technical_config_for_preset(preset_name)
    merged.update({key: value for key, value in values.items() if key in merged})
    return TechnicalScoreConfig(**merged)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_price(value: Any) -> float | None:
    try:
        number = float(value)
        if not np.isfinite(number):
            return None
        if number < 100:
            return round(number, 1)
        return float(round(number))
    except (TypeError, ValueError):
        return None


def _pct(value: float, base: float) -> float | None:
    if not base:
        return None
    return (value / base - 1) * 100


def _score_add(
    score: int,
    points: int,
    condition: bool,
    reasons: List[str],
    text: str,
) -> int:
    if condition:
        reasons.append(f"{text} +{points}")
        return score + points
    return score


def _error_record(record: Dict[str, Any], fetch_target: str, error_type: str, error_message: str) -> Dict[str, Any]:
    normalized = normalize_jp_symbol(record.get("normalized_symbol") or record.get("code") or record.get("raw_code"))
    return {
        "code": str(record.get("code", "") or "").replace(".T", ""),
        "name": str(record.get("name", "") or ""),
        "normalized_symbol": normalized,
        "fetch_target": fetch_target,
        "error_type": error_type or "exception",
        "error_message": error_message or "日足データ取得に失敗しました。",
        "last_attempt_at": now_jst_display(),
    }


def calculate_macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


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


def add_daily_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_index()
    out["date"] = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    out["open"] = out["Open"]
    out["high"] = out["High"]
    out["low"] = out["Low"]
    out["close"] = out["Close"]
    out["volume"] = out["Volume"].fillna(0)
    out["previous_close"] = out["close"].shift(1)
    out["daily_change_pct"] = (out["close"] / out["previous_close"] - 1) * 100
    out["gap_pct"] = (out["open"] / out["previous_close"] - 1) * 100

    out["ma5"] = out["close"].rolling(5, min_periods=5).mean()
    out["ma25"] = out["close"].rolling(25, min_periods=25).mean()
    out["ma75"] = out["close"].rolling(75, min_periods=75).mean()
    out["ma5_slope"] = out["ma5"] - out["ma5"].shift(3)
    out["ma25_slope"] = out["ma25"] - out["ma25"].shift(5)
    out["ma75_slope"] = out["ma75"] - out["ma75"].shift(10)
    out["volume_ma5"] = out["volume"].rolling(5, min_periods=5).mean()
    out["volume_ma20"] = out["volume"].rolling(20, min_periods=20).mean()
    out["volume_ratio_20"] = out["volume"] / out["volume_ma20"].replace(0, np.nan)
    out["rsi14"] = calculate_rsi(out["close"], 14)
    macd, macd_signal, macd_hist = calculate_macd(out["close"])
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    high_low = out["high"] - out["low"]
    high_prev_close = (out["high"] - out["previous_close"]).abs()
    low_prev_close = (out["low"] - out["previous_close"]).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    out["atr14"] = true_range.rolling(14, min_periods=14).mean()
    out["atr_pct"] = out["atr14"] / out["close"] * 100

    out["recent_high_5"] = out["high"].rolling(5, min_periods=5).max()
    out["recent_high_20"] = out["high"].rolling(20, min_periods=20).max()
    out["recent_high_20_prev"] = out["high"].shift(1).rolling(20, min_periods=20).max()
    out["recent_low_5"] = out["low"].rolling(5, min_periods=5).min()
    out["recent_low_20"] = out["low"].rolling(20, min_periods=20).min()
    out["previous_recent_low_20"] = out["low"].shift(5).rolling(20, min_periods=10).min()

    out["distance_from_ma5_pct"] = (out["close"] / out["ma5"] - 1) * 100
    out["distance_from_ma25_pct"] = (out["close"] / out["ma25"] - 1) * 100
    out["distance_from_ma75_pct"] = (out["close"] / out["ma75"] - 1) * 100
    out["distance_from_recent_high_20_pct"] = (out["close"] / out["recent_high_20"] - 1) * 100
    out["distance_from_recent_low_20_pct"] = (out["close"] / out["recent_low_20"] - 1) * 100

    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    body = (out["close"] - out["open"]).abs()
    upper_shadow = out["high"] - np.maximum(out["open"], out["close"])
    lower_shadow = np.minimum(out["open"], out["close"]) - out["low"]
    out["candle_body_pct"] = body / out["close"] * 100
    out["upper_shadow_pct"] = upper_shadow.clip(lower=0) / out["close"] * 100
    out["lower_shadow_pct"] = lower_shadow.clip(lower=0) / out["close"] * 100
    out["body_to_range_ratio"] = (body / candle_range).fillna(0)
    out["is_bullish_candle"] = out["close"] > out["open"]
    out["is_bearish_candle"] = out["close"] < out["open"]
    out["is_long_bullish_candle"] = (
        out["is_bullish_candle"] & (out["body_to_range_ratio"] >= 0.6) & (out["daily_change_pct"] >= 2.0)
    )
    out["is_long_bearish_candle"] = (
        out["is_bearish_candle"] & (out["body_to_range_ratio"] >= 0.6) & (out["daily_change_pct"] <= -2.0)
    )
    close_in_range = (out["close"] - out["low"]) / candle_range
    out["is_lower_shadow_bullish"] = (lower_shadow.clip(lower=0) >= body) & (
        out["is_bullish_candle"] | (close_in_range >= 0.7)
    )
    out["is_upper_shadow_warning"] = (upper_shadow.clip(lower=0) >= body) | (out["upper_shadow_pct"] >= 2.0)
    prev_open = out["open"].shift(1)
    prev_close = out["close"].shift(1)
    prev_bearish = prev_close < prev_open
    out["is_engulfing_bullish"] = (
        prev_bearish
        & out["is_bullish_candle"]
        & (out["open"] <= prev_close)
        & (out["close"] >= prev_open)
    )
    out["is_three_bearish_days"] = (
        out["is_bearish_candle"]
        & out["is_bearish_candle"].shift(1, fill_value=False)
        & out["is_bearish_candle"].shift(2, fill_value=False)
    )
    out["trading_value"] = out["close"] * out["volume"]
    out["return_3d_pct"] = (out["close"] / out["close"].shift(3) - 1) * 100
    return out


def build_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return add_daily_technical_indicators(df)


def fetch_daily_technical_data(
    records: Iterable[Dict[str, Any]],
    period: str = DEFAULT_DAILY_FETCH_PERIOD,
    min_rows: int = MIN_DAILY_HISTORY_ROWS,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    daily_data: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []
    for record in records:
        normalized = normalize_jp_symbol(record.get("normalized_symbol") or record.get("code") or record.get("raw_code"))
        if not normalized:
            errors.append(_error_record(record, "daily_technical", "invalid_symbol", "銘柄コードを正規化できませんでした。"))
            continue
        code = normalized.replace(".T", "")
        fetched = fetch_price_data(code, period=period, interval="1d")
        if fetched.error:
            errors.append(
                _error_record(
                    {**record, "code": code, "normalized_symbol": normalized},
                    "daily_technical",
                    fetched.error_type or "exception",
                    fetched.error_message or fetched.error or "日足データ取得に失敗しました。",
                )
            )
            continue
        if fetched.data.empty:
            errors.append(_error_record({**record, "code": code}, "daily_technical", "empty_dataframe", "日足データが空でした。"))
            continue
        if len(fetched.data.dropna(subset=["Open", "High", "Low", "Close"])) == 0:
            errors.append(
                _error_record({**record, "code": code}, "daily_technical", "no_rows_after_dropna", "OHLC有効行が0件でした。")
            )
            continue
        if len(fetched.data) < int(min_rows):
            errors.append(
                _error_record(
                    {**record, "code": code},
                    "daily_technical",
                    "insufficient_history",
                    f"日足履歴が不足しています。必要{min_rows}本、取得{len(fetched.data)}本。",
                )
            )
            continue
        try:
            prepared = add_daily_technical_indicators(fetched.data)
        except Exception as exc:
            errors.append(
                _error_record({**record, "code": code}, "daily_technical", "exception", f"日足指標計算に失敗しました: {exc}")
            )
            continue
        latest = prepared.iloc[-1].to_dict()
        daily_data[code] = {
            "code": code,
            "name": str(record.get("name", "") or code),
            "normalized_symbol": normalized,
            "data": prepared,
            "latest": latest,
            "fetched_rows": len(prepared),
            "last_attempt_at": fetched.last_attempt_at or now_jst_display(),
        }
    return daily_data, errors


def _stop_and_target(row: pd.Series) -> Dict[str, Any]:
    close = _num(row.get("close"))
    candidates = [
        _num(row.get("recent_low_5")),
        _num(row.get("recent_low_20")),
        _num(row.get("ma25")),
    ]
    valid_stops = [value for value in candidates if value and value < close]
    stop_loss = max(valid_stops) if valid_stops else None
    atr_target = close + _num(row.get("atr14")) * 2 if _num(row.get("atr14")) else None
    high_target = _num(row.get("recent_high_20"))
    target_candidates = [value for value in (high_target, atr_target) if value and value > close]
    target = max(target_candidates) if target_candidates else None
    risk = close - stop_loss if stop_loss else None
    reward = target - close if target else None
    risk_reward = reward / risk if risk and reward and risk > 0 else None
    return {
        "stop_loss_candidate": _round_price(stop_loss),
        "target_price_candidate": _round_price(target),
        "stop_loss_distance_pct": round(risk / close * 100, 2) if risk and close else None,
        "risk_reward": round(risk_reward, 2) if risk_reward is not None else None,
    }


def _trend_score(row: pd.Series, breakdown: Dict[str, List[str]]) -> int:
    reasons: List[str] = []
    score = 0
    close = _num(row.get("close"))
    score = _score_add(score, 2, close > _num(row.get("ma5")), reasons, "終値が5日線より上")
    score = _score_add(score, 4, close > _num(row.get("ma25")), reasons, "終値が25日線より上")
    score = _score_add(score, 3, close > _num(row.get("ma75")), reasons, "終値が75日線より上")
    score = _score_add(score, 2, _num(row.get("ma5_slope")) > 0, reasons, "5日線が上向き")
    score = _score_add(score, 3, _num(row.get("ma25_slope")) > 0, reasons, "25日線が上向き")
    score = _score_add(score, 2, _num(row.get("ma5")) > _num(row.get("ma25")), reasons, "5日線が25日線より上")
    score = _score_add(
        score,
        1,
        _num(row.get("recent_low_5")) > _num(row.get("previous_recent_low_20")),
        reasons,
        "直近5日安値が前回安値目安より上",
    )
    breakdown["trend"] = reasons
    return min(score, 14)


def _entry_position_score(row: pd.Series, rr: Dict[str, Any], breakdown: Dict[str, List[str]]) -> int:
    reasons: List[str] = []
    score = 0
    dist_ma25 = _num(row.get("distance_from_ma25_pct"), 999)
    dist_ma5 = _num(row.get("distance_from_ma5_pct"), 999)
    stop_loss_distance = _num(rr.get("stop_loss_distance_pct"), 999)
    upside = abs(_num(row.get("distance_from_recent_high_20_pct"), 0))
    score = _score_add(score, 4, -2 <= dist_ma25 <= 8, reasons, f"25日線から{dist_ma25:.1f}%")
    score = _score_add(score, 2, -1 <= dist_ma5 <= 4, reasons, f"5日線から{dist_ma5:.1f}%")
    score = _score_add(score, 2, stop_loss_distance <= 5, reasons, f"損切り幅{stop_loss_distance:.1f}%")
    score = _score_add(score, 1, upside >= 2, reasons, "20日高値まで上昇余地2%以上")
    score = _score_add(score, 2, _num(rr.get("risk_reward")) >= 1.2, reasons, f"リスクリワード{rr.get('risk_reward')}倍")
    breakdown["entry_position"] = reasons
    return min(score, 10)


def _volume_score(row: pd.Series, breakdown: Dict[str, List[str]]) -> int:
    reasons: List[str] = []
    score = 0
    ratio = _num(row.get("volume_ratio_20"))
    change = _num(row.get("daily_change_pct"))
    score = _score_add(score, 2, ratio >= 1.0, reasons, f"出来高20日平均の{ratio:.1f}倍")
    score = _score_add(score, 2, ratio >= 1.2, reasons, "出来高20日平均の1.2倍以上")
    score = _score_add(score, 2, ratio >= 1.5, reasons, "出来高20日平均の1.5倍以上")
    score = _score_add(score, 2, change > 0 and _num(row.get("volume")) > _num(row.get("volume_ma20")), reasons, "上昇日に出来高増")
    score = _score_add(score, 2, _down_day_volume_decreased(row), reasons, "下落日の出来高が20日平均未満")
    score = _score_add(score, 1, _num(row.get("volume_ma5")) > _num(row.get("volume_ma20")), reasons, "5日出来高平均が20日平均より上")
    score = _score_add(score, 1, _num(row.get("trading_value")) >= 1_000_000_000, reasons, "売買代金10億円以上")
    breakdown["volume"] = reasons
    return min(score, 10)


def _down_day_volume_decreased(row: pd.Series) -> bool:
    return _num(row.get("daily_change_pct")) < 0 and _num(row.get("volume")) < _num(row.get("volume_ma20"))


def _candle_score(row: pd.Series, breakdown: Dict[str, List[str]]) -> int:
    reasons: List[str] = []
    score = 0
    upper_shadow_short = _num(row.get("upper_shadow_pct"), 999) < 1.0 or _num(row.get("upper_shadow_pct")) < _num(row.get("candle_body_pct"))
    close_in_top = (_num(row.get("close")) - _num(row.get("low"))) / max(_num(row.get("high")) - _num(row.get("low")), 1e-9) >= 0.75
    score = _score_add(score, 2, bool(row.get("is_bullish_candle")), reasons, "陽線")
    score = _score_add(score, 1, bool(row.get("is_long_bullish_candle")), reasons, "大陽線")
    score = _score_add(score, 2, bool(row.get("is_lower_shadow_bullish")), reasons, "下ヒゲ陽線")
    score = _score_add(score, 1, upper_shadow_short, reasons, "上ヒゲが短い")
    score = _score_add(score, 2, bool(row.get("is_engulfing_bullish")), reasons, "包み陽線")
    score = _score_add(score, 2, close_in_top, reasons, "寄り底に近い")
    breakdown["candle"] = reasons
    return min(score, 8)


def _breakout_score(row: pd.Series, breakdown: Dict[str, List[str]]) -> int:
    reasons: List[str] = []
    score = 0
    close = _num(row.get("close"))
    high20 = _num(row.get("recent_high_20"))
    prev_high20 = _num(row.get("recent_high_20_prev"))
    near_round = _nearest_round_level(close)
    round_break = close > near_round and close / max(near_round, 1) - 1 <= 0.03
    ma25_rebound = _num(row.get("low")) <= _num(row.get("ma25")) * 1.01 and close > _num(row.get("ma25"))
    near_high = close >= high20 * 0.97 if high20 else False
    breakout = close > prev_high20 if prev_high20 else False
    maintained = close >= prev_high20 * 0.995 if prev_high20 else False
    score = _score_add(score, 1, near_high, reasons, "20日高値から3%以内")
    score = _score_add(score, 2, breakout, reasons, "20日高値を上抜け")
    score = _score_add(score, 2, maintained, reasons, "高値突破後に維持")
    score = _score_add(score, 2, ma25_rebound, reasons, "25日線付近で反発")
    score = _score_add(score, 1, round_break, reasons, f"キリ番{near_round:,.0f}円突破")
    score = _score_add(score, 1, (breakout or round_break) and _num(row.get("volume_ratio_20")) >= 1.5, reasons, "出来高を伴う節目突破")
    breakdown["breakout"] = reasons
    return min(score, 8)


def _nearest_round_level(price: float) -> float:
    if price < 1000:
        step = 100
    elif price < 3000:
        step = 500
    elif price < 10000:
        step = 1000
    else:
        step = 5000
    return float(np.floor(price / step) * step)


def _momentum_score(df: pd.DataFrame, row: pd.Series, breakdown: Dict[str, List[str]]) -> int:
    reasons: List[str] = []
    score = 0
    rsi = _num(row.get("rsi14"))
    prev_rsi = _num(df["rsi14"].iloc[-2] if len(df) >= 2 else np.nan)
    score = _score_add(score, 3, 45 <= rsi <= 65, reasons, f"RSI{rsi:.1f}が45〜65")
    score = _score_add(score, 1, rsi >= 50 and rsi > prev_rsi, reasons, "RSIが50以上で上向き")
    score = _score_add(score, 2, _num(row.get("macd")) > _num(row.get("macd_signal")), reasons, "MACDがシグナルより上")
    score = _score_add(score, 1, _num(row.get("macd")) > 0, reasons, "MACDが0より上")
    breakdown["momentum"] = reasons
    return min(score, 6)


def _risk_reward_score(rr: Dict[str, Any], breakdown: Dict[str, List[str]]) -> int:
    reasons: List[str] = []
    score = 0
    score = _score_add(score, 1, rr.get("stop_loss_candidate") is not None, reasons, "損切り候補あり")
    score = _score_add(score, 1, _num(rr.get("stop_loss_distance_pct"), 999) <= 5, reasons, "損切り幅5%以内")
    score = _score_add(score, 1, _num(rr.get("risk_reward")) >= 1.2, reasons, "リスクリワード1.2倍以上")
    score = _score_add(score, 1, _num(rr.get("risk_reward")) >= 1.5, reasons, "リスクリワード1.5倍以上")
    score = _score_add(score, 1, rr.get("target_price_candidate") is not None, reasons, "利確候補あり")
    breakdown["risk_reward"] = reasons
    return min(score, 4)


def _penalties(df: pd.DataFrame, row: pd.Series) -> Tuple[int, List[str]]:
    reasons: List[str] = []
    penalty = 0

    def add(points: int, condition: bool, text: str) -> None:
        nonlocal penalty
        if condition:
            penalty += points
            reasons.append(f"{text} {points}")

    close = _num(row.get("close"))
    add(-8, close < _num(row.get("ma25")), "25日線割れ")
    add(-5, close < _num(row.get("ma75")), "75日線割れ")
    add(-5, bool(row.get("is_upper_shadow_warning")), "長い上ヒゲ")
    add(-6, bool(row.get("is_long_bearish_candle")), "大陰線")
    add(-3, bool(row.get("is_three_bearish_days")), "陰線3連続")
    add(-8, bool(row.get("is_bearish_candle")) and _num(row.get("volume_ratio_20")) >= 1.5, "出来高急増の陰線")
    add(-5, _num(row.get("return_3d_pct")) >= 15, "直近3日で15%以上急騰")
    add(-5, _num(row.get("distance_from_ma5_pct")) >= 8, "5日線から8%以上乖離")
    add(-4, _num(row.get("rsi14")) >= 75, "RSI75以上")
    add(-3, _num(row.get("rsi14")) >= 80, "RSI80以上")
    add(-5, _num(row.get("gap_pct")) >= 3 and close < _num(row.get("open")), "窓開け急騰後に失速")
    add(-8, _num(row.get("daily_change_pct")) >= 8 and close < _num(row.get("open")), "急騰後の寄り天気味")
    add(-3, _num(row.get("trading_value")) < 300_000_000, "売買代金3億円未満")
    return penalty, reasons


def _hard_filters(row: pd.Series, rr: Dict[str, Any], missing_data: List[str]) -> List[str]:
    reasons: List[str] = []
    close = _num(row.get("close"))
    if rr.get("stop_loss_candidate") is None:
        reasons.append("損切り候補が作れない")
    if rr.get("stop_loss_distance_pct") is None or _num(rr.get("stop_loss_distance_pct"), 999) > 5:
        reasons.append("損切り幅が5%超")
    if rr.get("risk_reward") is None or _num(rr.get("risk_reward")) < 1.0:
        reasons.append("リスクリワード1.0未満")
    if _num(row.get("trading_value")) < 300_000_000:
        reasons.append("売買代金3億円未満")
    if bool(row.get("is_bearish_candle")) and _num(row.get("volume_ratio_20")) >= 1.5:
        reasons.append("出来高急増の陰線")
    if bool(row.get("is_long_bearish_candle")) and bool(row.get("is_upper_shadow_warning")):
        reasons.append("長い上ヒゲ大陰線")
    if close < _num(row.get("ma25")) and _num(row.get("ma25_slope")) < 0:
        reasons.append("25日線を明確に割って下落継続")
    if close < _num(row.get("ma75")):
        reasons.append("75日線割れ")
    if _num(row.get("return_3d_pct")) >= 15 and _num(row.get("distance_from_ma5_pct")) >= 8:
        reasons.append("直近3日急騰かつ5日線から大きく乖離")
    if _num(row.get("rsi14")) >= 80:
        reasons.append("RSI80以上")
    if missing_data:
        reasons.append("データ不足")
    return list(dict.fromkeys(reasons))


def _judgement(final_score: int, hard_filters: List[str]) -> str:
    if hard_filters:
        return "見送り" if final_score < 42 else "監視"
    if final_score >= 50:
        return "テクニカル強い"
    if final_score >= 42:
        return "買い候補"
    if final_score >= 36:
        return "条件付き買い"
    if final_score >= 30:
        return "監視強化"
    if final_score >= 20:
        return "監視のみ"
    return "触らない"


def _hold_days_hint(row: pd.Series, hard_filters: List[str]) -> str:
    if hard_filters:
        return "見送り。形が整うまで待ち"
    rsi = _num(row.get("rsi14"))
    if _num(row.get("volume_ratio_20")) >= 2.0 and _num(row.get("close")) > _num(row.get("recent_high_20")) and bool(row.get("is_long_bullish_candle")) and 65 <= rsi <= 75:
        return "当日〜翌日。ブレイク型なので伸びなければ早めに撤退"
    if _num(row.get("close")) > _num(row.get("ma5")) and _num(row.get("ma25_slope")) > 0 and _num(row.get("volume_ratio_20")) >= 1.3 and 50 <= rsi <= 70:
        return "2〜3営業日。短期トレンド継続型。5日線維持を確認"
    if _num(row.get("close")) > _num(row.get("ma25")) and 0 <= _num(row.get("distance_from_ma25_pct")) <= 5 and bool(row.get("is_lower_shadow_bullish")):
        return "3〜5営業日。押し目反発型。直近安値を割らなければ継続"
    if _num(row.get("ma25_slope")) > 0 and _num(row.get("ma75_slope")) > 0 and rsi < 70:
        return "5〜10営業日。トレンド継続型。25日線割れまでは継続候補"
    return "2〜3営業日。条件を確認しながら短期で判断"


def _buy_timing_hint(row: pd.Series, hard_filters: List[str]) -> str:
    if hard_filters:
        return "寄り付き直後はノイズが多いため原則見送り。形が整うまで待ち"
    if _num(row.get("close")) > _num(row.get("recent_high_20")):
        return "9:20〜10:30にVWAP上で5分足高値切り上げなら買い検討"
    if 0 <= _num(row.get("distance_from_ma25_pct")) <= 5:
        return "10:30〜11:15にVWAP上で押し目反発を確認"
    if _num(row.get("volume_ratio_20")) >= 1.5:
        return "12:35〜13:30に後場再上昇があれば買い検討"
    return "14:30以降は高値圏維持かつ上ヒゲ短い場合のみ持ち越し前提"


def _confidence(missing_data: List[str], hard_filters: List[str]) -> int:
    confidence = 100 - len(missing_data) * 8 - len(hard_filters) * 5
    return int(max(30, min(100, confidence)))


def _required_metrics_for_config(config: TechnicalScoreConfig) -> List[str]:
    required = set()
    if config.use_trend:
        required.update(["ma5", "ma25", "ma75"])
    if config.use_entry_position:
        required.update(["ma5", "ma25", "recent_high_20", "recent_low_20"])
    if config.use_volume:
        required.update(["volume_ma20", "volume_ratio_20"])
    if config.use_momentum:
        required.update(["rsi14", "macd", "macd_signal"])
    if config.use_risk_reward:
        required.update(["atr14", "recent_high_20", "recent_low_20"])
    return sorted(required)


def _disabled_section(breakdown: Dict[str, List[str]], key: str) -> int:
    breakdown[key] = ["設定で未使用"]
    return 0


def score_daily_technical_item(item: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    cfg = normalize_technical_config(config)
    df = item["data"]
    row = df.iloc[-1]
    required = _required_metrics_for_config(cfg)
    missing_data = [key for key in required if pd.isna(row.get(key))]
    score_breakdown: Dict[str, List[str]] = {}
    rr = _stop_and_target(row)

    trend = _trend_score(row, score_breakdown) if cfg.use_trend else _disabled_section(score_breakdown, "trend")
    entry_position = (
        _entry_position_score(row, rr, score_breakdown)
        if cfg.use_entry_position
        else _disabled_section(score_breakdown, "entry_position")
    )
    volume = _volume_score(row, score_breakdown) if cfg.use_volume else _disabled_section(score_breakdown, "volume")
    candle = _candle_score(row, score_breakdown) if cfg.use_candle else _disabled_section(score_breakdown, "candle")
    breakout = _breakout_score(row, score_breakdown) if cfg.use_breakout else _disabled_section(score_breakdown, "breakout")
    momentum = _momentum_score(df, row, score_breakdown) if cfg.use_momentum else _disabled_section(score_breakdown, "momentum")
    risk_reward = (
        _risk_reward_score(rr, score_breakdown)
        if cfg.use_risk_reward
        else _disabled_section(score_breakdown, "risk_reward")
    )
    raw = trend + entry_position + volume + candle + breakout + momentum + risk_reward
    penalty_score, penalty_reasons = _penalties(df, row) if cfg.use_penalty else (0, [])
    score_breakdown["penalty"] = penalty_reasons
    final_score = max(0, min(60, raw + penalty_score))
    hard_filter_reasons = _hard_filters(row, rr, missing_data) if cfg.use_hard_filter else []

    return {
        "code": item["code"],
        "name": item["name"],
        "normalized_symbol": item.get("normalized_symbol", ""),
        "technical_preset": cfg.preset_name,
        "technical_config": asdict(cfg),
        "close": _round_price(row.get("close")),
        "technical_score_raw": int(raw),
        "penalty_score": int(penalty_score),
        "technical_score_final": int(final_score),
        "trend_score": trend,
        "entry_position_score": entry_position,
        "volume_score": volume,
        "candle_score": candle,
        "breakout_score": breakout,
        "momentum_score": momentum,
        "risk_reward_score": risk_reward,
        "technical_judgement": _judgement(int(final_score), hard_filter_reasons),
        "confidence": _confidence(missing_data, hard_filter_reasons),
        "hold_days_hint": _hold_days_hint(row, hard_filter_reasons),
        "buy_timing_hint": _buy_timing_hint(row, hard_filter_reasons),
        "stop_loss_candidate": rr.get("stop_loss_candidate"),
        "stop_loss_distance_pct": rr.get("stop_loss_distance_pct"),
        "target_price_candidate": rr.get("target_price_candidate"),
        "risk_reward": rr.get("risk_reward"),
        "penalty_reasons": penalty_reasons,
        "hard_filter_reason": hard_filter_reasons,
        "missing_data": missing_data,
        "score_breakdown": score_breakdown,
        "last_updated_at": item.get("last_attempt_at") or now_jst_display(),
        "latest_metrics": {key: row.get(key) for key in row.index},
    }


def score_daily_technical_data(daily_data: Dict[str, Dict[str, Any]], config: Any = None) -> List[Dict[str, Any]]:
    results = [score_daily_technical_item(item, config=config) for item in daily_data.values()]
    return sorted(results, key=lambda item: item.get("technical_score_final", 0), reverse=True)


def score_technical_item(
    df: pd.DataFrame,
    config: Any = None,
    code: str = "",
    name: str = "",
    normalized_symbol: str = "",
    as_of: Any = None,
) -> Dict[str, Any]:
    scoped = df.copy().sort_index()
    if as_of is not None:
        scoped = scoped.loc[scoped.index <= pd.Timestamp(as_of)]
    if scoped.empty:
        raise ValueError("テクニカル採点用の日足データが空です。")
    prepared = build_daily_indicators(scoped)
    return score_daily_technical_item(
        {
            "code": str(code or normalized_symbol).replace(".T", ""),
            "name": name or str(code or normalized_symbol),
            "normalized_symbol": normalized_symbol,
            "data": prepared,
            "last_attempt_at": now_jst_display(),
        },
        config=config,
    )


def score_technical_batch(data_dict: Dict[str, Dict[str, Any]], config: Any = None) -> List[Dict[str, Any]]:
    return score_daily_technical_data(data_dict, config=config)
