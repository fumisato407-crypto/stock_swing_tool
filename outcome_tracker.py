from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from data_fetcher import clean_ohlcv_data, fetch_price_data
from time_utils import now_jst_iso, parse_trade_datetime_to_jst_naive
from virtual_trade_store import load_open_virtual_trades, update_virtual_trade_outcome


CHECKPOINTS = {
    "close": 0,
    "1d": 1,
    "3d": 3,
    "5d": 5,
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "") or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _entry_datetime(value: Any) -> Optional[datetime]:
    return parse_trade_datetime_to_jst_naive(value)


def _normalize_ohlcv_index_to_jst_naive(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    normalized = data.copy()
    index = pd.to_datetime(normalized.index)
    try:
        if index.tz is not None:
            index = index.tz_convert("Asia/Tokyo").tz_localize(None)
        else:
            index = index.tz_localize(None)
    except Exception:
        index = pd.to_datetime(normalized.index).tz_localize(None)
    normalized.index = index
    return normalized.sort_index()


def _fetch_intraday_for_1h(symbol: str) -> pd.DataFrame:
    try:
        import yfinance as yf

        raw = yf.download(
            symbol,
            period="5d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            threads=False,
            prepost=False,
        )
    except Exception:
        return pd.DataFrame()
    cleaned, error = clean_ohlcv_data(raw, normalize_index=False)
    if error:
        return pd.DataFrame()
    index = pd.to_datetime(cleaned.index)
    try:
        if index.tz is not None:
            index = index.tz_convert("Asia/Tokyo").tz_localize(None)
        else:
            index = index.tz_localize(None)
    except Exception:
        index = pd.to_datetime(cleaned.index).tz_localize(None)
    cleaned.index = index
    return cleaned.sort_index()


def _pct(price: float, entry_price: float) -> Optional[float]:
    if not entry_price or not price:
        return None
    return round((price / entry_price - 1) * 100, 2)


def _checkpoint_returns(daily: pd.DataFrame, entry_dt: datetime, entry_price: float) -> Dict[str, Optional[float]]:
    if daily.empty:
        return {key: None for key in CHECKPOINTS}
    after = daily[daily.index.date >= entry_dt.date()].copy()
    results: Dict[str, Optional[float]] = {}
    for key, offset in CHECKPOINTS.items():
        if len(after) > offset:
            results[key] = _pct(float(after["Close"].iloc[offset]), entry_price)
        else:
            results[key] = None
    return results


def _one_hour_return(symbol: str, entry_dt: datetime, entry_price: float) -> Optional[float]:
    intraday = _fetch_intraday_for_1h(symbol)
    if intraday.empty:
        return None
    after = intraday[intraday.index >= entry_dt].copy()
    if after.empty:
        return None
    target_time = entry_dt + pd.Timedelta(hours=1)
    after_1h = after[after.index >= target_time]
    if after_1h.empty:
        return None
    return _pct(float(after_1h["Close"].iloc[0]), entry_price)


def _evaluate_path(
    daily: pd.DataFrame,
    entry_dt: datetime,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> Dict[str, Any]:
    after = daily[daily.index.date >= entry_dt.date()].copy()
    if after.empty:
        return {
            "return_pct": None,
            "max_profit_pct": None,
            "max_drawdown_pct": None,
            "hit_stop_loss": False,
            "hit_take_profit": False,
            "outcome": "tracking",
        }

    max_profit_pct = _pct(float(after["High"].max()), entry_price)
    max_drawdown_pct = _pct(float(after["Low"].min()), entry_price)
    hit_stop = False
    hit_target = False
    outcome = "tracking"
    return_pct = _pct(float(after["Close"].iloc[-1]), entry_price)

    for _, row in after.iterrows():
        low = _num(row.get("Low"))
        high = _num(row.get("High"))
        if stop_loss and low <= stop_loss:
            hit_stop = True
            outcome = "stop_loss"
            return_pct = _pct(stop_loss, entry_price)
            break
        if take_profit and high >= take_profit:
            hit_target = True
            outcome = "take_profit"
            return_pct = _pct(take_profit, entry_price)
            break

    if outcome == "tracking" and len(after) >= 6:
        outcome = "time_exit_5d"

    return {
        "return_pct": return_pct,
        "max_profit_pct": max_profit_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "hit_stop_loss": hit_stop,
        "hit_take_profit": hit_target,
        "outcome": outcome,
    }


def update_open_virtual_trade_outcomes() -> Dict[str, Any]:
    trades = load_open_virtual_trades()
    updated = 0
    skipped = 0
    errors: List[str] = []

    for trade in trades:
        trade_id = int(trade.get("id"))
        symbol = str(trade.get("symbol", ""))
        entry_dt = _entry_datetime(trade.get("timestamp"))
        entry_price = _num(trade.get("entry_price"))
        if not symbol or entry_dt is None or not entry_price:
            skipped += 1
            continue

        fetched = fetch_price_data(symbol, period="6mo", interval="1d")
        if fetched.error or fetched.data.empty:
            skipped += 1
            errors.append(f"{symbol}: daily data unavailable")
            continue
        daily = _normalize_ohlcv_index_to_jst_naive(fetched.data)

        checkpoints = _checkpoint_returns(daily, entry_dt, entry_price)
        checkpoints["1h"] = _one_hour_return(symbol, entry_dt, entry_price)
        path_result = _evaluate_path(
            daily=daily,
            entry_dt=entry_dt,
            entry_price=entry_price,
            stop_loss=_num(trade.get("stop_loss")),
            take_profit=_num(trade.get("take_profit")),
        )
        status = "closed" if path_result["outcome"] in {"stop_loss", "take_profit", "time_exit_5d"} else "open"
        outcome_json = {
            "checkpoints": checkpoints,
            "conservative_rule": "same_candle_stop_loss_first",
        }
        update_virtual_trade_outcome(
            trade_id,
            {
                "status": status,
                "return_pct": path_result["return_pct"],
                "max_profit_pct": path_result["max_profit_pct"],
                "max_drawdown_pct": path_result["max_drawdown_pct"],
                "hit_stop_loss": 1 if path_result["hit_stop_loss"] else 0,
                "hit_take_profit": 1 if path_result["hit_take_profit"] else 0,
                "outcome": path_result["outcome"],
                "outcome_json": json.dumps(outcome_json, ensure_ascii=False, default=str),
                "outcome_updated_at": now_jst_iso(),
                "outcome_updated_at_jst": now_jst_iso(),
            },
        )
        updated += 1

    return {"checked": len(trades), "updated": updated, "skipped": skipped, "errors": errors[:5]}
