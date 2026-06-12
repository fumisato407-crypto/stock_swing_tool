from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from alert_builder import build_intraday_alert_text
from data_fetcher import clean_ohlcv_data, normalize_jp_symbol
from entry_rules import evaluate_intraday_entry


@dataclass
class IntradayFetchResult:
    code: str
    ticker: str
    data: pd.DataFrame
    previous_low: Optional[float] = None
    error: Optional[str] = None
    error_type: str = ""
    error_message: str = ""
    fetched_rows: int = 0
    last_attempt_at: str = ""


def _attempt_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _result(
    code: object,
    ticker: str,
    data: Optional[pd.DataFrame],
    last_attempt_at: str,
    previous_low: Optional[float] = None,
    error_type: str = "",
    error_message: str = "",
) -> IntradayFetchResult:
    frame = data if data is not None else pd.DataFrame()
    return IntradayFetchResult(
        code=str(code or ""),
        ticker=ticker,
        data=frame,
        previous_low=previous_low,
        error=error_message or None,
        error_type=error_type,
        error_message=error_message,
        fetched_rows=0 if frame is None else len(frame),
        last_attempt_at=last_attempt_at,
    )


def _normalize_intraday_index(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    index = pd.to_datetime(normalized.index)
    try:
        if index.tz is None:
            index = index.tz_localize("Asia/Tokyo")
        else:
            index = index.tz_convert("Asia/Tokyo")
        index = index.tz_localize(None)
    except Exception:
        index = pd.to_datetime(normalized.index).tz_localize(None)
    normalized.index = index
    return normalized.sort_index()


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df, error = clean_ohlcv_data(df, normalize_index=False)
    if error:
        raise ValueError(error)
    return _normalize_intraday_index(df)


def _today_session(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[float]]:
    if df.empty:
        return pd.DataFrame(), None

    dates = pd.Series(df.index.date, index=df.index)
    unique_dates = list(pd.unique(dates))
    today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    today_data = df[dates == today].copy()

    previous_low: Optional[float] = None
    previous_dates = [session_date for session_date in unique_dates if session_date < today]
    if previous_dates:
        previous_date = previous_dates[-1]
        previous = df[dates == previous_date]
        if not previous.empty:
            previous_low = float(previous["Low"].min())

    return today_data, previous_low


def fetch_intraday_data(code: object, interval: str = "5m", period: str = "5d") -> IntradayFetchResult:
    last_attempt_at = _attempt_timestamp()
    ticker = normalize_jp_symbol(code)
    if not ticker:
        return _result(
            code,
            ticker,
            pd.DataFrame(),
            last_attempt_at,
            error_type="invalid_code",
            error_message="銘柄コードが空、またはyfinance用シンボルへ変換できませんでした。",
        )

    try:
        import yfinance as yf
    except ImportError:
        return _result(
            code,
            ticker,
            pd.DataFrame(),
            last_attempt_at,
            error_type="missing_dependency",
            error_message="yfinanceがインストールされていません。",
        )

    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            prepost=False,
        )
    except Exception as exc:
        return _result(
            code,
            ticker,
            pd.DataFrame(),
            last_attempt_at,
            error_type="download_exception",
            error_message=f"場中データ取得に失敗しました（{exc.__class__.__name__}）。",
        )

    if raw is None or raw.empty:
        return _result(
            code,
            ticker,
            pd.DataFrame(),
            last_attempt_at,
            error_type="empty_data",
            error_message=f"場中データが空でした。symbol={ticker}, period={period}, interval={interval}",
        )

    try:
        cleaned = _clean_ohlcv(raw)
    except Exception as exc:
        return _result(
            code,
            ticker,
            pd.DataFrame(),
            last_attempt_at,
            error_type="cleaning_error",
            error_message=f"場中データの整形に失敗しました: {exc}",
        )

    today_data, previous_low = _today_session(cleaned)
    if today_data.empty:
        return _result(
            code,
            ticker,
            cleaned,
            last_attempt_at,
            previous_low=previous_low,
            error_type="no_today_intraday",
            error_message="市場時間外または当日5分足なし。JST基準の当日データが見つかりません。",
        )

    return _result(code, ticker, today_data, last_attempt_at, previous_low=previous_low)


def _failure_signal(record: Dict[str, Any], fetched: IntradayFetchResult) -> Dict[str, Any]:
    error_message = fetched.error_message or fetched.error or "場中データ取得に失敗しました。"
    return {
        "code": str(record.get("code", "")),
        "name": str(record.get("name", "")),
        "raw_code": str(record.get("raw_code", record.get("code", ""))),
        "normalized_symbol": fetched.ticker,
        "error_type": fetched.error_type,
        "error_message": error_message,
        "fetched_rows": fetched.fetched_rows,
        "last_attempt_at": fetched.last_attempt_at,
        "current_price": None,
        "judgement": "取得失敗",
        "signal_type": "-",
        "intraday_score": 0,
        "buy_zone": "-",
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "reasons": [],
        "risk_notes": [error_message],
        "no_buy_conditions": [error_message],
        "notification_text": f"{record.get('code', '')} {record.get('name', '')}: {error_message}",
        "data": pd.DataFrame(),
        "error": error_message,
    }


def scan_intraday_entries(
    records: Iterable[Dict[str, Any]],
    interval: str = "5m",
    period: str = "5d",
) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    for record in records:
        raw_code = record.get("raw_code") or record.get("code", "")
        normalized_symbol = normalize_jp_symbol(record.get("normalized_symbol") or raw_code)
        if not normalized_symbol:
            continue
        code = normalized_symbol.replace(".T", "")
        fetched = fetch_intraday_data(code, interval=interval, period=period)
        if fetched.error:
            failure_record = dict(record)
            failure_record["code"] = code
            failure_record["raw_code"] = raw_code
            signals.append(_failure_signal(failure_record, fetched))
            continue

        signal = evaluate_intraday_entry(
            code=code,
            name=str(record.get("name", "")),
            intraday_data=fetched.data,
            previous_low=fetched.previous_low,
        )
        signal["notification_text"] = build_intraday_alert_text(signal)
        signal["raw_code"] = raw_code
        signal["normalized_symbol"] = fetched.ticker
        signal["error_type"] = ""
        signal["error_message"] = ""
        signal["fetched_rows"] = fetched.fetched_rows
        signal["last_attempt_at"] = fetched.last_attempt_at
        signals.append(signal)

    order = {"買い検討OK": 0, "監視強化": 1, "見送り": 2, "取得失敗": 3}
    return sorted(signals, key=lambda item: (order.get(item.get("judgement", ""), 9), -int(item.get("intraday_score", 0))))
