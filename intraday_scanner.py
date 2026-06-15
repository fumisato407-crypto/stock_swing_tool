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
    cache_hit: bool = False


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


def _clean_error_type(message: str) -> str:
    if "必要な列" in str(message):
        return "invalid_columns"
    if "空" in str(message):
        return "empty_dataframe"
    return "exception"


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
        text = str(exc).lower()
        error_type = "rate_limit_or_connection_error" if any(word in text for word in ("rate", "limit", "connection", "timeout")) else "exception"
        return _result(
            code,
            ticker,
            pd.DataFrame(),
            last_attempt_at,
            error_type=error_type,
            error_message=f"場中データ取得に失敗しました（{exc.__class__.__name__}）。",
        )

    if raw is None or raw.empty:
        return _result(
            code,
            ticker,
            pd.DataFrame(),
            last_attempt_at,
            error_type="empty_dataframe",
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
            error_type=_clean_error_type(str(exc)),
            error_message=f"場中データの整形に失敗しました: {exc}",
        )
    if cleaned.empty:
        return _result(
            code,
            ticker,
            cleaned,
            last_attempt_at,
            error_type="no_rows_after_dropna",
            error_message=f"場中データ整形後に有効行が0件でした。symbol={ticker}, period={period}, interval={interval}",
        )

    today_data, previous_low = _today_session(cleaned)
    if today_data.empty:
        latest_text = cleaned.index[-1].strftime("%Y-%m-%d %H:%M") if not cleaned.empty and hasattr(cleaned.index[-1], "strftime") else "-"
        return _result(
            code,
            ticker,
            cleaned,
            last_attempt_at,
            previous_low=previous_low,
            error_type="no_latest_bar",
            error_message=f"市場時間外または当日5分足なし。JST基準の当日データが見つかりません。latest={latest_text}",
        )

    return _result(code, ticker, today_data, last_attempt_at, previous_low=previous_low)


def _latest_debug_from_data(data: pd.DataFrame) -> Dict[str, Any]:
    if data is None or data.empty:
        return {
            "latest_5m_jst": "-",
            "latest_close": None,
            "latest_volume": None,
            "vwap": None,
        }
    frame = data.copy().sort_index()
    if "vwap" not in frame.columns:
        try:
            from entry_rules import add_intraday_indicators

            frame = add_intraday_indicators(frame)
        except Exception:
            pass
    latest = frame.iloc[-1]
    latest_time = frame.index[-1].strftime("%Y-%m-%d %H:%M") if hasattr(frame.index[-1], "strftime") else str(frame.index[-1])
    return {
        "latest_5m_jst": latest_time,
        "latest_close": float(latest.get("Close")) if pd.notna(latest.get("Close")) else None,
        "latest_volume": int(float(latest.get("Volume", 0) or 0)),
        "vwap": float(latest.get("vwap")) if "vwap" in frame.columns and pd.notna(latest.get("vwap")) else None,
    }


def _failure_signal(record: Dict[str, Any], fetched: IntradayFetchResult) -> Dict[str, Any]:
    error_message = fetched.error_message or fetched.error or "場中データ取得に失敗しました。"
    error_type = fetched.error_type or "unknown_intraday_fetch_error"
    latest_debug = _latest_debug_from_data(fetched.data)
    return {
        "code": str(record.get("code", "")),
        "name": str(record.get("name", "")),
        "raw_code": str(record.get("raw_code", record.get("code", ""))),
        "normalized_symbol": fetched.ticker,
        "ticker_for_intraday": fetched.ticker,
        "intraday_fetch_ok": False,
        "intraday_rows": fetched.fetched_rows,
        "latest_5m_jst": latest_debug["latest_5m_jst"],
        "latest_close": latest_debug["latest_close"],
        "latest_volume": latest_debug["latest_volume"],
        "vwap": latest_debug["vwap"],
        "cache_hit": fetched.cache_hit,
        "error_type": error_type,
        "error_message": error_message,
        "fetched_rows": fetched.fetched_rows,
        "last_attempt_at": fetched.last_attempt_at,
        "current_price": None,
        "judgement": "取得失敗",
        "signal_type": "5分足データ取得失敗",
        "intraday_score": 0,
        "intraday_ok_display": "データなし",
        "multi_timeframe_status": "判定不可",
        "entry_type": "5分足データ取得失敗",
        "reject_reason": error_message,
        "buy_zone": "-",
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "reasons": [],
        "risk_notes": [error_message],
        "no_buy_conditions": [error_message],
        "notification_text": f"{record.get('code', '')} {record.get('name', '')}: {error_message}",
        "data": fetched.data if fetched.data is not None else pd.DataFrame(),
        "error": error_message,
    }


def _attach_fetch_debug(signal: Dict[str, Any], fetched: IntradayFetchResult) -> Dict[str, Any]:
    latest_debug = _latest_debug_from_data(signal.get("data") if isinstance(signal.get("data"), pd.DataFrame) else fetched.data)
    signal["ticker_for_intraday"] = fetched.ticker
    signal["intraday_fetch_ok"] = True
    signal["intraday_rows"] = fetched.fetched_rows
    signal["latest_5m_jst"] = latest_debug["latest_5m_jst"]
    signal["latest_close"] = latest_debug["latest_close"]
    signal["latest_volume"] = latest_debug["latest_volume"]
    signal["vwap"] = latest_debug["vwap"]
    signal["cache_hit"] = fetched.cache_hit
    signal["intraday_ok_display"] = "-"
    signal["multi_timeframe_status"] = "5分足判定済み"
    return signal


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
        signal = _attach_fetch_debug(signal, fetched)
        if signal.get("judgement") == "取得失敗":
            signal["error_type"] = "insufficient_intraday_rows"
            signal["error_message"] = signal.get("error") or "5分足データが不足しています。"
            signal["signal_type"] = signal.get("signal_type") or "5分足データ取得失敗"
            signal["entry_type"] = "5分足データ取得失敗"
            signal["reject_reason"] = signal["error_message"]
            signal["intraday_ok_display"] = "データなし"
            signal["multi_timeframe_status"] = "判定不可"
        else:
            signal["error_type"] = ""
            signal["error_message"] = ""
        signal["fetched_rows"] = fetched.fetched_rows
        signal["last_attempt_at"] = fetched.last_attempt_at
        signals.append(signal)

    order = {"買い検討OK": 0, "監視強化": 1, "見送り": 2, "取得失敗": 3}
    return sorted(signals, key=lambda item: (order.get(item.get("judgement", ""), 9), -int(item.get("intraday_score", 0))))


def run_raw_intraday_fetch_test(
    symbols: Iterable[object],
    period: str = "1d",
    interval: str = "5m",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        import yfinance as yf
    except ImportError as exc:
        for symbol in symbols:
            normalized = normalize_jp_symbol(symbol)
            rows.append(
                {
                    "symbol": normalized or str(symbol or ""),
                    "period": period,
                    "interval": interval,
                    "rows": 0,
                    "columns": "",
                    "first_index": "-",
                    "last_index": "-",
                    "latest_close": None,
                    "latest_volume": None,
                    "exception_message": f"yfinanceがインストールされていません: {exc}",
                }
            )
        return rows

    for symbol in symbols:
        normalized = normalize_jp_symbol(symbol)
        row = {
            "symbol": normalized or str(symbol or ""),
            "period": period,
            "interval": interval,
            "rows": 0,
            "columns": "",
            "first_index": "-",
            "last_index": "-",
            "latest_close": None,
            "latest_volume": None,
            "exception_message": "",
        }
        if not normalized:
            row["exception_message"] = "銘柄コードを正規化できませんでした。"
            rows.append(row)
            continue
        try:
            raw = yf.download(
                normalized,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                prepost=False,
            )
            row["rows"] = 0 if raw is None else len(raw)
            row["columns"] = ", ".join(str(col) for col in getattr(raw, "columns", []))
            if raw is not None and not raw.empty:
                row["first_index"] = str(raw.index[0])
                row["last_index"] = str(raw.index[-1])
                cleaned = _clean_ohlcv(raw)
                if not cleaned.empty:
                    latest = cleaned.iloc[-1]
                    row["latest_close"] = float(latest["Close"])
                    row["latest_volume"] = int(float(latest["Volume"] or 0))
        except Exception as exc:
            row["exception_message"] = f"{exc.__class__.__name__}: {exc}"
        rows.append(row)
    return rows
