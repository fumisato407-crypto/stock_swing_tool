from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import unicodedata
from typing import Optional

import pandas as pd


@dataclass
class PriceFetchResult:
    code: str
    ticker: str
    data: pd.DataFrame
    error: Optional[str] = None
    error_type: str = ""
    error_message: str = ""
    fetched_rows: int = 0
    last_attempt_at: str = ""


def _attempt_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _is_missing_code(code: object) -> bool:
    if code is None:
        return True
    try:
        return bool(pd.isna(code))
    except (TypeError, ValueError):
        return False


def normalize_jp_symbol(code: object) -> str:
    """Normalize a Japanese stock code to the yfinance Tokyo ticker format."""
    if _is_missing_code(code):
        return ""

    raw = unicodedata.normalize("NFKC", str(code))
    normalized = re.sub(r"[\s\u3000]+", "", raw).upper()
    if not normalized or normalized in {"NAN", "NONE", "NULL"}:
        return ""

    while normalized.endswith(".T"):
        normalized = normalized[:-2]

    if re.fullmatch(r"\d+\.0+", normalized):
        normalized = normalized.split(".", 1)[0]

    normalized = normalized.strip(".")
    if not normalized:
        return ""
    return f"{normalized}.T"


def to_yfinance_ticker(code: object) -> str:
    """Backward-compatible alias for the shared Japanese ticker normalizer."""
    return normalize_jp_symbol(code)


def _result(
    code: object,
    ticker: str,
    data: Optional[pd.DataFrame],
    last_attempt_at: str,
    error_type: str = "",
    error_message: str = "",
) -> PriceFetchResult:
    frame = data if data is not None else pd.DataFrame()
    return PriceFetchResult(
        code=str(code or ""),
        ticker=ticker,
        data=frame,
        error=error_message or None,
        error_type=error_type,
        error_message=error_message,
        fetched_rows=0 if frame is None else len(frame),
        last_attempt_at=last_attempt_at,
    )


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    columns = df.columns
    first_level = [str(value) for value in columns.get_level_values(0)]
    second_level = [str(value) for value in columns.get_level_values(1)]
    if "Close" in first_level:
        df.columns = first_level
    elif "Close" in second_level:
        df.columns = second_level
    else:
        df.columns = ["_".join(map(str, col)).strip("_") for col in columns]
    return df


def _drop_timezone(index: pd.Index) -> pd.DatetimeIndex:
    converted = pd.to_datetime(index)
    try:
        if converted.tz is None:
            return converted.tz_localize(None)
        return converted.tz_convert(None)
    except Exception:
        return pd.to_datetime(index).tz_localize(None)


def clean_ohlcv_data(df: pd.DataFrame, normalize_index: bool = True) -> tuple[pd.DataFrame, Optional[str]]:
    required = ["Open", "High", "Low", "Close", "Volume"]
    if df is None or df.empty:
        return pd.DataFrame(), "株価データが空でした。"

    cleaned = _flatten_yfinance_columns(df.copy())
    missing = [col for col in required if col not in cleaned.columns]
    if missing:
        return pd.DataFrame(), f"必要な列がありません: {', '.join(missing)}"

    cleaned = cleaned[required].copy()
    for col in required:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close"])
    cleaned["Volume"] = cleaned["Volume"].fillna(0)
    if normalize_index:
        cleaned.index = _drop_timezone(cleaned.index)
    cleaned = cleaned.sort_index()
    return cleaned, None


def fetch_price_data(
    code: object,
    period: str = "6mo",
    interval: str = "1d",
) -> PriceFetchResult:
    """
    Fetch daily OHLCV data.

    The function intentionally returns a result object instead of raising, so the
    Streamlit app can keep running even when one ticker fails.
    """
    last_attempt_at = _attempt_timestamp()
    ticker = normalize_jp_symbol(code)
    if not ticker:
        return _result(
            code,
            ticker,
            pd.DataFrame(),
            last_attempt_at,
            "invalid_code",
            "銘柄コードが空、またはyfinance用シンボルへ変換できませんでした。",
        )

    try:
        import yfinance as yf
    except ImportError:
        return _result(
            code,
            ticker,
            pd.DataFrame(),
            last_attempt_at,
            "missing_dependency",
            "yfinanceがインストールされていません。",
        )

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        return _result(
            code,
            ticker,
            pd.DataFrame(),
            last_attempt_at,
            "download_exception",
            f"株価取得に失敗しました（{exc.__class__.__name__}）。",
        )

    if df is None or df.empty:
        return _result(
            code,
            ticker,
            pd.DataFrame(),
            last_attempt_at,
            "empty_data",
            f"株価データが空でした。symbol={ticker}, period={period}, interval={interval}",
        )

    cleaned, clean_error = clean_ohlcv_data(df)
    if clean_error:
        return _result(
            code,
            ticker,
            cleaned,
            last_attempt_at,
            "missing_columns",
            clean_error,
        )

    if len(cleaned) < 80:
        return _result(
            code,
            ticker,
            cleaned,
            last_attempt_at,
            "insufficient_rows",
            f"分析に必要な日足データが不足しています。取得行数: {len(cleaned)}",
        )

    return _result(code, ticker, cleaned, last_attempt_at)
