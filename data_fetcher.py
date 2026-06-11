from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class PriceFetchResult:
    code: str
    ticker: str
    data: pd.DataFrame
    error: Optional[str] = None


def to_yfinance_ticker(code: str) -> str:
    """Convert a Japanese stock code to a yfinance Tokyo ticker."""
    normalized = str(code).strip()
    if normalized.upper().endswith(".T"):
        return normalized.upper()
    return f"{normalized}.T"


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


def fetch_price_data(
    code: str,
    period: str = "9mo",
    interval: str = "1d",
) -> PriceFetchResult:
    """
    Fetch daily OHLCV data.

    The function intentionally returns a result object instead of raising, so the
    Streamlit app can keep running even when one ticker fails.
    """
    ticker = to_yfinance_ticker(code)
    try:
        import yfinance as yf
    except ImportError:
        return PriceFetchResult(
            code=str(code),
            ticker=ticker,
            data=pd.DataFrame(),
            error="yfinanceがインストールされていません。",
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
        return PriceFetchResult(
            code=str(code),
            ticker=ticker,
            data=pd.DataFrame(),
            error=f"株価取得に失敗しました: {exc}",
        )

    if df is None or df.empty:
        return PriceFetchResult(
            code=str(code),
            ticker=ticker,
            data=pd.DataFrame(),
            error="株価データが空でした。",
        )

    df = _flatten_yfinance_columns(df.copy())
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return PriceFetchResult(
            code=str(code),
            ticker=ticker,
            data=pd.DataFrame(),
            error=f"必要な列がありません: {', '.join(missing)}",
        )

    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    if len(df) < 80:
        return PriceFetchResult(
            code=str(code),
            ticker=ticker,
            data=df,
            error="分析に必要な日足データが不足しています。",
        )

    return PriceFetchResult(code=str(code), ticker=ticker, data=df, error=None)

