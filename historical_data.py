from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from config import BASE_DIR
from data_fetcher import normalize_jp_symbol


DATA_CACHE_DIR = BASE_DIR / "data_cache"
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
JST_TZ = "Asia/Tokyo"


@dataclass
class HistoricalDataResult:
    symbol: str
    normalized_symbol: str
    period: str
    interval: str
    data: pd.DataFrame
    cache_path: Path
    from_cache: bool = False
    error_type: str = ""
    error_message: str = ""

    @property
    def fetched_rows(self) -> int:
        return int(len(self.data))

    @property
    def first_timestamp(self) -> str:
        if self.data.empty:
            return "-"
        return _format_timestamp(self.data.index[0])

    @property
    def last_timestamp(self) -> str:
        if self.data.empty:
            return "-"
        return _format_timestamp(self.data.index[-1])


def _format_timestamp(value: object) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value or "-")


def _safe_cache_part(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return text.strip("._") or "unknown"


def _cache_path(symbol: object, period: str, interval: str) -> Path:
    normalized = normalize_jp_symbol(symbol) or str(symbol or "")
    filename = f"{_safe_cache_part(normalized)}_{_safe_cache_part(period)}_{_safe_cache_part(interval)}.csv"
    return DATA_CACHE_DIR / filename


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    first_level = [str(value) for value in out.columns.get_level_values(0)]
    second_level = [str(value) for value in out.columns.get_level_values(1)]
    if "Close" in first_level:
        out.columns = first_level
    elif "Close" in second_level:
        out.columns = second_level
    else:
        out.columns = ["_".join(map(str, col)).strip("_") for col in out.columns]
    return out


def _normalize_index_to_jst_naive(index: pd.Index) -> pd.DatetimeIndex:
    converted = pd.to_datetime(index)
    try:
        if converted.tz is None:
            return converted.tz_localize(None)
        return converted.tz_convert(JST_TZ).tz_localize(None)
    except Exception:
        return pd.to_datetime(index).tz_localize(None)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    cleaned = _flatten_yfinance_columns(df.copy())
    missing = [column for column in REQUIRED_COLUMNS if column not in cleaned.columns]
    if missing:
        raise ValueError(f"必要な列がありません: {', '.join(missing)}")

    cleaned = cleaned[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close"])
    cleaned["Volume"] = cleaned["Volume"].fillna(0)
    cleaned.index = _normalize_index_to_jst_naive(cleaned.index)
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    return cleaned.sort_index()


def save_historical_cache(symbol: object, period: str, interval: str, df: pd.DataFrame) -> Path:
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, period, interval)
    df.to_csv(path, index_label="Datetime")
    return path


def load_historical_cache(symbol: object, period: str, interval: str) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol, period, interval)
    if not path.exists():
        return None
    try:
        cached = pd.read_csv(path, index_col="Datetime", parse_dates=True)
        return normalize_ohlcv(cached)
    except Exception:
        return None


def fetch_historical_data(symbol: object, period: str, interval: str) -> HistoricalDataResult:
    normalized_symbol = normalize_jp_symbol(symbol)
    path = _cache_path(symbol, period, interval)
    if not normalized_symbol:
        return HistoricalDataResult(
            symbol=str(symbol or ""),
            normalized_symbol="",
            period=period,
            interval=interval,
            data=pd.DataFrame(columns=REQUIRED_COLUMNS),
            cache_path=path,
            error_type="invalid_symbol",
            error_message="銘柄コードが空、またはyfinance用シンボルへ変換できませんでした。",
        )

    try:
        import yfinance as yf
    except ImportError:
        return HistoricalDataResult(
            symbol=str(symbol or ""),
            normalized_symbol=normalized_symbol,
            period=period,
            interval=interval,
            data=pd.DataFrame(columns=REQUIRED_COLUMNS),
            cache_path=path,
            error_type="missing_dependency",
            error_message="yfinanceがインストールされていません。",
        )

    try:
        raw = yf.download(
            normalized_symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            prepost=False,
        )
    except Exception as exc:
        return HistoricalDataResult(
            symbol=str(symbol or ""),
            normalized_symbol=normalized_symbol,
            period=period,
            interval=interval,
            data=pd.DataFrame(columns=REQUIRED_COLUMNS),
            cache_path=path,
            error_type="download_exception",
            error_message=f"過去データ取得に失敗しました（{exc.__class__.__name__}）。",
        )

    try:
        cleaned = normalize_ohlcv(raw)
    except Exception as exc:
        return HistoricalDataResult(
            symbol=str(symbol or ""),
            normalized_symbol=normalized_symbol,
            period=period,
            interval=interval,
            data=pd.DataFrame(columns=REQUIRED_COLUMNS),
            cache_path=path,
            error_type="normalize_error",
            error_message=f"過去データの整形に失敗しました: {exc}",
        )

    if cleaned.empty:
        return HistoricalDataResult(
            symbol=str(symbol or ""),
            normalized_symbol=normalized_symbol,
            period=period,
            interval=interval,
            data=cleaned,
            cache_path=path,
            error_type="empty_data",
            error_message=f"過去データが空でした。symbol={normalized_symbol}, period={period}, interval={interval}",
        )

    save_historical_cache(normalized_symbol, period, interval, cleaned)
    return HistoricalDataResult(
        symbol=str(symbol or ""),
        normalized_symbol=normalized_symbol,
        period=period,
        interval=interval,
        data=cleaned,
        cache_path=path,
        from_cache=False,
    )


def load_or_fetch_historical_data(symbol: object, period: str, interval: str) -> HistoricalDataResult:
    normalized_symbol = normalize_jp_symbol(symbol)
    path = _cache_path(symbol, period, interval)
    if normalized_symbol:
        cached = load_historical_cache(normalized_symbol, period, interval)
        if cached is not None and not cached.empty:
            return HistoricalDataResult(
                symbol=str(symbol or ""),
                normalized_symbol=normalized_symbol,
                period=period,
                interval=interval,
                data=cached,
                cache_path=path,
                from_cache=True,
            )
    return fetch_historical_data(symbol, period, interval)
