from __future__ import annotations

import numpy as np
import pandas as pd


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


def add_indicators(price_data: pd.DataFrame) -> pd.DataFrame:
    df = price_data.copy().sort_index()
    df["sma_5"] = df["Close"].rolling(5, min_periods=1).mean()
    df["sma_25"] = df["Close"].rolling(25, min_periods=1).mean()
    df["sma_75"] = df["Close"].rolling(75, min_periods=1).mean()
    df["rsi_14"] = calculate_rsi(df["Close"], 14)
    df["volume_ma_20"] = df["Volume"].rolling(20, min_periods=1).mean()

    df["prev_close"] = df["Close"].shift(1)
    df["prev_high"] = df["High"].shift(1)
    df["prev_low"] = df["Low"].shift(1)
    df["change_pct"] = (df["Close"] / df["prev_close"] - 1) * 100

    df["recent_5_high"] = df["High"].rolling(5, min_periods=1).max()
    df["recent_20_high"] = df["High"].rolling(20, min_periods=1).max()
    df["recent_5_low"] = df["Low"].rolling(5, min_periods=1).min()
    df["recent_20_low"] = df["Low"].rolling(20, min_periods=1).min()
    df["recent_20_high_prev"] = df["High"].shift(1).rolling(20, min_periods=1).max()

    candle_range = (df["High"] - df["Low"]).replace(0, np.nan)
    lower_base = np.minimum(df["Open"], df["Close"])
    upper_base = np.maximum(df["Open"], df["Close"])
    df["lower_wick_ratio"] = ((lower_base - df["Low"]) / candle_range).clip(lower=0).fillna(0)
    df["upper_wick_ratio"] = ((df["High"] - upper_base) / candle_range).clip(lower=0).fillna(0)

    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["prev_close"]).abs()
    low_prev_close = (df["Low"] - df["prev_close"]).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14, min_periods=1).mean()

    df["return_3d_pct"] = (df["Close"] / df["Close"].shift(3) - 1) * 100
    df["return_5d_pct"] = (df["Close"] / df["Close"].shift(5) - 1) * 100
    df["return_20d_pct"] = (df["Close"] / df["Close"].shift(20) - 1) * 100
    df["drawdown_from_20d_high_pct"] = (df["Close"] / df["recent_20_high"] - 1) * 100
    return df

