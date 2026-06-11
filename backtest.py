from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from data_fetcher import fetch_price_data
from indicators import add_indicators
from scanner import load_watchlist
from scoring import score_stock


def _max_losing_streak(returns: pd.Series) -> int:
    streak = 0
    max_streak = 0
    for value in returns:
        if value <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def run_backtest(period: str = "1y") -> Dict[str, Any]:
    watchlist = load_watchlist()
    rows: List[Dict[str, Any]] = []
    for record in watchlist.to_dict("records"):
        fetched = fetch_price_data(record["code"], period=period)
        if fetched.error or fetched.data.empty:
            continue
        df = add_indicators(fetched.data)
        for idx in range(80, len(df) - 5):
            hist = df.iloc[: idx + 1]
            signal = score_stock(hist, record)
            if signal["score"] < 70:
                continue
            close = df["Close"].iloc[idx]
            rows.append(
                {
                    "date": df.index[idx].date().isoformat(),
                    "code": record["code"],
                    "name": record["name"],
                    "score": signal["score"],
                    "entry_type": signal["entry_type"],
                    "return_1d": (df["Close"].iloc[idx + 1] / close - 1) * 100,
                    "return_3d": (df["Close"].iloc[idx + 3] / close - 1) * 100,
                    "return_5d": (df["Close"].iloc[idx + 5] / close - 1) * 100,
                }
            )

    trades = pd.DataFrame(rows)
    if trades.empty:
        return {"trades": trades, "summary": {"候補数": 0}}

    ret = trades["return_5d"]
    wins = ret[ret > 0]
    losses = ret[ret <= 0]
    cumulative = ret.fillna(0).cumsum()
    drawdown = cumulative - cumulative.cummax()
    summary = {
        "候補数": int(len(trades)),
        "5日後勝率": round((ret > 0).mean() * 100, 1),
        "平均利益": round(wins.mean(), 2) if not wins.empty else 0,
        "平均損失": round(losses.mean(), 2) if not losses.empty else 0,
        "期待値": round(ret.mean(), 2),
        "最大連敗": _max_losing_streak(ret),
        "簡易最大ドローダウン": round(drawdown.min(), 2),
    }
    return {"trades": trades, "summary": summary}


if __name__ == "__main__":
    result = run_backtest()
    print(result["summary"])
    if not result["trades"].empty:
        print(result["trades"].tail(20).to_string(index=False))

