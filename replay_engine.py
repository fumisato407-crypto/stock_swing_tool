from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from entry_rules import evaluate_intraday_entry


INTERVAL_MAX_HOLD_BARS = {
    "5m": 390,
    "15m": 130,
    "60m": 30,
    "1h": 30,
    "1d": 5,
}


@dataclass
class ReplayRuleConfig:
    min_score: int = 70
    target_rule: str = "すべて"
    max_trades: int = 10
    interval: str = "5m"
    cooldown_bars: int = 12


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(price: Optional[float], entry_price: float) -> Optional[float]:
    if price is None or not entry_price:
        return None
    return round((float(price) / entry_price - 1) * 100, 2)


def _timestamp_text(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value or "")


def _normalize_rule_config(rule_config: Optional[Dict[str, Any] | ReplayRuleConfig]) -> ReplayRuleConfig:
    if isinstance(rule_config, ReplayRuleConfig):
        return rule_config
    values = dict(rule_config or {})
    return ReplayRuleConfig(
        min_score=int(values.get("min_score", 70) or 70),
        target_rule=str(values.get("target_rule", "すべて") or "すべて"),
        max_trades=int(values.get("max_trades", 10) or 10),
        interval=str(values.get("interval", "5m") or "5m"),
        cooldown_bars=int(values.get("cooldown_bars", 12) or 12),
    )


def _max_hold_bars(interval: str) -> int:
    return INTERVAL_MAX_HOLD_BARS.get(str(interval), 30)


def _previous_session_low(history_df: pd.DataFrame, current_time: pd.Timestamp) -> Optional[float]:
    if history_df.empty:
        return None
    dates = pd.Series(history_df.index.date, index=history_df.index)
    current_date = current_time.date()
    previous_dates = [date for date in pd.unique(dates) if date < current_date]
    if not previous_dates:
        return None
    previous = history_df[dates == previous_dates[-1]]
    if previous.empty:
        return None
    return float(previous["Low"].min())


def evaluate_replay_step(
    history_df: pd.DataFrame,
    current_time: Any,
    rule_config: Dict[str, Any] | ReplayRuleConfig,
    symbol: str = "",
    name: str = "",
) -> Dict[str, Any]:
    config = _normalize_rule_config(rule_config)
    if history_df.empty or len(history_df) < 25:
        return {"judgement": "見送り", "intraday_score": 0, "reason": "判定に必要な履歴足が不足"}

    current_ts = pd.Timestamp(current_time)
    safe_history = history_df.loc[history_df.index <= current_ts].copy()
    if len(safe_history) < 25:
        return {"judgement": "見送り", "intraday_score": 0, "reason": "判定に必要な履歴足が不足"}

    signal = evaluate_intraday_entry(
        code=symbol,
        name=name,
        intraday_data=safe_history.tail(160),
        previous_low=_previous_session_low(safe_history, current_ts),
    )
    score = int(signal.get("intraday_score", 0) or 0)
    signal_type = str(signal.get("signal_type", ""))
    if score < config.min_score:
        signal["replay_skip_reason"] = "min_score未満"
        return signal
    if config.target_rule != "すべて" and signal_type != config.target_rule:
        signal["replay_skip_reason"] = "対象ルール外"
        return signal
    if signal.get("judgement") not in {"買い検討OK", "監視強化"}:
        signal["replay_skip_reason"] = "買い判定未達"
        return signal
    return signal


def create_replay_trade(signal: Dict[str, Any], current_bar: pd.Series, current_time: Any) -> Dict[str, Any]:
    entry_price = _num(signal.get("current_price"), _num(current_bar.get("Close")))
    return {
        "signal_time": _timestamp_text(current_time),
        "entry_price": entry_price,
        "stop_loss": signal.get("stop_loss"),
        "take_profit": signal.get("take_profit_1"),
        "score": int(signal.get("intraday_score", 0) or 0),
        "entry_type": signal.get("signal_type", "-"),
        "rule_name": "intraday_replay_rule",
        "signal": {
            key: value
            for key, value in signal.items()
            if key not in {"data"}
        },
    }


def evaluate_replay_trade_outcome(
    trade: Dict[str, Any],
    future_df: pd.DataFrame,
    interval: str = "5m",
) -> Dict[str, Any]:
    entry_price = _num(trade.get("entry_price"))
    stop_loss = _num(trade.get("stop_loss"), 0)
    take_profit = _num(trade.get("take_profit"), 0)
    max_bars = _max_hold_bars(interval)
    horizon = future_df.head(max_bars).copy()

    if horizon.empty or not entry_price:
        return {
            "status": "open",
            "outcome": "open",
            "return_pct": None,
            "max_profit_pct": None,
            "max_drawdown_pct": None,
            "hit_stop_loss": False,
            "hit_take_profit": False,
            "evaluated_until": "",
            "holding_period": "-",
            "exit_reason": "検証中",
        }

    max_profit_pct = _pct(float(horizon["High"].max()), entry_price)
    max_drawdown_pct = _pct(float(horizon["Low"].min()), entry_price)
    return_pct = _pct(float(horizon["Close"].iloc[-1]), entry_price)
    outcome = "timeout"
    status = "closed"
    hit_stop = False
    hit_target = False
    evaluated_until = horizon.index[-1]
    exit_reason = "timeout"

    for ts, row in horizon.iterrows():
        low = _num(row.get("Low"))
        high = _num(row.get("High"))
        if stop_loss and low <= stop_loss:
            hit_stop = True
            outcome = "hit_stop_loss"
            return_pct = _pct(stop_loss, entry_price)
            evaluated_until = ts
            exit_reason = "hit_stop_loss"
            break
        if take_profit and high >= take_profit:
            hit_target = True
            outcome = "hit_take_profit"
            return_pct = _pct(take_profit, entry_price)
            evaluated_until = ts
            exit_reason = "hit_take_profit"
            break

    if len(future_df) < max_bars and outcome == "timeout":
        status = "open"
        outcome = "open"
        exit_reason = "検証中"

    signal_time = pd.Timestamp(trade.get("signal_time"))
    holding_delta = pd.Timestamp(evaluated_until) - signal_time
    holding_period = f"{holding_delta.days}日 {holding_delta.seconds // 3600}時間"

    return {
        "status": status,
        "outcome": outcome,
        "return_pct": return_pct,
        "max_profit_pct": max_profit_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "hit_stop_loss": hit_stop,
        "hit_take_profit": hit_target,
        "evaluated_until": _timestamp_text(evaluated_until),
        "holding_period": holding_period,
        "exit_reason": exit_reason,
    }


def run_replay(
    symbol: str,
    df: pd.DataFrame,
    start_at: Any = None,
    end_at: Any = None,
    rule_config: Optional[Dict[str, Any] | ReplayRuleConfig] = None,
    name: str = "",
) -> Dict[str, Any]:
    config = _normalize_rule_config(rule_config)
    if df is None or df.empty:
        return {
            "trades": [],
            "summary": summarize_replay_results([]) | {"evaluated_steps": 0, "error": "過去データが空です。"},
        }

    data = df.copy().sort_index()
    if start_at:
        data = data[data.index >= pd.Timestamp(start_at)]
    if end_at:
        data = data[data.index <= pd.Timestamp(end_at)]

    trades: List[Dict[str, Any]] = []
    evaluated_steps = 0
    last_trade_index = -10**9
    all_data = df.copy().sort_index()

    for position, (current_time, current_bar) in enumerate(data.iterrows()):
        global_position = all_data.index.get_indexer([current_time])[0]
        if global_position < 24:
            continue
        if len(trades) >= config.max_trades:
            break
        if position - last_trade_index < config.cooldown_bars:
            continue

        history = all_data.loc[all_data.index <= current_time]
        signal = evaluate_replay_step(history, current_time, config, symbol=symbol, name=name)
        evaluated_steps += 1
        if signal.get("replay_skip_reason") or signal.get("judgement") not in {"買い検討OK", "監視強化"}:
            continue

        trade = create_replay_trade(signal, current_bar, current_time)
        future = all_data.loc[all_data.index > current_time]
        trade.update(evaluate_replay_trade_outcome(trade, future, interval=config.interval))
        trade.update(
            {
                "symbol": symbol,
                "name": name,
                "replay_start_at": _timestamp_text(data.index[0]) if not data.empty else "",
                "replay_end_at": _timestamp_text(data.index[-1]) if not data.empty else "",
            }
        )
        trades.append(trade)
        last_trade_index = position

    summary = summarize_replay_results(trades)
    summary["evaluated_steps"] = evaluated_steps
    summary["data_rows"] = len(data)
    summary["first_timestamp"] = _timestamp_text(data.index[0]) if not data.empty else "-"
    summary["last_timestamp"] = _timestamp_text(data.index[-1]) if not data.empty else "-"
    return {"trades": trades, "summary": summary}


def summarize_replay_results(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(trades)
    closed = [trade for trade in trades if trade.get("status") == "closed"]
    returns = [_num(trade.get("return_pct")) for trade in closed if trade.get("return_pct") is not None]
    wins = [trade for trade in closed if _num(trade.get("return_pct")) > 0]
    avg_return = round(sum(returns) / len(returns), 2) if returns else 0.0
    avg_profit = round(
        sum(_num(trade.get("max_profit_pct")) for trade in trades if trade.get("max_profit_pct") is not None)
        / max(1, len([trade for trade in trades if trade.get("max_profit_pct") is not None])),
        2,
    )
    avg_drawdown = round(
        sum(_num(trade.get("max_drawdown_pct")) for trade in trades if trade.get("max_drawdown_pct") is not None)
        / max(1, len([trade for trade in trades if trade.get("max_drawdown_pct") is not None])),
        2,
    )
    return {
        "trade_count": total,
        "closed_count": len(closed),
        "open_count": total - len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
        "avg_return_pct": avg_return,
        "avg_max_profit_pct": avg_profit,
        "avg_max_drawdown_pct": avg_drawdown,
        "hit_stop_loss_count": sum(1 for trade in trades if trade.get("hit_stop_loss")),
        "hit_take_profit_count": sum(1 for trade in trades if trade.get("hit_take_profit")),
    }
