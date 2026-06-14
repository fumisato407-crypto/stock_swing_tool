from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from entry_rules import evaluate_intraday_entry
from multi_timeframe_rules import build_replay_daily_context, evaluate_multi_timeframe_signal, evaluate_risk_filter


INTERVAL_MAX_HOLD_BARS = {
    "5m": 390,
    "15m": 130,
    "60m": 30,
    "1h": 30,
    "1d": 5,
}
DEFAULT_REPLAY_SHARES = 100


@dataclass
class ReplayRuleConfig:
    min_score: int = 70
    target_rule: str = "すべて"
    max_trades: int = 10
    interval: str = "5m"
    cooldown_bars: int = 12
    debug: bool = False
    use_multi_timeframe: bool = True
    daily_min_ok: int = 3
    intraday_min_ok: int = 2
    use_vwap: bool = True
    use_volume_spike: bool = True
    shares: int = DEFAULT_REPLAY_SHARES
    use_risk_filter: bool = True
    max_stop_loss_pct: float = 3.0
    max_loss_yen_limit: float = 20000.0
    min_risk_reward: float = 1.2


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
    def _int_setting(key: str, default: int) -> int:
        return default if values.get(key) is None else int(values.get(key))

    return ReplayRuleConfig(
        min_score=int(values.get("min_score", 70) or 70),
        target_rule=str(values.get("target_rule", "すべて") or "すべて"),
        max_trades=int(values.get("max_trades", 10) or 10),
        interval=str(values.get("interval", "5m") or "5m"),
        cooldown_bars=int(values.get("cooldown_bars", 12) or 12),
        debug=bool(values.get("debug", False)),
        use_multi_timeframe=bool(values.get("use_multi_timeframe", True)),
        daily_min_ok=_int_setting("daily_min_ok", 3),
        intraday_min_ok=_int_setting("intraday_min_ok", 2),
        use_vwap=bool(values.get("use_vwap", True)),
        use_volume_spike=bool(values.get("use_volume_spike", True)),
        shares=_int_setting("shares", DEFAULT_REPLAY_SHARES),
        use_risk_filter=bool(values.get("use_risk_filter", True)),
        max_stop_loss_pct=float(values.get("max_stop_loss_pct", 3.0) or 3.0),
        max_loss_yen_limit=float(values.get("max_loss_yen_limit", 20000.0) or 20000.0),
        min_risk_reward=float(values.get("min_risk_reward", 1.2) or 1.2),
    )


def _multi_timeframe_config(config: ReplayRuleConfig) -> Dict[str, Any]:
    return {
        "daily_min_ok": config.daily_min_ok,
        "intraday_min_ok": config.intraday_min_ok,
        "use_vwap": config.use_vwap,
        "use_volume_spike": config.use_volume_spike,
    }


def _risk_filter_config(config: ReplayRuleConfig) -> Dict[str, Any]:
    return {
        "enabled": config.use_risk_filter,
        "max_stop_loss_pct": config.max_stop_loss_pct,
        "max_loss_yen_limit": config.max_loss_yen_limit,
        "min_risk_reward": config.min_risk_reward,
    }


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
    daily_df: Optional[pd.DataFrame] = None,
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
    if config.use_multi_timeframe:
        daily_context = build_replay_daily_context(safe_history, current_ts, prior_daily_df=daily_df)
        mtf = evaluate_multi_timeframe_signal(
            daily_context,
            safe_history.tail(160),
            current_time=current_ts,
            config=_multi_timeframe_config(config),
        )
        decision = str(mtf.get("decision_category", "avoid"))
        if decision == "buy":
            signal["judgement"] = "買い検討OK"
        elif decision == "watch":
            signal["judgement"] = "監視強化"
        else:
            signal["judgement"] = "見送り"
        signal["intraday_score"] = int(mtf.get("score", signal.get("intraday_score", 0)) or 0)
        signal["signal_type"] = str(mtf.get("entry_type") or signal.get("signal_type", "見送り"))
        signal["multi_timeframe_enabled"] = True
        signal["multi_timeframe_pass"] = bool(mtf.get("multi_timeframe_pass"))
        signal["multi_timeframe_score"] = mtf.get("score")
        signal["multi_timeframe_decision"] = decision
        signal["daily_filter"] = mtf.get("daily_filter", {})
        signal["intraday_entry"] = mtf.get("intraday_entry", {})
        signal["daily_ok_count"] = signal["daily_filter"].get("daily_ok_count", 0)
        signal["daily_total_count"] = signal["daily_filter"].get("daily_total_count", 4)
        signal["intraday_ok_count"] = signal["intraday_entry"].get("intraday_ok_count", 0)
        signal["intraday_total_count"] = signal["intraday_entry"].get("intraday_total_count", 4)
        signal["daily_filter_json"] = signal["daily_filter"]
        signal["intraday_entry_json"] = signal["intraday_entry"]
        signal["multi_timeframe_detail"] = mtf.get("detail_json", {})
        signal["reasons"] = list(dict.fromkeys(list(signal.get("reasons", [])) + list(mtf.get("reasons", []))))
    risk_filter = evaluate_risk_filter(signal, shares=config.shares, config=_risk_filter_config(config))
    signal["risk_filter"] = risk_filter
    signal["risk_filter_json"] = risk_filter.get("risk_filter_json", risk_filter)
    signal["risk_pass"] = risk_filter.get("risk_pass")
    signal["stop_loss_pct"] = risk_filter.get("stop_loss_pct")
    signal["max_loss_yen"] = risk_filter.get("max_loss_yen")
    signal["expected_profit_yen"] = risk_filter.get("expected_profit_yen")
    signal["risk_reward_ratio"] = risk_filter.get("risk_reward_ratio")
    signal["risk_reasons"] = risk_filter.get("risk_reasons", [])
    if config.use_risk_filter and not risk_filter.get("risk_pass"):
        signal["judgement"] = "監視強化"
        signal["replay_skip_reason"] = "risk_filter_ng"
        signal["risk_exclusion_reason"] = "、".join(risk_filter.get("risk_reasons", []))
        return signal
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
        "daily_ok_count": signal.get("daily_ok_count"),
        "daily_total_count": signal.get("daily_total_count"),
        "daily_score": signal.get("daily_score"),
        "daily_rank_at_scan": signal.get("daily_rank_at_scan"),
        "daily_rank_total": signal.get("daily_rank_total"),
        "daily_top_n_pass": signal.get("daily_top_n_pass"),
        "daily_top_n": signal.get("daily_top_n"),
        "intraday_ok_count": signal.get("intraday_ok_count"),
        "intraday_total_count": signal.get("intraday_total_count"),
        "daily_filter_json": signal.get("daily_filter_json", signal.get("daily_filter", {})),
        "intraday_entry_json": signal.get("intraday_entry_json", signal.get("intraday_entry", {})),
        "multi_timeframe_pass": signal.get("multi_timeframe_pass"),
        "risk_pass": signal.get("risk_pass"),
        "stop_loss_pct": signal.get("stop_loss_pct"),
        "max_loss_yen": signal.get("max_loss_yen"),
        "expected_profit_yen": signal.get("expected_profit_yen"),
        "risk_reward_ratio": signal.get("risk_reward_ratio"),
        "risk_filter_json": signal.get("risk_filter_json", signal.get("risk_filter", {})),
        "risk_reasons": signal.get("risk_reasons", []),
        "buy_condition_json": signal.get("buy_condition_json", {}),
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
    debug: bool = False,
    symbol: str = "",
) -> Dict[str, Any]:
    entry_price = _num(trade.get("entry_price"))
    stop_loss = _num(trade.get("stop_loss"), 0)
    take_profit = _num(trade.get("take_profit"), 0)
    max_bars = _max_hold_bars(interval)
    horizon = pd.DataFrame() if future_df is None else future_df.copy().sort_index().head(max_bars)
    future_len = 0 if future_df is None else len(future_df)
    signal_time = trade.get("signal_time")

    if horizon.empty or not entry_price:
        result = {
            "status": "open",
            "outcome": "open",
            "return_pct": None,
            "max_profit_pct": None,
            "max_drawdown_pct": None,
            "hit_stop_loss": False,
            "hit_take_profit": False,
            "evaluated_until": "",
            "exit_time": "",
            "exit_price": None,
            "holding_period": "-",
            "exit_reason": "検証中",
        }
        if debug:
            result["debug"] = {
                "symbol": symbol,
                "signal_time": signal_time,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "future_df_rows": 0,
                "outcome": "open",
            }
        return result

    return_pct = _pct(float(horizon["Close"].iloc[-1]), entry_price)
    outcome = "timeout"
    status = "closed"
    hit_stop = False
    hit_target = False
    evaluated_until = horizon.index[-1]
    exit_price = float(horizon["Close"].iloc[-1])
    exit_reason = "timeout"

    for ts, row in horizon.iterrows():
        low = _num(row.get("Low"))
        high = _num(row.get("High"))
        # Conservative assumption: if one candle touches both stop and target,
        # the stop-loss is treated as hit first.
        if stop_loss and low <= stop_loss:
            hit_stop = True
            outcome = "hit_stop_loss"
            return_pct = _pct(stop_loss, entry_price)
            evaluated_until = ts
            exit_price = stop_loss
            exit_reason = "hit_stop_loss"
            break
        if take_profit and high >= take_profit:
            hit_target = True
            outcome = "hit_take_profit"
            return_pct = _pct(take_profit, entry_price)
            evaluated_until = ts
            exit_price = take_profit
            exit_reason = "hit_take_profit"
            break

    if future_len < max_bars and outcome == "timeout":
        status = "open"
        outcome = "open"
        exit_reason = "検証中"

    outcome_window = horizon.loc[horizon.index <= evaluated_until].copy()
    if outcome_window.empty:
        outcome_window = horizon.head(1).copy()

    min_low = float(outcome_window["Low"].min())
    max_high = float(outcome_window["High"].max())
    min_low_time = outcome_window["Low"].idxmin()
    max_high_time = outcome_window["High"].idxmax()
    max_profit_pct = _pct(max_high, entry_price)
    max_drawdown_pct = _pct(min_low, entry_price)

    signal_time = pd.Timestamp(trade.get("signal_time"))
    holding_delta = pd.Timestamp(evaluated_until) - signal_time
    holding_period = f"{holding_delta.days}日 {holding_delta.seconds // 3600}時間"

    anomaly_rows = outcome_window[
        (outcome_window["High"] >= entry_price * 1.2)
        | (outcome_window["Low"] <= entry_price * 0.8)
    ]

    result = {
        "status": status,
        "outcome": outcome,
        "return_pct": return_pct,
        "max_profit_pct": max_profit_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "hit_stop_loss": hit_stop,
        "hit_take_profit": hit_target,
        "evaluated_until": _timestamp_text(evaluated_until),
        "exit_time": _timestamp_text(evaluated_until),
        "exit_price": exit_price,
        "holding_period": holding_period,
        "exit_reason": exit_reason,
    }
    if debug:
        result["debug"] = {
            "symbol": symbol,
            "signal_time": _timestamp_text(signal_time),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "future_df_start": _timestamp_text(horizon.index[0]),
            "future_df_end": _timestamp_text(horizon.index[-1]),
            "future_df_rows": len(horizon),
            "outcome_window_start": _timestamp_text(outcome_window.index[0]),
            "outcome_window_end": _timestamp_text(outcome_window.index[-1]),
            "outcome_window_rows": len(outcome_window),
            "min_low": min_low,
            "min_low_time": _timestamp_text(min_low_time),
            "max_high": max_high,
            "max_high_time": _timestamp_text(max_high_time),
            "exit_time": _timestamp_text(evaluated_until),
            "exit_price": exit_price,
            "outcome": outcome,
            "return_pct": return_pct,
            "max_profit_pct": max_profit_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "entry_relative_anomaly_count": len(anomaly_rows),
        }
    return result


def run_replay(
    symbol: str,
    df: pd.DataFrame,
    start_at: Any = None,
    end_at: Any = None,
    rule_config: Optional[Dict[str, Any] | ReplayRuleConfig] = None,
    name: str = "",
    daily_df: Optional[pd.DataFrame] = None,
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
        signal = evaluate_replay_step(history, current_time, config, symbol=symbol, name=name, daily_df=daily_df)
        evaluated_steps += 1
        if signal.get("replay_skip_reason") or signal.get("judgement") not in {"買い検討OK", "監視強化"}:
            continue

        trade = create_replay_trade(signal, current_bar, current_time)
        future = all_data.loc[all_data.index > current_time]
        trade.update(
            evaluate_replay_trade_outcome(
                trade,
                future,
                interval=config.interval,
                debug=config.debug,
                symbol=symbol,
            )
        )
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


def _trade_sort_key(trade: Dict[str, Any]) -> pd.Timestamp:
    try:
        return pd.Timestamp(trade.get("signal_time"))
    except Exception:
        return pd.Timestamp.max


def _is_closed_money_trade(trade: Dict[str, Any]) -> bool:
    if str(trade.get("status", "")) != "closed":
        return False
    return trade.get("entry_price") is not None and trade.get("exit_price") is not None


def apply_replay_money_metrics(
    trades: List[Dict[str, Any]],
    shares: int = DEFAULT_REPLAY_SHARES,
) -> List[Dict[str, Any]]:
    share_count = int(shares or DEFAULT_REPLAY_SHARES)
    cumulative = 0.0
    rows: List[Dict[str, Any]] = []
    for trade in sorted([dict(item) for item in trades], key=_trade_sort_key):
        entry_price = _num(trade.get("entry_price"), 0)
        exit_price = _num(trade.get("exit_price"), 0)
        required_capital = entry_price * share_count if entry_price else None
        profit_yen = None
        cumulative_profit = None
        if _is_closed_money_trade(trade) and entry_price and exit_price:
            profit_yen = round((exit_price - entry_price) * share_count)
            cumulative += profit_yen
            cumulative_profit = round(cumulative)
        trade["shares"] = share_count
        trade["required_capital_yen"] = round(required_capital) if required_capital is not None else None
        trade["profit_yen"] = profit_yen
        trade["cumulative_profit_yen"] = cumulative_profit
        rows.append(trade)
    return rows


def _max_streak(values: List[float], positive: bool) -> int:
    best = 0
    current = 0
    for value in values:
        is_hit = value > 0 if positive else value < 0
        if is_hit:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize_replay_money(
    trades: List[Dict[str, Any]],
    shares: int = DEFAULT_REPLAY_SHARES,
) -> Dict[str, Any]:
    prepared = apply_replay_money_metrics(trades, shares)
    closed = [trade for trade in prepared if trade.get("profit_yen") is not None]
    profits = [float(trade["profit_yen"]) for trade in closed]
    wins = [value for value in profits if value > 0]
    losses = [value for value in profits if value < 0]
    required_capitals = [
        float(trade["required_capital_yen"])
        for trade in closed
        if trade.get("required_capital_yen") is not None
    ]
    avg_profit = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    profit_loss_ratio = None
    if avg_profit is not None and avg_loss not in (None, 0):
        profit_loss_ratio = round(avg_profit / abs(avg_loss), 2)

    return {
        "shares": int(shares or DEFAULT_REPLAY_SHARES),
        "closed_trade_count": len(closed),
        "average_required_capital_yen": round(sum(required_capitals) / len(required_capitals)) if required_capitals else None,
        "gross_profit_yen": round(sum(wins)) if wins else 0,
        "gross_loss_yen": round(sum(losses)) if losses else 0,
        "net_profit_yen": round(sum(profits)) if profits else 0,
        "profit_loss_ratio": profit_loss_ratio,
        "average_profit_yen": round(avg_profit) if avg_profit is not None else None,
        "average_loss_yen": round(avg_loss) if avg_loss is not None else None,
        "max_profit_yen": round(max(profits)) if profits else None,
        "max_loss_yen": round(min(profits)) if profits else None,
        "max_win_streak": _max_streak(profits, positive=True),
        "max_loss_streak": _max_streak(profits, positive=False),
    }
