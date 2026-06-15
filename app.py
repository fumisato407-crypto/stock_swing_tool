from __future__ import annotations

import inspect
import re
import time
from datetime import datetime, time as dt_time, timedelta
from typing import Any, Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alert_builder import alert_key, append_alert_log, build_buy_candidate_discord_text, send_discord_webhook
from auto_virtual_logger import run_rule_based_virtual_logging, select_rule_buy_candidates
from config import (
    AI_VIRTUAL_MODEL,
    ALERTS_LOG_PATH,
    DEFAULT_PRICE_PERIOD,
    OPENAI_API_ENABLED,
    REPLAY_TRADES_DB_PATH,
    TRADES_PATH,
    VIRTUAL_TRADES_DB_PATH,
    get_setting,
)
from data_fetcher import fetch_price_data, normalize_jp_symbol
from daily_technical import (
    DEFAULT_DAILY_FETCH_PERIOD,
    MIN_DAILY_HISTORY_ROWS,
    fetch_daily_technical_data,
    get_technical_config_for_preset,
    get_technical_preset_options,
    score_daily_technical_data,
)
from historical_scan_replay import HistoricalScanReplayConfig, attach_watchlist_replay_exports, run_historical_scan_replay
from historical_data import HistoricalDataResult, YFINANCE_AUTO_ADJUST, clear_historical_cache, load_or_fetch_historical_data, validate_historical_ohlcv
from intraday_scanner import fetch_intraday_data, run_raw_intraday_fetch_test, scan_intraday_entries
from market_hours import is_market_open_jst, market_status_label, next_market_open_hint, now_jst as market_now_jst
from notifier import format_yen
from outcome_tracker import update_open_virtual_trade_outcomes
from paper_trader import process_virtual_trade_signals
from pattern_stats import calculate_pattern_stats
from replay_batch import (
    BatchReplayConfig,
    ConditionVariant,
    build_condition_variants,
    run_multi_symbol_replay_comparison,
)
from replay_cache import clear_replay_cache, replay_cache_stats
from replay_engine import (
    CANDIDATE_MODE_EXISTING,
    CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL,
    CANDIDATE_MODE_LABELS,
    CANDIDATE_MODE_TECHNICAL_ONLY,
    apply_replay_money_metrics,
    run_replay,
    summarize_replay_money,
    summarize_replay_results,
)
from replay_store import insert_replay_run, insert_replay_trades, load_replay_trades
from scanner import (
    append_trade_candidate,
    build_signal_table,
    ensure_trades_file,
    load_watchlist,
    scan_watchlist,
)
from scoring import BUY_SCORE_THRESHOLD, WATCH_SCORE_THRESHOLD
from stock_personality import generate_stock_personalities
from time_utils import now_jst_display, now_jst_iso, parse_trade_datetime_to_jst_naive
from virtual_trade_store import (
    get_virtual_database_list,
    insert_virtual_trade,
    load_virtual_trades,
    rows_to_display,
)
from watchlist_filter import filter_watchlist_symbols, parse_symbol_input

st.set_page_config(
    page_title="日本株 1〜5日スイング候補ツール",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1120px; }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.55rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.12rem; }
    .signal-line { margin: 0.18rem 0 0.42rem 0; line-height: 1.55; }
    .summary-list { line-height: 1.85; }
    @media (max-width: 640px) {
        .block-container { padding-left: 0.7rem; padding-right: 0.7rem; }
        h1 { font-size: 1.38rem !important; }
        h2, h3 { font-size: 1.04rem !important; }
        div[data-testid="stMetricValue"] { font-size: 0.95rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.76rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SCORE_LABELS = {
    "market_score": "地合い",
    "trend_score": "トレンド",
    "pullback_score": "押し目",
    "volume_score": "出来高",
    "theme_score": "テーマ",
    "risk_penalty": "リスク減点",
    "total_score": "合計",
}

COMPACT_COLUMNS = [
    "code",
    "name",
    "price",
    "score",
    "daily_rank",
    "daily_score",
    "daily_ok",
    "intraday_ok",
    "multi_timeframe",
    "entry_type",
    "expected_value_label",
    "risk_reward",
]

INTRADAY_TABLE_COLUMNS = [
    "code",
    "name",
    "current_price",
    "intraday_score",
    "judgement",
    "signal_type",
    "buy_zone",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "reason",
    "normalized_symbol",
    "error_type",
    "error_message",
    "fetched_rows",
    "intraday_rows",
    "latest_5m_jst",
    "latest_close",
    "latest_volume",
    "vwap",
    "cache_hit",
    "last_attempt_at",
]

PLOTLY_CHART_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
}
REPLAY_PERIOD_OPTIONS = {
    "1ヶ月": "1mo",
    "3ヶ月": "3mo",
    "6ヶ月": "6mo",
    "1年": "1y",
}
REPLAY_INTERVAL_OPTIONS = {
    "5分足": "5m",
    "15分足": "15m",
    "1時間足": "60m",
    "日足": "1d",
}
REPLAY_RULE_OPTIONS = ["すべて", "押し目反発", "ブレイク狙い", "後場V字回復"]
REPLAY_RULE_LABEL_TO_SIGNAL = {
    "すべて": "すべて",
    "押し目反発": "押し目再反発",
    "ブレイク狙い": "節目ブレイク",
    "後場V字回復": "後場V字回復",
}
HISTORICAL_SCAN_MAX_SYMBOL_OPTIONS = {
    "10": 10,
    "30": 30,
    "50": 50,
    "全件": None,
}
HISTORICAL_SCAN_INTERVAL_OPTIONS = {
    "5分": "5m",
    "15分": "15m",
    "30分": "30m",
    "1時間": "60m",
    "1日": "1d",
}
HISTORICAL_SCAN_TARGET_OPTIONS = {
    "買い候補のみ": "buy_only",
    "買い候補＋監視": "buy_watch",
}
HISTORICAL_SCAN_COOLDOWN_OPTIONS = {
    "30分": "30m",
    "1時間": "1h",
    "3時間": "3h",
    "当日中": "day",
}
DAILY_TOP_N_OPTIONS = [5, 10, 15, 20, 30]
NORMAL_SCAN_BUY_LIMIT_OPTIONS = [1, 3, 5, 10]
DISCORD_BUY_SCORE_THRESHOLD = 70
DISCORD_COOLDOWN_MINUTES = 30
JST = ZoneInfo("Asia/Tokyo")


def _render_plotly_chart(fig: go.Figure, key: str) -> None:
    width_kwargs: Dict[str, Any]
    if "width" in inspect.signature(st.plotly_chart).parameters:
        width_kwargs = {"width": "stretch"}
    else:
        width_kwargs = {"use_container_width": True}
    st.plotly_chart(fig, config=PLOTLY_CHART_CONFIG, key=key, **width_kwargs)


def _now_jst() -> datetime:
    return datetime.now(JST)


def _is_jpx_market_time(now: datetime | None = None) -> bool:
    current = now or _now_jst()
    if current.weekday() >= 5:
        return False
    current_time = current.time()
    morning = dt_time(9, 0) <= current_time <= dt_time(11, 30)
    afternoon = dt_time(12, 30) <= current_time <= dt_time(15, 30)
    return morning or afternoon


def _intraday_score(signal: Dict[str, Any]) -> int:
    try:
        return int(float(signal.get("intraday_score", signal.get("score", 0)) or 0))
    except (TypeError, ValueError):
        return 0


def _notification_symbol(signal: Dict[str, Any]) -> str:
    return str(signal.get("normalized_symbol") or signal.get("code") or "").strip()


def _is_discord_buy_candidate(signal: Dict[str, Any]) -> bool:
    judgement = str(signal.get("judgement") or "")
    category = str(signal.get("category") or "")
    return (
        _intraday_score(signal) >= DISCORD_BUY_SCORE_THRESHOLD
        and (judgement in {"買い検討OK", "買い候補"} or category == "買い候補")
    )


def _parse_state_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=JST)
    except ValueError:
        return None


def _notification_cooldown_elapsed(last_sent_at: datetime | None, now: datetime) -> bool:
    if last_sent_at is None:
        return True
    return now - last_sent_at >= timedelta(minutes=DISCORD_COOLDOWN_MINUTES)


def _format_jst(dt: datetime) -> str:
    return dt.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S")


def _parse_display_datetime(value: Any) -> datetime | None:
    if not value or value == "-":
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").replace(tzinfo=JST)
    except ValueError:
        return None


def _next_check_text(auto_monitor_enabled: bool, refresh_seconds: int) -> str:
    if not auto_monitor_enabled:
        return "-"
    last_checked_at = _parse_display_datetime(st.session_state.get("intraday_last_checked_at", "-"))
    if last_checked_at is None:
        return f"{refresh_seconds}秒以内"
    return _format_jst(last_checked_at + timedelta(seconds=refresh_seconds))


def _recently_checked(min_seconds: int = 5) -> bool:
    last_checked_at = _parse_display_datetime(st.session_state.get("intraday_last_checked_at", "-"))
    if last_checked_at is None:
        return False
    return (_now_jst() - last_checked_at) < timedelta(seconds=min_seconds)


def _records_key(df: pd.DataFrame) -> Tuple[Tuple[str, str, str, str, str, str], ...]:
    return tuple(
        (
            str(getattr(row, "code", "")),
            str(getattr(row, "name", "")),
            str(getattr(row, "theme", "")),
            str(getattr(row, "market", "")),
            str(getattr(row, "raw_code", "")),
            str(getattr(row, "normalized_symbol", "")),
        )
        for row in df.itertuples(index=False)
    )


def _records_from_key(records_key: Tuple[Tuple[str, str, str, str, str, str], ...]) -> List[Dict[str, str]]:
    return [
        {
            "code": code,
            "name": name,
            "theme": theme,
            "market": market,
            "raw_code": raw_code,
            "normalized_symbol": normalized_symbol,
        }
        for code, name, theme, market, raw_code, normalized_symbol in records_key
    ]


def _settings_key(settings: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(settings.items()))


def _settings_from_key(settings_key: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    return dict(settings_key)


@st.cache_data(ttl=900, show_spinner=False)
def _scan_cached(
    records_key: Tuple[Tuple[str, str, str, str, str, str], ...],
    period: str,
    refresh_token: int,
    settings_key: Tuple[Tuple[str, Any], ...],
) -> List[Dict[str, Any]]:
    del refresh_token
    settings = _settings_from_key(settings_key)
    return scan_watchlist(
        _records_from_key(records_key),
        period=period,
        use_daily_top_n=bool(settings.get("use_daily_top_n", True)),
        daily_top_n=int(settings.get("daily_top_n", 10)),
        min_daily_score=int(settings.get("min_daily_score", 70)),
        daily_min_ok=int(settings.get("daily_min_ok", 3)),
        intraday_min_ok=int(settings.get("intraday_min_ok", 2)),
        use_vwap=bool(settings.get("use_vwap", True)),
        use_volume_spike=bool(settings.get("use_volume_spike", True)),
        use_risk_filter=bool(settings.get("use_risk_filter", True)),
        max_stop_loss_pct=float(settings.get("max_stop_loss_pct", 3.0)),
        max_loss_yen_limit=float(settings.get("max_loss_yen_limit", 20000.0)),
        min_risk_reward=float(settings.get("min_risk_reward", 1.2)),
        max_buy_candidates=int(settings.get("max_buy_candidates", 1)),
    )


def _score(signal: Dict[str, Any]) -> int:
    try:
        return int(signal.get("score", 0))
    except (TypeError, ValueError):
        return 0


def _run_stock_scan(watchlist: pd.DataFrame, period: str, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    scan_start = now_jst_display()
    started = time.perf_counter()
    signals = _scan_cached(_records_key(watchlist), period, st.session_state.get("refresh_token", 0), _settings_key(settings))
    elapsed = round(time.perf_counter() - started, 2)
    failed_count = sum(1 for signal in signals if signal.get("category") == "取得失敗")
    daily_top_n_pass_count = sum(1 for signal in signals if signal.get("daily_top_n_pass"))
    st.session_state["latest_signals"] = signals
    st.session_state["scan_status"] = {
        "scan_start_jst": scan_start,
        "scan_end_jst": now_jst_display(),
        "elapsed_seconds": elapsed,
        "signals_count": len(signals),
        "failed_count": failed_count,
        "daily_top_n_pass_count": daily_top_n_pass_count,
        "settings": dict(settings),
    }
    return signals


def _render_scan_status() -> None:
    status = st.session_state.get("scan_status", {})
    if not status:
        st.info("初期表示では株価スキャンを実行しません。必要なときに「株価スキャンを実行」を押してください。")
        return
    cols = st.columns(6)
    cols[0].metric("signals", status.get("signals_count", 0))
    cols[1].metric("failed", status.get("failed_count", 0))
    cols[2].metric("日足上位N通過", status.get("daily_top_n_pass_count", 0))
    cols[3].metric("elapsed", f"{status.get('elapsed_seconds', '-')}秒")
    cols[4].metric("scan_start_jst", status.get("scan_start_jst", "-"))
    cols[5].metric("scan_end_jst", status.get("scan_end_jst", "-"))


def _condition_settings_rows(settings: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"条件": "日足上位N", "設定": "ON" if settings.get("use_daily_top_n") else "OFF"},
            {"条件": "日足スコア上位N", "設定": settings.get("daily_top_n", "-")},
            {"条件": "日足最低スコア", "設定": settings.get("min_daily_score", "-")},
            {"条件": "日足OK最低", "設定": f"{settings.get('daily_min_ok', '-')}/4"},
            {"条件": "5分足OK最低", "設定": f"{settings.get('intraday_min_ok', '-')}/4"},
            {"条件": "VWAP", "設定": "使用" if settings.get("use_vwap") else "未使用"},
            {"条件": "出来高急増", "設定": "使用" if settings.get("use_volume_spike") else "未使用"},
            {"条件": "リスク条件", "設定": "ON" if settings.get("use_risk_filter") else "OFF"},
            {"条件": "最大損切り幅", "設定": f"{float(settings.get('max_stop_loss_pct', 0)):.1f}%"},
            {"条件": "最大損失", "設定": f"{float(settings.get('max_loss_yen_limit', 0)):,.0f}円"},
            {"条件": "最低損益比", "設定": f"{float(settings.get('min_risk_reward', 0)):.2f}"},
            {"条件": "1回の最大買い候補", "設定": f"{settings.get('max_buy_candidates', '-')}件"},
        ]
    )


def _render_condition_settings_summary(title: str, settings: Dict[str, Any]) -> None:
    st.markdown(f"**{title}**")
    st.dataframe(_safe_dataframe(_condition_settings_rows(settings)), width="stretch", hide_index=True)


def _render_normal_scan_settings() -> Dict[str, Any]:
    with st.expander("通常スキャン条件", expanded=False):
        st.caption("過去リプレイ検証と同じ意味の条件です。保存機能はまだなく、この画面を開いている間はsession_stateで保持します。")
        cols = st.columns(4)
        use_daily_top_n = cols[0].toggle("日足スコア上位Nを使う", value=True, key="normal_scan_use_daily_top_n")
        daily_top_n = int(cols[1].selectbox("日足スコア上位N", DAILY_TOP_N_OPTIONS, index=1, key="normal_scan_daily_top_n"))
        min_daily_score = int(cols[2].number_input("日足最低スコア", min_value=0, max_value=100, value=70, step=1, key="normal_scan_min_daily_score"))
        max_buy_candidates = int(cols[3].selectbox("1回の最大買い候補件数", NORMAL_SCAN_BUY_LIMIT_OPTIONS, index=0, key="normal_scan_max_buy_candidates"))

        cols = st.columns(4)
        daily_min_ok = int(cols[0].selectbox("日足OK最低数", [2, 3, 4], index=1, key="normal_scan_daily_min_ok"))
        intraday_min_ok = int(cols[1].selectbox("5分足OK最低数", [2, 3, 4], index=0, key="normal_scan_intraday_min_ok"))
        use_vwap = cols[2].toggle("VWAP条件を使う", value=True, key="normal_scan_use_vwap")
        use_volume_spike = cols[3].toggle("出来高急増条件を使う", value=True, key="normal_scan_use_volume_spike")

        cols = st.columns(4)
        use_risk_filter = cols[0].toggle("リスク条件ON/OFF", value=True, key="normal_scan_use_risk_filter")
        max_stop_loss_pct = float(cols[1].number_input("最大損切り幅 %", min_value=0.1, max_value=20.0, value=3.0, step=0.1, key="normal_scan_max_stop_loss_pct"))
        max_loss_yen_limit = float(cols[2].number_input("最大想定損失 円", min_value=1000, max_value=500000, value=20000, step=1000, key="normal_scan_max_loss_yen_limit"))
        min_risk_reward = float(cols[3].number_input("最低損益比", min_value=0.1, max_value=10.0, value=1.2, step=0.1, key="normal_scan_min_risk_reward"))

    settings = {
        "use_daily_top_n": bool(use_daily_top_n),
        "daily_top_n": int(daily_top_n),
        "min_daily_score": int(min_daily_score),
        "daily_min_ok": int(daily_min_ok),
        "intraday_min_ok": int(intraday_min_ok),
        "use_vwap": bool(use_vwap),
        "use_volume_spike": bool(use_volume_spike),
        "use_risk_filter": bool(use_risk_filter),
        "max_stop_loss_pct": float(max_stop_loss_pct),
        "max_loss_yen_limit": float(max_loss_yen_limit),
        "min_risk_reward": float(min_risk_reward),
        "max_buy_candidates": int(max_buy_candidates),
    }
    st.session_state["current_buy_condition_settings"] = settings
    _render_condition_settings_summary("現在の通常スキャン条件", settings)
    return settings


def _split_signals(signals: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], ...]:
    failed = [s for s in signals if s.get("category") == "取得失敗"]
    analyzable = [s for s in signals if s.get("category") != "取得失敗"]
    buy = [s for s in analyzable if s.get("category") == "買い候補" or _score(s) >= BUY_SCORE_THRESHOLD]
    buy_ids = {id(s) for s in buy}
    watch = [
        s
        for s in analyzable
        if id(s) not in buy_ids
        and (s.get("category") == "監視" or WATCH_SCORE_THRESHOLD <= _score(s) < BUY_SCORE_THRESHOLD)
    ]
    watch_ids = {id(s) for s in watch}
    avoid = [s for s in analyzable if id(s) not in buy_ids and id(s) not in watch_ids]
    return buy, watch, avoid, failed


def _format_targets(signal: Dict[str, Any]) -> str:
    return f"{format_yen(signal.get('target_1'))} / {format_yen(signal.get('target_2'))}"


def _format_rr(signal: Dict[str, Any]) -> str:
    return str(signal.get("risk_reward_label") or signal.get("risk_reward") or "-")


def _make_price_chart(signal: Dict[str, Any]) -> go.Figure:
    df = signal.get("history")
    fig = go.Figure()
    if df is None or getattr(df, "empty", True):
        fig.update_layout(height=260, margin=dict(l=8, r=8, t=28, b=8))
        return fig

    chart = df.tail(90)
    fig.add_trace(
        go.Candlestick(
            x=chart.index,
            open=chart["Open"],
            high=chart["High"],
            low=chart["Low"],
            close=chart["Close"],
            name="日足",
            increasing_line_color="#dc2626",
            decreasing_line_color="#2563eb",
        )
    )
    for col, color, label in [
        ("sma_5", "#f59e0b", "5日線"),
        ("sma_25", "#059669", "25日線"),
        ("sma_75", "#6b7280", "75日線"),
    ]:
        if col in chart.columns:
            fig.add_trace(
                go.Scatter(
                    x=chart.index,
                    y=chart[col],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.5),
                )
            )
    fig.update_layout(
        height=330,
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _score_breakdown_rows(signal: Dict[str, Any]) -> pd.DataFrame:
    breakdown = signal.get("score_breakdown") or {}
    return pd.DataFrame(
        [{"項目": label, "点": breakdown.get(key, 0)} for key, label in SCORE_LABELS.items()]
    )


def _score_breakdown_chart(signal: Dict[str, Any]) -> go.Figure:
    rows = _score_breakdown_rows(signal)
    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in rows["点"]]
    fig = go.Figure(go.Bar(x=rows["点"], y=rows["項目"], orientation="h", marker_color=colors))
    fig.update_layout(height=250, margin=dict(l=8, r=8, t=8, b=8), xaxis_title="点")
    return fig


def _make_intraday_chart(signal: Dict[str, Any]) -> go.Figure:
    df = signal.get("data")
    fig = go.Figure()
    if df is None or getattr(df, "empty", True):
        fig.update_layout(height=260, margin=dict(l=8, r=8, t=28, b=8))
        return fig

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="5分足",
            increasing_line_color="#dc2626",
            decreasing_line_color="#2563eb",
        )
    )
    for col, color, label in [
        ("ma_short", "#f59e0b", "短期線"),
        ("ma_mid", "#059669", "中期線"),
        ("ma_long", "#6b7280", "長期線"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=label, line=dict(color=color, width=1.4)))
    level = signal.get("level_price")
    if level:
        fig.add_hline(y=level, line_dash="dot", line_color="#7c3aed", annotation_text=f"節目 {format_yen(level)}")
    fig.update_layout(
        height=310,
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _compact_table(signals: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        intraday_ok_display = signal.get("intraday_ok_display")
        if not intraday_ok_display:
            intraday_ok_display = f"{signal.get('intraday_ok_count', '-')}/{signal.get('intraday_total_count', '-')}"
        multi_timeframe_display = signal.get("multi_timeframe_status")
        if not multi_timeframe_display:
            multi_timeframe_display = "OK" if signal.get("multi_timeframe_pass") else "NG"
        rows.append(
            {
                "code": signal.get("code", ""),
                "name": signal.get("name", ""),
                "price": format_yen(signal.get("price")),
                "score": signal.get("score", "-"),
                "daily_rank": (
                    f"{signal.get('daily_rank_at_scan')}位/{signal.get('daily_rank_total')}銘柄"
                    if signal.get("daily_rank_at_scan") not in (None, "")
                    else "-"
                ),
                "daily_score": _format_score_value(signal.get("daily_score")),
                "daily_ok": f"{signal.get('daily_ok_count', '-')}/{signal.get('daily_total_count', '-')}",
                "intraday_ok": intraday_ok_display,
                "multi_timeframe": multi_timeframe_display,
                "entry_type": signal.get("entry_type", "-"),
                "expected_value_label": signal.get("expected_value_label", "-"),
                "risk_reward": _format_rr(signal),
            }
        )
    return pd.DataFrame(rows, columns=COMPACT_COLUMNS)


def _parse_manual_codes(raw_text: str) -> List[str]:
    if not raw_text.strip():
        return []
    normalized = raw_text.replace(",", " ").replace("\n", " ")
    return [part.strip().replace(".T", "") for part in normalized.split() if part.strip()]


def _build_intraday_targets(
    watchlist: pd.DataFrame,
    buy: List[Dict[str, Any]],
    watch: List[Dict[str, Any]],
    manual_codes: List[str],
    include_watchlist: bool,
    include_swing_focus: bool,
) -> List[Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}

    def add_record(code: Any, name: Any = "", theme: Any = "", market: Any = "", raw_code: Any = "") -> None:
        normalized_symbol = normalize_jp_symbol(code)
        if not normalized_symbol:
            return
        normalized = normalized_symbol.replace(".T", "")
        if normalized not in records:
            records[normalized] = {
                "code": normalized,
                "name": str(name or normalized),
                "theme": str(theme or ""),
                "market": str(market or ""),
                "raw_code": str(raw_code or code or ""),
                "normalized_symbol": normalized_symbol,
            }

    if include_watchlist and not watchlist.empty:
        for row in watchlist.to_dict("records"):
            add_record(
                row.get("normalized_symbol") or row.get("code"),
                row.get("name"),
                row.get("theme"),
                row.get("market"),
                row.get("raw_code") or row.get("code"),
            )

    if include_swing_focus:
        for signal in buy + watch:
            add_record(
                signal.get("normalized_symbol") or signal.get("code"),
                signal.get("name"),
                signal.get("theme"),
                signal.get("market"),
                signal.get("raw_code") or signal.get("code"),
            )

    for code in manual_codes:
        add_record(code)

    return list(records.values())


def _intraday_table(signals: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        rows.append(
            {
                "code": signal.get("code", ""),
                "name": signal.get("name", ""),
                "current_price": format_yen(signal.get("current_price")),
                "intraday_score": signal.get("intraday_score", 0),
                "judgement": signal.get("judgement", "-"),
                "signal_type": signal.get("signal_type", "-"),
                "buy_zone": signal.get("buy_zone", "-"),
                "stop_loss": format_yen(signal.get("stop_loss")),
                "take_profit_1": format_yen(signal.get("take_profit_1")),
                "take_profit_2": format_yen(signal.get("take_profit_2")),
                "reason": "、".join(signal.get("reasons", [])) if signal.get("reasons") else signal.get("error", "-"),
                "normalized_symbol": signal.get("normalized_symbol", ""),
                "error_type": signal.get("error_type", ""),
                "error_message": signal.get("error_message", signal.get("error", "")),
                "fetched_rows": signal.get("fetched_rows", ""),
                "intraday_rows": signal.get("intraday_rows", signal.get("fetched_rows", "")),
                "latest_5m_jst": signal.get("latest_5m_jst", signal.get("last_time", "")),
                "latest_close": format_yen(signal.get("latest_close", signal.get("current_price"))),
                "latest_volume": signal.get("latest_volume", signal.get("current_volume", "")),
                "vwap": format_yen(signal.get("vwap")),
                "cache_hit": signal.get("cache_hit", False),
                "last_attempt_at": signal.get("last_attempt_at", ""),
            }
        )
    return pd.DataFrame(rows, columns=INTRADAY_TABLE_COLUMNS)


def _intraday_fetch_debug_table(signals: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        rows.append(
            {
                "code": signal.get("code", ""),
                "ticker_for_intraday": signal.get("ticker_for_intraday", signal.get("normalized_symbol", "")),
                "normalized_symbol": signal.get("normalized_symbol", ""),
                "intraday_fetch_ok": signal.get("intraday_fetch_ok", False),
                "intraday_rows": signal.get("intraday_rows", signal.get("fetched_rows", 0)),
                "latest_5m_jst": signal.get("latest_5m_jst", signal.get("last_time", "-")),
                "latest_close": format_yen(signal.get("latest_close", signal.get("current_price"))),
                "latest_volume": signal.get("latest_volume", signal.get("current_volume", "-")),
                "vwap": format_yen(signal.get("vwap")),
                "fetch_error_type": signal.get("error_type", ""),
                "fetch_error_message": signal.get("error_message", signal.get("error", "")),
                "cache_hit": signal.get("cache_hit", False),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "code",
            "ticker_for_intraday",
            "normalized_symbol",
            "intraday_fetch_ok",
            "intraday_rows",
            "latest_5m_jst",
            "latest_close",
            "latest_volume",
            "vwap",
            "fetch_error_type",
            "fetch_error_message",
            "cache_hit",
        ],
    )


def _intraday_recent_bar_table(signal: Dict[str, Any]) -> pd.DataFrame:
    data = signal.get("data")
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame(
            columns=[
                "bar_time_jst",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "is_bullish",
                "close_above_vwap",
                "volume_spike_ok",
                "high_break_ok",
                "pullback_rebound_ok",
                "five_min_bar_ok",
                "ng_reason",
            ]
        )
    frame = data.copy().sort_index()
    if "vwap" not in frame.columns:
        typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3
        volume = frame["Volume"].fillna(0).clip(lower=0)
        frame["vwap"] = ((typical * volume).cumsum() / volume.cumsum().replace(0, pd.NA)).fillna(
            typical.expanding(min_periods=1).mean()
        )
    rows = []
    for pos, (ts, row) in enumerate(frame.tail(4).iterrows()):
        loc = frame.index.get_loc(ts)
        if isinstance(loc, slice):
            loc = loc.stop - 1
        prior = frame.iloc[: int(loc)] if int(loc) > 0 else pd.DataFrame()
        prior_high = float(prior["High"].tail(12).max()) if not prior.empty else None
        avg_volume = float(prior["Volume"].tail(12).mean()) if not prior.empty else None
        previous_close = float(frame.iloc[int(loc) - 1]["Close"]) if int(loc) > 0 else None
        close = _to_float(row.get("Close"))
        high = _to_float(row.get("High"))
        low = _to_float(row.get("Low"))
        open_price = _to_float(row.get("Open"))
        volume_value = _to_float(row.get("Volume"), 0) or 0
        vwap = _to_float(row.get("vwap"))
        is_bullish = close is not None and open_price is not None and close >= open_price
        close_above_vwap = close is not None and vwap is not None and close > vwap
        volume_spike_ok = avg_volume not in (None, 0) and volume_value >= avg_volume * 1.5
        high_break_ok = prior_high not in (None, 0) and close is not None and close > prior_high
        pullback_rebound_ok = (
            low is not None
            and close is not None
            and previous_close is not None
            and vwap is not None
            and low <= vwap * 1.003
            and close > vwap
            and close > previous_close
        )
        five_min_bar_ok = bool(close_above_vwap and (high_break_ok or pullback_rebound_ok or volume_spike_ok))
        ng_reasons = []
        if not close_above_vwap:
            ng_reasons.append("VWAP未達")
        if not high_break_ok:
            ng_reasons.append("高値突破なし")
        if not pullback_rebound_ok:
            ng_reasons.append("押し目反発なし")
        if not volume_spike_ok:
            ng_reasons.append("出来高急増なし")
        rows.append(
            {
                "bar_time_jst": ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts),
                "open": format_yen(open_price),
                "high": format_yen(high),
                "low": format_yen(low),
                "close": format_yen(close),
                "volume": int(volume_value),
                "vwap": format_yen(vwap),
                "is_bullish": "OK" if is_bullish else "NG",
                "close_above_vwap": "OK" if close_above_vwap else "NG",
                "volume_spike_ok": "OK" if volume_spike_ok else "NG",
                "high_break_ok": "OK" if high_break_ok else "NG",
                "pullback_rebound_ok": "OK" if pullback_rebound_ok else "NG",
                "five_min_bar_ok": "OK" if five_min_bar_ok else "NG",
                "ng_reason": "、".join(ng_reasons) if ng_reasons else "-",
            }
        )
    return pd.DataFrame(rows)


def _failure_debug_table(signals: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        rows.append(
            {
                "code": signal.get("raw_code") or signal.get("code", ""),
                "ticker_for_intraday": signal.get("ticker_for_intraday", signal.get("normalized_symbol", "")),
                "normalized_symbol": signal.get("normalized_symbol", ""),
                "error_type": signal.get("error_type", ""),
                "error_message": signal.get("error_message", signal.get("error", signal.get("comment", ""))),
                "fetched_rows": signal.get("fetched_rows", 0),
                "intraday_rows": signal.get("intraday_rows", signal.get("fetched_rows", 0)),
                "latest_5m_jst": signal.get("latest_5m_jst", "-"),
                "latest_close": format_yen(signal.get("latest_close")),
                "latest_volume": signal.get("latest_volume", "-"),
                "vwap": format_yen(signal.get("vwap")),
                "last_attempt_at": signal.get("last_attempt_at", ""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "code",
            "ticker_for_intraday",
            "normalized_symbol",
            "error_type",
            "error_message",
            "fetched_rows",
            "intraday_rows",
            "latest_5m_jst",
            "latest_close",
            "latest_volume",
            "vwap",
            "last_attempt_at",
        ],
    )


def _render_failure_debug(signals: List[Dict[str, Any]], title: str = "取得失敗の理由") -> None:
    if not signals:
        return
    with st.expander(title, expanded=True):
        st.dataframe(_failure_debug_table(signals), width="stretch", hide_index=True)


def _render_all_failed_warning(signals: List[Dict[str, Any]]) -> None:
    if not signals:
        return
    st.error("全銘柄の取得に失敗しています。銘柄コード形式、yfinance接続、watchlist.csvの形式を確認してください。")
    st.caption("デバッグ用に最初の5件だけ表示します。")
    st.dataframe(_failure_debug_table(signals).head(5), width="stretch", hide_index=True)


def _render_count_metrics(
    buy: List[Dict[str, Any]],
    watch: List[Dict[str, Any]],
    avoid: List[Dict[str, Any]],
    failed: List[Dict[str, Any]],
) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("買い候補", len(buy))
    metric_cols[1].metric("監視", len(watch))
    metric_cols[2].metric("触らない", len(avoid))
    metric_cols[3].metric("取得失敗", len(failed))


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _condition_status(value: bool | None, unavailable: str = "未使用") -> str:
    if value is None:
        return unavailable
    return "OK" if value else "NG"


def _condition_table(rows: List[Dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["条件", "判定", "補足"])


def _latest_daily_row(signal: Dict[str, Any]) -> pd.Series | None:
    history = signal.get("history")
    if isinstance(history, pd.DataFrame) and not history.empty:
        return history.iloc[-1]
    return None


def _daily_condition_rows(signal: Dict[str, Any]) -> Tuple[pd.DataFrame, bool | None]:
    saved = signal.get("daily_filter") or signal.get("daily_filter_json")
    if isinstance(saved, dict) and saved:
        rows = [
            {
                "条件": "25日線より上",
                "判定": _condition_status(bool(saved.get("ma25_ok"))),
                "補足": f"終値 {format_yen(saved.get('close'))} / 25日線 {format_yen(saved.get('ma25'))}",
            },
            {
                "条件": "前日終値より上",
                "判定": _condition_status(bool(saved.get("above_prev_close_ok"))),
                "補足": f"終値 {format_yen(saved.get('close'))} / 前日終値 {format_yen(saved.get('previous_close'))}",
            },
            {
                "条件": "直近5日高値圏",
                "判定": _condition_status(bool(saved.get("near_5day_high_ok"))),
                "補足": f"終値 {format_yen(saved.get('close'))} / 5日高値 {format_yen(saved.get('recent_5day_high'))}",
            },
            {
                "条件": "日足出来高増加",
                "判定": _condition_status(bool(saved.get("daily_volume_increase_ok"))),
                "補足": f"出来高 {int(_to_float(saved.get('current_volume'), 0) or 0):,} / 5日平均 {int(_to_float(saved.get('avg_volume_5d'), 0) or 0):,}",
            },
        ]
        return _condition_table(rows), bool(saved.get("daily_pass"))

    row = _latest_daily_row(signal)
    if row is None:
        rows = [
            {"条件": "25日線より上", "判定": "未取得", "補足": "日足履歴なし"},
            {"条件": "前日終値より上", "判定": "未取得", "補足": "日足履歴なし"},
            {"条件": "直近5日高値圏", "判定": "未取得", "補足": "日足履歴なし"},
            {"条件": "出来高増加", "判定": "未取得", "補足": "日足履歴なし"},
        ]
        return _condition_table(rows), None

    close = _to_float(row.get("Close"))
    sma_25 = _to_float(row.get("sma_25"))
    prev_close = _to_float(row.get("prev_close"))
    recent_5_high = _to_float(row.get("recent_5_high"))
    volume = _to_float(row.get("Volume"))
    volume_ma = _to_float(row.get("volume_ma_20"))

    above_25 = close is not None and sma_25 is not None and close > sma_25
    above_prev_close = close is not None and prev_close is not None and close > prev_close
    near_5_high = close is not None and recent_5_high is not None and close >= recent_5_high * 0.98
    volume_increased = volume is not None and volume_ma is not None and volume >= volume_ma

    rows = [
        {
            "条件": "25日線より上",
            "判定": _condition_status(above_25 if close is not None and sma_25 is not None else None, "未取得"),
            "補足": f"終値 {format_yen(close)} / 25日線 {format_yen(sma_25)}",
        },
        {
            "条件": "前日終値より上",
            "判定": _condition_status(above_prev_close if close is not None and prev_close is not None else None, "未取得"),
            "補足": f"終値 {format_yen(close)} / 前日終値 {format_yen(prev_close)}",
        },
        {
            "条件": "直近5日高値圏",
            "判定": _condition_status(near_5_high if close is not None and recent_5_high is not None else None, "未取得"),
            "補足": f"終値 {format_yen(close)} / 5日高値 {format_yen(recent_5_high)}",
        },
        {
            "条件": "出来高増加",
            "判定": _condition_status(volume_increased if volume is not None and volume_ma is not None else None, "未取得"),
            "補足": f"出来高 {int(volume or 0):,} / 20日平均 {int(volume_ma or 0):,}",
        },
    ]
    daily_ok = _score(signal) >= WATCH_SCORE_THRESHOLD and above_25
    return _condition_table(rows), bool(daily_ok)


def _intraday_condition_rows(signal: Dict[str, Any]) -> Tuple[pd.DataFrame, bool | None]:
    saved = signal.get("intraday_entry") or signal.get("intraday_entry_json")
    if isinstance(saved, dict) and saved:
        if saved.get("intraday_available") is False or saved.get("intraday_data_status") in {"データなし", "データ不足"}:
            reason = saved.get("fetch_error_message") or saved.get("reason") or "5分足データが空です"
            rows = [
                {"条件": "VWAP上", "判定": "データなし", "補足": reason},
                {"条件": "直近高値突破", "判定": "データなし", "補足": "過去高値 N/A"},
                {"条件": "押し目反発", "判定": "データなし", "補足": "判定不可"},
                {"条件": "出来高急増", "判定": "データなし", "補足": "出来高 N/A / 平均 N/A"},
            ]
            return _condition_table(rows), None
        rows = [
            {
                "条件": "VWAP上",
                "判定": _condition_status(saved.get("vwap_ok")),
                "補足": f"現在値 {format_yen(saved.get('current_close'))} / VWAP {format_yen(saved.get('vwap'))}",
            },
            {
                "条件": "直近高値突破",
                "判定": _condition_status(saved.get("breakout_ok")),
                "補足": f"過去高値 {format_yen(saved.get('past_n_bars_high'))}",
            },
            {
                "条件": "押し目反発",
                "判定": _condition_status(saved.get("pullback_rebound_ok")),
                "補足": "VWAP付近への押し後に再上抜け",
            },
            {
                "条件": "出来高急増",
                "判定": _condition_status(saved.get("intraday_volume_spike_ok")),
                "補足": f"出来高 {int(_to_float(saved.get('current_volume'), 0) or 0):,} / 平均 {int(_to_float(saved.get('avg_volume_12bars'), 0) or 0):,}",
            },
        ]
        return _condition_table(rows), bool(saved.get("intraday_pass"))

    has_intraday = any(
        key in signal
        for key in (
            "intraday_score",
            "above_mas",
            "level_breakout",
            "level_hold",
            "higher_low",
            "volume_ratio",
        )
    )
    if not has_intraday:
        rows = [
            {"条件": "VWAP上", "判定": "未取得", "補足": "日足上位N対象外、または5分足未取得"},
            {"条件": "直近高値突破", "判定": "未取得", "補足": "日足上位N通過後に5分足で確認"},
            {"条件": "押し目反発", "判定": "未取得", "補足": "日足上位N通過後に5分足で確認"},
            {"条件": "出来高急増", "判定": "未取得", "補足": "日足上位N通過後に5分足で確認"},
        ]
        return _condition_table(rows), None

    level_breakout = bool(signal.get("level_breakout") or signal.get("level_hold"))
    higher_low = bool(signal.get("higher_low"))
    signal_type = str(signal.get("signal_type", ""))
    pullback_rebound = higher_low or signal_type in {"押し目再反発", "後場V字回復"}
    volume_ratio = _to_float(signal.get("volume_ratio"))
    volume_spike = volume_ratio is not None and volume_ratio >= 1.2
    above_mas = bool(signal.get("above_mas"))
    intraday_ok = above_mas and (level_breakout or pullback_rebound) and volume_spike

    rows = [
        {"条件": "VWAP上", "判定": "未使用", "補足": "現行MVPではVWAPを計算していません"},
        {
            "条件": "直近高値突破",
            "判定": _condition_status(level_breakout),
            "補足": "節目突破/維持と直近高値圏を代替条件にしています",
        },
        {
            "条件": "押し目反発",
            "判定": _condition_status(pullback_rebound),
            "補足": "安値切り上げ、短期線回復、再反発型を確認",
        },
        {
            "条件": "出来高急増",
            "判定": _condition_status(volume_spike if volume_ratio is not None else None, "未取得"),
            "補足": f"直近平均比 {volume_ratio:.2f}倍" if volume_ratio is not None else "出来高倍率なし",
        },
    ]
    return _condition_table(rows), bool(intraday_ok)


def _risk_condition_rows(signal: Dict[str, Any]) -> Tuple[pd.DataFrame, bool | None]:
    saved = signal.get("risk_filter") or signal.get("risk_filter_json")
    if not isinstance(saved, dict) or not saved:
        rows = [
            {"条件": "損切り幅", "判定": "未取得", "補足": "リスク条件未計算"},
            {"条件": "最大損失", "判定": "未取得", "補足": "リスク条件未計算"},
            {"条件": "損益比", "判定": "未取得", "補足": "リスク条件未計算"},
        ]
        return _condition_table(rows), None
    rows = [
        {
            "条件": "損切り幅",
            "判定": _condition_status(bool(saved.get("stop_loss_pct_ok"))),
            "補足": f"{_format_plain_pct(saved.get('stop_loss_pct'))} / 上限 {_format_plain_pct(saved.get('max_stop_loss_pct'))}",
        },
        {
            "条件": "最大損失",
            "判定": _condition_status(bool(saved.get("max_loss_yen_ok"))),
            "補足": f"{_format_plain_yen(saved.get('max_loss_yen'))} / 上限 {_format_plain_yen(saved.get('max_loss_yen_limit'))}",
        },
        {
            "条件": "損益比",
            "判定": _condition_status(bool(saved.get("risk_reward_ok"))),
            "補足": f"{_format_profit_loss_ratio(saved.get('risk_reward_ratio'))} / 最低 {_format_profit_loss_ratio(saved.get('min_risk_reward'))}",
        },
    ]
    return _condition_table(rows), bool(saved.get("risk_pass"))


def _combined_judgement_text(daily_ok: bool | None, intraday_ok: bool | None) -> str:
    if daily_ok is None and intraday_ok is None:
        return "判定データが不足しています。"
    if daily_ok is None:
        return "場中エントリー監視は5分足判定です。日足フィルターは通常株価スキャンとは別判定です。"
    if intraday_ok is None:
        return "日足上位N対象外、または5分足データ未取得のため、5分足エントリーは未判定です。"
    if daily_ok and intraday_ok:
        return "日足OK + 5分足OK = マルチ時間足買い候補"
    if daily_ok and not intraday_ok:
        return "日足OK + 5分足NG = 監視"
    return "日足NG = 除外または触らない"


def _render_judgement_breakdown(signal: Dict[str, Any]) -> None:
    st.markdown("**判定内訳**")
    daily_df, daily_ok = _daily_condition_rows(signal)
    intraday_df, intraday_ok = _intraday_condition_rows(signal)
    risk_df, risk_ok = _risk_condition_rows(signal)
    condition = signal.get("buy_condition_json") if isinstance(signal.get("buy_condition_json"), dict) else {}
    intraday_ok_display = signal.get("intraday_ok_display") or f"{signal.get('intraday_ok_count', '-')}/{signal.get('intraday_total_count', '-')}"
    mtf_status = signal.get("multi_timeframe_status") or signal.get("category") or signal.get("judgement", "-")
    metric_cols = st.columns(7)
    metric_cols[0].metric(
        "日足",
        f"{signal.get('daily_ok_count', '-')}/{signal.get('daily_total_count', '-')}",
    )
    metric_cols[1].metric(
        "5分足",
        intraday_ok_display,
    )
    metric_cols[2].metric("日足順位", f"{signal.get('daily_rank_at_scan', '-')}/{signal.get('daily_rank_total', '-')}")
    metric_cols[3].metric("日足スコア", _format_score_value(signal.get("daily_score")))
    metric_cols[4].metric("上位N", _display_risk_pass(signal.get("daily_top_n_pass", condition.get("daily_top_n_pass"))))
    metric_cols[5].metric("最小スコア", _display_risk_pass(condition.get("min_score_pass")))
    metric_cols[6].metric("買い上限", _display_risk_pass(condition.get("max_buy_candidates_pass")))
    st.caption(f"統合判定: {mtf_status}")
    if signal.get("reject_reason"):
        st.caption(f"判定不可理由: {signal.get('reject_reason')}")
    st.caption(_buy_condition_text(signal))
    cols = st.columns(3)
    with cols[0]:
        st.caption("日足フィルター")
        st.dataframe(daily_df, width="stretch", hide_index=True)
    with cols[1]:
        st.caption("5分足エントリー")
        st.dataframe(intraday_df, width="stretch", hide_index=True)
    with cols[2]:
        st.caption("リスク条件")
        st.dataframe(risk_df, width="stretch", hide_index=True)
    st.caption(f"判定: {_combined_judgement_text(daily_ok, intraday_ok)}")
    if risk_ok is False:
        st.warning("リスク条件NG: " + "、".join(str(item) for item in signal.get("risk_reasons", [])))


def _render_logic_confirmation_section() -> None:
    with st.expander("判定ロジック確認", expanded=False):
        st.caption("現在の実装が実際に見ている時間足と条件です。ここではロジックを変更せず、確認用に表示しています。")

        st.markdown("**通常株価スキャン**")
        st.dataframe(
            _condition_table(
                [
                    {"条件": "使用データ", "判定": "日足 + 5分足", "補足": "日足でフィルターし、取得できる場合は当日5分足でエントリー判定"},
                    {"条件": "日足上位N", "判定": "使用", "補足": "日足スコア上位10銘柄、最低70点を5分足判定の母集団にします"},
                    {"条件": "25日線", "判定": "使用", "補足": "終値 > 25日線"},
                    {"条件": "前日終値", "判定": "使用", "補足": "現在の日足終値 > 前日終値"},
                    {"条件": "直近5日高値", "判定": "使用", "補足": "直近5日高値から2%以内"},
                    {"条件": "出来高増加", "判定": "使用", "補足": "日足出来高 >= 5日平均の1.2倍"},
                    {"条件": "VWAP", "判定": "使用", "補足": "当日5分足VWAPを計算"},
                    {"条件": "5分足直近高値突破", "判定": "使用", "補足": "現在足を除く過去12本高値を突破"},
                    {"条件": "押し目反発", "判定": "使用", "補足": "VWAP付近への押し後に再上抜け"},
                    {"条件": "出来高急増", "判定": "使用", "補足": "5分足出来高 >= 過去12本平均の1.5倍"},
                    {"条件": "リスク条件", "判定": "使用", "補足": "損切り幅3%以内、想定損失2万円以内、損益比1.2以上"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown("**場中エントリー監視**")
        st.dataframe(
            _condition_table(
                [
                    {"条件": "使用データ", "判定": "5分足中心", "補足": "yfinance period=5d / interval=5mが基本"},
                    {"条件": "VWAP", "判定": "未使用", "補足": "場中監視タブ単体の既存ロジックではまだ計算していません"},
                    {"条件": "移動平均線", "判定": "使用", "補足": "5本、10本、20本の5分足移動平均"},
                    {"条件": "直近高値突破", "判定": "使用", "補足": "直近12本高値圏、節目突破/維持で判定"},
                    {"条件": "押し目反発", "判定": "使用", "補足": "安値切り上げ、短期線回復、再反発型"},
                    {"条件": "出来高急増", "判定": "使用", "補足": "直近平均出来高比1.2倍以上を加点"},
                    {"条件": "日足フィルター", "判定": "別判定", "補足": "通常株価スキャンとは統合していません"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown("**過去スキャン再現**")
        st.dataframe(
            _condition_table(
                [
                    {"条件": "使用データ", "判定": "日足 + 選択時間足", "補足": "マルチ時間足ON時は6か月日足と選択足を組み合わせます"},
                    {"条件": "買い判定", "判定": "未来データ未使用", "補足": "前日までの確定日足 + scan_time以前の足だけで評価"},
                    {"条件": "結果検証", "判定": "未来データ使用", "補足": "entry後のfuture_dfだけで利確/損切り/期限到達を検証"},
                    {"条件": "日足+5分足統合", "判定": "使用可能", "補足": "初期値は日足3/4以上 + 日足上位N + 5分足2/4以上"},
                    {"条件": "OpenAI API", "判定": "未使用", "補足": "過去検証はルールベースのみです"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.info("通常株価スキャンと過去検証ではマルチ時間足判定を使えます。場中エントリー監視タブ単体は既存の5分足監視ロジックを維持しています。")


def _render_signal_metrics(signal: Dict[str, Any]) -> None:
    cols = st.columns(3)
    cols[0].metric("現在値", format_yen(signal.get("price")))
    cols[1].metric("スコア", f"{signal.get('score', '-')}点")
    cols[2].metric("分類", signal.get("category", "-"))

    cols = st.columns(3)
    cols[0].metric("期待値", signal.get("expected_value_label", "-"))
    cols[1].metric("確度", signal.get("confidence", "-"))
    cols[2].metric("R/R", _format_rr(signal))


def _render_reason_list(title: str, values: Any) -> None:
    values = values or []
    if not values:
        st.markdown(f"**{title}**：-")
        return
    st.markdown(f"**{title}**")
    for value in values:
        st.markdown(f"- {value}")


def _render_detail_content(signal: Dict[str, Any], key_prefix: str, show_chart: bool = False) -> None:
    _render_signal_metrics(signal)
    buy_condition_text = _buy_condition_text(signal)
    st.markdown(f"<div class='signal-line'><b>狙い</b>：{signal.get('entry_type', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>買い条件</b>：{buy_condition_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>待ち条件</b>：{signal.get('wait_condition', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>損切り</b>：{format_yen(signal.get('stop_loss'))}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>利確目安</b>：{_format_targets(signal)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>ハイエナ心理</b>：{signal.get('comment', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>無効条件</b>：{signal.get('invalidation_condition', signal.get('invalid_conditions', '-'))}</div>", unsafe_allow_html=True)

    _render_judgement_breakdown(signal)

    st.markdown("**スコア内訳**")
    st.dataframe(_score_breakdown_rows(signal), width="stretch", hide_index=True)

    reason_cols = st.columns(3)
    with reason_cols[0]:
        _render_reason_list("プラス材料", signal.get("positive_reasons"))
    with reason_cols[1]:
        _render_reason_list("マイナス材料", signal.get("negative_reasons"))
    with reason_cols[2]:
        _render_reason_list("待つ理由", signal.get("wait_reasons"))

    if show_chart:
        chart_key = f"price_chart_{key_prefix}_{signal.get('code')}"
        score_key = f"score_chart_{key_prefix}_{signal.get('code')}"
        _render_plotly_chart(_make_price_chart(signal), key=chart_key)
        _render_plotly_chart(_score_breakdown_chart(signal), key=score_key)

    st.text_area(
        "ChatGPTへ貼り付ける通知文",
        value=signal.get("notification_text", ""),
        height=260,
        key=f"note_{key_prefix}_{signal.get('code')}",
    )

    memo = st.text_input(
        "記録メモ",
        key=f"memo_{key_prefix}_{signal.get('code')}",
        placeholder="例: 前日高値突破なら監視強化",
    )
    if st.button("trades.csvに候補として記録", key=f"record_{key_prefix}_{signal.get('code')}"):
        try:
            append_trade_candidate(signal, memo=memo)
            st.success("trades.csvに記録しました。")
        except Exception as exc:
            st.warning(
                "trades.csvへ記録できませんでした。Streamlit Cloudではファイル保存が一時的で、"
                f"再起動や再デプロイで消える場合があります。詳細: {exc}"
            )


def _expander_title(signal: Dict[str, Any]) -> str:
    return (
        f"{signal.get('code')} {signal.get('name')}｜{signal.get('score', '-')}点"
        f"｜R/R {_format_rr(signal)}｜{signal.get('category', '-')}"
    )


def _render_signal_expanders(
    signals: Iterable[Dict[str, Any]],
    key_prefix: str,
    show_chart: bool = False,
) -> None:
    signals = list(signals)
    if not signals:
        st.info("該当銘柄はありません。")
        return
    for idx, signal in enumerate(signals):
        with st.expander(_expander_title(signal), expanded=False):
            _render_detail_content(signal, key_prefix=f"{key_prefix}_{idx}_{signal.get('code')}", show_chart=show_chart)


def _render_summary_tab(
    buy: List[Dict[str, Any]],
    watch: List[Dict[str, Any]],
    avoid: List[Dict[str, Any]],
    failed: List[Dict[str, Any]],
) -> None:
    st.subheader("サマリー")
    _render_count_metrics(buy, watch, avoid, failed)
    st.info("詳細は買い候補タブを開いて確認してください。")

    st.markdown("**買い候補 上位3件**")
    if not buy:
        st.write("買い候補はありません。")
        return

    lines = []
    for signal in buy[:3]:
        lines.append(
            f"{signal.get('code')} {signal.get('name')}｜{signal.get('score')}点｜R/R {_format_rr(signal)}"
        )
    st.markdown("<div class='summary-list'>" + "<br>".join(lines) + "</div>", unsafe_allow_html=True)


def _render_candidate_tab(title: str, signals: List[Dict[str, Any]], key_prefix: str) -> None:
    st.subheader(title)
    if not signals:
        st.info("該当銘柄はありません。")
        return
    st.dataframe(_compact_table(signals), width="stretch", hide_index=True)
    st.caption("詳細は各銘柄の折りたたみを開いて確認してください。")
    _render_signal_expanders(signals, key_prefix=key_prefix, show_chart=False)


def _render_avoid_tab(avoid: List[Dict[str, Any]]) -> None:
    st.subheader("触らない")
    if not avoid:
        st.info("該当銘柄はありません。")
        return
    columns = ["code", "name", "price", "score", "entry_type", "monitoring_reason", "risk_reward_label"]
    table = build_signal_table(avoid)
    st.dataframe(table[columns], width="stretch", hide_index=True)


def _render_details_tab(
    buy: List[Dict[str, Any]],
    watch: List[Dict[str, Any]],
    avoid: List[Dict[str, Any]],
    failed: List[Dict[str, Any]],
) -> None:
    st.subheader("詳細分析")
    st.caption("チャート込みの詳細確認用です。初期状態では閉じています。")
    _render_signal_expanders(buy, key_prefix="detail_buy", show_chart=True)
    _render_signal_expanders(watch, key_prefix="detail_watch", show_chart=True)
    _render_signal_expanders(avoid, key_prefix="detail_avoid", show_chart=True)
    if failed:
        st.subheader("取得失敗")
        _render_failure_debug(failed, "取得失敗の理由")
        with st.expander("取得失敗の詳細一覧", expanded=False):
            st.dataframe(build_signal_table(failed), width="stretch", hide_index=True)


def _render_intraday_signal(signal: Dict[str, Any], key_prefix: str) -> None:
    if signal.get("judgement") == "取得失敗":
        st.warning(signal.get("error_message") or signal.get("error") or "場中データ取得に失敗しました。")
        st.markdown("**5分足デバッグ情報**")
        st.dataframe(_intraday_fetch_debug_table([signal]), width="stretch", hide_index=True)
        recent_bars = _intraday_recent_bar_table(signal)
        if not recent_bars.empty:
            st.markdown("**直近4本の5分足判定**")
            st.dataframe(_safe_dataframe(recent_bars), width="stretch", hide_index=True)
        return

    cols = st.columns(3)
    cols[0].metric("現在値", format_yen(signal.get("current_price")))
    cols[1].metric("場中スコア", f"{signal.get('intraday_score', 0)}点")
    cols[2].metric("判定", signal.get("judgement", "-"))

    cols = st.columns(3)
    cols[0].metric("買い候補", signal.get("buy_zone", "-"))
    cols[1].metric("損切り", format_yen(signal.get("stop_loss")))
    cols[2].metric("利確", f"{format_yen(signal.get('take_profit_1'))} / {format_yen(signal.get('take_profit_2'))}")

    st.markdown(f"**狙い**：{signal.get('signal_type', '-')}")
    st.markdown(f"**最終足**：{signal.get('last_time', '-')}")
    st.markdown(
        f"**節目**：{format_yen(signal.get('level_price'))} / "
        f"突破: {signal.get('level_breakout', False)} / 維持: {signal.get('level_hold', False)}"
    )
    st.markdown(
        f"**5分足状態**：短期 {format_yen(signal.get('ma_short'))} / "
        f"中期 {format_yen(signal.get('ma_mid'))} / 長期 {format_yen(signal.get('ma_long'))}"
    )
    st.markdown(
        f"**出来高**：{int(_to_float(signal.get('current_volume'), 0) or 0):,} / "
        f"平均 {int(_to_float(signal.get('volume_avg'), 0) or 0):,} / 倍率 {signal.get('volume_ratio', '-')}"
    )
    st.markdown(f"**VWAP**：{format_yen(signal.get('vwap'))} / 終値>VWAP: {signal.get('close_above_vwap', '-')}")
    st.markdown(
        f"**当日レンジ**：高値 {format_yen(signal.get('day_high'))} / "
        f"安値 {format_yen(signal.get('day_low'))} / 安値から {signal.get('rebound_from_day_low_pct', 0)}%"
    )
    st.markdown(f"**RSI**：{signal.get('rsi', '-')}")

    _render_judgement_breakdown(signal)

    st.markdown("**5分足デバッグ情報**")
    st.dataframe(_intraday_fetch_debug_table([signal]), width="stretch", hide_index=True)
    st.markdown("**直近4本の5分足判定**")
    st.dataframe(_safe_dataframe(_intraday_recent_bar_table(signal)), width="stretch", hide_index=True)

    _render_reason_list("判定理由", signal.get("reasons"))
    _render_reason_list("見送り・警戒理由", signal.get("risk_notes"))
    _render_reason_list("買ってはいけない条件", signal.get("no_buy_conditions"))

    _render_plotly_chart(_make_intraday_chart(signal), key=f"intraday_chart_{key_prefix}_{signal.get('code')}")
    st.text_area(
        "通知文",
        value=signal.get("notification_text", ""),
        height=220,
        key=f"intraday_note_{key_prefix}_{signal.get('code')}",
    )


def _render_intraday_section(title: str, signals: List[Dict[str, Any]], key_prefix: str) -> None:
    st.subheader(f"{title}（{len(signals)}件）")
    if not signals:
        st.info("該当銘柄はありません。")
        return
    for idx, signal in enumerate(signals):
        label = (
            f"{signal.get('code')} {signal.get('name')}｜{signal.get('intraday_score', 0)}点"
            f"｜{signal.get('judgement', '-')}｜{signal.get('signal_type', '-')}"
        )
        with st.expander(label, expanded=False):
            _render_intraday_signal(signal, key_prefix=f"{key_prefix}_{idx}")


def _process_intraday_notifications(signals: List[Dict[str, Any]], discord_enabled: bool) -> None:
    if "intraday_notified_keys" not in st.session_state:
        st.session_state.intraday_notified_keys = []
    if "discord_last_notified_by_symbol" not in st.session_state:
        st.session_state.discord_last_notified_by_symbol = {}
    if "discord_initial_suppressed_at_by_symbol" not in st.session_state:
        st.session_state.discord_initial_suppressed_at_by_symbol = {}

    now = _now_jst()
    notified = set(st.session_state.intraday_notified_keys)
    missing_webhook_warned = False
    market_time_warned = False
    webhook_configured = bool(get_setting("DISCORD_WEBHOOK_URL", "").strip())
    is_market_time = _is_jpx_market_time(now)
    current_candidate_symbols = {
        _notification_symbol(signal)
        for signal in signals
        if _is_discord_buy_candidate(signal) and _notification_symbol(signal)
    }
    previous_candidate_symbols_raw = st.session_state.get("discord_previous_buy_candidate_symbols")
    first_candidate_scan = previous_candidate_symbols_raw is None
    previous_candidate_symbols = set(previous_candidate_symbols_raw or [])
    last_notified_by_symbol = dict(st.session_state.discord_last_notified_by_symbol)
    initial_suppressed_by_symbol = dict(st.session_state.discord_initial_suppressed_at_by_symbol)

    for signal in signals:
        if signal.get("judgement") not in {"買い検討OK", "監視強化"}:
            continue

        key = alert_key(signal)
        is_new_screen_alert = key not in notified
        discord_status = "disabled" if not discord_enabled else ""
        discord_error = ""
        discord_attempted = False

        if is_new_screen_alert:
            message = (
                f"{signal.get('code')} {signal.get('name')} "
                f"{signal.get('judgement')} / {signal.get('intraday_score')}点"
            )
            if signal.get("judgement") == "買い検討OK":
                st.toast(message)
                st.success(message)
            else:
                st.toast(message)
                st.warning(message)
            notified.add(key)

        if discord_enabled and _is_discord_buy_candidate(signal):
            symbol = _notification_symbol(signal)
            if not symbol:
                discord_status = "skipped_no_symbol"
            elif first_candidate_scan:
                discord_status = "initial_suppressed"
                initial_suppressed_by_symbol.setdefault(symbol, now.isoformat())
            elif not webhook_configured:
                discord_status = "skipped_no_webhook"
                st.session_state.discord_last_status = "スキップ"
                st.session_state.discord_last_error = "Webhook設定がないため通知をスキップしました。"
                if not missing_webhook_warned:
                    st.warning("Webhook設定がないためDiscord通知をスキップしました。")
                    missing_webhook_warned = True
            elif not is_market_time:
                discord_status = "skipped_market_closed"
                st.session_state.discord_last_status = "市場時間外"
                st.session_state.discord_last_error = ""
                if not market_time_warned:
                    st.caption("市場時間外のため、本番Discord通知はスキップしました。テスト通知は送信できます。")
                    market_time_warned = True
            else:
                is_new_candidate = symbol not in previous_candidate_symbols
                last_sent_at = _parse_state_datetime(last_notified_by_symbol.get(symbol))
                initial_suppressed_at = _parse_state_datetime(initial_suppressed_by_symbol.get(symbol))
                cooldown_ok = last_sent_at is not None and _notification_cooldown_elapsed(last_sent_at, now)
                initial_suppressed_elapsed = (
                    last_sent_at is None
                    and initial_suppressed_at is not None
                    and _notification_cooldown_elapsed(initial_suppressed_at, now)
                )
                should_send_discord = is_new_candidate or cooldown_ok or initial_suppressed_elapsed
                if not should_send_discord:
                    discord_status = "skipped_cooldown"
                else:
                    discord_attempted = True
                    ok, error = send_discord_webhook(build_buy_candidate_discord_text(signal))
                    st.session_state.discord_last_notified_at = now.strftime("%Y-%m-%d %H:%M:%S")
                    if ok:
                        discord_status = "success"
                        last_notified_by_symbol[symbol] = now.isoformat()
                        st.session_state.discord_last_status = "成功"
                        st.session_state.discord_last_error = ""
                        st.success(f"Discordへ買い候補を通知しました: {signal.get('code')} {signal.get('name')}")
                    else:
                        discord_status = "failed"
                        discord_error = error
                        st.session_state.discord_last_status = "失敗"
                        st.session_state.discord_last_error = error
                        st.warning(error)

        if is_new_screen_alert or discord_attempted:
            try:
                append_alert_log(signal, discord_sent=discord_status, discord_error=discord_error)
            except Exception as exc:
                st.caption(f"alerts_log.csvへ保存できませんでした: {exc}")

    st.session_state.intraday_notified_keys = sorted(notified)
    st.session_state.discord_previous_buy_candidate_symbols = sorted(current_candidate_symbols)
    st.session_state.discord_last_notified_by_symbol = last_notified_by_symbol
    st.session_state.discord_initial_suppressed_at_by_symbol = initial_suppressed_by_symbol


def _render_alerts_log() -> None:
    with st.expander("通知履歴 alerts_log.csv", expanded=False):
        if not ALERTS_LOG_PATH.exists():
            st.info("通知履歴はまだありません。")
            return
        try:
            log_df = pd.read_csv(ALERTS_LOG_PATH, dtype={"code": str}).fillna("")
        except Exception as exc:
            st.warning(f"通知履歴を読み込めませんでした: {exc}")
            return
        if log_df.empty:
            st.info("通知履歴はまだありません。")
        else:
            st.dataframe(_safe_dataframe(log_df.tail(100).iloc[::-1]), width="stretch", hide_index=True)


def _render_fetch_test() -> None:
    with st.expander("取得テスト", expanded=False):
        st.caption("まず主要3銘柄で、yfinanceの5分足がこの環境から取れるか確認できます。")
        test_cols = st.columns(2)
        run_three_test = test_cols[0].button("主要3銘柄の5分足単体取得テスト", key="fetch_three_symbols_test_button")
        three_period = test_cols[1].selectbox("単体取得テストperiod", ["1d", "5d"], index=0, key="fetch_three_symbols_period")
        if run_three_test:
            with st.spinner("6501.T / 5803.T / 7011.T の5分足を直接取得中です..."):
                rows = run_raw_intraday_fetch_test(["6501.T", "5803.T", "7011.T"], period=three_period, interval="5m")
            st.dataframe(_safe_dataframe(pd.DataFrame(rows)), width="stretch", hide_index=True)

        test_code = st.text_input("取得テストコード", value="5803", key="fetch_test_code")
        if not st.button("取得テスト", key="fetch_test_button"):
            return

        symbol = normalize_jp_symbol(test_code)
        with st.spinner("yfinance取得テスト中です..."):
            daily = fetch_price_data(test_code, period="6mo", interval="1d")
            intraday = fetch_intraday_data(test_code, interval="5m", period="5d")

        latest_close = "-"
        if not intraday.data.empty:
            latest_close = format_yen(intraday.data["Close"].iloc[-1])
        elif not daily.data.empty:
            latest_close = format_yen(daily.data["Close"].iloc[-1])

        error_messages = []
        if daily.error:
            error_messages.append(f"日足: {daily.error_message or daily.error}")
        if intraday.error:
            error_messages.append(f"5分足: {intraday.error_message or intraday.error}")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "入力コード": test_code,
                        "yfinance用シンボル": symbol,
                        "日足取得行数": daily.fetched_rows,
                        "5分足取得行数": intraday.fetched_rows,
                        "最新Close": latest_close,
                        "5分足error_type": intraday.error_type,
                        "5分足latest_5m_jst": intraday.data.index[-1].strftime("%Y-%m-%d %H:%M") if not intraday.data.empty else "-",
                        "5分足latest_volume": int(intraday.data["Volume"].iloc[-1]) if not intraday.data.empty else "-",
                        "エラー": " / ".join(error_messages) if error_messages else "-",
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def _run_intraday_scan(
    targets: List[Dict[str, Any]],
    interval: str,
    discord_enabled: bool,
    source_label: str,
) -> None:
    if not targets:
        st.session_state.intraday_results = []
        st.session_state.intraday_last_scan_status = "監視対象なし"
        st.session_state.intraday_last_scan_summary = {
            "fetch_success_count": 0,
            "fetch_failed_count": 0,
            "rows_zero_count": 0,
            "intraday_ok_positive_count": 0,
            "latest_5m_jst_max": "-",
            "scan_elapsed_sec": 0,
            "cache_used": "なし",
            "preflight_ok": False,
        }
        st.warning("監視対象がありません。watchlist.csvまたは手動入力コードを確認してください。")
        return

    started = time.perf_counter()
    with st.spinner(f"{source_label}で5分足データを取得して場中エントリー条件を判定中です..."):
        preflight_rows = run_raw_intraday_fetch_test(["6501.T", "5803.T", "7011.T"], period="1d", interval=interval)
        st.session_state.intraday_preflight_rows = preflight_rows
        preflight_success = [row for row in preflight_rows if int(row.get("rows", 0) or 0) > 0 and not row.get("exception_message")]
        if not preflight_success:
            elapsed = round(time.perf_counter() - started, 2)
            st.session_state.intraday_results = []
            checked_at = _format_jst(_now_jst())
            st.session_state.intraday_last_updated = checked_at
            st.session_state.intraday_last_checked_at = checked_at
            st.session_state.intraday_last_scan_status = "主要3銘柄の5分足取得に失敗したため停止"
            st.session_state.intraday_last_scan_summary = {
                "fetch_success_count": 0,
                "fetch_failed_count": len(targets),
                "rows_zero_count": len(targets),
                "intraday_ok_positive_count": 0,
                "latest_5m_jst_max": "-",
                "scan_elapsed_sec": elapsed,
                "cache_used": "なし",
                "preflight_ok": False,
            }
            st.error("6501.T / 5803.T / 7011.T の5分足取得に失敗しました。yfinance接続、interval指定、ネットワークを確認してください。")
            st.dataframe(_safe_dataframe(pd.DataFrame(preflight_rows)), width="stretch", hide_index=True)
            return

        results = scan_intraday_entries(targets, interval=interval, period="5d")
        elapsed = round(time.perf_counter() - started, 2)
        success_count = sum(1 for signal in results if signal.get("intraday_fetch_ok"))
        failed_count = sum(1 for signal in results if not signal.get("intraday_fetch_ok"))
        rows_zero_count = sum(1 for signal in results if int(signal.get("intraday_rows", signal.get("fetched_rows", 0)) or 0) == 0)
        intraday_positive_count = sum(1 for signal in results if int(signal.get("intraday_ok_count", signal.get("intraday_score", 0)) or 0) > 0)
        latest_times = [str(signal.get("latest_5m_jst") or signal.get("last_time") or "") for signal in results]
        latest_times = [value for value in latest_times if value and value != "-"]
        cache_used = "あり" if any(signal.get("cache_hit") for signal in results) else "なし"
        st.session_state.intraday_results = results
        checked_at = _format_jst(_now_jst())
        st.session_state.intraday_last_updated = checked_at
        st.session_state.intraday_last_checked_at = checked_at
        st.session_state.intraday_last_scan_status = f"{len(targets)}銘柄チェック完了"
        st.session_state.intraday_last_scan_summary = {
            "fetch_success_count": success_count,
            "fetch_failed_count": failed_count,
            "rows_zero_count": rows_zero_count,
            "intraday_ok_positive_count": intraday_positive_count,
            "latest_5m_jst_max": max(latest_times) if latest_times else "-",
            "scan_elapsed_sec": elapsed,
            "cache_used": cache_used,
            "preflight_ok": True,
        }
    _process_intraday_notifications(st.session_state.intraday_results, discord_enabled=discord_enabled)


def _render_intraday_tab(
    watchlist: pd.DataFrame,
    buy: List[Dict[str, Any]],
    watch: List[Dict[str, Any]],
) -> None:
    st.subheader("場中エントリー監視")
    st.caption("5分足で、場中に買える形になった銘柄を検知します。自動売買は行いません。")

    col_a, col_b = st.columns(2)
    with col_a:
        interval = st.selectbox("足種", ["5m", "1m"], index=0)
        include_watchlist = st.checkbox("watchlist.csv全銘柄を対象", value=True)
    with col_b:
        include_swing_focus = st.checkbox("買い候補・監視銘柄を対象に含める", value=True)
        auto_refresh = st.toggle("自動監視ON/OFF", value=False)

    refresh_seconds = st.selectbox("監視間隔", [60, 120, 300], index=0, format_func=lambda value: f"{value}秒")
    manual_codes_text = st.text_area("手動入力コード（任意、カンマ・空白・改行区切り）", placeholder="例: 5803, 3774, 6501")
    manual_codes = _parse_manual_codes(manual_codes_text)

    _render_fetch_test()

    if "discord_last_notified_at" not in st.session_state:
        st.session_state.discord_last_notified_at = "-"
    if "discord_last_status" not in st.session_state:
        st.session_state.discord_last_status = "未送信"
    if "discord_last_error" not in st.session_state:
        st.session_state.discord_last_error = ""

    st.markdown("**Discord通知**")
    webhook_configured = bool(get_setting("DISCORD_WEBHOOK_URL", "").strip())
    mention_configured = bool(get_setting("DISCORD_MENTION_ID", "").strip())
    st.caption(f"Webhook設定：{'あり' if webhook_configured else 'なし'}")
    st.caption(f"メンション設定：{'あり' if mention_configured else 'なし'}")
    st.caption("本番通知対象：市場時間内の買い検討OK・70点以上のみ。同一銘柄は30分クールダウン。")
    discord_cols = st.columns(2)
    with discord_cols[0]:
        discord_enabled = st.checkbox("Discord通知ON/OFF", value=webhook_configured)
    with discord_cols[1]:
        test_discord_clicked = st.button("テスト通知")

    if test_discord_clicked:
        test_message = "\n".join(
            [
                "【テスト通知】",
                "stock_swing_tool からDiscord通知テストです。",
                "このメッセージが届けばWebhook設定OKです。",
            ]
        )
        ok, error = send_discord_webhook(test_message)
        st.session_state.discord_last_notified_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ok:
            st.session_state.discord_last_status = "成功"
            st.session_state.discord_last_error = ""
            st.success("Discordテスト通知を送信しました。")
        else:
            st.session_state.discord_last_status = "失敗"
            st.session_state.discord_last_error = error
            st.warning(error)

    status_text = f"通知状態: {st.session_state.discord_last_status}"
    if st.session_state.discord_last_error:
        status_text += f"（{st.session_state.discord_last_error}）"
    st.caption(f"最終Discord通知時刻: {st.session_state.discord_last_notified_at}")
    st.caption(status_text)

    targets = _build_intraday_targets(
        watchlist=watchlist,
        buy=buy,
        watch=watch,
        manual_codes=manual_codes,
        include_watchlist=include_watchlist,
        include_swing_focus=include_swing_focus,
    )
    st.caption(f"今回の監視対象: {len(targets)}銘柄")

    update_clicked = st.button("場中データを更新", type="primary")
    should_scan = update_clicked

    if "intraday_results" not in st.session_state:
        st.session_state.intraday_results = []
    if "intraday_last_updated" not in st.session_state:
        st.session_state.intraday_last_updated = "-"
    if "intraday_last_checked_at" not in st.session_state:
        st.session_state.intraday_last_checked_at = "-"
    if "intraday_last_scan_status" not in st.session_state:
        st.session_state.intraday_last_scan_status = "未チェック"
    if "intraday_last_scan_summary" not in st.session_state:
        st.session_state.intraday_last_scan_summary = {}

    status_cols = st.columns(5)
    status_cols[0].metric("自動監視", "ON" if auto_refresh else "OFF")
    status_cols[1].metric("監視間隔", f"{refresh_seconds}秒")
    status_cols[2].metric("最終チェック", st.session_state.intraday_last_checked_at)
    status_cols[3].metric("次回チェック目安", _next_check_text(auto_refresh, int(refresh_seconds)))
    status_cols[4].metric("Discord通知", "ON" if discord_enabled else "OFF")
    st.caption(
        "PCスリープ中、Streamlit停止中、またはブラウザを閉じている場合は自動監視できません。"
        "ブラウザでこのアプリを開いている間だけ動作します。"
    )
    st.caption(f"監視状態：{st.session_state.intraday_last_scan_status}")
    summary = st.session_state.get("intraday_last_scan_summary", {})
    summary_cols = st.columns(7)
    summary_cols[0].metric("5分足取得成功", summary.get("fetch_success_count", 0))
    summary_cols[1].metric("5分足取得失敗", summary.get("fetch_failed_count", 0))
    summary_cols[2].metric("rows=0", summary.get("rows_zero_count", 0))
    summary_cols[3].metric("5分足OK>=1", summary.get("intraday_ok_positive_count", 0))
    summary_cols[4].metric("最新5分足", summary.get("latest_5m_jst_max", "-"))
    summary_cols[5].metric("scan elapsed", f"{summary.get('scan_elapsed_sec', 0)}秒")
    summary_cols[6].metric("cache", summary.get("cache_used", "なし"))
    if st.session_state.get("intraday_preflight_rows"):
        with st.expander("主要3銘柄の事前取得テスト結果", expanded=False):
            st.dataframe(_safe_dataframe(pd.DataFrame(st.session_state.get("intraday_preflight_rows", []))), width="stretch", hide_index=True)

    if should_scan:
        _run_intraday_scan(targets, interval=interval, discord_enabled=discord_enabled, source_label="手動更新")

    if auto_refresh:
        @st.fragment(run_every=timedelta(seconds=int(refresh_seconds)))
        def _auto_intraday_scan_fragment() -> None:
            if _recently_checked():
                return
            _run_intraday_scan(targets, interval=interval, discord_enabled=discord_enabled, source_label="自動監視")
            st.rerun()

        _auto_intraday_scan_fragment()

    st.markdown(f"**最終更新**：{st.session_state.intraday_last_updated}")
    results = list(st.session_state.intraday_results)
    ok = [s for s in results if s.get("judgement") == "買い検討OK"]
    strong_watch = [s for s in results if s.get("judgement") == "監視強化"]
    skip = [s for s in results if s.get("judgement") == "見送り"]
    failed = [s for s in results if s.get("judgement") == "取得失敗"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("買い検討OK", len(ok))
    metric_cols[1].metric("監視強化", len(strong_watch))
    metric_cols[2].metric("見送り", len(skip))
    metric_cols[3].metric("取得失敗", len(failed))

    if results and len(failed) == len(results):
        _render_all_failed_warning(failed)

    if results:
        with st.expander("5分足取得デバッグ一覧", expanded=False):
            st.dataframe(_safe_dataframe(_intraday_fetch_debug_table(results)), width="stretch", hide_index=True)

    _render_intraday_section("買い検討OK", ok, "intraday_ok")
    _render_intraday_section("監視強化", strong_watch, "intraday_watch")
    _render_intraday_section("見送り", skip, "intraday_skip")
    if failed:
        _render_intraday_section("取得失敗・データ不足", failed, "intraday_failed")

    with st.expander("PC向け一覧表", expanded=False):
        if results:
            st.dataframe(_safe_dataframe(_intraday_table(results)), width="stretch", hide_index=True)
        else:
            st.info("更新ボタンを押すと一覧を表示します。")

    _render_alerts_log()


def _safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.astype(object).where(pd.notna(df), "").astype(str)


def _ai_candidate_table(signals: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        rows.append(
            {
                "code": signal.get("code", ""),
                "name": signal.get("name", ""),
                "price": format_yen(signal.get("price", signal.get("current_price"))),
                "score": signal.get("score", signal.get("intraday_score", 0)),
                "category": signal.get("category", signal.get("judgement", "")),
                "entry_type": signal.get("entry_type", signal.get("signal_type", "")),
            }
        )
    return pd.DataFrame(rows)


def _ai_virtual_json_safe_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
        return None
    if isinstance(value, dict):
        return {
            str(key): _ai_virtual_json_safe_value(item)
            for key, item in value.items()
            if not isinstance(item, (pd.DataFrame, pd.Series))
        }
    if isinstance(value, (list, tuple)):
        return [
            _ai_virtual_json_safe_value(item)
            for item in value
            if not isinstance(item, (pd.DataFrame, pd.Series))
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _snapshot_ai_virtual_candidates(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    snapshots = []
    for signal in signals:
        compact = {
            str(key): _ai_virtual_json_safe_value(value)
            for key, value in signal.items()
            if not isinstance(value, (pd.DataFrame, pd.Series))
        }
        snapshots.append(compact)
    return snapshots


def _render_virtual_trade_result(result: Dict[str, Any], idx: int) -> None:
    record = result.get("record", {})
    decision = result.get("decision", {})
    title = (
        f"{record.get('symbol', record.get('code', '-'))} {record.get('name', '')}"
        f"｜{decision.get('decision', '-')}"
        f"｜{decision.get('entry_type', '-')}"
        f"｜{'保存' if result.get('saved') else result.get('reason', '-')}"
    )
    with st.expander(title, expanded=False):
        cols = st.columns(4)
        cols[0].metric("仮想判断", decision.get("decision", "-"))
        cols[1].metric("確度", decision.get("confidence", "-"))
        cols[2].metric("仮想買値", format_yen(decision.get("entry_price")))
        cols[3].metric("最大保有", f"{decision.get('max_hold_days', '-')}日")
        st.markdown(
            f"**判断ソース**：{record.get('judge_source', '-')} / "
            f"**model**：{record.get('model_used', '-')} / "
            f"**AI生成**：{record.get('is_ai_generated', '-')}"
        )
        if record.get("fallback_reason") or record.get("fallback_error_type"):
            st.markdown(
                f"**fallback**：{record.get('fallback_reason', '-')} / "
                f"{record.get('fallback_error_type', '-')}"
            )
            if record.get("fallback_error_message"):
                st.caption(record.get("fallback_error_message"))
        st.markdown(f"**JST時刻**：{record.get('timestamp_jst', record.get('timestamp', '-'))}")
        st.markdown(f"**損切り**：{format_yen(decision.get('stop_loss'))}")
        st.markdown(f"**利確**：{format_yen(decision.get('take_profit'))}")
        _render_reason_list("理由", decision.get("reasons"))
        _render_reason_list("リスク", decision.get("risk_factors"))
        st.caption(f"保存結果: {result.get('reason', '-')}")


def _virtual_result_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        decision = result.get("decision") or {}
        record = result.get("record") or {}
        rows.append(
            {
                "saved": result.get("saved", False),
                "reason": result.get("reason", ""),
                "error_message": result.get("error_message", ""),
                "trade_id": result.get("trade_id", ""),
                "duplicate_id": result.get("duplicate_id", ""),
                "db_path": result.get("db_path", str(VIRTUAL_TRADES_DB_PATH)),
                "timestamp_jst": record.get("timestamp_jst", ""),
                "symbol": record.get("symbol") or record.get("code") or "",
                "name": record.get("name", ""),
                "decision": decision.get("decision", record.get("decision", "")),
                "entry_type": decision.get("entry_type", record.get("entry_type", "")),
                "judge_source": record.get("judge_source", decision.get("judge_source", "")),
                "model_used": record.get("model_used", decision.get("model_used", "")),
                "is_ai_generated": record.get("is_ai_generated", decision.get("is_ai_generated", "")),
                "fallback_reason": record.get("fallback_reason", decision.get("fallback_reason", "")),
                "fallback_error_type": record.get("fallback_error_type", decision.get("fallback_error_type", "")),
                "fallback_error_message": record.get("fallback_error_message", decision.get("fallback_error_message", "")),
                "source_score": record.get("source_score", ""),
            }
        )
    return pd.DataFrame(rows)


def _format_trade_datetime(value: Any) -> str:
    parsed = parse_trade_datetime_to_jst_naive(value)
    if parsed is None:
        return str(value or "-")
    return parsed.strftime("%Y-%m-%d %H:%M")


def _format_score_value(value: Any) -> str:
    try:
        if value in (None, "") or pd.isna(value):
            return "-"
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "-"


def _format_pct_value(value: Any) -> str:
    try:
        if value in (None, "") or pd.isna(value):
            return "-"
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:+.1f}%"


def _format_signed_yen(value: Any) -> str:
    try:
        if value in (None, "") or pd.isna(value):
            return "-"
        return f"{float(value):+,.0f}円"
    except (TypeError, ValueError):
        return "-"


def _format_plain_yen(value: Any) -> str:
    return format_yen(value)


def _format_plain_pct(value: Any) -> str:
    try:
        if value in (None, "") or pd.isna(value):
            return "-"
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "-"


def _format_elapsed_days(value: Any) -> str:
    parsed = parse_trade_datetime_to_jst_naive(value)
    if parsed is None:
        return "-"
    elapsed = max(0, (market_now_jst().replace(tzinfo=None) - parsed).days)
    return f"{elapsed}日"


def _display_decision(value: Any) -> str:
    return {
        "virtual_buy": "仮想買い",
        "virtual_watch": "仮想監視",
        "virtual_avoid": "見送り",
    }.get(str(value or ""), str(value or "-"))


def _display_status(value: Any) -> str:
    return {
        "open": "保有中",
        "closed": "終了",
        "logged": "記録のみ",
    }.get(str(value or ""), str(value or "-"))


def _display_outcome(value: Any) -> str:
    return {
        "tracking": "検証中",
        "stop_loss": "損切り",
        "take_profit": "利確",
        "time_exit_5d": "5日経過",
    }.get(str(value or ""), str(value or "-") if value not in (None, "") else "-")


def _display_judge_method(judge_source: Any, model_used: Any = "") -> str:
    source = str(judge_source or "").lower()
    model = str(model_used or "").lower()
    if source == "gpt" or (model and model != "rule_based_fallback" and "fallback" not in model):
        return "GPT"
    if source == "fallback" or model == "rule_based_fallback":
        return "ルールベース"
    if source == "test":
        return "テスト"
    return "-"


def _display_target_mode(value: Any) -> str:
    return "買い候補＋監視" if str(value) == "buy_watch" else "買い候補のみ"


def _display_bool(value: Any) -> str:
    return "ON" if bool(value) else "OFF"


def _virtual_trade_display_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "仮想買い時刻",
                "銘柄",
                "判定",
                "型",
                "買値",
                "損切り",
                "利確目標",
                "スコア",
                "状態",
                "現在結果",
                "最大利益率",
                "最大下落率",
                "判定方式",
            ]
        )
    rows = []
    for _, row in df.iterrows():
        symbol = str(row.get("symbol", "") or "")
        name = str(row.get("name", "") or "")
        rows.append(
            {
                "仮想買い時刻": _format_trade_datetime(row.get("timestamp_jst") or row.get("timestamp")),
                "銘柄": f"{symbol} {name}".strip() or "-",
                "判定": _display_decision(row.get("decision")),
                "型": row.get("entry_type") or "-",
                "買値": format_yen(row.get("entry_price")),
                "損切り": format_yen(row.get("stop_loss")),
                "利確目標": format_yen(row.get("take_profit")),
                "スコア": _format_score_value(row.get("source_score")),
                "状態": _display_status(row.get("status")),
                "現在結果": _display_outcome(row.get("outcome")),
                "最大利益率": _format_pct_value(row.get("max_profit_pct")),
                "最大下落率": _format_pct_value(row.get("max_drawdown_pct")),
                "判定方式": _display_judge_method(row.get("judge_source"), row.get("model_used")),
            }
        )
    return pd.DataFrame(rows)


def _operational_virtual_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    if "judge_source" in filtered.columns:
        filtered = filtered[filtered["judge_source"].astype(str) != "test"]
    if "symbol" in filtered.columns:
        filtered = filtered[filtered["symbol"].astype(str) != "DB_TEST"]
    if "entry_type" in filtered.columns:
        filtered = filtered[filtered["entry_type"].astype(str) != "db_connection_test"]
    return filtered


def _open_virtual_trade_display_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["仮想買い時刻", "銘柄", "型", "買値", "損切り", "利確目標", "スコア", "経過日数", "現在結果"])
    rows = []
    for _, row in df.iterrows():
        symbol = str(row.get("symbol", "") or "")
        name = str(row.get("name", "") or "")
        rows.append(
            {
                "仮想買い時刻": _format_trade_datetime(row.get("timestamp_jst") or row.get("timestamp")),
                "銘柄": f"{symbol} {name}".strip() or "-",
                "型": row.get("entry_type") or "-",
                "買値": format_yen(row.get("entry_price")),
                "損切り": format_yen(row.get("stop_loss")),
                "利確目標": format_yen(row.get("take_profit")),
                "スコア": _format_score_value(row.get("source_score")),
                "経過日数": _format_elapsed_days(row.get("timestamp_jst") or row.get("timestamp")),
                "現在結果": _display_outcome(row.get("outcome")),
            }
        )
    return pd.DataFrame(rows)


def _closed_virtual_trade_display_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["仮想買い時刻", "銘柄", "型", "買値", "結果", "リターン", "最大利益率", "最大下落率", "終了理由"])
    rows = []
    for _, row in df.iterrows():
        symbol = str(row.get("symbol", "") or "")
        name = str(row.get("name", "") or "")
        rows.append(
            {
                "仮想買い時刻": _format_trade_datetime(row.get("timestamp_jst") or row.get("timestamp")),
                "銘柄": f"{symbol} {name}".strip() or "-",
                "型": row.get("entry_type") or "-",
                "買値": format_yen(row.get("entry_price")),
                "結果": _display_outcome(row.get("outcome")),
                "リターン": _format_pct_value(row.get("return_pct")),
                "最大利益率": _format_pct_value(row.get("max_profit_pct")),
                "最大下落率": _format_pct_value(row.get("max_drawdown_pct")),
                "終了理由": _display_outcome(row.get("outcome")),
            }
        )
    return pd.DataFrame(rows)


def _filter_virtual_trades(df: pd.DataFrame, limit: int, status_label: str, query: str, entry_type: str, judge_method: str) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    if status_label == "保有中":
        filtered = filtered[filtered["status"].astype(str) == "open"]
    elif status_label == "終了":
        filtered = filtered[filtered["status"].astype(str) == "closed"]

    if query.strip():
        needle = query.strip().lower()
        haystack = (
            filtered.get("symbol", pd.Series("", index=filtered.index)).astype(str)
            + " "
            + filtered.get("name", pd.Series("", index=filtered.index)).astype(str)
        ).str.lower()
        filtered = filtered[haystack.str.contains(re.escape(needle), na=False)]

    if entry_type != "すべて" and "entry_type" in filtered.columns:
        filtered = filtered[filtered["entry_type"].astype(str) == entry_type]

    if judge_method != "すべて":
        methods = filtered.apply(lambda row: _display_judge_method(row.get("judge_source"), row.get("model_used")), axis=1)
        filtered = filtered[methods == judge_method]

    return filtered.head(int(limit))


def _render_last_ai_virtual_run() -> None:
    run_state = st.session_state.get("last_ai_virtual_run")
    results = st.session_state.get("last_ai_virtual_results", [])
    if not run_state:
        return

    st.markdown("**直近のフォワード検証ログ保存結果**")
    event_label = run_state.get("event_label", "フォワード検証ログ保存を検知しました")
    st.info(f"{event_label}（{run_state.get('timestamp', '-')}）")
    st.write(f"候補数: {run_state.get('candidate_count', 0)}")
    st.write(f"DBパス: {run_state.get('db_path', VIRTUAL_TRADES_DB_PATH)}")

    if run_state.get("error"):
        st.error(run_state.get("error"))

    saved_count = int(run_state.get("saved_count", 0))
    failed_count = int(run_state.get("failed_count", 0))
    st.success(f"保存成功: {saved_count}件 / 失敗: {failed_count}件")

    if results:
        result_table = _virtual_result_table(results)
        failed = result_table[result_table["saved"].astype(str) != "True"]
        if not failed.empty:
            st.warning("保存できなかった候補があります。reasonを確認してください。")
            st.dataframe(
                _safe_dataframe(failed[["symbol", "name", "reason", "error_message", "duplicate_id"]]),
                width="stretch",
                hide_index=True,
            )
        st.dataframe(_safe_dataframe(result_table), width="stretch", hide_index=True)
        with st.expander("process_virtual_trade_signals の戻り値", expanded=False):
            st.json(results)
    else:
        st.warning("process_virtual_trade_signals の戻り値は空です。")

    recent_records = st.session_state.get("last_ai_virtual_recent", [])
    st.markdown("**保存後に再取得した最近のフォワード検証ログ**")
    if recent_records:
        st.dataframe(_safe_dataframe(pd.DataFrame(recent_records)), width="stretch", hide_index=True)
    else:
        st.info("保存後に再取得した最近ログは空です。")


def _mask_sensitive_openai_error(message: Any, *secrets: str) -> str:
    masked = str(message or "")
    for secret in secrets:
        if secret:
            masked = masked.replace(secret, "[masked]")
    return re.sub(r"sk-[A-Za-z0-9_\-\*]{4,}", "[masked]", masked)


def _current_ai_virtual_model() -> str:
    return get_setting("AI_VIRTUAL_MODEL", AI_VIRTUAL_MODEL)


def _initialize_openai_api_enabled_state() -> None:
    if "openai_api_enabled_ui" not in st.session_state:
        st.session_state["openai_api_enabled_ui"] = bool(OPENAI_API_ENABLED)


def _openai_api_enabled() -> bool:
    return bool(st.session_state.get("openai_api_enabled_ui", OPENAI_API_ENABLED))


def _openai_mode_label() -> str:
    return "GPT" if _openai_api_enabled() else "rule_based_fallback"


def _increment_openai_call_count(amount: int = 1) -> None:
    st.session_state["openai_call_count"] = int(st.session_state.get("openai_call_count", 0)) + int(amount)


def _on_openai_connection_test_click() -> None:
    tested_at = now_jst_display()
    api_key = get_setting("OPENAI_API_KEY", "")
    model = _current_ai_virtual_model()
    if not _openai_api_enabled():
        st.session_state["openai_test_result"] = {
            "status": "disabled",
            "model_used": model,
            "error_type": "",
            "error_message": "",
            "message": "OpenAI API使用がOFFのため接続テストは実行しません。ONにするとテストできます。",
            "tested_at_jst": tested_at,
        }
        return

    if not api_key.strip():
        st.session_state["openai_test_result"] = {
            "status": "missing_key",
            "model_used": model,
            "error_type": "",
            "error_message": "OPENAI_API_KEYが未設定です。",
            "tested_at_jst": tested_at,
        }
        return

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=10)
        _increment_openai_call_count()
        client.responses.create(
            model=model,
            input='JSONで {"ok": true} だけ返してください',
            max_output_tokens=30,
        )
        st.session_state["openai_test_result"] = {
            "status": "ok",
            "model_used": model,
            "error_type": "",
            "error_message": "",
            "tested_at_jst": tested_at,
        }
    except Exception as exc:
        st.session_state["openai_test_result"] = {
            "status": "error",
            "model_used": model,
            "error_type": exc.__class__.__name__,
            "error_message": _mask_sensitive_openai_error(str(exc), api_key),
            "tested_at_jst": tested_at,
        }


def _render_openai_connection_test_state() -> None:
    st.markdown("**OpenAI接続テスト**")
    result = st.session_state.get("openai_test_result")
    if not result:
        st.info("まだOpenAI接続テストは実行していません。")
        return
    if result.get("status") == "ok":
        st.success("OpenAI接続テスト: ok")
    elif result.get("status") == "disabled":
        st.info("OpenAI connection test: disabled")
    elif result.get("status") == "missing_key":
        st.warning("OpenAI接続テスト: missing_key")
    else:
        st.error("OpenAI接続テスト: error")
    st.json(
        {
            "status": result.get("status", ""),
            "model_used": result.get("model_used", ""),
            "error_type": result.get("error_type", ""),
            "error_message": result.get("error_message", ""),
            "message": result.get("message", ""),
            "tested_at_jst": result.get("tested_at_jst", ""),
        }
    )


def _insert_ai_virtual_db_test_record() -> int:
    now = now_jst_iso()
    return insert_virtual_trade(
        {
            "timestamp": now,
            "timestamp_jst": now,
            "created_at_jst": now,
            "symbol": "DB_TEST",
            "name": "DB疎通テスト",
            "decision": "virtual_watch",
            "entry_type": "db_connection_test",
            "confidence": "テスト",
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "max_hold_days": 0,
            "reasons": ["DB疎通テスト用のdummyレコード"],
            "risk_factors": ["実売買・AI判断ではありません"],
            "source_score": 0,
            "model_used": "db_test",
            "is_ai_generated": 0,
            "judge_source": "test",
            "fallback_reason": "",
            "fallback_error_type": "",
            "fallback_error_message": "",
            "market_snapshot_json": {
                "test": True,
                "created_at": now,
                "source": "ai_virtual_trade_db_test",
            },
            "status": "logged",
        }
    )


def _ai_virtual_now_text() -> str:
    return now_jst_display()


def _refresh_ai_virtual_db_debug_state() -> None:
    st.session_state["last_ai_virtual_recent"] = rows_to_display(load_virtual_trades(limit=10)).to_dict("records")
    st.session_state["ai_virtual_database_list"] = get_virtual_database_list()


def _mark_ai_virtual_callback_click(source: str) -> str:
    clicked_at = _ai_virtual_now_text()
    st.session_state["ai_virtual_click_count"] = int(st.session_state.get("ai_virtual_click_count", 0)) + 1
    st.session_state["ai_virtual_last_click_at"] = clicked_at
    st.session_state["ai_virtual_last_click_source"] = source
    return clicked_at


def _on_ai_virtual_db_test_click() -> None:
    clicked_at = _mark_ai_virtual_callback_click("db_test")
    try:
        trade_id = _insert_ai_virtual_db_test_record()
        _refresh_ai_virtual_db_debug_state()
        st.session_state["last_ai_virtual_db_test"] = {
            "ok": True,
            "trade_id": trade_id,
            "db_path": str(VIRTUAL_TRADES_DB_PATH),
            "timestamp": clicked_at,
        }
        st.session_state["ai_virtual_last_error"] = ""
    except Exception as exc:
        st.session_state["last_ai_virtual_db_test"] = {
            "ok": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "db_path": str(VIRTUAL_TRADES_DB_PATH),
            "timestamp": clicked_at,
        }
        st.session_state["ai_virtual_last_error"] = f"{exc.__class__.__name__}: {exc}"


def _on_ai_virtual_run_click(candidates_snapshot: List[Dict[str, Any]] | None = None) -> None:
    clicked_at = _mark_ai_virtual_callback_click("ai_virtual_run")
    candidates = list(candidates_snapshot or st.session_state.get("ai_virtual_candidates_snapshot", []))
    use_openai = _openai_api_enabled()
    try:
        if use_openai and get_setting("OPENAI_API_KEY", "").strip():
            _increment_openai_call_count(len(candidates))
        results = process_virtual_trade_signals(candidates, use_openai=use_openai)
        saved_count = sum(1 for result in results if result.get("saved"))
        failed_count = len(results) - saved_count
        _refresh_ai_virtual_db_debug_state()
        st.session_state["last_ai_virtual_results"] = results
        st.session_state["last_ai_virtual_run"] = {
            "timestamp": clicked_at,
            "candidate_count": len(candidates),
            "saved_count": saved_count,
            "failed_count": failed_count,
            "db_path": str(VIRTUAL_TRADES_DB_PATH),
            "openai_api_enabled": use_openai,
            "ai_virtual_mode": _openai_mode_label(),
            "event_label": "フォワード検証ログ保存コールバックを検知しました",
            "reason_summary": [result.get("reason", "") for result in results],
            "error": "",
        }
        st.session_state["ai_virtual_last_error"] = ""
    except Exception as exc:
        try:
            _refresh_ai_virtual_db_debug_state()
        except Exception:
            pass
        st.session_state["last_ai_virtual_results"] = []
        st.session_state["last_ai_virtual_run"] = {
            "timestamp": clicked_at,
            "candidate_count": len(candidates),
            "saved_count": 0,
            "failed_count": len(candidates),
            "db_path": str(VIRTUAL_TRADES_DB_PATH),
            "openai_api_enabled": use_openai,
            "ai_virtual_mode": _openai_mode_label(),
            "event_label": "フォワード検証ログ保存コールバックを検知しました",
            "reason_summary": [],
            "error": f"フォワード検証ログ保存中に例外が発生しました: {exc.__class__.__name__}: {exc}",
        }
        st.session_state["ai_virtual_last_error"] = f"{exc.__class__.__name__}: {exc}"


def _render_ai_virtual_db_test_result() -> None:
    result = st.session_state.get("last_ai_virtual_db_test")
    if not result:
        return
    if result.get("ok"):
        st.success(f"DB疎通テスト成功: trade_id={result.get('trade_id')} / db_path={result.get('db_path')}")
    else:
        st.error(f"DB疎通テスト失敗: {result.get('error', '-')}")


def _render_ai_virtual_callback_debug_state() -> None:
    st.markdown("**callback方式テスト中**")
    cols = st.columns(3)
    cols[0].metric("click_count", int(st.session_state.get("ai_virtual_click_count", 0)))
    cols[1].metric("last_click_at", st.session_state.get("ai_virtual_last_click_at", "-"))
    cols[2].metric("last_source", st.session_state.get("ai_virtual_last_click_source", "-"))

    if st.session_state.get("ai_virtual_last_error"):
        st.error(st.session_state["ai_virtual_last_error"])

    db_state = st.session_state.get("last_ai_virtual_db_test", {})
    run_state = st.session_state.get("last_ai_virtual_run", {})
    results = st.session_state.get("last_ai_virtual_results", [])

    st.markdown("**last_ai_virtual_db_test**")
    st.json(db_state)
    st.markdown("**last_ai_virtual_run**")
    st.json(run_state)
    st.markdown("**last_ai_virtual_results**")
    if results:
        st.dataframe(_safe_dataframe(_virtual_result_table(results)), width="stretch", hide_index=True)
    else:
        st.info("last_ai_virtual_results はまだ空です。")

    database_list = st.session_state.get("ai_virtual_database_list")
    if database_list is None:
        try:
            database_list = get_virtual_database_list()
            st.session_state["ai_virtual_database_list"] = database_list
        except Exception as exc:
            database_list = [{"error": f"{exc.__class__.__name__}: {exc}"}]
    st.markdown("**SQLite PRAGMA database_list**")
    st.dataframe(_safe_dataframe(pd.DataFrame(database_list)), width="stretch", hide_index=True)

    st.markdown("**load_virtual_trades(limit=10) の直近ログ**")
    try:
        recent = rows_to_display(load_virtual_trades(limit=10))
        if recent.empty:
            st.info("直近ログはまだありません。")
        else:
            st.dataframe(_safe_dataframe(recent), width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"直近ログ取得失敗: {exc.__class__.__name__}: {exc}")


def _auto_market_now() -> datetime:
    override = st.session_state.get("ai_auto_market_now_override")
    if override:
        try:
            parsed = datetime.fromisoformat(str(override))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
            return parsed.astimezone(ZoneInfo("Asia/Tokyo"))
        except ValueError:
            pass
    return market_now_jst()


def _select_rule_auto_log_candidates(
    signals: List[Dict[str, Any]],
    target_mode: str,
    min_score: int,
    max_count: int,
) -> List[Dict[str, Any]]:
    return select_rule_buy_candidates(signals, target_mode, min_score, max_count)


def _summarize_virtual_trade_results(results: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "saved_count": sum(1 for result in results if result.get("saved")),
        "duplicate_count": sum(1 for result in results if result.get("reason") == "duplicate_recent"),
        "failed_count": sum(
            1
            for result in results
            if not result.get("saved") and result.get("reason") != "duplicate_recent"
        ),
    }


def _run_rule_auto_log_once() -> Dict[str, Any]:
    now = _auto_market_now()
    interval_minutes = int(st.session_state.get("gpt_interval_minutes", 1) or 1)
    interval_seconds = interval_minutes * 60
    target_mode = str(st.session_state.get("ai_auto_target_mode", "buy_only"))
    min_score = int(st.session_state.get("ai_auto_min_score", 70) or 70)
    max_count = int(st.session_state.get("gpt_max_candidates", 3) or 3)
    enabled = bool(st.session_state.get("ai_virtual_logging_enabled", False))
    st.session_state["auto_run_enabled"] = enabled
    signals = list(st.session_state.get("latest_signals", []))
    candidates = _select_rule_auto_log_candidates(signals, target_mode, min_score, max_count) if signals else []
    last_attempt_epoch = st.session_state.get("ai_rule_auto_last_attempt_epoch")

    summary: Dict[str, Any] = {
        "checked_at_jst": now.strftime("%Y-%m-%d %H:%M:%S JST"),
        "market_status": market_status_label(now),
        "auto_enabled": enabled,
        "interval_minutes": interval_minutes,
        "target_mode": target_mode,
        "min_score": min_score,
        "max_count": max_count,
        "last_auto_save_at": st.session_state.get("ai_rule_auto_last_save_at", "-"),
        "next_run_hint": "-",
        "latest_signals_count": len(signals),
        "target_count": len(candidates),
        "saved_count": 0,
        "failed_count": 0,
        "duplicate_count": 0,
        "openai_call_count": int(st.session_state.get("openai_call_count", 0)),
        "last_error": "",
        "db_path": str(VIRTUAL_TRADES_DB_PATH),
        "status": "skipped",
        "reason": "",
    }

    st.session_state["ai_rule_auto_last_check_at"] = summary["checked_at_jst"]

    if enabled:
        if is_market_open_jst(now):
            base_epoch = float(last_attempt_epoch or now.timestamp())
            next_dt = datetime.fromtimestamp(base_epoch + interval_seconds, tz=ZoneInfo("Asia/Tokyo"))
            summary["next_run_hint"] = next_dt.strftime("%Y-%m-%d %H:%M:%S JST")
        else:
            summary["next_run_hint"] = next_market_open_hint(now)

    if not enabled:
        summary["reason"] = "auto_disabled"
        st.session_state["ai_rule_auto_last_summary"] = summary
        return summary
    if not is_market_open_jst(now):
        summary["reason"] = "market_closed"
        st.session_state["ai_rule_auto_last_summary"] = summary
        return summary
    if not signals:
        summary["reason"] = "signals_missing"
        st.session_state["ai_rule_auto_last_error"] = "まだ株価スキャン結果がありません。先に株価スキャンを実行してください"
        summary["last_error"] = st.session_state["ai_rule_auto_last_error"]
        st.session_state["ai_rule_auto_last_summary"] = summary
        return summary

    now_epoch = now.timestamp()
    if last_attempt_epoch and now_epoch - float(last_attempt_epoch) < interval_seconds:
        summary["reason"] = "interval_wait"
        st.session_state["ai_rule_auto_last_summary"] = summary
        return summary

    st.session_state["ai_rule_auto_last_attempt_epoch"] = now_epoch
    next_dt = datetime.fromtimestamp(now_epoch + interval_seconds, tz=ZoneInfo("Asia/Tokyo"))
    summary["next_run_hint"] = next_dt.strftime("%Y-%m-%d %H:%M:%S JST")
    if not candidates:
        summary["reason"] = "no_candidates"
        st.session_state["ai_rule_auto_last_error"] = ""
        st.session_state["ai_rule_auto_last_summary"] = summary
        return summary

    try:
        log_summary = run_rule_based_virtual_logging(signals, target_mode, min_score, max_count)
        results = log_summary.get("results", [])
        summary.update(
            {
                "target_count": log_summary.get("selected_count", len(candidates)),
                "saved_count": log_summary.get("saved_count", 0),
                "failed_count": log_summary.get("failed_count", 0),
                "duplicate_count": log_summary.get("duplicate_count", 0),
                "db_path": log_summary.get("db_path", str(VIRTUAL_TRADES_DB_PATH)),
                "last_error": log_summary.get("error", ""),
            }
        )
        summary["status"] = "executed"
        summary["reason"] = "saved" if summary["saved_count"] else "no_new_saved"
        st.session_state["last_ai_virtual_results"] = results
        st.session_state["ai_rule_auto_last_results"] = results
        st.session_state["ai_rule_auto_last_save_at"] = summary["checked_at_jst"]
        st.session_state["ai_rule_auto_last_error"] = str(summary.get("last_error", ""))
        _refresh_ai_virtual_db_debug_state()
    except Exception as exc:
        summary["status"] = "error"
        summary["reason"] = "exception"
        summary["last_error"] = f"{exc.__class__.__name__}: {exc}"
        st.session_state["ai_rule_auto_last_error"] = summary["last_error"]

    st.session_state["ai_rule_auto_last_summary"] = summary
    return summary


def _render_rule_auto_log_status(summary: Dict[str, Any]) -> None:
    metrics = [
        ("現在時刻", summary.get("checked_at_jst", "-")),
        ("市場", summary.get("market_status", "-")),
        ("自動保存", _display_bool(summary.get("auto_enabled"))),
        ("間隔", f"{summary.get('interval_minutes', '-')}分"),
        ("対象", _display_target_mode(summary.get("target_mode", "buy_only"))),
        ("最小スコア", summary.get("min_score", "-")),
        ("最大保存", f"{summary.get('max_count', '-')}件"),
        ("スキャン候補", f"{summary.get('latest_signals_count', 0)}件"),
        ("対象候補", f"{summary.get('target_count', 0)}件"),
        ("最終保存", summary.get("last_auto_save_at", "-")),
        ("次回予定", summary.get("next_run_hint", "-")),
        ("保存成功", summary.get("saved_count", 0)),
        ("重複スキップ", summary.get("duplicate_count", 0)),
        ("保存失敗", summary.get("failed_count", 0)),
        ("API calls", summary.get("openai_call_count", 0)),
    ]
    for start in range(0, len(metrics), 5):
        cols = st.columns(5)
        for col, (label, value) in zip(cols, metrics[start : start + 5]):
            col.metric(label, value)

    if summary.get("reason") == "signals_missing":
        st.warning("まだスキャン結果がありません。先に『株価スキャンを実行』を押してください。")
    elif summary.get("reason") == "market_closed":
        st.info("市場時間外、昼休み、土日は自動保存を実行しません。")
    elif summary.get("reason") == "interval_wait":
        st.info("実行間隔の待機中です。")
    elif summary.get("status") == "executed":
        st.success("相場中ルール買いログ自動保存を実行しました。")

    if summary.get("last_error"):
        st.error(f"最終エラー: {summary['last_error']}")


def _render_rule_auto_log_fragment() -> None:
    interval_minutes = int(st.session_state.get("gpt_interval_minutes", 1) or 1)

    @st.fragment(run_every=timedelta(minutes=interval_minutes))
    def _fragment() -> None:
        summary = _run_rule_auto_log_once()
        _render_rule_auto_log_status(summary)
        recent_results = st.session_state.get("ai_rule_auto_last_results", [])
        if recent_results:
            with st.expander("直近の相場中自動保存結果", expanded=False):
                st.dataframe(_safe_dataframe(_virtual_result_table(recent_results)), width="stretch", hide_index=True)

    _fragment()


def _run_rule_auto_log_test_once() -> Dict[str, Any]:
    signals = list(st.session_state.get("latest_signals", []))
    target_mode = str(st.session_state.get("ai_auto_target_mode", "buy_only"))
    min_score = int(st.session_state.get("ai_auto_min_score", 70) or 70)
    max_count = int(st.session_state.get("gpt_max_candidates", 3) or 3)
    if not signals:
        summary = {
            "timestamp_jst": now_jst_display(),
            "market_status": market_status_label(_auto_market_now()),
            "input_signals_count": 0,
            "selected_count": 0,
            "saved_count": 0,
            "duplicate_count": 0,
            "failed_count": 0,
            "results": [],
            "error": "まだ株価スキャン結果がありません。先に株価スキャンを実行してください",
            "db_path": str(VIRTUAL_TRADES_DB_PATH),
            "openai_api_enabled": False,
            "use_openai": False,
        }
        st.session_state["ai_rule_auto_test_summary"] = summary
        st.session_state["ai_rule_auto_last_error"] = summary["error"]
        return summary

    summary = run_rule_based_virtual_logging(signals, target_mode, min_score, max_count)
    st.session_state["ai_rule_auto_test_summary"] = summary
    st.session_state["ai_rule_auto_last_results"] = summary.get("results", [])
    st.session_state["last_ai_virtual_results"] = summary.get("results", [])
    st.session_state["ai_rule_auto_last_error"] = summary.get("error", "")
    _refresh_ai_virtual_db_debug_state()
    return summary


def _render_rule_auto_log_test_result() -> None:
    summary = st.session_state.get("ai_rule_auto_test_summary")
    if not summary:
        return
    if summary.get("error"):
        st.warning(summary["error"])
    st.caption(f"保存テスト時刻：{summary.get('timestamp_jst', '-')}")
    cols = st.columns(5)
    cols[0].metric("対象候補", summary.get("selected_count", 0))
    cols[1].metric("保存成功", summary.get("saved_count", 0))
    cols[2].metric("重複スキップ", summary.get("duplicate_count", 0))
    cols[3].metric("保存失敗", summary.get("failed_count", 0))
    cols[4].metric("API calls", int(st.session_state.get("openai_call_count", 0)))


def _render_rule_auto_logger_section() -> None:
    st.markdown("**相場中ルール買いログ自動保存**")
    st.caption("画面を開いている間だけ、市場時間中にルールベースのpaper tradingログを保存します。実売買・発注は行いません。")
    has_signals = bool(st.session_state.get("latest_signals"))
    if not has_signals:
        st.info("まだスキャン結果がありません。先に『株価スキャンを実行』を押してください。")
        st.session_state["ai_virtual_logging_enabled"] = False
    auto_enabled = st.toggle(
        "相場中ルール買いログ自動保存",
        value=False,
        key="ai_virtual_logging_enabled",
        disabled=not has_signals,
    )
    st.session_state["auto_run_enabled"] = bool(auto_enabled)

    col_a, col_b = st.columns(2)
    with col_a:
        st.selectbox(
            "実行間隔",
            [1, 3, 5, 10],
            index=0,
            key="gpt_interval_minutes",
            format_func=lambda value: f"{value}分",
        )
        target_label = st.selectbox(
            "対象",
            ["買い候補のみ", "買い候補＋監視"],
            index=0,
            key="ai_auto_target_label",
        )
        st.session_state["ai_auto_target_mode"] = "buy_watch" if "監視" in target_label else "buy_only"
    with col_b:
        st.selectbox("最小スコア", [70, 75, 80], index=0, key="ai_auto_min_score")
        st.selectbox("最大保存件数", [1, 3, 5, 10], index=1, key="gpt_max_candidates")

    st.caption("自動保存は安全運用のため常に rule_based_fallback で実行します。OpenAI APIは呼びません。")
    st.caption("相場時間外でも、現在のスキャン結果からルールベースで仮想ログを保存します。")
    if st.button("手動で1回保存", key="run_rule_buy_log_once_test_button", disabled=not has_signals):
        _run_rule_auto_log_test_once()
    _render_rule_auto_log_test_result()
    _render_rule_auto_log_fragment()


def _render_ai_operation_status(trades: pd.DataFrame) -> None:
    st.markdown("**現在の運用状態**")
    open_count = int((trades["status"] == "open").sum()) if not trades.empty and "status" in trades.columns else 0
    closed_count = int((trades["status"] == "closed").sum()) if not trades.empty and "status" in trades.columns else 0
    cols = st.columns(6)
    cols[0].metric("OpenAI API", "ON" if _openai_api_enabled() else "OFF")
    cols[1].metric("判定方式", "GPT" if _openai_api_enabled() else "ルールベース")
    cols[2].metric("API calls", int(st.session_state.get("openai_call_count", 0)))
    cols[3].metric("仮想ログ", len(trades))
    cols[4].metric("open", open_count)
    cols[5].metric("closed", closed_count)
    if not _openai_api_enabled():
        st.info("OpenAI API使用OFFのため、GPT判断は行わず、ルールベースで仮想取引ログを保存します。API料金は発生しません。")


def _render_manual_virtual_judgement(
    buy: List[Dict[str, Any]],
    watch: List[Dict[str, Any]],
    has_signals: bool,
) -> None:
    with st.expander("手動のフォワード検証ログ保存", expanded=False):
        if not has_signals:
            st.info("まだスキャン結果がありません。先に『株価スキャンを実行』を押してください。")
        include_watch = st.checkbox("監視銘柄もフォワード検証に含める", value=False, disabled=not has_signals)
        max_candidates = st.selectbox("保存する候補件数", [1, 3, 5, 10], index=1, disabled=not has_signals)
        selected = list(buy) + (list(watch) if include_watch else [])
        selected = sorted(selected, key=lambda item: _score(item), reverse=True)[: int(max_candidates)]
        candidates_snapshot = _snapshot_ai_virtual_candidates(selected)
        st.session_state["ai_virtual_candidates_snapshot"] = candidates_snapshot

        if selected:
            st.caption("今回のフォワード検証候補")
            st.dataframe(_safe_dataframe(_ai_candidate_table(selected)), width="stretch", hide_index=True)
        elif has_signals:
            st.info("フォワード検証ログとして保存できる候補がありません。")

        if not _openai_api_enabled():
            st.caption("手動実行モード：rule_based_fallback（GPT判断なし）")
        else:
            st.caption("OpenAI API使用ONのため、手動実行ではGPT判定を使います。")
        st.button(
            "フォワード検証ログを保存",
            key="run_ai_virtual_trade_callback_button",
            type="primary",
            disabled=(not has_signals or not candidates_snapshot),
            on_click=_on_ai_virtual_run_click,
            args=(candidates_snapshot,),
        )


def _render_virtual_trade_log_tables(trades: pd.DataFrame) -> None:
    st.markdown("**仮想買いログ一覧**")
    if trades.empty:
        st.info("まだ仮想買いログはありません。")
        return

    filter_cols = st.columns([1, 1, 1.4, 1.2, 1])
    limit = filter_cols[0].selectbox("表示件数", [10, 30, 50, 100], index=1, key="virtual_log_limit")
    status_label = filter_cols[1].selectbox("状態", ["すべて", "保有中", "終了"], index=0, key="virtual_log_status_filter")
    query = filter_cols[2].text_input("銘柄検索", key="virtual_log_symbol_search")
    entry_types = ["すべて"]
    if "entry_type" in trades.columns:
        entry_types += sorted([item for item in trades["entry_type"].dropna().astype(str).unique().tolist() if item])
    entry_type = filter_cols[3].selectbox("型", entry_types, index=0, key="virtual_log_entry_type_filter")
    judge_method = filter_cols[4].selectbox("判定方式", ["すべて", "ルールベース", "GPT"], index=0, key="virtual_log_judge_filter")

    filtered = _filter_virtual_trades(trades, int(limit), status_label, query, entry_type, judge_method)
    display = _virtual_trade_display_table(filtered)
    st.dataframe(_safe_dataframe(display), width="stretch", hide_index=True)


def _render_open_virtual_trades(trades: pd.DataFrame) -> None:
    st.markdown("**open中の仮想取引**")
    open_df = trades[trades["status"].astype(str) == "open"] if not trades.empty and "status" in trades.columns else pd.DataFrame()
    if open_df.empty:
        st.info("open中の仮想取引はありません。")
        return
    st.dataframe(_safe_dataframe(_open_virtual_trade_display_table(open_df)), width="stretch", hide_index=True)


def _render_closed_virtual_trades(trades: pd.DataFrame) -> None:
    st.markdown("**終了済みの仮想取引**")
    closed_df = trades[trades["status"].astype(str) == "closed"] if not trades.empty and "status" in trades.columns else pd.DataFrame()
    if closed_df.empty:
        st.info("終了済みの仮想取引はありません。")
        return
    st.dataframe(_safe_dataframe(_closed_virtual_trade_display_table(closed_df)), width="stretch", hide_index=True)


def _render_ai_debug_expander() -> None:
    with st.expander("詳細デバッグ情報", expanded=False):
        st.button(
            "OpenAI接続テスト",
            key="openai_connection_test_button",
            on_click=_on_openai_connection_test_click,
        )
        _render_openai_connection_test_state()
        st.button(
            "DB疎通テスト",
            key="ai_virtual_db_test_button",
            on_click=_on_ai_virtual_db_test_click,
        )
        _render_ai_virtual_db_test_result()
        _render_ai_virtual_callback_debug_state()
        if st.button("open仮想取引の結果を更新", key="update_virtual_outcomes_button"):
            with st.spinner("仮想取引の結果を追跡中です..."):
                summary = update_open_virtual_trade_outcomes()
            st.info(f"確認 {summary['checked']}件 / 更新 {summary['updated']}件 / スキップ {summary['skipped']}件")


def _render_ai_virtual_trade_tab(
    buy: List[Dict[str, Any]],
    watch: List[Dict[str, Any]],
) -> None:
    st.subheader("フォワード検証")
    st.caption(
        "フォワード検証は、通常スキャンで出た買い候補を実際の運用時間中に仮想ログとして保存し、"
        "後日結果を確認するための機能です。実売買・発注・自動売買は行いません。"
        "OpenAI APIをOFFにしている場合は、GPT判断ではなくルールベースで保存します。"
    )

    trades = _operational_virtual_trades(load_virtual_trades(limit=1000))
    has_signals = bool(st.session_state.get("latest_signals"))
    _render_ai_operation_status(trades)
    _render_condition_settings_summary(
        "現在のフォワード検証条件",
        st.session_state.get("current_buy_condition_settings", {}),
    )
    _render_manual_virtual_judgement(buy, watch, has_signals)
    _render_rule_auto_logger_section()
    _render_virtual_trade_log_tables(trades)
    _render_open_virtual_trades(trades)
    _render_closed_virtual_trades(trades)
    _render_ai_debug_expander()


def _combine_date_time(date_value: Any, time_value: Any) -> datetime:
    return datetime.combine(date_value, time_value)


def _display_replay_outcome(value: Any) -> str:
    return {
        "win": "勝ち",
        "loss": "負け",
        "open": "検証中",
        "hit_take_profit": "利確到達",
        "hit_stop_loss": "損切り到達",
        "take_profit_hit": "利確到達",
        "stop_loss_hit": "損切り到達",
        "time_exit": "期限到達",
        "timeout": "期限到達",
    }.get(str(value or ""), str(value or "-"))


def _replay_result_table(trades: List[Dict[str, Any]], shares: int | None = None) -> pd.DataFrame:
    default_shares = 100
    if shares:
        default_shares = int(shares)
    elif trades:
        try:
            default_shares = int(trades[0].get("shares") or 100)
        except (TypeError, ValueError):
            default_shares = 100
    display_trades = apply_replay_money_metrics(trades, default_shares)
    rows = []
    for trade in display_trades:
        symbol = str(trade.get("symbol", "") or "")
        name = str(trade.get("name", "") or "")
        rows.append(
            {
                "仮想買い時刻": trade.get("signal_time", "-"),
                "銘柄": f"{symbol} {name}".strip() or "-",
                "条件": trade.get("condition_name", "-"),
                "型": trade.get("entry_type", "-"),
                "買値": format_yen(trade.get("entry_price")),
                "株数": f"{trade.get('shares', shares or 100)}株",
                "必要資金": _format_plain_yen(trade.get("required_capital_yen")),
                "損切り": format_yen(trade.get("stop_loss")),
                "利確目標": format_yen(trade.get("take_profit")),
                "スコア": _format_score_value(trade.get("score")),
                "日足順位": (
                    f"{trade.get('daily_rank_at_scan')}位/{trade.get('daily_rank_total')}銘柄"
                    if trade.get("daily_rank_at_scan") not in (None, "")
                    else "-"
                ),
                "日足スコア": _format_score_value(trade.get("daily_score")),
                "日足OK": f"{trade.get('daily_ok_count', '-')}/{trade.get('daily_total_count', '-')}",
                "5分足OK": f"{trade.get('intraday_ok_count', '-')}/{trade.get('intraday_total_count', '-')}",
                "マルチ通過": "OK" if trade.get("multi_timeframe_pass") else "NG",
                "損切り幅": _format_plain_pct(trade.get("stop_loss_pct")),
                "最大損失": _format_plain_yen(trade.get("max_loss_yen")),
                "想定利益": _format_signed_yen(trade.get("expected_profit_yen")),
                "損益比": _format_profit_loss_ratio(trade.get("risk_reward_ratio")),
                "リスク判定": _display_risk_pass(trade.get("risk_pass")),
                "リスク除外理由": _risk_reason_text(trade),
                "買い条件": _buy_condition_text(trade),
                "テクニカルプリセット": trade.get("technical_preset", "-"),
                "テクニカル素点": _format_score_value(trade.get("technical_score_raw")),
                "テクニカル減点": _format_score_value(trade.get("technical_penalty_score")),
                "テクニカル最終": _format_score_value(trade.get("technical_score_final")),
                "テクニカル判定": trade.get("technical_judgement", "-"),
                "確度": f"{trade.get('technical_confidence')}%" if trade.get("technical_confidence") not in (None, "") else "-",
                "score_breakdown": _technical_breakdown_text(trade.get("technical_score_breakdown")),
                "技術損切り": format_yen(trade.get("technical_stop_loss_candidate")),
                "技術利確": format_yen(trade.get("technical_target_price_candidate")),
                "技術R/R": _format_profit_loss_ratio(trade.get("technical_risk_reward")),
                "技術減点理由": _join_reason_text(trade.get("technical_penalty_reasons")),
                "技術見送り理由": _join_reason_text(trade.get("technical_hard_filter_reason")),
                "不足データ": _join_reason_text(trade.get("technical_missing_data")),
                "リーク検査": trade.get("leak_check_result", "-"),
                "cache": "hit" if trade.get("technical_cache_hit") else "miss",
                "結果": _display_replay_outcome(trade.get("outcome")),
                "終了時刻": trade.get("exit_time") or trade.get("evaluated_until", "-"),
                "終了価格": format_yen(trade.get("exit_price")),
                "損益円": _format_signed_yen(trade.get("profit_yen")),
                "累計損益": _format_signed_yen(trade.get("cumulative_profit_yen")),
                "リターン": _format_pct_value(trade.get("return_pct")),
                "最大利益率": _format_pct_value(trade.get("max_profit_pct")),
                "最大下落率": _format_pct_value(trade.get("max_drawdown_pct")),
                "保有期間": trade.get("holding_period", "-"),
                "終了理由": _display_replay_outcome(trade.get("exit_reason")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "仮想買い時刻",
            "銘柄",
            "条件",
            "型",
            "買値",
            "株数",
            "必要資金",
            "損切り",
            "利確目標",
            "スコア",
            "日足順位",
            "日足スコア",
            "日足OK",
            "5分足OK",
            "マルチ通過",
            "損切り幅",
            "最大損失",
            "想定利益",
            "損益比",
            "リスク判定",
            "リスク除外理由",
            "買い条件",
            "テクニカルプリセット",
            "テクニカル素点",
            "テクニカル減点",
            "テクニカル最終",
            "テクニカル判定",
            "確度",
            "score_breakdown",
            "技術損切り",
            "技術利確",
            "技術R/R",
            "技術減点理由",
            "技術見送り理由",
            "不足データ",
            "リーク検査",
            "cache",
            "結果",
            "終了時刻",
            "終了価格",
            "損益円",
            "累計損益",
            "リターン",
            "最大利益率",
            "最大下落率",
            "保有期間",
            "終了理由",
        ],
    )


def _replay_db_display_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return _replay_result_table(df.to_dict("records"))


def _historical_scan_trade_table(trades: List[Dict[str, Any]], shares: int) -> pd.DataFrame:
    rows = []
    display_trades = apply_replay_money_metrics(trades, shares)
    for trade in display_trades:
        symbol = str(trade.get("symbol", "") or "")
        name = str(trade.get("name", "") or "")
        rows.append(
            {
                "スキャン時刻": trade.get("scan_time", "-"),
                "仮想買い時刻": trade.get("signal_time", "-"),
                "銘柄": f"{symbol} {name}".strip() or "-",
                "条件": trade.get("condition_name", "-"),
                "順位": trade.get("selected_rank", "-"),
                "型": trade.get("entry_type", "-"),
                "買値": format_yen(trade.get("entry_price")),
                "損切り": format_yen(trade.get("stop_loss")),
                "利確目標": format_yen(trade.get("take_profit")),
                "スコア": _format_score_value(trade.get("scan_score", trade.get("score"))),
                "日足順位": (
                    f"{trade.get('daily_rank_at_scan')}位/{trade.get('daily_rank_total')}銘柄"
                    if trade.get("daily_rank_at_scan") not in (None, "")
                    else "-"
                ),
                "日足スコア": _format_score_value(trade.get("daily_score")),
                "日足OK": f"{trade.get('daily_ok_count', '-')}/{trade.get('daily_total_count', '-')}",
                "5分足OK": f"{trade.get('intraday_ok_count', '-')}/{trade.get('intraday_total_count', '-')}",
                "マルチ通過": "OK" if trade.get("multi_timeframe_pass") else "NG",
                "損切り幅": _format_plain_pct(trade.get("stop_loss_pct")),
                "最大損失": _format_plain_yen(trade.get("max_loss_yen")),
                "想定利益": _format_signed_yen(trade.get("expected_profit_yen")),
                "損益比": _format_profit_loss_ratio(trade.get("risk_reward_ratio")),
                "リスク判定": _display_risk_pass(trade.get("risk_pass")),
                "買い条件": _buy_condition_text(trade),
                "テクニカルプリセット": trade.get("technical_preset", "-"),
                "テクニカル素点": _format_score_value(trade.get("technical_score_raw")),
                "テクニカル減点": _format_score_value(trade.get("technical_penalty_score")),
                "テクニカル最終": _format_score_value(trade.get("technical_score_final")),
                "テクニカル判定": trade.get("technical_judgement", "-"),
                "確度": f"{trade.get('technical_confidence')}%" if trade.get("technical_confidence") not in (None, "") else "-",
                "score_breakdown": _technical_breakdown_text(trade.get("technical_score_breakdown")),
                "技術R/R": _format_profit_loss_ratio(trade.get("technical_risk_reward")),
                "リーク検査": trade.get("leak_check_result", "-"),
                "cache": "hit" if trade.get("technical_cache_hit") else "miss",
                "結果": _display_replay_outcome(trade.get("outcome")),
                "終了時刻": trade.get("exit_time") or trade.get("evaluated_until", "-"),
                "終了価格": format_yen(trade.get("exit_price")),
                "終了理由": _display_replay_outcome(trade.get("exit_reason")),
                "リターン": _format_pct_value(trade.get("return_pct")),
                "株数": f"{trade.get('shares', shares)}株",
                "必要資金": _format_plain_yen(trade.get("required_capital_yen")),
                "損益円": _format_signed_yen(trade.get("profit_yen")),
                "累計損益": _format_signed_yen(trade.get("cumulative_profit_yen")),
            }
        )
    columns = [
        "スキャン時刻",
        "仮想買い時刻",
        "銘柄",
        "条件",
        "順位",
        "型",
        "買値",
        "損切り",
        "利確目標",
        "スコア",
        "日足順位",
        "日足スコア",
        "日足OK",
        "5分足OK",
        "マルチ通過",
        "損切り幅",
        "最大損失",
        "想定利益",
        "損益比",
        "リスク判定",
        "買い条件",
        "テクニカルプリセット",
        "テクニカル素点",
        "テクニカル減点",
        "テクニカル最終",
        "テクニカル判定",
        "確度",
        "score_breakdown",
        "技術R/R",
        "リーク検査",
        "cache",
        "結果",
        "終了時刻",
        "終了価格",
        "終了理由",
        "リターン",
        "株数",
        "必要資金",
        "損益円",
        "累計損益",
    ]
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def _historical_scan_symbol_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        symbol = str(row.get("symbol", "") or "")
        name = str(row.get("name", "") or "")
        rows.append(
            {
                "銘柄": f"{symbol} {name}".strip() or "-",
                "データ件数": int(row.get("data_rows", 0) or 0),
                "仮想買い件数": int(row.get("trade_count", 0) or 0),
                "勝率": _format_pct_value(row.get("win_rate_pct")),
                "純損益": _format_signed_yen(row.get("net_profit_yen")),
                "累計利益": _format_signed_yen(row.get("gross_profit_yen")),
                "累計損失": _format_signed_yen(row.get("gross_loss_yen")),
                "損益比": _format_profit_loss_ratio(row.get("profit_loss_ratio")),
                "平均リターン": _format_pct_value(row.get("avg_return_pct")),
                "最大利益": _format_signed_yen(row.get("max_profit_yen")),
                "最大損失": _format_signed_yen(row.get("max_loss_yen")),
                "最大連勝": int(row.get("max_win_streak", 0) or 0),
                "最大連敗": int(row.get("max_loss_streak", 0) or 0),
                "エラー": row.get("error", ""),
            }
        )
    return pd.DataFrame(rows)


def _replay_group_performance_table(trades: List[Dict[str, Any]], group_key: str, label: str, shares: int) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    values = sorted({str(trade.get(group_key, "-") or "-") for trade in trades})
    for value in values:
        scoped = [trade for trade in trades if str(trade.get(group_key, "-") or "-") == value]
        scoped_with_money = apply_replay_money_metrics(scoped, shares)
        perf = summarize_replay_results(scoped)
        money = summarize_replay_money(scoped_with_money, shares)
        closed_money = [trade for trade in scoped_with_money if trade.get("profit_yen") is not None]
        win_count = sum(1 for trade in closed_money if float(trade.get("profit_yen") or 0) > 0)
        loss_count = sum(1 for trade in closed_money if float(trade.get("profit_yen") or 0) < 0)
        rows.append(
            {
                label: value,
                "仮想買い件数": len(scoped),
                "確定損益対象": money.get("closed_trade_count", perf.get("closed_count", 0)),
                "勝ち件数": win_count,
                "負け件数": loss_count,
                "勝率": _format_pct_value(perf.get("win_rate_pct")),
                "累計利益": _format_signed_yen(money.get("gross_profit_yen")),
                "累計損失": _format_signed_yen(money.get("gross_loss_yen")),
                "純損益": _format_signed_yen(money.get("net_profit_yen")),
                "損益比": _format_profit_loss_ratio(money.get("profit_loss_ratio")),
                "平均利益": _format_signed_yen(money.get("average_profit_yen")),
                "平均損失": _format_signed_yen(money.get("average_loss_yen")),
                "最大利益": _format_signed_yen(money.get("max_profit_yen")),
                "最大損失": _format_signed_yen(money.get("max_loss_yen")),
                "最大連勝": money.get("max_win_streak", 0),
                "最大連敗": money.get("max_loss_streak", 0),
            }
        )
    return pd.DataFrame(rows)


def _render_multi_timeframe_performance(trades: List[Dict[str, Any]], shares: int) -> None:
    if not trades:
        return
    prepared = []
    for trade in trades:
        row = dict(trade)
        row["daily_ok_group"] = f"{row.get('daily_ok_count', '-')}/{row.get('daily_total_count', '-')}"
        row["intraday_ok_group"] = f"{row.get('intraday_ok_count', '-')}/{row.get('intraday_total_count', '-')}"
        row["multi_pass_group"] = "通過" if row.get("multi_timeframe_pass") else "未通過"
        prepared.append(row)

    st.markdown("**マルチ時間足別 成績**")
    cols = st.columns(2)
    with cols[0]:
        st.caption("日足OK数別")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "daily_ok_group", "日足OK", shares)), width="stretch", hide_index=True)
        st.caption("entry_type別")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "entry_type", "型", shares)), width="stretch", hide_index=True)
    with cols[1]:
        st.caption("5分足OK数別")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "intraday_ok_group", "5分足OK", shares)), width="stretch", hide_index=True)
        st.caption("マルチ時間足通過別")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "multi_pass_group", "マルチ判定", shares)), width="stretch", hide_index=True)


def _daily_rank_bucket(value: Any) -> str:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return "未計算"
    if number <= 5:
        return "1〜5位"
    if number <= 10:
        return "6〜10位"
    if number <= 20:
        return "11〜20位"
    return "21位以下"


def _daily_score_bucket(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "未計算"
    if number < 70:
        return "70未満"
    if number < 75:
        return "70〜74"
    if number < 80:
        return "75〜79"
    return "80以上"


def _render_daily_top_n_performance(trades: List[Dict[str, Any]], shares: int) -> None:
    if not trades:
        return
    prepared = []
    for trade in trades:
        row = dict(trade)
        row["daily_rank_bucket"] = _daily_rank_bucket(row.get("daily_rank_at_scan"))
        row["daily_score_bucket"] = _daily_score_bucket(row.get("daily_score"))
        row["daily_top_n_group"] = "通過" if row.get("daily_top_n_pass") else "未通過"
        prepared.append(row)

    st.markdown("**日足上位N別 成績**")
    cols = st.columns(2)
    with cols[0]:
        st.caption("日足順位別")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "daily_rank_bucket", "日足順位", shares)), width="stretch", hide_index=True)
        st.caption("日足スコア帯別")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "daily_score_bucket", "日足スコア", shares)), width="stretch", hide_index=True)
    with cols[1]:
        st.caption("日足上位N通過別")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "daily_top_n_group", "日足上位N", shares)), width="stretch", hide_index=True)


def _stop_loss_bucket(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "未計算"
    if number < 1:
        return "0〜1%"
    if number < 2:
        return "1〜2%"
    if number <= 3:
        return "2〜3%"
    return "3%以上"


def _max_loss_bucket(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "未計算"
    if number <= 10000:
        return "1万円以内"
    if number <= 20000:
        return "2万円以内"
    if number <= 30000:
        return "3万円以内"
    return "3万円超"


def _render_risk_performance(trades: List[Dict[str, Any]], shares: int) -> None:
    if not trades:
        return
    prepared = []
    for trade in trades:
        row = dict(trade)
        risk_status = _display_risk_pass(row.get("risk_pass"))
        row["risk_pass_group"] = "リスクOK" if risk_status == "OK" else ("リスクNG" if risk_status == "NG" else "未計算")
        row["stop_loss_bucket"] = _stop_loss_bucket(row.get("stop_loss_pct"))
        row["max_loss_bucket"] = _max_loss_bucket(row.get("max_loss_yen"))
        prepared.append(row)

    st.markdown("**リスク条件別 成績**")
    risk_ok = [trade for trade in prepared if trade.get("risk_pass")]
    cols = st.columns(2)
    with cols[0]:
        st.caption("リスクOKのみの成績")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(risk_ok, "risk_pass_group", "リスク判定", shares)), width="stretch", hide_index=True)
        st.caption("損切り幅別")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "stop_loss_bucket", "損切り幅", shares)), width="stretch", hide_index=True)
    with cols[1]:
        st.caption("リスクOK/NG別")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "risk_pass_group", "リスク判定", shares)), width="stretch", hide_index=True)
        st.caption("最大損失額別")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "max_loss_bucket", "最大損失", shares)), width="stretch", hide_index=True)


def _technical_score_bucket(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "未計算"
    if number < 30:
        return "30未満"
    if number < 36:
        return "30〜35"
    if number < 42:
        return "36〜41"
    if number < 50:
        return "42〜49"
    return "50以上"


def _breakdown_has(trade: Dict[str, Any], section: str, pattern: str) -> bool:
    breakdown = trade.get("technical_score_breakdown")
    if not isinstance(breakdown, dict):
        return False
    values = breakdown.get(section) or []
    if isinstance(values, str):
        values = [values]
    return any(pattern in str(value) for value in values)


def _render_technical_performance(trades: List[Dict[str, Any]], shares: int) -> None:
    technical_trades = [trade for trade in trades if trade.get("technical_score_final") not in (None, "")]
    if not technical_trades:
        return
    prepared = []
    for trade in technical_trades:
        row = dict(trade)
        row["technical_score_bucket"] = _technical_score_bucket(row.get("technical_score_final"))
        row["technical_preset_group"] = row.get("technical_preset") or "未設定"
        row["technical_judgement_group"] = row.get("technical_judgement") or "未設定"
        row["technical_25ma_group"] = "25日線上" if _breakdown_has(row, "trend", "25日線より上") else "25日線上なし"
        breakdown = row.get("technical_score_breakdown") if isinstance(row.get("technical_score_breakdown"), dict) else {}
        row["technical_volume_group"] = "出来高加点あり" if breakdown.get("volume") else "出来高加点なし"
        row["technical_breakout_group"] = "ブレイク加点あり" if breakdown.get("breakout") else "ブレイク加点なし"
        row["technical_lower_shadow_group"] = "下ヒゲ陽線あり" if _breakdown_has(row, "candle", "下ヒゲ陽線") else "下ヒゲ陽線なし"
        intraday = row.get("intraday_entry_json") if isinstance(row.get("intraday_entry_json"), dict) else {}
        row["technical_vwap_group"] = "VWAPあり" if intraday.get("vwap_ok") else "VWAPなし"
        prepared.append(row)

    st.markdown("**テクニカル採点別 成績**")
    cols = st.columns(2)
    with cols[0]:
        st.caption("プリセット別勝率")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "technical_preset_group", "プリセット", shares)), width="stretch", hide_index=True)
        st.caption("スコア帯別勝率")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "technical_score_bucket", "スコア帯", shares)), width="stretch", hide_index=True)
        st.caption("25日線上/下")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "technical_25ma_group", "25日線", shares)), width="stretch", hide_index=True)
    with cols[1]:
        st.caption("テクニカル判定別")
        st.dataframe(_safe_dataframe(_replay_group_performance_table(prepared, "technical_judgement_group", "判定", shares)), width="stretch", hide_index=True)
        st.caption("条件別: VWAP / 出来高 / ブレイク / 下ヒゲ")
        condition_rows = []
        for key, label in [
            ("technical_vwap_group", "VWAP"),
            ("technical_volume_group", "出来高"),
            ("technical_breakout_group", "ブレイク"),
            ("technical_lower_shadow_group", "下ヒゲ"),
        ]:
            table = _replay_group_performance_table(prepared, key, label, shares)
            if not table.empty:
                condition_rows.extend(table.to_dict("records"))
        st.dataframe(_safe_dataframe(pd.DataFrame(condition_rows)), width="stretch", hide_index=True)


def _render_replay_fetch_summary(meta: Dict[str, Any]) -> None:
    if not meta:
        return
    if meta.get("error_message"):
        st.warning(f"{meta.get('error_type', 'error')}: {meta.get('error_message')}")
    cols = st.columns(5)
    cols[0].metric("取得データ件数", f"{meta.get('fetched_rows', 0)}件")
    cols[1].metric("期間", meta.get("period_label", "-"))
    cols[2].metric("時間足", meta.get("interval_label", "-"))
    cols[3].metric("最初の日時", meta.get("first_timestamp", "-"))
    cols[4].metric("最後の日時", meta.get("last_timestamp", "-"))
    st.caption(
        f"データ元：{'キャッシュ' if meta.get('from_cache') else 'yfinance'} / "
        f"symbol={meta.get('normalized_symbol', '-')} / auto_adjust={YFINANCE_AUTO_ADJUST}"
    )


def _render_replay_anomaly_summary(validation: Dict[str, Any]) -> None:
    if not validation:
        return
    st.markdown("**異常値チェック結果**")
    cols = st.columns(5)
    cols[0].metric("元データ", f"{validation.get('original_rows', 0)}本")
    cols[1].metric("除外バー", f"{validation.get('excluded_count', 0)}本")
    cols[2].metric("High<Low", validation.get("invalid_ohlc_count", 0))
    cols[3].metric("Close<=0", validation.get("non_positive_close_count", 0))
    cols[4].metric("Close急変", validation.get("close_jump_count", 0))
    warnings = validation.get("warnings") or []
    if warnings:
        st.warning(" / ".join(str(item) for item in warnings))
    else:
        st.success("異常値チェック: 大きな異常は見つかりませんでした。")


def _replay_extreme_result_warnings(trades: List[Dict[str, Any]]) -> None:
    large_drawdown = [
        trade for trade in trades
        if trade.get("max_drawdown_pct") is not None and float(trade.get("max_drawdown_pct")) <= -20
    ]
    large_profit = [
        trade for trade in trades
        if trade.get("max_profit_pct") is not None and float(trade.get("max_profit_pct")) >= 20
    ]
    if large_drawdown:
        st.warning(f"最大下落率が-20%以下の結果が {len(large_drawdown)} 件あります。詳細デバッグ情報で確認してください。")
    if large_profit:
        st.warning(f"最大利益率が+20%以上の結果が {len(large_profit)} 件あります。詳細デバッグ情報で確認してください。")


def _replay_debug_table(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for trade in trades:
        debug = trade.get("debug") or {}
        if not debug:
            continue
        rows.append(debug)
    return pd.DataFrame(rows)


def _render_replay_summary(summary: Dict[str, Any], saved_count: int) -> None:
    cols = st.columns(5)
    cols[0].metric("リプレイ実行件数", summary.get("evaluated_steps", 0))
    cols[1].metric("仮想買い件数", summary.get("trade_count", 0))
    cols[2].metric("勝率", _format_pct_value(summary.get("win_rate_pct")))
    cols[3].metric("平均リターン", _format_pct_value(summary.get("avg_return_pct")))
    cols[4].metric("DB保存", f"{saved_count}件")

    cols = st.columns(5)
    cols[0].metric("最大利益率平均", _format_pct_value(summary.get("avg_max_profit_pct")))
    cols[1].metric("最大下落率平均", _format_pct_value(summary.get("avg_max_drawdown_pct")))
    cols[2].metric("損切り到達", summary.get("hit_stop_loss_count", 0))
    cols[3].metric("利確到達", summary.get("hit_take_profit_count", 0))
    cols[4].metric("検証データ", f"{summary.get('data_rows', 0)}本")


def _format_profit_loss_ratio(value: Any) -> str:
    try:
        if value in (None, "") or pd.isna(value):
            return "-"
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _display_risk_pass(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, str):
        if value.strip() in {"1", "True", "true", "OK"}:
            return "OK"
        if value.strip() in {"0", "False", "false", "NG"}:
            return "NG"
    return "OK" if bool(value) else "NG"


def _optional_float(value: Any) -> float | None:
    try:
        if value in (None, "") or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _risk_reason_text(trade: Dict[str, Any]) -> str:
    reasons = trade.get("risk_reasons")
    if isinstance(reasons, list) and reasons:
        return "、".join(str(item) for item in reasons)
    risk_filter = trade.get("risk_filter")
    if isinstance(risk_filter, dict) and risk_filter.get("risk_reasons"):
        return "、".join(str(item) for item in risk_filter.get("risk_reasons", []))
    return "-"


def _buy_condition_text(item: Dict[str, Any]) -> str:
    condition = item.get("buy_condition_json")
    if not isinstance(condition, dict):
        condition = {}
    daily_rank = item.get("daily_rank_at_scan", condition.get("daily_rank_at_scan"))
    daily_total = item.get("daily_rank_total", condition.get("daily_rank_total"))
    daily_score = item.get("daily_score", condition.get("daily_score"))
    daily_ok = item.get("daily_ok_count", condition.get("daily_ok_count", "-"))
    daily_total_count = item.get("daily_total_count", condition.get("daily_total_count", "-"))
    intraday_ok = item.get("intraday_ok_display") or item.get("intraday_ok_count", condition.get("intraday_ok_count", "-"))
    intraday_total_count = item.get("intraday_total_count", condition.get("intraday_total_count", "-"))
    risk_pass = item.get("risk_pass", condition.get("risk_pass"))
    daily_top_n_pass = item.get("daily_top_n_pass", condition.get("daily_top_n_pass"))
    min_score_pass = condition.get("min_score_pass")
    top_n = item.get("daily_top_n", condition.get("daily_top_n"))
    min_score = condition.get("min_score")
    max_buy_candidates = condition.get("max_buy_candidates")
    max_buy_candidates_pass = condition.get("max_buy_candidates_pass")
    max_stop_loss_pct = condition.get("max_stop_loss_pct")
    max_loss_yen_limit = condition.get("max_loss_yen_limit")
    min_risk_reward = condition.get("min_risk_reward")
    parts = []
    if daily_rank not in (None, ""):
        parts.append(f"日足順位 {daily_rank}位/{daily_total or '-'}銘柄中")
    if daily_score not in (None, ""):
        parts.append(f"日足スコア {_format_score_value(daily_score)}点")
    parts.append(f"日足条件 {daily_ok}/{daily_total_count} OK")
    if intraday_ok == "データなし":
        parts.append("5分足条件 データなし")
    else:
        parts.append(f"5分足条件 {intraday_ok}/{intraday_total_count} OK")
    parts.append(f"リスク条件 {_display_risk_pass(risk_pass)}")
    parts.append(f"日足上位N通過 {_display_risk_pass(daily_top_n_pass)}")
    if min_score not in (None, ""):
        parts.append(f"最小スコア {min_score}以上 {_display_risk_pass(min_score_pass)}")
    if top_n not in (None, ""):
        parts.append(f"日足上位N {top_n}以内")
    if max_buy_candidates not in (None, ""):
        parts.append(f"買い候補上限 {max_buy_candidates}件 {_display_risk_pass(max_buy_candidates_pass)}")
    parsed_max_stop = _optional_float(max_stop_loss_pct)
    parsed_max_loss = _optional_float(max_loss_yen_limit)
    parsed_min_rr = _optional_float(min_risk_reward)
    if parsed_max_stop is not None:
        parts.append(f"最大損切り幅 {parsed_max_stop:.1f}%以内")
    if parsed_max_loss is not None:
        parts.append(f"最大損失 {parsed_max_loss:,.0f}円以内")
    if parsed_min_rr is not None:
        parts.append(f"最低損益比 {parsed_min_rr:.2f}以上")
    return " / ".join(parts)


def _render_replay_money_summary(money_summary: Dict[str, Any]) -> None:
    st.markdown("**想定株数ベース損益**")
    cols = st.columns(5)
    cols[0].metric("想定株数", f"{money_summary.get('shares', 100)}株")
    cols[1].metric("必要資金平均", _format_plain_yen(money_summary.get("average_required_capital_yen")))
    cols[2].metric("累計利益", _format_signed_yen(money_summary.get("gross_profit_yen")))
    cols[3].metric("累計損失", _format_signed_yen(money_summary.get("gross_loss_yen")))
    cols[4].metric("純損益", _format_signed_yen(money_summary.get("net_profit_yen")))

    cols = st.columns(5)
    cols[0].metric("損益比", _format_profit_loss_ratio(money_summary.get("profit_loss_ratio")))
    cols[1].metric("平均利益", _format_signed_yen(money_summary.get("average_profit_yen")))
    cols[2].metric("平均損失", _format_signed_yen(money_summary.get("average_loss_yen")))
    cols[3].metric("最大利益", _format_signed_yen(money_summary.get("max_profit_yen")))
    cols[4].metric("最大損失", _format_signed_yen(money_summary.get("max_loss_yen")))

    cols = st.columns(3)
    cols[0].metric("確定損益対象", f"{money_summary.get('closed_trade_count', 0)}件")
    cols[1].metric("最大連勝数", money_summary.get("max_win_streak", 0))
    cols[2].metric("最大連敗数", money_summary.get("max_loss_streak", 0))


def _render_replay_profit_curve(trades: List[Dict[str, Any]], shares: int) -> None:
    prepared = apply_replay_money_metrics(trades, shares)
    rows = [
        {
            "仮想買い時刻": pd.Timestamp(trade.get("signal_time")),
            "累計損益円": trade.get("cumulative_profit_yen"),
        }
        for trade in prepared
        if trade.get("cumulative_profit_yen") is not None
    ]
    if not rows:
        return
    curve = pd.DataFrame(rows).sort_values("仮想買い時刻")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["仮想買い時刻"],
            y=curve["累計損益円"],
            mode="lines+markers",
            name="累計損益",
            line={"color": "#0f766e", "width": 2},
            marker={"size": 7},
        )
    )
    fig.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title="円",
        xaxis_title="",
    )
    _render_plotly_chart(fig, key="replay_profit_curve")


def _render_symbol_profit_bar(symbol_summary: pd.DataFrame) -> None:
    if symbol_summary.empty or "net_profit_yen" not in symbol_summary.columns:
        return
    top = symbol_summary.copy()
    top["label"] = (top["symbol"].astype(str) + " " + top["name"].fillna("").astype(str)).str.strip()
    top = top.sort_values("net_profit_yen", ascending=False).head(20)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["label"],
            y=top["net_profit_yen"],
            name="銘柄別純損益",
            marker_color=["#16a34a" if value >= 0 else "#dc2626" for value in top["net_profit_yen"]],
        )
    )
    fig.update_layout(height=340, margin={"l": 10, "r": 10, "t": 30, "b": 90}, yaxis_title="円", xaxis_title="")
    _render_plotly_chart(fig, key="historical_scan_symbol_profit_bar")


def _render_selected_count_chart(selected_by_scan_time: List[Dict[str, Any]]) -> None:
    if not selected_by_scan_time:
        return
    df = pd.DataFrame(selected_by_scan_time)
    if df.empty:
        return
    df["scan_time"] = pd.to_datetime(df["scan_time"], errors="coerce")
    df = df.dropna(subset=["scan_time"])
    if df.empty:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["scan_time"],
            y=df["selected_count"],
            mode="lines",
            name="採用銘柄数",
            line={"color": "#7c3aed", "width": 1.8},
        )
    )
    fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title="件", xaxis_title="")
    _render_plotly_chart(fig, key="historical_scan_selected_count_chart")


def _render_replay_chart(df: pd.DataFrame, trades: List[Dict[str, Any]]) -> None:
    if df.empty:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="終値",
            line={"color": "#2563eb", "width": 1.6},
        )
    )
    if trades:
        buy_x = [pd.Timestamp(trade["signal_time"]) for trade in trades if trade.get("signal_time")]
        buy_y = [trade.get("entry_price") for trade in trades if trade.get("signal_time")]
        fig.add_trace(
            go.Scatter(
                x=buy_x,
                y=buy_y,
                mode="markers",
                name="仮想買い",
                marker={"color": "#16a34a", "size": 9, "symbol": "triangle-up"},
            )
        )
        for index, trade in enumerate(trades):
            start = pd.Timestamp(trade.get("signal_time"))
            end = pd.Timestamp(trade.get("evaluated_until") or df.index[-1])
            stop_loss = trade.get("stop_loss")
            take_profit = trade.get("take_profit")
            if stop_loss:
                fig.add_trace(
                    go.Scatter(
                        x=[start, end],
                        y=[stop_loss, stop_loss],
                        mode="lines",
                        name="損切り" if index == 0 else None,
                        showlegend=index == 0,
                        line={"color": "#dc2626", "width": 1, "dash": "dot"},
                    )
                )
            if take_profit:
                fig.add_trace(
                    go.Scatter(
                        x=[start, end],
                        y=[take_profit, take_profit],
                        mode="lines",
                        name="利確" if index == 0 else None,
                        showlegend=index == 0,
                        line={"color": "#059669", "width": 1, "dash": "dot"},
                    )
                )
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        yaxis_title="価格",
        xaxis_title="",
    )
    _render_plotly_chart(fig, key="replay_price_chart")


def _historical_result_to_meta(
    result: HistoricalDataResult,
    period_label: str,
    interval_label: str,
) -> Dict[str, Any]:
    return {
        "symbol": result.symbol,
        "normalized_symbol": result.normalized_symbol,
        "period": result.period,
        "period_label": period_label,
        "interval": result.interval,
        "interval_label": interval_label,
        "cache_path": str(result.cache_path),
        "from_cache": result.from_cache,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "fetched_rows": result.fetched_rows,
        "first_timestamp": result.first_timestamp,
        "last_timestamp": result.last_timestamp,
    }


def _render_historical_scan_overall_summary(result: Dict[str, Any]) -> None:
    summary = result.get("summary", {})
    st.markdown("**全体サマリー**")
    cols = st.columns(6)
    cols[0].metric("対象銘柄", summary.get("symbols_count", 0))
    cols[1].metric("取得成功", summary.get("fetch_success_count", 0))
    cols[2].metric("取得失敗", summary.get("fetch_failed_count", 0))
    cols[3].metric("スキャン回数", summary.get("total_scan_steps", 0))
    cols[4].metric("仮想買い", summary.get("total_trades", 0))
    cols[5].metric("勝率", _format_pct_value(summary.get("win_rate_pct")))

    cols = st.columns(4)
    cols[0].metric("日足上位N", "ON" if summary.get("use_daily_top_n") else "OFF")
    cols[1].metric("上位N", summary.get("daily_top_n", "-"))
    cols[2].metric("日足最低点", summary.get("min_daily_score", "-"))
    cols[3].metric("watchlist上限", summary.get("watchlist_limit", "-"))

    cols = st.columns(6)
    cols[0].metric("データ取得秒数", f"{summary.get('fetch_seconds', 0)}秒")
    cols[1].metric("特徴量計算秒数", f"{summary.get('feature_seconds', 0)}秒")
    cols[2].metric("スキャン判定秒数", f"{summary.get('scan_seconds', 0)}秒")
    cols[3].metric("DB保存秒数", f"{summary.get('db_save_seconds', 0)}秒")
    cols[4].metric("合計秒数", f"{summary.get('total_seconds', 0)}秒")
    cols[5].metric("1スキャン平均", f"{summary.get('avg_seconds_per_scan', 0)}秒")

    _render_replay_money_summary(result.get("money_summary", {}))


def _render_historical_scan_rankings(symbol_summary: pd.DataFrame) -> None:
    if symbol_summary.empty:
        return
    st.markdown("**ランキング**")
    tradable = symbol_summary[pd.to_numeric(symbol_summary.get("trade_count", 0), errors="coerce").fillna(0) > 0].copy()
    if tradable.empty:
        st.info("ランキング対象の仮想買い結果がありません。")
        return
    ranking_cols = st.columns(2)
    with ranking_cols[0]:
        st.caption("純損益ランキング")
        st.dataframe(
            _safe_dataframe(_historical_scan_symbol_summary_table(tradable.sort_values("net_profit_yen", ascending=False).head(10))),
            width="stretch",
            hide_index=True,
        )
        st.caption("勝率ランキング")
        st.dataframe(
            _safe_dataframe(_historical_scan_symbol_summary_table(tradable.sort_values("win_rate_pct", ascending=False).head(10))),
            width="stretch",
            hide_index=True,
        )
    with ranking_cols[1]:
        st.caption("損益比ランキング")
        ratio_sorted = tradable.copy()
        ratio_sorted["profit_loss_ratio"] = pd.to_numeric(ratio_sorted["profit_loss_ratio"], errors="coerce").fillna(-1)
        st.dataframe(
            _safe_dataframe(_historical_scan_symbol_summary_table(ratio_sorted.sort_values("profit_loss_ratio", ascending=False).head(10))),
            width="stretch",
            hide_index=True,
        )
        st.caption("最大損失が小さいランキング")
        loss_sorted = tradable.copy()
        loss_sorted["max_loss_yen"] = pd.to_numeric(loss_sorted["max_loss_yen"], errors="coerce").fillna(0)
        st.dataframe(
            _safe_dataframe(_historical_scan_symbol_summary_table(loss_sorted.sort_values("max_loss_yen", ascending=False).head(10))),
            width="stretch",
            hide_index=True,
        )


def _render_watchlist_filter_status(filter_result: Dict[str, Any], target_record_count: int) -> None:
    st.markdown("**対象銘柄フィルター**")
    cols = st.columns(4)
    cols[0].metric("元watchlist", f"{filter_result.get('original_count', 0)}銘柄")
    cols[1].metric("対象指定後", f"{filter_result.get('included_count', 0)}銘柄")
    cols[2].metric("除外後", f"{filter_result.get('final_count', 0)}銘柄")
    cols[3].metric("実行対象", f"{target_record_count}銘柄")

    excluded_labels = filter_result.get("excluded_labels", [])
    missing_symbols = filter_result.get("missing_symbols", [])
    if excluded_labels:
        st.caption("除外: " + ", ".join(str(item) for item in excluded_labels))
    if missing_symbols:
        st.warning("watchlistに存在しない入力銘柄: " + ", ".join(str(item) for item in missing_symbols))


def _historical_scan_compare_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    money = result.get("money_summary", {}) if isinstance(result, dict) else {}
    return {
        "symbols_count": summary.get("symbols_count", 0),
        "fetch_success_count": summary.get("fetch_success_count", 0),
        "fetch_failed_count": summary.get("fetch_failed_count", 0),
        "scan_steps": summary.get("total_scan_steps", 0),
        "trades": summary.get("total_trades", 0),
        "win_rate_pct": summary.get("win_rate_pct", 0),
        "gross_profit_yen": money.get("gross_profit_yen", 0),
        "gross_loss_yen": money.get("gross_loss_yen", 0),
        "net_profit_yen": money.get("net_profit_yen", 0),
        "profit_loss_ratio": money.get("profit_loss_ratio"),
        "max_loss_yen": money.get("max_loss_yen"),
        "avg_loss_yen": money.get("average_loss_yen"),
        "total_seconds": summary.get("total_seconds", 0),
    }


def _render_historical_scan_ab_comparison(comparison: Dict[str, Any]) -> None:
    if not comparison:
        return
    before = comparison.get("before_exclude", {})
    after = comparison.get("after_exclude", {})
    rows = [
        {
            "比較": "A: 除外前",
            "対象銘柄": before.get("symbols_count", 0),
            "仮想買い": before.get("trades", 0),
            "勝率": _format_pct_value(before.get("win_rate_pct")),
            "累計利益": _format_signed_yen(before.get("gross_profit_yen")),
            "累計損失": _format_signed_yen(before.get("gross_loss_yen")),
            "純損益": _format_signed_yen(before.get("net_profit_yen")),
            "損益比": _format_profit_loss_ratio(before.get("profit_loss_ratio")),
            "最大損失": _format_signed_yen(before.get("max_loss_yen")),
            "処理秒数": before.get("total_seconds", 0),
        },
        {
            "比較": "B: 除外後",
            "対象銘柄": after.get("symbols_count", 0),
            "仮想買い": after.get("trades", 0),
            "勝率": _format_pct_value(after.get("win_rate_pct")),
            "累計利益": _format_signed_yen(after.get("gross_profit_yen")),
            "累計損失": _format_signed_yen(after.get("gross_loss_yen")),
            "純損益": _format_signed_yen(after.get("net_profit_yen")),
            "損益比": _format_profit_loss_ratio(after.get("profit_loss_ratio")),
            "最大損失": _format_signed_yen(after.get("max_loss_yen")),
            "処理秒数": after.get("total_seconds", 0),
        },
    ]
    before_net = float(before.get("net_profit_yen") or 0)
    after_net = float(after.get("net_profit_yen") or 0)
    before_loss = float(before.get("max_loss_yen") or 0)
    after_loss = float(after.get("max_loss_yen") or 0)
    diff_net = after_net - before_net
    diff_loss = after_loss - before_loss
    rows.append(
        {
            "比較": "差分 B-A",
            "対象銘柄": int(after.get("symbols_count", 0) or 0) - int(before.get("symbols_count", 0) or 0),
            "仮想買い": int(after.get("trades", 0) or 0) - int(before.get("trades", 0) or 0),
            "勝率": _format_pct_value(float(after.get("win_rate_pct") or 0) - float(before.get("win_rate_pct") or 0)),
            "累計利益": _format_signed_yen(float(after.get("gross_profit_yen") or 0) - float(before.get("gross_profit_yen") or 0)),
            "累計損失": _format_signed_yen(float(after.get("gross_loss_yen") or 0) - float(before.get("gross_loss_yen") or 0)),
            "純損益": _format_signed_yen(diff_net),
            "損益比": "-",
            "最大損失": _format_signed_yen(diff_loss),
            "処理秒数": "-",
        }
    )
    st.markdown("**除外前 / 除外後 比較**")
    st.caption("Aは対象銘柄指定だけを適用した結果、Bは対象銘柄指定に加えて除外銘柄も適用した結果です。A側はDBへ保存しません。")
    st.dataframe(_safe_dataframe(pd.DataFrame(rows)), width="stretch", hide_index=True)
    if diff_net > 0 and after_loss >= before_loss:
        st.success("簡易判定: 除外後のほうが純損益は改善しています。最大損失の変化もあわせて確認してください。")
    elif diff_net > 0:
        st.info("簡易判定: 純損益は改善していますが、最大損失が悪化していないか確認してください。")
    elif diff_net < 0:
        st.warning("簡易判定: 除外後は純損益が悪化しています。除外条件を見直す候補です。")
    else:
        st.info("簡易判定: 純損益は同水準です。勝率、損益比、最大損失で比較してください。")


def _render_technical_config_controls(prefix: str, default_enabled: bool = True) -> Dict[str, Any]:
    preset_options = get_technical_preset_options()
    preset_keys = list(preset_options.keys())
    preset_labels = [preset_options[key] for key in preset_keys]
    with st.expander("テクニカル指標設定", expanded=False):
        top_cols = st.columns(4)
        use_technical_score = top_cols[0].toggle(
            "テクニカル採点を使う",
            value=default_enabled,
            key=f"{prefix}_use_technical_score",
            help="ONの場合、過去の各判定時点で見えていた日足だけを使って採点します。",
        )
        preset_label = top_cols[1].selectbox(
            "使用プリセット",
            preset_labels,
            index=0,
            key=f"{prefix}_technical_preset_label",
        )
        selected_preset = preset_keys[preset_labels.index(preset_label)]
        technical_min_score = int(
            top_cols[2].number_input(
                "最小テクニカルスコア",
                min_value=0,
                max_value=60,
                value=36,
                step=1,
                key=f"{prefix}_technical_min_score",
            )
        )
        technical_show_breakdown = top_cols[3].toggle(
            "score_breakdown表示",
            value=True,
            key=f"{prefix}_technical_show_breakdown",
        )

        preset_config = get_technical_config_for_preset(selected_preset)
        key_prefix = f"{prefix}_{selected_preset}"
        st.caption("プリセットを土台に、下の指標グループON/OFFで検証条件を調整できます。")
        cols = st.columns(3)
        technical_config = {
            "preset_name": selected_preset,
            "use_trend": cols[0].checkbox("トレンドを使う", value=bool(preset_config.get("use_trend", True)), key=f"{key_prefix}_tech_use_trend"),
            "use_entry_position": cols[0].checkbox("エントリー位置を使う", value=bool(preset_config.get("use_entry_position", True)), key=f"{key_prefix}_tech_use_entry_position"),
            "use_volume": cols[0].checkbox("出来高を使う", value=bool(preset_config.get("use_volume", True)), key=f"{key_prefix}_tech_use_volume"),
            "use_candle": cols[1].checkbox("ローソク足を使う", value=bool(preset_config.get("use_candle", True)), key=f"{key_prefix}_tech_use_candle"),
            "use_breakout": cols[1].checkbox("節目・ブレイクを使う", value=bool(preset_config.get("use_breakout", True)), key=f"{key_prefix}_tech_use_breakout"),
            "use_momentum": cols[1].checkbox("RSI/MACDを使う", value=bool(preset_config.get("use_momentum", True)), key=f"{key_prefix}_tech_use_momentum"),
            "use_risk_reward": cols[2].checkbox("損切り・リスクリワードを使う", value=bool(preset_config.get("use_risk_reward", True)), key=f"{key_prefix}_tech_use_risk_reward"),
            "use_penalty": cols[2].checkbox("減点条件を使う", value=bool(preset_config.get("use_penalty", True)), key=f"{key_prefix}_tech_use_penalty"),
            "use_hard_filter": cols[2].checkbox("強制見送り条件を使う", value=bool(preset_config.get("use_hard_filter", True)), key=f"{key_prefix}_tech_use_hard_filter"),
        }
        if use_technical_score:
            st.info("この採点は過去リプレイ内だけで使います。OpenAI API、Discord通知、実売買には接続しません。")
        else:
            st.caption("OFFの場合、従来どおりマルチ時間足/リスク条件だけで検証します。")
    return {
        "use_technical_score": bool(use_technical_score),
        "technical_preset": selected_preset,
        "technical_min_score": int(technical_min_score),
        "technical_show_breakdown": bool(technical_show_breakdown),
        "technical_config": technical_config,
    }


def _render_candidate_generation_controls(prefix: str) -> Dict[str, Any]:
    mode_label_to_value = {
        "既存ロジック": CANDIDATE_MODE_EXISTING,
        "テクニカルのみ": CANDIDATE_MODE_TECHNICAL_ONLY,
        "既存ロジック＋テクニカル": CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL,
    }
    cols = st.columns([1.5, 1, 1.3])
    selected_label = cols[0].selectbox(
        "候補発生モード",
        list(mode_label_to_value.keys()),
        index=0,
        key=f"{prefix}_candidate_generation_mode_label",
    )
    show_stage_counts = cols[1].toggle(
        "段階別通過件数を表示",
        value=True,
        key=f"{prefix}_show_stage_counts",
    )
    cols[2].caption("0件時は原因候補を自動表示します。")
    mode = mode_label_to_value[selected_label]
    if mode == CANDIDATE_MODE_TECHNICAL_ONLY:
        st.info(
            "テクニカルのみモードでは、既存の最小スコア、対象、5分足エントリー、"
            "マルチ時間足、日足上位N、使用条件、リスク条件は候補発生に使いません。"
            "その時点までに見えているOHLCVだけでテクニカル採点し、最小テクニカルスコア以上なら仮想買い候補にします。"
        )
    elif mode == CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL:
        st.caption("既存ロジックを通過した候補に、テクニカル採点の最小点・強制見送り条件を追加で掛けます。")
    else:
        st.caption("従来どおり、既存の5分足/マルチ時間足/リスク条件を中心に候補を出します。")
    return {
        "candidate_generation_mode": mode,
        "candidate_generation_mode_label": selected_label,
        "show_stage_counts": bool(show_stage_counts),
    }


def _candidate_mode_label(value: Any) -> str:
    return CANDIDATE_MODE_LABELS.get(str(value or ""), str(value or "-"))


def _render_replay_stage_counts(summary: Dict[str, Any]) -> None:
    stage_counts = summary.get("stage_counts", {})
    if not isinstance(stage_counts, dict) or not stage_counts:
        return
    show_stage_counts = bool(summary.get("show_stage_counts", True))
    zero_reasons = summary.get("zero_trade_reasons", [])
    st.markdown("**候補発生の段階別通過件数**")
    cols = st.columns(5)
    cols[0].metric("候補発生モード", _candidate_mode_label(summary.get("candidate_generation_mode")))
    cols[1].metric("スキャン総回数", stage_counts.get("scan_total_count", 0))
    cols[2].metric("対象銘柄数", stage_counts.get("target_symbols_count", 0))
    cols[3].metric("最終仮想買い", stage_counts.get("final_virtual_buy_count", 0))
    cols[4].metric("最大件数で除外", stage_counts.get("max_trades_excluded_count", 0))
    if zero_reasons and int(stage_counts.get("final_virtual_buy_count", 0) or 0) <= 0:
        st.warning(" / ".join(str(item) for item in zero_reasons))
    if not show_stage_counts:
        return
    rows = [
        ("既存ロジック判定回数", "existing_logic_checked_count"),
        ("既存ロジック通過", "existing_logic_pass_count"),
        ("既存ロジック除外", "existing_logic_reject_count"),
        ("テクニカル判定回数", "technical_score_checked_count"),
        ("テクニカル通過", "technical_score_pass_count"),
        ("テクニカル除外", "technical_score_reject_count"),
        ("強制見送り除外", "hard_filter_reject_count"),
        ("テクニカル採点実行回数", "technical_score_attempt_count"),
        ("テクニカル採点成功回数", "technical_score_success_count"),
        ("最小テクニカルスコア通過", "technical_min_score_pass_count"),
        ("強制見送りで除外", "technical_hard_filter_excluded_count"),
        ("confidence不足で除外", "technical_confidence_excluded_count"),
        ("既存の最小スコア通過", "existing_min_score_pass_count"),
        ("対象フィルター通過", "target_filter_pass_count"),
        ("5分足エントリー通過", "intraday_entry_pass_count"),
        ("使用条件通過", "use_conditions_pass_count"),
        ("リスク条件通過", "risk_filter_pass_count"),
    ]
    table = pd.DataFrame(
        [{"段階": label, "件数": stage_counts.get(key, 0)} for label, key in rows]
    )
    st.dataframe(_safe_dataframe(table), width="stretch", hide_index=True)
    skip_reasons = stage_counts.get("skip_reasons", {})
    if isinstance(skip_reasons, dict) and skip_reasons:
        with st.expander("除外理由の内訳", expanded=False):
            reason_rows = [{"理由": key, "件数": value} for key, value in skip_reasons.items()]
            st.dataframe(_safe_dataframe(pd.DataFrame(reason_rows)), width="stretch", hide_index=True)


def _render_watchlist_historical_scan_tab(watchlist: pd.DataFrame) -> None:
    st.markdown("**watchlist過去スキャン再現**")
    st.caption("market_runner.pyの過去版です。watchlist全体を過去の各時刻でスキャンし、当時の買い候補を仮想買いとして検証します。")
    if watchlist.empty:
        st.warning("watchlist.csvに有効な銘柄がありません。")
        return

    option_cols = st.columns(4)
    max_symbols_label = option_cols[0].selectbox(
        "watchlistから使う最大銘柄数",
        list(HISTORICAL_SCAN_MAX_SYMBOL_OPTIONS.keys()),
        index=1,
        key="historical_scan_max_symbols",
        help="watchlistから読み込む銘柄数の上限です。日足スコア上位Nとは別です。",
    )
    period_label = option_cols[1].selectbox("期間", list(REPLAY_PERIOD_OPTIONS.keys()), index=0, key="historical_scan_period")
    interval_label = option_cols[2].selectbox("時間足", list(REPLAY_INTERVAL_OPTIONS.keys()), index=0, key="historical_scan_interval")
    scan_interval_label = option_cols[3].selectbox("スキャン間隔", list(HISTORICAL_SCAN_INTERVAL_OPTIONS.keys()), index=0, key="historical_scan_scan_interval")

    today = datetime.now().date()
    date_cols = st.columns(4)
    start_date = date_cols[0].date_input("開始日", value=today - timedelta(days=30), key="historical_scan_start_date")
    start_time = date_cols[1].time_input("開始時刻", value=dt_time(9, 0), key="historical_scan_start_time")
    end_date = date_cols[2].date_input("終了日", value=today, key="historical_scan_end_date")
    end_time = date_cols[3].time_input("終了時刻", value=dt_time(15, 30), key="historical_scan_end_time")

    candidate_settings = _render_candidate_generation_controls("historical_scan")
    candidate_mode = candidate_settings["candidate_generation_mode"]

    rule_cols = st.columns(5)
    min_score = int(rule_cols[0].selectbox("最小スコア", [60, 70, 75, 80], index=1, key="historical_scan_min_score"))
    target_label = rule_cols[1].selectbox("対象", list(HISTORICAL_SCAN_TARGET_OPTIONS.keys()), index=0, key="historical_scan_target_mode")
    max_buys = int(rule_cols[2].selectbox("1回の最大仮想買い件数", [1, 3, 5, 10], index=1, key="historical_scan_max_buys"))
    cooldown_label = rule_cols[3].selectbox("同一銘柄の再シグナル抑制", list(HISTORICAL_SCAN_COOLDOWN_OPTIONS.keys()), index=0, key="historical_scan_cooldown")
    shares = int(rule_cols[4].selectbox("想定株数", [100, 200, 300, 500, 1000], index=0, key="historical_scan_shares"))

    mtf_cols = st.columns(5)
    use_multi_timeframe = mtf_cols[0].toggle("マルチ時間足判定を使う", value=True, key="historical_scan_use_multi_timeframe")
    daily_filter_required = mtf_cols[1].toggle("日足フィルター必須", value=True, key="historical_scan_daily_filter_required")
    daily_min_ok = int(mtf_cols[2].selectbox("日足OK数の最低条件", [2, 3, 4], index=1, key="historical_scan_daily_min_ok"))
    intraday_min_ok = int(mtf_cols[3].selectbox("5分足OK数の最低条件", [2, 3, 4], index=0, key="historical_scan_intraday_min_ok"))
    mtf_options = mtf_cols[4].multiselect(
        "使用条件",
        ["VWAP", "出来高急増"],
        default=["VWAP", "出来高急増"],
        key="historical_scan_mtf_options",
    )
    use_vwap = "VWAP" in mtf_options
    use_volume_spike = "出来高急増" in mtf_options

    topn_cols = st.columns(4)
    use_daily_top_n = topn_cols[0].toggle("日足スコア上位N銘柄に絞る", value=True, key="historical_scan_use_daily_top_n")
    daily_top_n = int(topn_cols[1].selectbox("日足スコア上位N", DAILY_TOP_N_OPTIONS, index=1, key="historical_scan_daily_top_n"))
    min_daily_score = int(topn_cols[2].number_input("日足スコアの最低点", min_value=0, max_value=100, value=70, step=1, key="historical_scan_min_daily_score"))
    topn_cols[3].metric("上位抽出後の銘柄数", f"最大{daily_top_n}銘柄/回" if use_daily_top_n else "制限なし")
    if candidate_mode == CANDIDATE_MODE_TECHNICAL_ONLY:
        st.caption("処理順: watchlist読み込み → 対象/除外フィルター → watchlist上限 → 各時刻のテクニカル採点 → 最小テクニカルスコア → 最大仮想買い件数")
    else:
        st.caption("処理順: watchlist読み込み → 対象/除外フィルター → watchlist上限 → 各時刻の日足スコア上位N → 5分足エントリー → リスク条件 → 最大仮想買い件数")

    risk_cols = st.columns(4)
    use_risk_filter = risk_cols[0].toggle("リスク条件を使う", value=True, key="historical_scan_use_risk_filter")
    max_stop_loss_pct = float(
        risk_cols[1].number_input("最大損切り幅 %", min_value=0.1, max_value=20.0, value=3.0, step=0.1, key="historical_scan_max_stop_loss_pct")
    )
    max_loss_yen_limit = float(
        risk_cols[2].number_input("最大損失 円", min_value=1000, max_value=500000, value=20000, step=1000, key="historical_scan_max_loss_yen_limit")
    )
    min_risk_reward = float(
        risk_cols[3].number_input("最低損益比", min_value=0.1, max_value=10.0, value=1.2, step=0.1, key="historical_scan_min_risk_reward")
    )
    technical_settings = _render_technical_config_controls("historical_scan", default_enabled=True)

    cache_now = replay_cache_stats()
    cache_cols = st.columns(5)
    use_cache = cache_cols[0].toggle("キャッシュを使う", value=True, key="historical_scan_use_cache")
    parallel = cache_cols[1].toggle("並列処理", value=True, key="historical_scan_parallel")
    workers = int(cache_cols[2].selectbox("workers数", [1, 2, 3, 4], index=1, key="historical_scan_workers"))
    cache_cols[3].metric("採点cache", cache_now.get("technical_score_cache_count", 0))
    if cache_cols[4].button("キャッシュ削除", key="historical_scan_clear_cache"):
        cleared_score = clear_replay_cache()
        cleared_data = clear_historical_cache()
        st.session_state["historical_scan_cache_clear_result"] = {
            "technical": cleared_score,
            "historical": cleared_data,
        }
    clear_result = st.session_state.get("historical_scan_cache_clear_result")
    if clear_result:
        st.caption(
            f"キャッシュ削除: 採点 {clear_result.get('technical', {}).get('deleted_technical_score_rows', 0)}件 / "
            f"過去データCSV {clear_result.get('historical', {}).get('deleted_files', 0)}件"
        )

    filter_cols = st.columns(2)
    include_symbols_text = filter_cols[0].text_area(
        "対象銘柄のみ",
        value="",
        placeholder="例: 6501.T,6723.T,6752.T\n空欄ならwatchlist全体",
        height=84,
        key="historical_scan_include_symbols_text",
    )
    exclude_symbols_text = filter_cols[1].text_area(
        "除外銘柄",
        value="",
        placeholder="例: 6501.T,7751.T,7735.T\n空欄なら除外なし",
        height=84,
        key="historical_scan_exclude_symbols_text",
    )
    include_symbols = parse_symbol_input(include_symbols_text)
    exclude_symbols = parse_symbol_input(exclude_symbols_text)
    compare_exclude_mode = st.toggle(
        "除外前/除外後を比較する",
        value=False,
        key="historical_scan_compare_exclude_mode",
        help="A=対象銘柄指定だけ、B=対象銘柄指定+除外銘柄で同じ条件を実行して比較します。A側はDB保存しません。",
    )
    baseline_filter_result = filter_watchlist_symbols(watchlist, include_symbols=include_symbols, exclude_symbols=[])
    filter_result = filter_watchlist_symbols(watchlist, include_symbols=include_symbols, exclude_symbols=exclude_symbols)
    baseline_watchlist = baseline_filter_result.get("filtered_watchlist", pd.DataFrame())
    filtered_watchlist = filter_result.get("filtered_watchlist", pd.DataFrame())
    max_symbols = HISTORICAL_SCAN_MAX_SYMBOL_OPTIONS[max_symbols_label]
    if max_symbols is None:
        baseline_limited_watchlist = baseline_watchlist
        limited_watchlist = filtered_watchlist
    else:
        baseline_limited_watchlist = baseline_watchlist.head(int(max_symbols))
        limited_watchlist = filtered_watchlist.head(int(max_symbols))
    baseline_target_records = baseline_limited_watchlist.to_dict("records")
    target_records = limited_watchlist.to_dict("records")
    _render_watchlist_filter_status(filter_result, len(target_records))
    if compare_exclude_mode:
        st.caption(f"比較A（除外前）: {len(baseline_target_records)}銘柄 / 比較B（除外後）: {len(target_records)}銘柄")
    if int(filter_result.get("final_count", 0)) <= 0:
        st.error("対象銘柄が0件です。対象銘柄のみ/除外銘柄の入力を確認してください。")
    elif len(target_records) < int(filter_result.get("final_count", 0)):
        st.caption(f"最大対象銘柄数により、除外後{filter_result.get('final_count', 0)}銘柄のうち{len(target_records)}銘柄を検証します。")
    if len(target_records) > 30 and REPLAY_INTERVAL_OPTIONS[interval_label] == "5m":
        st.warning("5分足で30銘柄超は時間がかかります。まずは10〜30銘柄での確認を推奨します。")

    start_at = _combine_date_time(start_date, start_time)
    end_at = _combine_date_time(end_date, end_time)
    run_disabled = start_at >= end_at or not target_records
    if run_disabled:
        if start_at >= end_at:
            st.warning("開始日時は終了日時より前にしてください。")

    if st.button("過去スキャン再現を実行", key="run_historical_scan_replay_button", type="primary", disabled=run_disabled):
        progress_bar = st.progress(0)
        status_box = st.empty()

        progress_state = {"last_update": 0.0, "stage": "", "started": time.perf_counter()}

        def _progress(stage: str, current: int, total: int, label: str) -> None:
            now_perf = time.perf_counter()
            should_update = (
                stage != progress_state["stage"]
                or current >= total
                or now_perf - progress_state["last_update"] >= 0.3
            )
            if not should_update:
                return
            progress_state["stage"] = stage
            progress_state["last_update"] = now_perf
            pct = int(min(100, max(0, current / max(1, total) * 100)))
            elapsed = max(0.0, now_perf - float(progress_state["started"]))
            eta = "-"
            if current > 0 and total > current:
                eta_seconds = elapsed / current * (total - current)
                eta = f"{eta_seconds:.0f}秒"
            progress_bar.progress(pct)
            status_box.info(f"{stage} {current}/{total} {label} / 経過 {elapsed:.1f}秒 / ETA {eta}")

        with st.spinner("watchlist過去スキャン再現を実行しています..."):
            effective_use_technical_score = candidate_mode in {
                CANDIDATE_MODE_TECHNICAL_ONLY,
                CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL,
            }
            config = HistoricalScanReplayConfig(
                candidate_generation_mode=candidate_mode,
                period=REPLAY_PERIOD_OPTIONS[period_label],
                interval=REPLAY_INTERVAL_OPTIONS[interval_label],
                start_at=start_at,
                end_at=end_at,
                scan_interval=HISTORICAL_SCAN_INTERVAL_OPTIONS[scan_interval_label],
                min_score=min_score,
                target_mode=HISTORICAL_SCAN_TARGET_OPTIONS[target_label],
                max_buys_per_scan=max_buys,
                cooldown=HISTORICAL_SCAN_COOLDOWN_OPTIONS[cooldown_label],
                shares=shares,
                max_symbols=max_symbols,
                source_watchlist="watchlist.csv",
                use_multi_timeframe=bool(use_multi_timeframe),
                daily_filter_required=bool(daily_filter_required),
                daily_min_ok=int(daily_min_ok),
                intraday_min_ok=int(intraday_min_ok),
                use_vwap=bool(use_vwap),
                use_volume_spike=bool(use_volume_spike),
                use_risk_filter=bool(use_risk_filter),
                max_stop_loss_pct=max_stop_loss_pct,
                max_loss_yen_limit=max_loss_yen_limit,
                min_risk_reward=min_risk_reward,
                use_daily_top_n=bool(use_daily_top_n),
                daily_top_n=int(daily_top_n),
                min_daily_score=int(min_daily_score),
                use_technical_score=effective_use_technical_score,
                technical_preset=technical_settings["technical_preset"],
                technical_min_score=technical_settings["technical_min_score"],
                technical_show_breakdown=technical_settings["technical_show_breakdown"],
                technical_config=technical_settings["technical_config"],
                use_replay_cache=bool(use_cache),
                parallel=bool(parallel),
                max_workers=int(workers),
            )
            ab_comparison = {}
            if compare_exclude_mode and baseline_target_records:
                status_box.info("比較A（除外前）を実行しています。A側はDB保存しません。")
                before_result = run_historical_scan_replay(baseline_target_records, config, progress_callback=_progress)
                ab_comparison["before_exclude"] = _historical_scan_compare_payload(before_result)
                progress_state["stage"] = ""
                progress_state["last_update"] = 0.0
                status_box.info("比較B（除外後）を実行しています。")
            result = run_historical_scan_replay(target_records, config, progress_callback=_progress)
            if ab_comparison:
                ab_comparison["after_exclude"] = _historical_scan_compare_payload(result)
                result["ab_comparison"] = ab_comparison
            watchlist_filter_summary = {
                "include_symbols_text": include_symbols_text,
                "exclude_symbols_text": exclude_symbols_text,
                "include_symbols": include_symbols,
                "exclude_symbols": exclude_symbols,
                "excluded_symbols": filter_result.get("excluded_symbols", []),
                "excluded_labels": filter_result.get("excluded_labels", []),
                "missing_symbols": filter_result.get("missing_symbols", []),
                "original_count": filter_result.get("original_count", 0),
                "included_count": filter_result.get("included_count", 0),
                "final_symbols_count": filter_result.get("final_count", 0),
                "target_records_count": len(target_records),
                "compare_exclude_mode": bool(compare_exclude_mode),
                "baseline_target_records_count": len(baseline_target_records),
            }
            result["summary"]["watchlist_filter"] = watchlist_filter_summary
            result["summary"]["include_symbols_text"] = include_symbols_text
            result["summary"]["exclude_symbols_text"] = exclude_symbols_text
            result["summary"]["target_symbols"] = ",".join(
                str(record.get("normalized_symbol") or normalize_jp_symbol(record.get("code")) or "")
                for record in target_records
            )
            result["summary"]["excluded_symbols"] = ",".join(filter_result.get("excluded_symbols", []))
            result["summary"]["final_symbols_count"] = filter_result.get("final_count", 0)
            result["summary"]["target_records_count"] = len(target_records)
            result["summary"]["missing_symbols"] = filter_result.get("missing_symbols", [])
            result["summary"]["show_stage_counts"] = candidate_settings["show_stage_counts"]
            db_save_started = time.perf_counter()
            store_result = insert_replay_trades(
                result["trades"],
                replay_run_id=result["summary"].get("replay_run_id"),
            )
            result["summary"]["db_save_seconds"] = round(time.perf_counter() - db_save_started, 3)
            result["summary"]["total_seconds"] = round(
                float(result["summary"].get("total_seconds", 0) or 0)
                + float(result["summary"].get("db_save_seconds", 0) or 0),
                3,
            )
            result["summary"]["db_saved_count"] = int(store_result.get("saved_count", 0) or 0)
            result = attach_watchlist_replay_exports(result)
            run_record = {
                **result["summary"],
                "created_at_jst": now_jst_iso(),
                "total_profit_yen": result["money_summary"].get("net_profit_yen"),
                "win_rate": result["summary"].get("win_rate_pct"),
                "summary_json": result["summary"],
            }
            insert_replay_run(run_record)
        st.session_state["historical_scan_replay_result"] = result
        st.session_state["historical_scan_replay_store"] = store_result
        status_box.success(f"完了: 仮想買い {len(result['trades'])}件 / DB保存 {store_result['saved_count']}件")

    result = st.session_state.get("historical_scan_replay_result")
    store = st.session_state.get("historical_scan_replay_store", {})
    if not result:
        st.info("まだ過去スキャン再現の結果はありません。条件を設定して実行してください。")
        return

    _render_historical_scan_overall_summary(result)
    _render_batch_condition_summary(result)
    _render_batch_count_summary(result.get("summary", {}))
    _render_replay_stage_counts(result.get("summary", {}))
    _render_historical_scan_ab_comparison(result.get("ab_comparison", {}))
    st.caption(f"DB保存: {store.get('saved_count', 0)}件 / {result.get('lookahead_note', '')}")

    trades = result.get("trades", [])
    detail_export_df = result.get("trade_detail_df", pd.DataFrame())
    summary_export_df = result.get("run_summary_df", pd.DataFrame())
    if isinstance(summary_export_df, pd.DataFrame) and not summary_export_df.empty:
        st.download_button(
            "watchlist検証条件サマリーCSVをダウンロード",
            data=summary_export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="watchlist_replay_run_summary.csv",
            mime="text/csv",
            key="watchlist_replay_download_run_summary",
        )
    if isinstance(detail_export_df, pd.DataFrame) and not detail_export_df.empty:
        st.download_button(
            "watchlistトレード明細CSVをダウンロード",
            data=detail_export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="watchlist_replay_trades.csv",
            mime="text/csv",
            key="watchlist_replay_download_trades",
        )
    symbol_summary = result.get("symbol_summary", pd.DataFrame())
    st.markdown("**グラフ**")
    _render_replay_profit_curve(trades, int(result["summary"].get("shares", shares)))
    _render_symbol_profit_bar(symbol_summary)
    _render_selected_count_chart(result.get("selected_by_scan_time", []))

    st.markdown("**銘柄別サマリー**")
    st.dataframe(_safe_dataframe(_historical_scan_symbol_summary_table(symbol_summary)), width="stretch", hide_index=True)

    st.markdown("**トレード一覧**")
    st.dataframe(
        _safe_dataframe(_historical_scan_trade_table(trades, int(result["summary"].get("shares", shares)))),
        width="stretch",
        hide_index=True,
    )
    _render_multi_timeframe_performance(trades, int(result["summary"].get("shares", shares)))
    _render_daily_top_n_performance(trades, int(result["summary"].get("shares", shares)))
    _render_risk_performance(trades, int(result["summary"].get("shares", shares)))
    _render_technical_performance(trades, int(result["summary"].get("shares", shares)))

    _render_historical_scan_rankings(symbol_summary)

    with st.expander("詳細デバッグ情報", expanded=False):
        failures = result.get("fetch_failures", [])
        if failures:
            st.markdown("**取得失敗銘柄**")
            st.dataframe(_safe_dataframe(pd.DataFrame(failures)), width="stretch", hide_index=True)
        else:
            st.caption("取得失敗銘柄はありません。")
        st.markdown("**取得メタ情報**")
        st.dataframe(_safe_dataframe(pd.DataFrame(result.get("fetch_meta", []))), width="stretch", hide_index=True)
        st.caption(result.get("lookahead_note", ""))


def _batch_context_value(context: Dict[str, Any], key: str, default: str = "-") -> str:
    value = context.get(key, default)
    if value in (None, ""):
        return default
    return str(value)


def _render_batch_condition_summary(result: Dict[str, Any]) -> None:
    context = result.get("run_context", {}) or {}
    condition_rows = result.get("condition_rows", pd.DataFrame())
    if (not isinstance(condition_rows, pd.DataFrame) or condition_rows.empty) and isinstance(result.get("run_summary_df"), pd.DataFrame):
        condition_rows = result.get("run_summary_df", pd.DataFrame())
    presets = "-"
    min_scores = "-"
    modes = "-"
    config_hashes = "-"
    if isinstance(condition_rows, pd.DataFrame) and not condition_rows.empty:
        presets = ", ".join(sorted({str(value) for value in condition_rows.get("preset", []) if str(value)})) or "-"
        min_scores = ", ".join(sorted({str(value) for value in condition_rows.get("min_technical_score", []) if str(value)})) or "-"
        modes = ", ".join(sorted({str(value) for value in condition_rows.get("candidate_mode", []) if str(value)})) or "-"
        config_hashes = ", ".join(sorted({str(value) for value in condition_rows.get("config_hash", []) if str(value)})) or "-"

    st.markdown("**検証条件**")
    cols = st.columns(4)
    cols[0].metric("候補発生モード", modes)
    cols[1].metric("プリセット", presets)
    cols[2].metric("最小テクニカル", min_scores)
    cols[3].metric("想定株数", f"{_batch_context_value(context, 'assumed_shares')}株")
    cols = st.columns(4)
    cols[0].metric("期間", _batch_context_value(context, "period"))
    cols[1].metric("時間足", _batch_context_value(context, "interval"))
    cols[2].metric("最大買い/銘柄", _batch_context_value(context, "max_trades_per_symbol"))
    cols[3].metric("連続抑制", _batch_context_value(context, "cooldown_bars_or_minutes"))
    with st.expander("検証条件の詳細", expanded=False):
        rows = [
            {"項目": "run_id", "値": _batch_context_value(context, "run_id")},
            {"項目": "run_datetime", "値": _batch_context_value(context, "run_datetime")},
            {"項目": "対象銘柄", "値": _batch_context_value(context, "target_symbols")},
            {"項目": "除外銘柄", "値": _batch_context_value(context, "excluded_symbols")},
            {"項目": "watchlist上限", "値": _batch_context_value(context, "watchlist_limit")},
            {"項目": "同一銘柄1ポジション", "値": "ON" if context.get("one_position_per_symbol") else "OFF"},
            {"項目": "キャッシュ", "値": "ON" if context.get("cache_enabled") else "OFF"},
            {"項目": "並列処理", "値": "ON" if context.get("parallel_enabled") else "OFF"},
            {"項目": "workers", "値": _batch_context_value(context, "workers")},
            {"項目": "config_hash", "値": config_hashes},
        ]
        st.dataframe(_safe_dataframe(pd.DataFrame(rows)), width="stretch", hide_index=True)


def _render_batch_count_summary(summary: Dict[str, Any]) -> None:
    st.markdown("**件数サマリー**")
    cols = st.columns(4)
    cols[0].metric("候補シグナル数", summary.get("raw_signal_count", 0))
    cols[1].metric("最終採用トレード数", summary.get("final_trade_count", summary.get("total_trades", 0)))
    cols[2].metric("確定損益対象", summary.get("settled_trade_count", 0))
    cols[3].metric("CSV出力件数", summary.get("csv_export_count", 0))
    cols = st.columns(4)
    cols[0].metric("連続抑制で除外", summary.get("cooldown_filtered_count", 0))
    cols[1].metric("1ポジ制限で除外", summary.get("position_filtered_count", 0))
    cols[2].metric("最大件数で除外", summary.get("max_trade_filtered_count", 0))
    cols[3].metric("DB保存件数", summary.get("db_saved_count", 0))
    cols = st.columns(4)
    cols[0].metric("既存ロジック通過", summary.get("existing_logic_pass_count", 0))
    cols[1].metric("既存ロジック除外", summary.get("existing_logic_reject_count", 0))
    cols[2].metric("テクニカル通過", summary.get("technical_score_pass_count", 0))
    cols[3].metric("強制見送り除外", summary.get("hard_filter_reject_count", 0))
    raw_count = int(summary.get("raw_signal_count", 0) or 0)
    csv_count = int(summary.get("csv_export_count", 0) or 0)
    if raw_count != csv_count:
        st.info(
            "候補シグナル数とCSV明細件数は異なる場合があります。CSVには、連続シグナル抑制、"
            "同一銘柄1ポジション制限、銘柄ごとの最大件数制限などを適用した後の最終採用トレードのみ出力しています。"
        )


def _render_multi_symbol_replay_comparison_tab(watchlist: pd.DataFrame) -> None:
    st.markdown("**複数銘柄比較**")
    st.caption("複数銘柄を同じOHLCVデータでまとめて検証します。OpenAI API、Discord通知、実売買・発注は行いません。")
    if watchlist.empty:
        st.warning("watchlist.csvに有効な銘柄がありません。")
        return

    cache_now = replay_cache_stats()
    cache_cols = st.columns(5)
    use_cache = cache_cols[0].toggle("キャッシュ使用", value=True, key="batch_replay_use_cache")
    if cache_cols[1].button("キャッシュクリア", key="batch_replay_clear_cache"):
        cleared = clear_replay_cache()
        st.session_state["batch_replay_cache_clear_result"] = cleared
    cache_cols[2].metric("採点キャッシュ", cache_now.get("technical_score_cache_count", 0))
    cache_cols[3].metric("hit", cache_now.get("technical_hits", 0))
    cache_cols[4].metric("miss", cache_now.get("technical_misses", 0))
    if st.session_state.get("batch_replay_cache_clear_result"):
        st.caption(f"削除: {st.session_state['batch_replay_cache_clear_result'].get('deleted_technical_score_rows', 0)}件")

    option_cols = st.columns(5)
    max_symbols_label = option_cols[0].selectbox(
        "watchlistから使う最大銘柄数",
        list(HISTORICAL_SCAN_MAX_SYMBOL_OPTIONS.keys()),
        index=1,
        key="batch_replay_max_symbols",
    )
    period_label = option_cols[1].selectbox("期間", list(REPLAY_PERIOD_OPTIONS.keys()), index=0, key="batch_replay_period")
    interval_label = option_cols[2].selectbox("時間足", list(REPLAY_INTERVAL_OPTIONS.keys()), index=0, key="batch_replay_interval")
    shares = int(option_cols[3].selectbox("想定株数", [100, 200, 300, 500, 1000], index=0, key="batch_replay_shares"))
    max_trades = int(option_cols[4].selectbox("銘柄ごとの最大仮想買い", [10, 20, 50, 100], index=2, key="batch_replay_max_trades"))

    filter_cols = st.columns(2)
    include_symbols_text = filter_cols[0].text_area(
        "対象銘柄のみ",
        value="",
        placeholder="例: 8035.T,5803.T,6501.T",
        height=84,
        key="batch_replay_include_symbols_text",
    )
    exclude_symbols_text = filter_cols[1].text_area(
        "除外銘柄",
        value="",
        placeholder="例: 9999.T",
        height=84,
        key="batch_replay_exclude_symbols_text",
    )
    filter_result = filter_watchlist_symbols(
        watchlist,
        include_symbols=parse_symbol_input(include_symbols_text),
        exclude_symbols=parse_symbol_input(exclude_symbols_text),
    )
    filtered_watchlist = filter_result.get("filtered_watchlist", pd.DataFrame())
    max_symbols = HISTORICAL_SCAN_MAX_SYMBOL_OPTIONS[max_symbols_label]
    target_watchlist = filtered_watchlist if max_symbols is None else filtered_watchlist.head(int(max_symbols))
    target_records = target_watchlist.to_dict("records")
    _render_watchlist_filter_status(filter_result, len(target_records))

    preset_options = get_technical_preset_options()
    preset_label_to_key = {label: key for key, label in preset_options.items()}
    condition_cols = st.columns(4)
    mode_labels = condition_cols[0].multiselect(
        "候補発生モード",
        ["テクニカルのみ", "既存ロジック", "既存ロジック＋テクニカル"],
        default=["テクニカルのみ"],
        key="batch_replay_candidate_modes",
    )
    preset_labels = condition_cols[1].multiselect(
        "プリセット",
        list(preset_label_to_key.keys()),
        default=["標準スイング"],
        key="batch_replay_presets",
    )
    min_scores = condition_cols[2].multiselect(
        "最小テクニカルスコア",
        [20, 25, 30, 32, 36, 42],
        default=[32],
        key="batch_replay_min_scores",
    )
    hard_filter_labels = condition_cols[3].multiselect(
        "強制見送り",
        ["ON", "OFF"],
        default=["ON"],
        key="batch_replay_hard_filter_options",
    )
    use_penalty = st.toggle("減点条件を使う", value=True, key="batch_replay_use_penalty")
    with st.expander("候補発生モードの違い", expanded=False):
        st.markdown(
            "- **テクニカルのみ**：既存ロジックを使わず、テクニカル採点だけで候補化します。\n"
            "- **既存ロジック**：従来の5分足/マルチ時間足/リスク条件だけで候補化し、テクニカル採点は必須にしません。\n"
            "- **既存ロジック＋テクニカル**：既存ロジックを通過した候補だけに、テクニカル採点の最小点と強制見送り条件を追加で掛けます。"
        )

    run_cols = st.columns(5)
    one_position = run_cols[0].toggle("同一銘柄1ポジション制限", value=True, key="batch_replay_one_position")
    parallel = run_cols[1].toggle("並列処理", value=False, key="batch_replay_parallel")
    workers = int(run_cols[2].selectbox("workers数", [1, 2, 3, 4], index=1, key="batch_replay_workers"))
    cooldown_bars = int(run_cols[3].selectbox("連続シグナル抑制", [3, 6, 12, 24], index=2, key="batch_replay_cooldown_bars"))
    save_to_db = run_cols[4].toggle("DB保存", value=False, key="batch_replay_save_to_db")

    mode_map = {
        "テクニカルのみ": CANDIDATE_MODE_TECHNICAL_ONLY,
        "既存ロジック": CANDIDATE_MODE_EXISTING,
        "既存ロジック＋テクニカル": CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL,
    }
    variants = build_condition_variants(
        modes=[mode_map[label] for label in mode_labels],
        presets=[preset_label_to_key[label] for label in preset_labels],
        min_scores=[int(score) for score in min_scores],
        hard_filter_options=[label == "ON" for label in hard_filter_labels],
        use_penalty=use_penalty,
    )
    st.caption(f"検証条件数: {len(variants)} / 対象銘柄数: {len(target_records)}")
    run_disabled = not target_records or not variants
    if run_disabled:
        st.warning("対象銘柄または検証条件が0件です。")

    if st.button("複数銘柄比較を実行", key="run_batch_replay_button", type="primary", disabled=run_disabled):
        progress_bar = st.progress(0)
        status_box = st.empty()

        def _progress(stage: str, current: int, total: int, label: str) -> None:
            pct = int(min(100, max(0, current / max(1, total) * 100)))
            progress_bar.progress(pct)
            status_box.info(f"{stage} {current}/{total} {label}")

        with st.spinner("複数銘柄比較を実行しています..."):
            result = run_multi_symbol_replay_comparison(
                target_records,
                BatchReplayConfig(
                    period=REPLAY_PERIOD_OPTIONS[period_label],
                    interval=REPLAY_INTERVAL_OPTIONS[interval_label],
                    max_trades=max_trades,
                    cooldown_bars=cooldown_bars,
                    shares=shares,
                    use_replay_cache=use_cache,
                    parallel=parallel,
                    max_workers=workers,
                    one_position_per_symbol=one_position,
                    variants=variants,
                    target_symbols=",".join(
                        str(record.get("normalized_symbol") or normalize_jp_symbol(record.get("code")) or "")
                        for record in target_records
                    ),
                    excluded_symbols=",".join(parse_symbol_input(exclude_symbols_text)),
                    watchlist_limit=max_symbols_label,
                ),
                progress_callback=_progress,
            )
            store_result = {}
            if save_to_db and result.get("trades"):
                store_result = insert_replay_trades(result["trades"])
            result["store_result"] = store_result
            result.setdefault("summary", {})["db_saved_count"] = int(store_result.get("saved_count", 0) or 0)
            if isinstance(result.get("run_summary_df"), pd.DataFrame) and not result["run_summary_df"].empty:
                saved_count = int(store_result.get("saved_count", 0) or 0)
                result["run_summary_df"]["db_saved_count"] = (
                    result["run_summary_df"]["final_trade_count"] if saved_count else 0
                )
        st.session_state["batch_replay_result"] = result
        status_box.success(f"完了: 仮想買い {len(result.get('trades', []))}件")

    result = st.session_state.get("batch_replay_result")
    if not result:
        st.info("まだ複数銘柄比較の結果はありません。条件を設定して実行してください。")
        return

    summary = result.get("summary", {})
    _render_batch_condition_summary(result)
    st.markdown("**比較サマリー**")
    cols = st.columns(6)
    cols[0].metric("対象銘柄", summary.get("symbols_count", 0))
    cols[1].metric("取得成功", summary.get("success_count", 0))
    cols[2].metric("取得失敗", summary.get("error_count", 0))
    cols[3].metric("処理秒", summary.get("elapsed_seconds", 0))
    cache = summary.get("cache", {})
    cols[4].metric("cache hit率", f"{cache.get('technical_hit_rate_pct', 0)}%")
    cols[5].metric("短縮秒", cache.get("technical_saved_seconds", 0))
    _render_batch_count_summary(summary)

    symbol_rows = result.get("symbol_rows", pd.DataFrame())
    condition_rows = result.get("condition_rows", pd.DataFrame())
    sort_key = st.selectbox(
        "銘柄別サマリーの並び替え",
        ["純損益", "勝率", "仮想買い件数", "最大連敗", "平均損益", "処理秒数"],
        index=0,
        key="batch_replay_sort_key",
    )
    if isinstance(symbol_rows, pd.DataFrame) and not symbol_rows.empty:
        ascending = sort_key in {"最大連敗", "処理秒数"}
        st.markdown("**銘柄別サマリー**")
        st.dataframe(_safe_dataframe(symbol_rows.sort_values(sort_key, ascending=ascending)), width="stretch", hide_index=True)
    if isinstance(condition_rows, pd.DataFrame) and not condition_rows.empty:
        st.markdown("**条件別サマリー**")
        st.dataframe(_safe_dataframe(condition_rows.sort_values("純損益", ascending=False)), width="stretch", hide_index=True)

    trades = result.get("trades", [])
    detail_export_df = result.get("trade_detail_df", pd.DataFrame())
    summary_export_df = result.get("run_summary_df", pd.DataFrame())
    if isinstance(summary_export_df, pd.DataFrame) and not summary_export_df.empty:
        st.download_button(
            "検証条件サマリーCSVをダウンロード",
            data=summary_export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="batch_replay_run_summary.csv",
            mime="text/csv",
            key="batch_replay_download_run_summary",
        )
    if trades:
        csv_df = detail_export_df if isinstance(detail_export_df, pd.DataFrame) and not detail_export_df.empty else _historical_scan_trade_table(trades, shares)
        st.download_button(
            "トレード明細CSVをダウンロード",
            data=csv_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="batch_replay_trades.csv",
            mime="text/csv",
            key="batch_replay_download_trades",
        )
        with st.expander("詳細トレード一覧", expanded=False):
            st.dataframe(_safe_dataframe(csv_df), width="stretch", hide_index=True)
    errors = result.get("errors", [])
    if errors:
        with st.expander("途中エラー一覧", expanded=False):
            st.dataframe(_safe_dataframe(pd.DataFrame(errors)), width="stretch", hide_index=True)


def _render_historical_replay_tab(watchlist: pd.DataFrame) -> None:
    st.subheader("過去リプレイ検証")
    st.caption("過去データを1本ずつ進め、その時点までの情報だけでルール買いを検証します。実売買・発注・OpenAI API呼び出しは行いません。")
    replay_mode = st.radio(
        "モード",
        ["単一銘柄リプレイ", "watchlist過去スキャン再現", "複数銘柄比較"],
        horizontal=True,
        key="historical_replay_mode",
    )
    if replay_mode == "watchlist過去スキャン再現":
        _render_watchlist_historical_scan_tab(watchlist)
        return
    if replay_mode == "複数銘柄比較":
        _render_multi_symbol_replay_comparison_tab(watchlist)
        return

    input_cols = st.columns([1.1, 1.1, 1, 1])
    symbol = input_cols[0].text_input("銘柄コード", value="5803", key="replay_symbol_input")
    name = input_cols[1].text_input("銘柄名（任意）", value="", key="replay_name_input")
    period_label = input_cols[2].selectbox("期間", list(REPLAY_PERIOD_OPTIONS.keys()), index=0, key="replay_period_label")
    interval_label = input_cols[3].selectbox("時間足", list(REPLAY_INTERVAL_OPTIONS.keys()), index=0, key="replay_interval_label")
    period = REPLAY_PERIOD_OPTIONS[period_label]
    interval = REPLAY_INTERVAL_OPTIONS[interval_label]

    fetch_cols = st.columns([1, 3])
    if fetch_cols[0].button("過去データ取得", key="fetch_historical_replay_data_button", type="primary"):
        with st.spinner("過去データを取得しています..."):
            result = load_or_fetch_historical_data(symbol, period, interval)
            daily_result = load_or_fetch_historical_data(symbol, "6mo", "1d")
        validation = validate_historical_ohlcv(result.data)
        daily_validation = validate_historical_ohlcv(daily_result.data)
        st.session_state["replay_raw_data"] = result.data
        st.session_state["replay_data"] = validation.get("cleaned_data", result.data)
        st.session_state["replay_daily_data"] = daily_validation.get("cleaned_data", daily_result.data)
        st.session_state["replay_data_validation"] = {
            key: value
            for key, value in validation.items()
            if key != "cleaned_data"
        }
        st.session_state["replay_fetch_meta"] = _historical_result_to_meta(result, period_label, interval_label)
        st.session_state["replay_last_error"] = result.error_message or ""

    fetch_cols[1].caption("無料データでは時間足と期間の組み合わせに制限があります。取得できない場合もアプリは落ちません。")

    replay_data = st.session_state.get("replay_data", pd.DataFrame())
    fetch_meta = st.session_state.get("replay_fetch_meta", {})
    _render_replay_fetch_summary(fetch_meta)
    _render_replay_anomaly_summary(st.session_state.get("replay_data_validation", {}))

    if replay_data is None or replay_data.empty:
        st.info("まだ過去データがありません。まず「過去データ取得」を押してください。")
        st.button("リプレイ検証を実行", key="run_historical_replay_button", type="primary", disabled=True)
        recent = load_replay_trades(limit=30)
        if not recent.empty:
            with st.expander("保存済みリプレイ結果", expanded=False):
                st.dataframe(_safe_dataframe(_replay_db_display_table(recent)), width="stretch", hide_index=True)
        return

    first_dt = pd.Timestamp(replay_data.index[0]).to_pydatetime()
    last_dt = pd.Timestamp(replay_data.index[-1]).to_pydatetime()
    range_cols = st.columns(4)
    start_date = range_cols[0].date_input("開始日", value=first_dt.date(), key="replay_start_date")
    start_time = range_cols[1].time_input("開始時刻", value=first_dt.time().replace(second=0, microsecond=0), key="replay_start_time")
    end_date = range_cols[2].date_input("終了日", value=last_dt.date(), key="replay_end_date")
    end_time = range_cols[3].time_input("終了時刻", value=last_dt.time().replace(second=0, microsecond=0), key="replay_end_time")

    candidate_settings = _render_candidate_generation_controls("single_replay")
    candidate_mode = candidate_settings["candidate_generation_mode"]

    rule_cols = st.columns(5)
    min_score = rule_cols[0].selectbox("最小スコア", [60, 70, 75, 80], index=1, key="replay_min_score")
    target_rule_label = rule_cols[1].selectbox("対象ルール", REPLAY_RULE_OPTIONS, index=0, key="replay_target_rule")
    max_trades = rule_cols[2].selectbox("最大仮想買い件数", [1, 3, 5, 10, 20, 50], index=3, key="replay_max_trades")
    cooldown_bars = rule_cols[3].selectbox("連続シグナル抑制", [3, 6, 12, 24], index=2, key="replay_cooldown_bars")
    replay_shares = int(rule_cols[4].selectbox("想定株数", [100, 200, 300, 500, 1000], index=0, key="replay_shares"))

    mtf_cols = st.columns(5)
    use_multi_timeframe = mtf_cols[0].toggle("マルチ時間足判定", value=True, key="replay_use_multi_timeframe")
    daily_filter_required = mtf_cols[1].toggle("日足フィルター必須", value=True, key="replay_daily_filter_required")
    daily_min_ok = int(mtf_cols[2].selectbox("日足OK数", [2, 3, 4], index=1, key="replay_daily_min_ok"))
    intraday_min_ok = int(mtf_cols[3].selectbox("5分足OK数", [2, 3, 4], index=0, key="replay_intraday_min_ok"))
    option_pack = mtf_cols[4].multiselect(
        "使用条件",
        ["VWAP", "出来高急増"],
        default=["VWAP", "出来高急増"],
        key="replay_mtf_options",
    )
    use_vwap = "VWAP" in option_pack
    use_volume_spike = "出来高急増" in option_pack

    risk_cols = st.columns(4)
    replay_use_risk_filter = risk_cols[0].toggle("リスク条件を使う", value=True, key="replay_use_risk_filter")
    replay_max_stop_loss_pct = float(
        risk_cols[1].number_input("最大損切り幅 %", min_value=0.1, max_value=20.0, value=3.0, step=0.1, key="replay_max_stop_loss_pct")
    )
    replay_max_loss_yen_limit = float(
        risk_cols[2].number_input("最大損失 円", min_value=1000, max_value=500000, value=20000, step=1000, key="replay_max_loss_yen_limit")
    )
    replay_min_risk_reward = float(
        risk_cols[3].number_input("最低損益比", min_value=0.1, max_value=10.0, value=1.2, step=0.1, key="replay_min_risk_reward")
    )
    technical_settings = _render_technical_config_controls("single_replay", default_enabled=True)

    start_at = _combine_date_time(start_date, start_time)
    end_at = _combine_date_time(end_date, end_time)
    run_disabled = start_at >= end_at
    if run_disabled:
        st.warning("開始日時は終了日時より前にしてください。")

    if st.button("リプレイ検証を実行", key="run_historical_replay_button", type="primary", disabled=run_disabled):
        with st.spinner("過去リプレイ検証を実行しています..."):
            effective_use_technical_score = candidate_mode in {
                CANDIDATE_MODE_TECHNICAL_ONLY,
                CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL,
            }
            result = run_replay(
                symbol=normalize_jp_symbol(symbol),
                name=name,
                df=replay_data,
                daily_df=st.session_state.get("replay_daily_data", pd.DataFrame()),
                start_at=start_at,
                end_at=end_at,
                rule_config={
                    "candidate_generation_mode": candidate_mode,
                    "min_score": int(min_score),
                    "target_rule": REPLAY_RULE_LABEL_TO_SIGNAL.get(target_rule_label, "すべて"),
                    "max_trades": int(max_trades),
                    "interval": interval,
                    "cooldown_bars": int(cooldown_bars),
                    "debug": True,
                    "use_multi_timeframe": bool(use_multi_timeframe),
                    "daily_min_ok": int(daily_min_ok) if daily_filter_required else 0,
                    "intraday_min_ok": int(intraday_min_ok),
                    "use_vwap": bool(use_vwap),
                    "use_volume_spike": bool(use_volume_spike),
                    "shares": int(replay_shares),
                    "use_risk_filter": bool(replay_use_risk_filter),
                    "max_stop_loss_pct": replay_max_stop_loss_pct,
                    "max_loss_yen_limit": replay_max_loss_yen_limit,
                    "min_risk_reward": replay_min_risk_reward,
                    "use_technical_score": effective_use_technical_score,
                    "technical_preset": technical_settings["technical_preset"],
                    "technical_min_score": technical_settings["technical_min_score"],
                    "technical_show_breakdown": technical_settings["technical_show_breakdown"],
                    "technical_config": technical_settings["technical_config"],
                },
            )
            result["summary"]["show_stage_counts"] = candidate_settings["show_stage_counts"]
            result["trades"] = apply_replay_money_metrics(result["trades"], replay_shares)
            result["money_summary"] = summarize_replay_money(result["trades"], replay_shares)
            store_result = insert_replay_trades(result["trades"])
        st.session_state["replay_last_result"] = result
        st.session_state["replay_last_store_result"] = store_result
        st.session_state["replay_last_shares"] = replay_shares
        st.success(f"リプレイ検証を完了しました。仮想買い {len(result['trades'])}件 / DB保存 {store_result['saved_count']}件")

    last_result = st.session_state.get("replay_last_result", {})
    last_store = st.session_state.get("replay_last_store_result", {})
    if last_result:
        summary = last_result.get("summary", {})
        display_shares = int(st.session_state.get("replay_last_shares", replay_shares))
        trades = apply_replay_money_metrics(last_result.get("trades", []), display_shares)
        money_summary = summarize_replay_money(trades, display_shares)
        _render_replay_summary(summary, int(last_store.get("saved_count", 0)))
        _render_replay_stage_counts(summary)
        _render_replay_money_summary(money_summary)
        _replay_extreme_result_warnings(trades)
        _render_replay_profit_curve(trades, display_shares)
        _render_replay_chart(replay_data, trades)
        st.markdown("**リプレイ結果表**")
        if trades:
            st.dataframe(_safe_dataframe(_replay_result_table(trades, display_shares)), width="stretch", hide_index=True)
            _render_multi_timeframe_performance(trades, display_shares)
            _render_risk_performance(trades, display_shares)
            _render_technical_performance(trades, display_shares)
        else:
            st.info("条件に一致する仮想買いポイントはありませんでした。")

        with st.expander("詳細デバッグ情報", expanded=False):
            debug_df = _replay_debug_table(trades)
            if debug_df.empty:
                st.info("デバッグ情報はありません。")
            else:
                st.dataframe(_safe_dataframe(debug_df), width="stretch", hide_index=True)
            validation = st.session_state.get("replay_data_validation", {})
            anomalies = validation.get("anomalies") or []
            st.caption(f"除外バー件数: {validation.get('excluded_count', 0)}")
            if anomalies:
                st.dataframe(_safe_dataframe(pd.DataFrame(anomalies)), width="stretch", hide_index=True)
            else:
                st.caption("異常バー詳細はありません。")

    with st.expander("保存済みリプレイ結果", expanded=False):
        recent = load_replay_trades(limit=100)
        if recent.empty:
            st.info("保存済みのリプレイ結果はありません。")
        else:
            st.dataframe(_safe_dataframe(_replay_db_display_table(recent)), width="stretch", hide_index=True)
        st.caption(f"保存先: {REPLAY_TRADES_DB_PATH}")


def _render_virtual_performance_tab() -> None:
    st.subheader("仮想成績")
    st.caption("フォワード検証で保存した仮想ログを後追いで検証します。実売買の履歴とは完全に分離しています。")
    if st.button("仮想成績を更新"):
        summary = update_open_virtual_trade_outcomes()
        st.info(f"確認 {summary['checked']}件 / 更新 {summary['updated']}件 / スキップ {summary['skipped']}件")

    trades = load_virtual_trades(limit=1000)
    if trades.empty:
        st.info("まだ仮想取引データはありません。")
        return

    returns = pd.to_numeric(trades["return_pct"], errors="coerce").dropna()
    cols = st.columns(4)
    cols[0].metric("総ログ", len(trades))
    cols[1].metric("open", int((trades["status"] == "open").sum()))
    cols[2].metric("評価済み", len(returns))
    cols[3].metric("平均リターン", f"{returns.mean():.2f}%" if not returns.empty else "-")

    with st.expander("仮想取引一覧", expanded=False):
        show_cols = [
            "timestamp_jst",
            "timestamp",
            "symbol",
            "name",
            "decision",
            "entry_type",
            "judge_source",
            "model_used",
            "is_ai_generated",
            "fallback_reason",
            "fallback_error_type",
            "fallback_error_message",
            "entry_price",
            "return_pct",
            "max_profit_pct",
            "max_drawdown_pct",
            "outcome",
            "outcome_updated_at_jst",
            "status",
        ]
        st.dataframe(_safe_dataframe(rows_to_display(trades)[[col for col in show_cols if col in trades.columns]]), width="stretch", hide_index=True)


def _render_pattern_stats_tab() -> None:
    st.subheader("パターン別勝率")
    st.caption("サンプル数が30未満は仮説、50未満は参考、50以上で信頼度を表示します。")
    stats = calculate_pattern_stats()
    sections = [
        ("entry_type別", stats.get("by_entry_type", pd.DataFrame())),
        ("symbol別", stats.get("by_symbol", pd.DataFrame())),
        ("地合い別", stats.get("by_market", pd.DataFrame())),
    ]
    for title, df in sections:
        with st.expander(title, expanded=title == "entry_type別"):
            if df.empty:
                st.info("評価済みサンプルがまだありません。")
            else:
                st.dataframe(_safe_dataframe(df), width="stretch", hide_index=True)


def _render_stock_personality_tab() -> None:
    st.subheader("銘柄別クセ")
    st.caption("仮想取引の検証結果から、得意パターン・苦手パターン・最適保有目安をまとめます。")
    personalities = generate_stock_personalities()
    if personalities.empty:
        st.info("銘柄別クセを出すには、評価済みの仮想取引サンプルが必要です。")
        return
    for row in personalities.to_dict("records"):
        title = (
            f"{row.get('symbol')} {row.get('name', '')}"
            f"｜期待値 {row.get('expected_value_pct')}%"
            f"｜{row.get('reliability')}"
        )
        with st.expander(title, expanded=False):
            cols = st.columns(4)
            cols[0].metric("サンプル", row.get("sample_count"))
            cols[1].metric("勝率", f"{row.get('win_rate_pct')}%")
            cols[2].metric("期待値", f"{row.get('expected_value_pct')}%")
            cols[3].metric("平均逆行", f"{row.get('avg_drawdown_pct')}%")
            st.markdown(f"**得意パターン**：{row.get('good_patterns')}")
            st.markdown(f"**苦手パターン**：{row.get('weak_patterns')}")
            st.markdown(f"**最適保有目安**：{row.get('best_hold_period')}")
            st.markdown(f"**注意点**：{row.get('caution')}")


def _join_reason_text(values: Any) -> str:
    if values in (None, ""):
        return "-"
    if isinstance(values, str):
        return values or "-"
    if isinstance(values, (list, tuple, set)):
        return " / ".join(str(value) for value in values if str(value)) or "-"
    return str(values)


def _technical_breakdown_text(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    parts = []
    for section, reasons in value.items():
        if not reasons:
            continue
        if isinstance(reasons, str):
            reason_text = reasons
        elif isinstance(reasons, (list, tuple, set)):
            reason_text = "、".join(str(item) for item in reasons if str(item))
        else:
            reason_text = str(reasons)
        if reason_text:
            parts.append(f"{section}: {reason_text}")
    return " / ".join(parts) if parts else "-"


def _technical_score_display_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "順位",
        "code",
        "name",
        "現在値",
        "総合点",
        "素点",
        "減点",
        "判定",
        "確度",
        "トレンド",
        "位置",
        "出来高",
        "ローソク",
        "ブレイク",
        "RSI/MACD",
        "R/R点",
        "損切り候補",
        "損切り幅",
        "利確候補",
        "R/R",
        "保有目安",
        "買いタイミング",
        "減点理由",
        "ハード除外",
        "不足データ",
        "更新時刻",
    ]
    rows = []
    for rank, row in enumerate(results, start=1):
        rows.append(
            {
                "順位": rank,
                "code": row.get("code", ""),
                "name": row.get("name", ""),
                "現在値": format_yen(row.get("close")),
                "総合点": f"{row.get('technical_score_final', 0)}/60",
                "素点": row.get("technical_score_raw", 0),
                "減点": row.get("penalty_score", 0),
                "判定": row.get("technical_judgement", "-"),
                "確度": f"{row.get('confidence', 0)}%",
                "トレンド": f"{row.get('trend_score', 0)}/14",
                "位置": f"{row.get('entry_position_score', 0)}/10",
                "出来高": f"{row.get('volume_score', 0)}/10",
                "ローソク": f"{row.get('candle_score', 0)}/8",
                "ブレイク": f"{row.get('breakout_score', 0)}/8",
                "RSI/MACD": f"{row.get('momentum_score', 0)}/6",
                "R/R点": f"{row.get('risk_reward_score', 0)}/4",
                "損切り候補": format_yen(row.get("stop_loss_candidate")),
                "損切り幅": _format_plain_pct(row.get("stop_loss_distance_pct")),
                "利確候補": format_yen(row.get("target_price_candidate")),
                "R/R": _format_profit_loss_ratio(row.get("risk_reward")),
                "保有目安": row.get("hold_days_hint", "-"),
                "買いタイミング": row.get("buy_timing_hint", "-"),
                "減点理由": _join_reason_text(row.get("penalty_reasons")),
                "ハード除外": _join_reason_text(row.get("hard_filter_reason")),
                "不足データ": _join_reason_text(row.get("missing_data")),
                "更新時刻": row.get("last_updated_at", "-"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _daily_fetch_error_table(errors: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "code",
        "name",
        "normalized_symbol",
        "fetch_target",
        "error_type",
        "error_message",
        "last_attempt_at",
    ]
    rows = []
    for error in errors:
        rows.append(
            {
                "code": str(error.get("code", "") or "-"),
                "name": str(error.get("name", "") or "-"),
                "normalized_symbol": str(error.get("normalized_symbol", "") or "-"),
                "fetch_target": str(error.get("fetch_target", "") or "daily_technical"),
                "error_type": str(error.get("error_type", "") or "unknown_error"),
                "error_message": str(error.get("error_message", "") or "エラー理由が空でした。"),
                "last_attempt_at": str(error.get("last_attempt_at", "") or now_jst_display()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _technical_score_breakdown_table(result: Dict[str, Any]) -> pd.DataFrame:
    labels = {
        "trend": "A トレンド",
        "entry_position": "B エントリー位置",
        "volume": "C 出来高",
        "candle": "D ローソク足",
        "breakout": "E ブレイク",
        "momentum": "F RSI/MACD",
        "risk_reward": "G リスク/リワード",
        "penalty": "減点",
    }
    points = {
        "trend": f"{result.get('trend_score', 0)}/14",
        "entry_position": f"{result.get('entry_position_score', 0)}/10",
        "volume": f"{result.get('volume_score', 0)}/10",
        "candle": f"{result.get('candle_score', 0)}/8",
        "breakout": f"{result.get('breakout_score', 0)}/8",
        "momentum": f"{result.get('momentum_score', 0)}/6",
        "risk_reward": f"{result.get('risk_reward_score', 0)}/4",
        "penalty": result.get("penalty_score", 0),
    }
    breakdown = result.get("score_breakdown") or {}
    return pd.DataFrame(
        [
            {
                "項目": label,
                "点数": points.get(key, "-"),
                "理由": _join_reason_text(breakdown.get(key)),
            }
            for key, label in labels.items()
        ]
    )


def _technical_latest_metrics_table(result: Dict[str, Any]) -> pd.DataFrame:
    metrics = result.get("latest_metrics") or {}
    rows = [
        ("終値", format_yen(metrics.get("close"))),
        ("前日比", _format_plain_pct(metrics.get("daily_change_pct"))),
        ("ギャップ", _format_plain_pct(metrics.get("gap_pct"))),
        ("5日線", format_yen(metrics.get("ma5"))),
        ("25日線", format_yen(metrics.get("ma25"))),
        ("75日線", format_yen(metrics.get("ma75"))),
        ("25日線乖離", _format_plain_pct(metrics.get("distance_from_ma25_pct"))),
        ("出来高20日比", _format_profit_loss_ratio(metrics.get("volume_ratio_20"))),
        ("RSI14", _format_profit_loss_ratio(metrics.get("rsi14"))),
        ("MACD", _format_profit_loss_ratio(metrics.get("macd"))),
        ("MACD signal", _format_profit_loss_ratio(metrics.get("macd_signal"))),
        ("ATR率", _format_plain_pct(metrics.get("atr_pct"))),
        ("20日高値", format_yen(metrics.get("recent_high_20"))),
        ("20日安値", format_yen(metrics.get("recent_low_20"))),
        ("陽線", "OK" if metrics.get("is_bullish_candle") else "NG"),
        ("下ヒゲ陽線", "OK" if metrics.get("is_lower_shadow_bullish") else "NG"),
        ("長い上ヒゲ警戒", "警戒" if metrics.get("is_upper_shadow_warning") else "なし"),
    ]
    return pd.DataFrame(rows, columns=["指標", "値"])


def _render_technical_score_detail(result: Dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("総合点", f"{result.get('technical_score_final', 0)}/60")
    cols[1].metric("判定", result.get("technical_judgement", "-"))
    cols[2].metric("確度", f"{result.get('confidence', 0)}%")
    cols[3].metric("現在値", format_yen(result.get("close")))

    cols = st.columns(4)
    cols[0].metric("損切り候補", format_yen(result.get("stop_loss_candidate")))
    cols[1].metric("損切り幅", _format_plain_pct(result.get("stop_loss_distance_pct")))
    cols[2].metric("利確候補", format_yen(result.get("target_price_candidate")))
    cols[3].metric("R/R", _format_profit_loss_ratio(result.get("risk_reward")))

    st.markdown(f"**保有目安**：{result.get('hold_days_hint', '-')}")
    st.markdown(f"**買いタイミング**：{result.get('buy_timing_hint', '-')}")
    hard_filter_text = _join_reason_text(result.get("hard_filter_reason"))
    penalty_text = _join_reason_text(result.get("penalty_reasons"))
    missing_text = _join_reason_text(result.get("missing_data"))
    if result.get("hard_filter_reason"):
        st.warning(f"強制見送り理由：{hard_filter_text}")
    else:
        st.caption("強制見送り理由：なし")
    st.caption(f"減点理由：{penalty_text if penalty_text != '-' else 'なし'}")
    st.caption(f"不足データ：{missing_text if missing_text != '-' else 'なし'}")

    st.markdown("**スコア内訳**")
    st.dataframe(_safe_dataframe(_technical_score_breakdown_table(result)), width="stretch", hide_index=True)
    with st.expander("日足指標詳細", expanded=False):
        st.dataframe(_safe_dataframe(_technical_latest_metrics_table(result)), width="stretch", hide_index=True)


def _render_daily_technical_tab(watchlist: pd.DataFrame) -> None:
    st.subheader("日足データ取得・テクニカル採点")
    st.caption("OpenAI APIは呼ばず、日足OHLCVからルールベースで採点します。実売買・自動売買ではありません。")

    if "daily_data" not in st.session_state:
        st.session_state["daily_data"] = {}
    if "fetch_errors" not in st.session_state:
        st.session_state["fetch_errors"] = []

    records = watchlist.to_dict("records") if not watchlist.empty else []
    limit_options = {
        "10件": 10,
        "30件": 30,
        "50件": 50,
        "全件": None,
    }
    setting_cols = st.columns([1, 3])
    limit_label = setting_cols[0].selectbox(
        "日足取得件数",
        list(limit_options.keys()),
        index=1,
        key="daily_technical_fetch_limit_label",
        help="まず10件、次に30件、必要なら50件/全件の順で段階的に取得できます。",
    )
    fetch_limit = limit_options[limit_label]
    target_records = records if fetch_limit is None else records[:fetch_limit]
    setting_cols[1].caption(
        f"watchlist全体: {len(records)}銘柄 / 今回の取得対象: {len(target_records)}銘柄。"
        "通信が重い場合は10件から確認してください。"
    )

    controls = st.columns([1, 1, 2])
    fetch_clicked = controls[0].button("日足データ取得", key="fetch_daily_technical_data_button", type="primary")
    score_clicked = controls[1].button(
        "テクニカル採点",
        key="score_daily_technical_data_button",
        disabled=not bool(st.session_state.get("daily_data")),
    )
    controls[2].caption(
        f"取得期間: {DEFAULT_DAILY_FETCH_PERIOD} / 必要本数: {MIN_DAILY_HISTORY_ROWS}本以上 / 段階取得: {limit_label}"
    )

    if fetch_clicked:
        if not target_records:
            st.error("watchlist.csvに銘柄がありません。日足データ取得を実行できません。")
        else:
            with st.spinner("watchlistの日足データを取得しています..."):
                daily_data, fetch_errors = fetch_daily_technical_data(target_records)
            st.session_state["daily_data"] = daily_data
            st.session_state["fetch_errors"] = fetch_errors
            st.session_state["technical_scores"] = []
            st.session_state["daily_data_last_fetched_at"] = now_jst_display()
            st.session_state["daily_data_fetch_limit_label"] = limit_label
            st.success(f"日足データ取得完了: 成功 {len(daily_data)}件 / 失敗 {len(fetch_errors)}件")

    if score_clicked:
        daily_data = st.session_state.get("daily_data") or {}
        if not daily_data:
            st.warning("先に「日足データ取得」を実行してください。")
        else:
            with st.spinner("日足テクニカルを採点しています..."):
                results = score_daily_technical_data(daily_data)
            st.session_state["technical_scores"] = results
            st.session_state["technical_scores_last_scored_at"] = now_jst_display()
            st.success(f"テクニカル採点完了: {len(results)}件")

    daily_data = st.session_state.get("daily_data") or {}
    fetch_errors = st.session_state.get("fetch_errors") or []
    results = st.session_state.get("technical_scores") or []

    st.markdown("**取得・採点サマリー**")
    cols = st.columns(7)
    scored_count = len(results)
    strong_count = sum(1 for row in results if row.get("technical_score_final", 0) >= 50)
    buy_count = sum(1 for row in results if row.get("technical_score_final", 0) >= 42)
    conditional_count = sum(1 for row in results if row.get("technical_score_final", 0) >= 36)
    watch_count = sum(1 for row in results if row.get("technical_judgement") == "監視強化")
    avoid_count = sum(1 for row in results if row.get("technical_judgement") in {"見送り", "触らない"})
    cols[0].metric("採点対象", f"{scored_count}件")
    cols[1].metric("50点以上", f"{strong_count}件")
    cols[2].metric("42点以上", f"{buy_count}件")
    cols[3].metric("36点以上", f"{conditional_count}件")
    cols[4].metric("監視強化", f"{watch_count}件")
    cols[5].metric("見送り", f"{avoid_count}件")
    cols[6].metric("取得エラー", f"{len(fetch_errors)}件")
    st.caption(
        f"最終取得: {st.session_state.get('daily_data_last_fetched_at', '-')} / "
        f"最終採点: {st.session_state.get('technical_scores_last_scored_at', '-')} / "
        f"前回取得件数: {st.session_state.get('daily_data_fetch_limit_label', '-')}"
    )

    if not daily_data and not fetch_errors:
        st.info("まだ日足データを取得していません。「日足データ取得」を押してください。")

    if results:
        st.markdown("**テクニカル採点結果（総合点順）**")
        st.dataframe(_safe_dataframe(_technical_score_display_table(results)), width="stretch", hide_index=True)
        st.markdown("**銘柄別詳細**")
        for result in results:
            title = (
                f"{result.get('code')} {result.get('name')}｜"
                f"{result.get('technical_score_final', 0)}/60｜"
                f"{result.get('technical_judgement', '-')}｜"
                f"R/R {_format_profit_loss_ratio(result.get('risk_reward'))}"
            )
            with st.expander(title, expanded=False):
                _render_technical_score_detail(result)

    if fetch_errors:
        with st.expander("日足データ取得失敗一覧", expanded=True):
            st.warning("取得失敗の理由を表示しています。error_type / error_message が空にならないように記録しています。")
            st.dataframe(_safe_dataframe(_daily_fetch_error_table(fetch_errors)), width="stretch", hide_index=True)


def _render_trades() -> None:
    ensure_trades_file()
    st.subheader("検証結果")
    st.caption("実際に買った候補や通知候補を記録し、1日後・3日後・5日後の結果を追記して検証します。")
    try:
        trades = pd.read_csv(TRADES_PATH, dtype={"code": str}).fillna("")
    except Exception as exc:
        st.error(f"trades.csvを読み込めませんでした: {exc}")
        return
    if trades.empty:
        st.info("まだ記録はありません。候補の詳細から記録できます。")
    else:
        st.dataframe(_safe_dataframe(trades), width="stretch", hide_index=True)


def main() -> None:
    st.title("日本株 1〜5日スイング候補ツール")
    st.caption("自動売買ではありません。最終判断は必ずご自身で行ってください。")
    if "openai_call_count" not in st.session_state:
        st.session_state["openai_call_count"] = 0
    if "latest_signals" not in st.session_state:
        st.session_state["latest_signals"] = []
    _initialize_openai_api_enabled_state()

    with st.sidebar:
        st.header("設定")
        period_options = ["6mo", "9mo", "1y", "2y"]
        default_index = period_options.index(DEFAULT_PRICE_PERIOD) if DEFAULT_PRICE_PERIOD in period_options else 1
        period = st.selectbox("株価取得期間", period_options, index=default_index)
        st.caption(f"買い候補: {BUY_SCORE_THRESHOLD}点以上 / 監視: {WATCH_SCORE_THRESHOLD}点以上")
        st.caption("初期スキャンではOpenAI APIを呼びません。")
        st.checkbox(
            "OpenAI API使用",
            key="openai_api_enabled_ui",
            help="OFFの場合、APIキーがあってもGPT判断は呼ばず、ルールベースfallbackで保存します。",
        )
        st.caption(f"OpenAI APIキー: {'あり' if get_setting('OPENAI_API_KEY', '').strip() else 'なし'}")
        st.caption(f"OpenAI API使用: {'ON' if _openai_api_enabled() else 'OFF'}")
        st.caption(f"現在のフォワード検証判定: {_openai_mode_label()}")
        st.caption(f"openai_call_count: {int(st.session_state.get('openai_call_count', 0))}")
        if "refresh_token" not in st.session_state:
            st.session_state.refresh_token = 0
        if st.button("キャッシュをクリア"):
            st.session_state.refresh_token += 1
            st.cache_data.clear()
        st.divider()
        st.button(
            "フォワード検証DB疎通テスト",
            key="sidebar_ai_virtual_db_test_button",
            on_click=_on_ai_virtual_db_test_click,
        )
        if st.session_state.get("last_ai_virtual_db_test", {}).get("ok"):
            st.caption(
                f"フォワード検証DB OK: trade_id={st.session_state['last_ai_virtual_db_test'].get('trade_id')}"
            )
        elif st.session_state.get("last_ai_virtual_db_test"):
            st.caption("フォワード検証DBテスト失敗")

    try:
        watchlist = load_watchlist()
    except Exception as exc:
        st.error(f"watchlist.csvを読み込めませんでした: {exc}")
        watchlist = pd.DataFrame(columns=["code", "name", "theme", "market", "raw_code", "normalized_symbol"])

    watchlist_error = watchlist.attrs.get("load_error", "")
    if watchlist_error:
        st.warning(watchlist_error)

    st.info("初期表示ではOpenAI APIも株価スキャンも実行しません。スキャンは下のボタンを押した時だけ実行します。")
    normal_scan_settings = _render_normal_scan_settings()
    scan_cols = st.columns([1, 3])
    if scan_cols[0].button("株価スキャンを実行", key="run_stock_scan_button", type="primary"):
        with st.spinner("株価取得とスコア計算を実行中です..."):
            _run_stock_scan(watchlist, period, normal_scan_settings)
    scan_cols[1].caption(f"初期表示直後の openai_call_count は 0 です。現在: {int(st.session_state.get('openai_call_count', 0))}")
    _render_scan_status()

    signals = list(st.session_state.get("latest_signals", []))

    buy, watch, avoid, failed = _split_signals(signals)

    _render_count_metrics(buy, watch, avoid, failed)
    st.info("買い候補は下のタブから確認してください。トップ画面には詳細カードを表示していません。")
    _render_logic_confirmation_section()
    if signals and len(failed) == len(signals):
        _render_all_failed_warning(failed)

    (
        tab_summary,
        tab_buy,
        tab_watch,
        tab_avoid,
        tab_intraday,
        tab_ai_virtual,
        tab_historical_replay,
        tab_virtual_performance,
        tab_stock_personality,
        tab_pattern_stats,
        tab_details,
        tab_watchlist,
        tab_trades,
    ) = st.tabs(
        [
            "サマリー",
            "買い候補",
            "監視",
            "触らない",
            "場中エントリー監視",
            "フォワード検証",
            "過去リプレイ検証",
            "仮想成績",
            "銘柄別クセ",
            "パターン別勝率",
            "詳細分析",
            "監視リスト",
            "検証結果",
        ]
    )

    with tab_summary:
        _render_summary_tab(buy, watch, avoid, failed)

    with tab_buy:
        _render_candidate_tab("買い候補", buy, "buy")

    with tab_watch:
        _render_candidate_tab("監視", watch, "watch")

    with tab_avoid:
        _render_avoid_tab(avoid)

    with tab_intraday:
        _render_intraday_tab(watchlist, buy, watch)

    with tab_ai_virtual:
        _render_ai_virtual_trade_tab(buy, watch)

    with tab_historical_replay:
        _render_historical_replay_tab(watchlist)

    with tab_virtual_performance:
        _render_virtual_performance_tab()

    with tab_stock_personality:
        _render_stock_personality_tab()

    with tab_pattern_stats:
        _render_pattern_stats_tab()

    with tab_details:
        _render_details_tab(buy, watch, avoid, failed)

    with tab_watchlist:
        st.subheader("watchlist.csvの銘柄一覧")
        st.dataframe(_safe_dataframe(watchlist), width="stretch", hide_index=True)

    with tab_trades:
        _render_trades()

    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
