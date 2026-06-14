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
from config import (
    AI_VIRTUAL_MODEL,
    ALERTS_LOG_PATH,
    DEFAULT_PRICE_PERIOD,
    OPENAI_API_ENABLED,
    TRADES_PATH,
    VIRTUAL_TRADES_DB_PATH,
    get_setting,
)
from data_fetcher import fetch_price_data, normalize_jp_symbol
from intraday_scanner import fetch_intraday_data, scan_intraday_entries
from market_hours import is_market_open_jst, market_status_label, next_market_open_hint, now_jst as market_now_jst
from notifier import format_yen
from outcome_tracker import update_open_virtual_trade_outcomes
from paper_trader import process_virtual_trade_signals
from pattern_stats import calculate_pattern_stats
from scanner import (
    append_trade_candidate,
    build_signal_table,
    ensure_trades_file,
    load_watchlist,
    scan_watchlist,
)
from scoring import BUY_SCORE_THRESHOLD, WATCH_SCORE_THRESHOLD
from stock_personality import generate_stock_personalities
from time_utils import now_jst_display, now_jst_iso
from virtual_trade_store import (
    get_virtual_database_list,
    insert_virtual_trade,
    load_virtual_trades,
    rows_to_display,
)

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
    "last_attempt_at",
]

PLOTLY_CHART_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
}
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


@st.cache_data(ttl=900, show_spinner=False)
def _scan_cached(
    records_key: Tuple[Tuple[str, str, str, str, str, str], ...],
    period: str,
    refresh_token: int,
) -> List[Dict[str, Any]]:
    del refresh_token
    return scan_watchlist(_records_from_key(records_key), period=period)


def _score(signal: Dict[str, Any]) -> int:
    try:
        return int(signal.get("score", 0))
    except (TypeError, ValueError):
        return 0


def _run_stock_scan(watchlist: pd.DataFrame, period: str) -> List[Dict[str, Any]]:
    scan_start = now_jst_display()
    started = time.perf_counter()
    signals = _scan_cached(_records_key(watchlist), period, st.session_state.get("refresh_token", 0))
    elapsed = round(time.perf_counter() - started, 2)
    failed_count = sum(1 for signal in signals if signal.get("category") == "取得失敗")
    st.session_state["latest_signals"] = signals
    st.session_state["scan_status"] = {
        "scan_start_jst": scan_start,
        "scan_end_jst": now_jst_display(),
        "elapsed_seconds": elapsed,
        "signals_count": len(signals),
        "failed_count": failed_count,
    }
    return signals


def _render_scan_status() -> None:
    status = st.session_state.get("scan_status", {})
    if not status:
        st.info("初期表示では株価スキャンを実行しません。必要なときに「株価スキャンを実行」を押してください。")
        return
    cols = st.columns(5)
    cols[0].metric("signals", status.get("signals_count", 0))
    cols[1].metric("failed", status.get("failed_count", 0))
    cols[2].metric("elapsed", f"{status.get('elapsed_seconds', '-')}秒")
    cols[3].metric("scan_start_jst", status.get("scan_start_jst", "-"))
    cols[4].metric("scan_end_jst", status.get("scan_end_jst", "-"))


def _split_signals(signals: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], ...]:
    failed = [s for s in signals if s.get("category") == "取得失敗"]
    analyzable = [s for s in signals if s.get("category") != "取得失敗"]
    buy = [s for s in analyzable if _score(s) >= BUY_SCORE_THRESHOLD]
    watch = [s for s in analyzable if WATCH_SCORE_THRESHOLD <= _score(s) < BUY_SCORE_THRESHOLD]
    avoid = [s for s in analyzable if _score(s) < WATCH_SCORE_THRESHOLD]
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
        rows.append(
            {
                "code": signal.get("code", ""),
                "name": signal.get("name", ""),
                "price": format_yen(signal.get("price")),
                "score": signal.get("score", "-"),
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
                "last_attempt_at": signal.get("last_attempt_at", ""),
            }
        )
    return pd.DataFrame(rows, columns=INTRADAY_TABLE_COLUMNS)


def _failure_debug_table(signals: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        rows.append(
            {
                "code": signal.get("raw_code") or signal.get("code", ""),
                "normalized_symbol": signal.get("normalized_symbol", ""),
                "error_type": signal.get("error_type", ""),
                "error_message": signal.get("error_message", signal.get("error", signal.get("comment", ""))),
                "fetched_rows": signal.get("fetched_rows", 0),
                "last_attempt_at": signal.get("last_attempt_at", ""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["code", "normalized_symbol", "error_type", "error_message", "fetched_rows", "last_attempt_at"],
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
    st.markdown(f"<div class='signal-line'><b>狙い</b>：{signal.get('entry_type', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>買い条件</b>：{signal.get('entry_trigger', signal.get('buy_condition', '-'))}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>待ち条件</b>：{signal.get('wait_condition', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>損切り</b>：{format_yen(signal.get('stop_loss'))}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>利確目安</b>：{_format_targets(signal)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>ハイエナ心理</b>：{signal.get('comment', '-')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signal-line'><b>無効条件</b>：{signal.get('invalidation_condition', signal.get('invalid_conditions', '-'))}</div>", unsafe_allow_html=True)

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
        st.dataframe(_failure_debug_table([signal]), width="stretch", hide_index=True)
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
        f"**出来高**：{signal.get('current_volume', 0):,} / "
        f"平均 {signal.get('volume_avg', 0):,} / 倍率 {signal.get('volume_ratio', '-')}"
    )
    st.markdown(
        f"**当日レンジ**：高値 {format_yen(signal.get('day_high'))} / "
        f"安値 {format_yen(signal.get('day_low'))} / 安値から {signal.get('rebound_from_day_low_pct', 0)}%"
    )
    st.markdown(f"**RSI**：{signal.get('rsi', '-')}")

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
        st.warning("監視対象がありません。watchlist.csvまたは手動入力コードを確認してください。")
        return

    with st.spinner(f"{source_label}で5分足データを取得して場中エントリー条件を判定中です..."):
        st.session_state.intraday_results = scan_intraday_entries(targets, interval=interval, period="5d")
        checked_at = _format_jst(_now_jst())
        st.session_state.intraday_last_updated = checked_at
        st.session_state.intraday_last_checked_at = checked_at
        st.session_state.intraday_last_scan_status = f"{len(targets)}銘柄チェック完了"
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
    return df.fillna("").astype(str)


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


def _render_last_ai_virtual_run() -> None:
    run_state = st.session_state.get("last_ai_virtual_run")
    results = st.session_state.get("last_ai_virtual_results", [])
    if not run_state:
        return

    st.markdown("**直近のAI仮想判断実行結果**")
    event_label = run_state.get("event_label", "フォーム送信を検知しました")
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
    st.markdown("**保存後に再取得した最近のAI仮想取引ログ**")
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
            "event_label": "AI仮想判断コールバックを検知しました",
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
            "event_label": "AI仮想判断コールバックを検知しました",
            "reason_summary": [],
            "error": f"AI仮想判断中に例外が発生しました: {exc.__class__.__name__}: {exc}",
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
    allowed_categories = {"買い候補"}
    if target_mode == "buy_watch":
        allowed_categories.add("監視")

    candidates = [
        signal
        for signal in signals
        if signal.get("category") in allowed_categories and _score(signal) >= int(min_score)
    ]
    return sorted(candidates, key=lambda item: _score(item), reverse=True)[: int(max_count)]


def _summarize_virtual_trade_results(results: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "saved_count": sum(1 for result in results if result.get("saved")),
        "failed_count": sum(1 for result in results if not result.get("saved")),
        "duplicate_count": sum(1 for result in results if result.get("reason") == "duplicate_recent"),
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
        "target_count": len(candidates),
        "saved_count": 0,
        "failed_count": 0,
        "duplicate_count": 0,
        "openai_call_count": int(st.session_state.get("openai_call_count", 0)),
        "last_error": "",
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
        results = process_virtual_trade_signals(candidates, use_openai=False)
        counts = _summarize_virtual_trade_results(results)
        summary.update(counts)
        summary["status"] = "executed"
        summary["reason"] = "saved" if counts["saved_count"] else "no_new_saved"
        st.session_state["last_ai_virtual_results"] = results
        st.session_state["ai_rule_auto_last_results"] = results
        st.session_state["ai_rule_auto_last_save_at"] = summary["checked_at_jst"]
        st.session_state["ai_rule_auto_last_error"] = ""
        _refresh_ai_virtual_db_debug_state()
    except Exception as exc:
        summary["status"] = "error"
        summary["reason"] = "exception"
        summary["last_error"] = f"{exc.__class__.__name__}: {exc}"
        st.session_state["ai_rule_auto_last_error"] = summary["last_error"]

    st.session_state["ai_rule_auto_last_summary"] = summary
    return summary


def _render_rule_auto_log_status(summary: Dict[str, Any]) -> None:
    st.caption(f"現在時刻JST：{summary.get('checked_at_jst', '-')}")
    st.caption(f"市場ステータス：{summary.get('market_status', '-')}")
    st.caption(f"自動保存：{'ON' if summary.get('auto_enabled') else 'OFF'}")
    st.caption(f"実行間隔：{summary.get('interval_minutes', '-')}分")
    st.caption(f"対象条件：{summary.get('target_mode', '-')}")
    st.caption(f"最小スコア：{summary.get('min_score', '-')}")
    st.caption(f"最大保存件数：{summary.get('max_count', '-')}")
    st.caption(f"最終自動保存時刻：{summary.get('last_auto_save_at', '-')}")
    st.caption(f"次回実行予定：{summary.get('next_run_hint', '-')}")

    cols = st.columns(5)
    cols[0].metric("対象候補数", summary.get("target_count", 0))
    cols[1].metric("保存成功", summary.get("saved_count", 0))
    cols[2].metric("保存失敗", summary.get("failed_count", 0))
    cols[3].metric("duplicate", summary.get("duplicate_count", 0))
    cols[4].metric("API calls", summary.get("openai_call_count", 0))

    if summary.get("reason") == "signals_missing":
        st.warning("まだ株価スキャン結果がありません。先に株価スキャンを実行してください")
    elif summary.get("reason") == "market_closed":
        st.info("市場時間外、昼休み、土日は自動保存を実行しません。")
    elif summary.get("reason") == "interval_wait":
        st.info("実行間隔の待機中です。")
    elif summary.get("status") == "executed":
        st.success("相場中ルール買いログ自動保存を実行しました。")

    if summary.get("last_error"):
        st.error(summary["last_error"])


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


def _render_rule_auto_logger_section() -> None:
    st.markdown("**相場中ルール買いログ自動保存**")
    st.caption("画面を開いている間だけ、市場時間中にルールベースのpaper tradingログを保存します。実売買・発注は行いません。")
    auto_enabled = st.toggle(
        "相場中ルール買いログ自動保存",
        value=False,
        key="ai_virtual_logging_enabled",
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

    st.caption(f"OpenAI API使用：{'ON' if _openai_api_enabled() else 'OFF'}")
    st.caption("自動保存は安全運用のため常に rule_based_fallback で実行します。OpenAI APIは呼びません。")
    _render_rule_auto_log_fragment()


def _render_ai_virtual_trade_tab(
    buy: List[Dict[str, Any]],
    watch: List[Dict[str, Any]],
) -> None:
    st.subheader("AI仮想取引")
    st.caption("GPTによるpaper trading / virtual trading専用です。実売買・発注・自動売買は行いません。")

    api_configured = bool(get_setting("OPENAI_API_KEY", "").strip())
    st.caption(f"OpenAI APIキー：{'あり' if api_configured else 'なし'}")
    st.caption(f"OpenAI API使用：{'ON' if _openai_api_enabled() else 'OFF'}")
    st.caption(f"現在のAI仮想判断：{_openai_mode_label()}")
    st.caption(f"API呼び出し回数：{int(st.session_state.get('openai_call_count', 0))}")
    if not _openai_api_enabled():
        st.info("OpenAI API使用OFFのため、GPT判断は行わず、ルールベースで仮想取引ログを保存します。API料金は発生しません。")
    st.button(
        "OpenAI接続テスト",
        key="openai_connection_test_button",
        on_click=_on_openai_connection_test_click,
    )
    _render_openai_connection_test_state()

    trades = load_virtual_trades(limit=1000)
    metric_cols = st.columns(4)
    metric_cols[0].metric("仮想ログ", len(trades))
    metric_cols[1].metric("open", int((trades["status"] == "open").sum()) if not trades.empty else 0)
    metric_cols[2].metric("closed", int((trades["status"] == "closed").sum()) if not trades.empty else 0)
    metric_cols[3].metric("logged", int((trades["status"] == "logged").sum()) if not trades.empty else 0)

    include_watch = st.checkbox("監視銘柄もAI仮想判断に含める", value=False)
    max_candidates = st.selectbox("AI仮想判断する件数", [1, 3, 5, 10], index=1)
    candidates = list(buy) + (list(watch) if include_watch else [])
    candidates = sorted(candidates, key=lambda item: _score(item), reverse=True)[: int(max_candidates)]
    candidates_snapshot = _snapshot_ai_virtual_candidates(candidates)
    st.session_state["ai_virtual_candidates_snapshot"] = candidates_snapshot

    st.markdown("**今回のAI仮想判断候補**")
    if candidates:
        st.dataframe(_safe_dataframe(_ai_candidate_table(candidates)), width="stretch", hide_index=True)
    else:
        st.info("AI仮想判断できる候補がありません。")
    st.caption(f"候補数: {len(candidates)} / 仮想取引DB: {VIRTUAL_TRADES_DB_PATH}")

    st.button(
        "DB疎通テスト",
        key="ai_virtual_db_test_button",
        on_click=_on_ai_virtual_db_test_click,
    )
    _render_ai_virtual_db_test_result()

    if not _openai_api_enabled():
        st.caption("手動実行モード：rule_based_fallback（GPT判断なし）")
    else:
        st.caption("OpenAI API使用ONのため、手動実行ではGPT判定を使います。")
    st.button(
        "AI仮想判断を実行",
        key="run_ai_virtual_trade_callback_button",
        type="primary",
        disabled=not candidates_snapshot,
        on_click=_on_ai_virtual_run_click,
        args=(candidates_snapshot,),
    )

    _render_rule_auto_logger_section()
    _render_ai_virtual_callback_debug_state()
    _render_last_ai_virtual_run()

    if st.button("open仮想取引の結果を更新", key="update_virtual_outcomes_button"):
        with st.spinner("仮想取引の結果を追跡中です..."):
            summary = update_open_virtual_trade_outcomes()
        st.info(
            f"確認 {summary['checked']}件 / 更新 {summary['updated']}件 / スキップ {summary['skipped']}件"
        )

    with st.expander("最近のAI仮想取引ログ", expanded=False):
        recent = rows_to_display(load_virtual_trades(limit=100))
        if recent.empty:
            st.info("まだAI仮想取引ログはありません。")
        else:
            columns = [
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
                "confidence",
                "entry_price",
                "stop_loss",
                "take_profit",
                "source_score",
                "status",
            ]
            st.dataframe(_safe_dataframe(recent[[col for col in columns if col in recent.columns]]), width="stretch", hide_index=True)


def _render_virtual_performance_tab() -> None:
    st.subheader("仮想成績")
    st.caption("AI仮想取引の保存結果を後追いで検証します。実売買の履歴とは完全に分離しています。")
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
        st.caption(f"現在のAI仮想判断: {_openai_mode_label()}")
        st.caption(f"openai_call_count: {int(st.session_state.get('openai_call_count', 0))}")
        if "refresh_token" not in st.session_state:
            st.session_state.refresh_token = 0
        if st.button("キャッシュをクリア"):
            st.session_state.refresh_token += 1
            st.cache_data.clear()
        st.divider()
        st.button(
            "AI DB疎通テスト",
            key="sidebar_ai_virtual_db_test_button",
            on_click=_on_ai_virtual_db_test_click,
        )
        if st.session_state.get("last_ai_virtual_db_test", {}).get("ok"):
            st.caption(
                f"AI DB OK: trade_id={st.session_state['last_ai_virtual_db_test'].get('trade_id')}"
            )
        elif st.session_state.get("last_ai_virtual_db_test"):
            st.caption("AI DBテスト失敗")

    try:
        watchlist = load_watchlist()
    except Exception as exc:
        st.error(f"watchlist.csvを読み込めませんでした: {exc}")
        watchlist = pd.DataFrame(columns=["code", "name", "theme", "market", "raw_code", "normalized_symbol"])

    watchlist_error = watchlist.attrs.get("load_error", "")
    if watchlist_error:
        st.warning(watchlist_error)

    st.info("初期表示ではOpenAI APIも株価スキャンも実行しません。スキャンは下のボタンを押した時だけ実行します。")
    scan_cols = st.columns([1, 3])
    if scan_cols[0].button("株価スキャンを実行", key="run_stock_scan_button", type="primary"):
        with st.spinner("株価取得とスコア計算を実行中です..."):
            _run_stock_scan(watchlist, period)
    scan_cols[1].caption(f"初期表示直後の openai_call_count は 0 です。現在: {int(st.session_state.get('openai_call_count', 0))}")
    _render_scan_status()

    signals = list(st.session_state.get("latest_signals", []))

    buy, watch, avoid, failed = _split_signals(signals)

    _render_count_metrics(buy, watch, avoid, failed)
    st.info("買い候補は下のタブから確認してください。トップ画面には詳細カードを表示していません。")
    if signals and len(failed) == len(signals):
        _render_all_failed_warning(failed)

    (
        tab_summary,
        tab_buy,
        tab_watch,
        tab_avoid,
        tab_intraday,
        tab_ai_virtual,
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
            "AI仮想取引",
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
