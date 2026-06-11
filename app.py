from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DEFAULT_PRICE_PERIOD, TRADES_PATH
from notifier import format_yen
from scanner import (
    append_trade_candidate,
    build_signal_table,
    ensure_trades_file,
    load_watchlist,
    scan_watchlist,
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
        padding: 0.6rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.16rem; }
    .signal-line {
        margin: 0.2rem 0 0.45rem 0;
        line-height: 1.55;
    }
    .muted { color: #64748b; font-size: 0.9rem; }
    @media (max-width: 640px) {
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
        h1 { font-size: 1.45rem !important; }
        h2, h3 { font-size: 1.08rem !important; }
        div[data-testid="stMetricValue"] { font-size: 0.98rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.78rem; }
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


def _records_key(df: pd.DataFrame) -> Tuple[Tuple[str, str, str, str], ...]:
    return tuple(
        (str(row.code), str(row.name), str(row.theme), str(row.market))
        for row in df.itertuples(index=False)
    )


def _records_from_key(records_key: Tuple[Tuple[str, str, str, str], ...]) -> List[Dict[str, str]]:
    return [
        {"code": code, "name": name, "theme": theme, "market": market}
        for code, name, theme, market in records_key
    ]


@st.cache_data(ttl=900, show_spinner=False)
def _scan_cached(
    records_key: Tuple[Tuple[str, str, str, str], ...],
    period: str,
    refresh_token: int,
) -> List[Dict[str, Any]]:
    del refresh_token
    return scan_watchlist(_records_from_key(records_key), period=period)


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
        height=340,
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _score_breakdown_rows(signal: Dict[str, Any]) -> pd.DataFrame:
    breakdown = signal.get("score_breakdown") or {}
    rows = [
        {"項目": label, "点": breakdown.get(key, 0)}
        for key, label in SCORE_LABELS.items()
    ]
    return pd.DataFrame(rows)


def _score_breakdown_chart(signal: Dict[str, Any]) -> go.Figure:
    rows = _score_breakdown_rows(signal)
    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in rows["点"]]
    fig = go.Figure(go.Bar(x=rows["点"], y=rows["項目"], orientation="h", marker_color=colors))
    fig.update_layout(height=260, margin=dict(l=8, r=8, t=8, b=8), xaxis_title="点")
    return fig


def _render_metrics(signal: Dict[str, Any]) -> None:
    cols = st.columns(3)
    cols[0].metric("現在値", format_yen(signal.get("price")))
    cols[1].metric("スコア", f"{signal.get('score', '-')}点")
    cols[2].metric("R/R", _format_rr(signal))

    cols = st.columns(3)
    cols[0].metric("損切り", format_yen(signal.get("stop_loss")))
    cols[1].metric("利確目安", _format_targets(signal))
    cols[2].metric("期待値 / 確度", f"{signal.get('expected_value_label', '-')} / {signal.get('confidence', '-')}")


def _render_reason_list(title: str, values: Any) -> None:
    values = values or []
    if not values:
        st.markdown(f"**{title}**：-")
        return
    st.markdown(f"**{title}**")
    for value in values:
        st.markdown(f"- {value}")


def _render_signal_card(signal: Dict[str, Any], key_prefix: str, show_chart: bool = False) -> None:
    with st.container(border=True):
        left, right = st.columns([0.68, 0.32])
        left.markdown(f"### {signal.get('code')} {signal.get('name')}")
        left.caption(f"{signal.get('theme', '')} / {signal.get('last_date', '-')}")
        right.metric(signal.get("category", "-"), f"{signal.get('score', '-')}点")

        _render_metrics(signal)
        st.markdown(f"<div class='signal-line'><b>狙い</b>：{signal.get('entry_type', '-')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='signal-line'><b>待ち条件</b>：{signal.get('wait_condition', '-')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='signal-line'><b>買い条件</b>：{signal.get('entry_trigger', signal.get('buy_condition', '-'))}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='signal-line'><b>エントリー目安</b>：{signal.get('entry_zone', '-')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='signal-line'><b>ハイエナ心理</b>：{signal.get('comment', '-')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='signal-line'><b>無効条件</b>：{signal.get('invalidation_condition', signal.get('invalid_conditions', '-'))}</div>", unsafe_allow_html=True)

        if show_chart:
            st.markdown("**判定理由**")
            reason_cols = st.columns(3)
            with reason_cols[0]:
                _render_reason_list("プラス材料", signal.get("positive_reasons"))
            with reason_cols[1]:
                _render_reason_list("マイナス材料", signal.get("negative_reasons"))
            with reason_cols[2]:
                _render_reason_list("待つ理由", signal.get("wait_reasons"))

            st.markdown("**スコア内訳**")
            st.dataframe(_score_breakdown_rows(signal), width="stretch", hide_index=True)
            chart_key = f"price_chart_{key_prefix}_{signal.get('code')}"
            score_key = f"score_chart_{key_prefix}_{signal.get('code')}"
            st.plotly_chart(_make_price_chart(signal), width="stretch", key=chart_key)
            st.plotly_chart(_score_breakdown_chart(signal), width="stretch", key=score_key)

        with st.expander("通知文をコピー"):
            st.text_area(
                "ChatGPTへ貼り付ける通知文",
                value=signal.get("notification_text", ""),
                height=300,
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


def _render_card_section(title: str, signals: Iterable[Dict[str, Any]], key_prefix: str) -> None:
    signals = list(signals)
    st.subheader(title)
    if not signals:
        st.info("該当銘柄はありません。")
        return
    for idx, signal in enumerate(signals):
        _render_signal_card(signal, key_prefix=f"{key_prefix}_{idx}_{signal.get('code')}", show_chart=False)


def _render_expanders(title: str, signals: Iterable[Dict[str, Any]], key_prefix: str) -> None:
    signals = list(signals)
    st.subheader(title)
    if not signals:
        st.info("該当銘柄はありません。")
        return
    for signal in signals:
        label = (
            f"{signal.get('score', '-')}点 | {signal.get('code')} {signal.get('name')}"
            f" | {signal.get('entry_type')} | {signal.get('wait_condition', '-')}"
        )
        with st.expander(label, expanded=False):
            _render_signal_card(signal, key_prefix=f"{key_prefix}_{signal.get('code')}", show_chart=True)


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
        st.info("まだ記録はありません。候補カードのボタンから記録できます。")
    else:
        st.dataframe(trades, width="stretch", hide_index=True)


def main() -> None:
    st.title("日本株 1〜5日スイング候補ツール")
    st.caption("自動売買ではありません。最終判断は必ずご自身で行ってください。")

    with st.sidebar:
        st.header("設定")
        period_options = ["6mo", "9mo", "1y", "2y"]
        default_index = period_options.index(DEFAULT_PRICE_PERIOD) if DEFAULT_PRICE_PERIOD in period_options else 1
        period = st.selectbox("株価取得期間", period_options, index=default_index)
        if "refresh_token" not in st.session_state:
            st.session_state.refresh_token = 0
        if st.button("株価を再取得"):
            st.session_state.refresh_token += 1
            st.cache_data.clear()
        st.caption("スマホから見る場合は `--server.address 0.0.0.0` で起動します。")

    try:
        watchlist = load_watchlist()
    except Exception as exc:
        st.error(f"watchlist.csvを読み込めませんでした: {exc}")
        return

    with st.spinner("株価取得とスコア計算を実行中です..."):
        signals = _scan_cached(_records_key(watchlist), period, st.session_state.refresh_token)

    buy = [s for s in signals if s.get("category") == "買い候補"]
    watch = [s for s in signals if s.get("category") == "監視"]
    avoid = [s for s in signals if s.get("category") == "触らない"]
    failed = [s for s in signals if s.get("category") == "取得失敗"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("買い候補", len(buy))
    metric_cols[1].metric("監視", len(watch))
    metric_cols[2].metric("触らない", len(avoid))
    metric_cols[3].metric("取得失敗", len(failed))

    _render_card_section("今日の買い候補", buy, "buy")
    _render_card_section("監視銘柄", watch, "watch_main")

    tab_summary, tab_details, tab_watchlist, tab_trades = st.tabs(["一覧", "詳細分析", "監視リスト", "検証結果"])
    with tab_summary:
        st.subheader("スコア一覧")
        table = build_signal_table(signals)
        st.dataframe(table, width="stretch", hide_index=True)
        if failed:
            st.warning("一部銘柄の株価取得に失敗しました。時間を置いて再取得してください。")

    with tab_details:
        _render_expanders("買い候補の詳細", buy, "buy_detail")
        _render_expanders("監視の詳細", watch, "watch_detail")
        _render_expanders("触らない", avoid, "avoid")
        _render_expanders("取得失敗", failed, "failed")

    with tab_watchlist:
        st.subheader("watchlist.csvの銘柄一覧")
        st.dataframe(watchlist, width="stretch", hide_index=True)

    with tab_trades:
        _render_trades()

    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
