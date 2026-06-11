from __future__ import annotations

from typing import Any, Dict, Optional


def format_yen(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.0f}円"
    except (TypeError, ValueError):
        return "-"


def _format_score_breakdown(signal: Dict[str, Any]) -> str:
    breakdown = signal.get("score_breakdown") or {}
    labels = [
        ("market_score", "地合い"),
        ("trend_score", "トレンド"),
        ("pullback_score", "押し目"),
        ("volume_score", "出来高"),
        ("theme_score", "テーマ"),
        ("risk_penalty", "リスク減点"),
        ("total_score", "合計"),
    ]
    parts = [f"{label}:{breakdown.get(key, '-')}" for key, label in labels]
    return " / ".join(parts)


def _join_reasons(values: Any) -> str:
    if not values:
        return "-"
    if isinstance(values, list):
        return "、".join(str(value) for value in values if value)
    return str(values)


def build_notification_text(signal: Dict[str, Any]) -> str:
    """Build text that can be pasted directly into ChatGPT."""
    targets = f"{format_yen(signal.get('target_1'))} / {format_yen(signal.get('target_2'))}"
    return "\n".join(
        [
            "【スイング候補通知】",
            f"銘柄：{signal.get('code', '')} {signal.get('name', '')}",
            f"現在値：{format_yen(signal.get('price'))}",
            f"スコア：{signal.get('score', '-')}点",
            f"スコア内訳：{_format_score_breakdown(signal)}",
            f"分類：{signal.get('category', '-')}",
            f"狙い：{signal.get('entry_type', '-')}",
            f"監視理由：{signal.get('monitoring_reason', '-')}",
            f"待ち条件：{signal.get('wait_condition', '-')}",
            f"買い条件：{signal.get('buy_condition', '-')}",
            f"損切り：{format_yen(signal.get('stop_loss'))}割れ",
            f"利確目安：{targets}",
            f"保有想定：{signal.get('expected_holding_days', '-')}",
            f"期待値：{signal.get('expected_value_label', '-')}",
            f"確度：{signal.get('confidence', '-')}",
            f"リスクリワード：{signal.get('risk_reward_label', signal.get('risk_reward', '-'))}",
            f"プラス材料：{_join_reasons(signal.get('positive_reasons'))}",
            f"マイナス材料：{_join_reasons(signal.get('negative_reasons'))}",
            f"待つ理由：{_join_reasons(signal.get('wait_reasons'))}",
            f"ハイエナ心理：{signal.get('comment', '-')}",
            f"無効条件：{signal.get('invalidation_condition', signal.get('invalid_conditions', '-'))}",
            f"買ってはいけない条件：{signal.get('no_buy_conditions', '-')}",
        ]
    )


def send_notification(message: str, channel: str = "console") -> None:
    """
    MVP notifier.

    External channels are intentionally not implemented yet. Streamlit rendering
    is handled in app.py so this function stays framework-light.
    """
    if channel == "console":
        print(message)
    elif channel == "streamlit":
        return
    elif channel == "chatgpt_text":
        print(message)
    else:
        raise ValueError(f"未対応の通知チャンネルです: {channel}")
