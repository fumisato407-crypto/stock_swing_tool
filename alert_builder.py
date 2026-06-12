from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import config
from config import ALERTS_LOG_PATH
from notifier import format_yen


ALERT_COLUMNS = [
    "datetime",
    "code",
    "name",
    "current_price",
    "intraday_score",
    "signal_type",
    "buy_zone",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "reason",
    "discord_sent",
    "discord_error",
]


def _join(values: List[str]) -> str:
    return "、".join(str(value) for value in values if value) if values else "-"


def _format_rr(signal: Dict[str, Any]) -> str:
    value = signal.get("risk_reward")
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def build_intraday_alert_text(signal: Dict[str, Any]) -> str:
    targets = f"{format_yen(signal.get('take_profit_1'))} / {format_yen(signal.get('take_profit_2'))}"
    return "\n".join(
        [
            "【場中エントリー候補】",
            f"銘柄：{signal.get('code', '')} {signal.get('name', '')}",
            f"現在値：{format_yen(signal.get('current_price'))}",
            f"判定：{signal.get('judgement', '-')}",
            f"スコア：{signal.get('intraday_score', '-')}",
            f"狙い：{signal.get('signal_type', '-')}",
            f"買い候補：{signal.get('buy_zone', '-')}",
            f"損切り：{format_yen(signal.get('stop_loss'))}割れ",
            f"利確：{targets}",
            f"R/R：{_format_rr(signal)}",
            f"時間軸：{signal.get('timeframe', '短期ブレイク〜1日スイング')}",
            f"理由：{_join(signal.get('reasons', []))}",
            f"買ってはいけない条件：{_join(signal.get('no_buy_conditions', []))}",
        ]
    )


def _build_expected_action(signal: Dict[str, Any]) -> str:
    trigger = str(signal.get("entry_trigger") or signal.get("buy_condition") or "").strip()
    if trigger and trigger != "-":
        return trigger

    signal_type = str(signal.get("signal_type") or "")
    if "押し目" in signal_type:
        return "押し目再反発を待ち、短期線回復または高値更新でエントリー検討"
    if "ブレイク" in signal_type or "節目" in signal_type:
        return "節目維持または直近高値更新でエントリー検討"
    if "V字" in signal_type or "反発" in signal_type:
        return "反発継続と出来高維持を確認してエントリー検討"
    return "押し目待ち、または高値更新でエントリー検討"


def build_buy_candidate_discord_text(signal: Dict[str, Any]) -> str:
    reasons = signal.get("reasons") or signal.get("positive_reasons") or []
    if isinstance(reasons, str):
        reasons_text = reasons
    else:
        reasons_text = _join(reasons)

    score = signal.get("intraday_score", signal.get("score", "-"))
    return "\n".join(
        [
            "【買い候補通知】",
            f"{signal.get('code', '')} {signal.get('name', '')}",
            f"現在値：{format_yen(signal.get('current_price', signal.get('price')))}",
            f"スコア：{score}",
            f"理由：{reasons_text}",
            f"想定：{_build_expected_action(signal)}",
        ]
    )


def alert_key(signal: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(signal.get("code", "")),
            str(signal.get("judgement", "")),
            str(signal.get("signal_type", "")),
            str(signal.get("level_price", "")),
            str(signal.get("intraday_score", "")),
        ]
    )


def discord_alert_key(signal: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(signal.get("code", "")),
            str(signal.get("judgement", "")),
        ]
    )


def _normalize_discord_mention_id(mention_id: str) -> str:
    normalized = str(mention_id or "").strip()
    normalized = normalized.removeprefix("<@!").removeprefix("<@").removesuffix(">")
    return normalized if normalized.isdigit() else ""


def _get_discord_webhook_url() -> str:
    return config.get_setting("DISCORD_WEBHOOK_URL", "")


def _get_discord_mention_id() -> str:
    return config.get_setting("DISCORD_MENTION_ID", "")


def _build_discord_payload(message: str, mention_id: str = "") -> Dict[str, Any]:
    normalized_mention_id = _normalize_discord_mention_id(mention_id)
    content = str(message)
    payload: Dict[str, Any] = {"content": content[:1900]}
    if normalized_mention_id:
        payload["content"] = f"<@{normalized_mention_id}> {content}"[:1900]
        payload["allowed_mentions"] = {"users": [normalized_mention_id]}
    return payload


def send_discord_webhook(
    message: str,
    webhook_url: str = "",
    mention_id: str = "",
) -> tuple[bool, str]:
    webhook_url = str(webhook_url or _get_discord_webhook_url()).strip()
    mention_id = str(mention_id or _get_discord_mention_id()).strip()
    if not webhook_url:
        return False, "DISCORD_WEBHOOK_URLが未設定です。"

    try:
        import requests

        response = requests.post(
            webhook_url,
            json=_build_discord_payload(message, mention_id=mention_id),
            timeout=10,
        )
    except Exception as exc:
        return False, f"Discord通知に失敗しました（{exc.__class__.__name__}）。"

    if 200 <= response.status_code < 300:
        return True, ""
    return False, f"Discord通知に失敗しました（HTTP {response.status_code}）。"


def ensure_alerts_log(path: Path = ALERTS_LOG_PATH) -> None:
    if not path.exists():
        pd.DataFrame(columns=ALERT_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
        return

    try:
        log_df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        pd.DataFrame(columns=ALERT_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
        return

    missing_columns = [column for column in ALERT_COLUMNS if column not in log_df.columns]
    if missing_columns:
        for column in missing_columns:
            log_df[column] = ""
        log_df[ALERT_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")


def append_alert_log(
    signal: Dict[str, Any],
    discord_sent: str = "",
    discord_error: str = "",
    path: Path = ALERTS_LOG_PATH,
) -> None:
    ensure_alerts_log(path)
    row = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "code": signal.get("code", ""),
        "name": signal.get("name", ""),
        "current_price": signal.get("current_price", ""),
        "intraday_score": signal.get("intraday_score", ""),
        "signal_type": signal.get("signal_type", ""),
        "buy_zone": signal.get("buy_zone", ""),
        "stop_loss": signal.get("stop_loss", ""),
        "take_profit_1": signal.get("take_profit_1", ""),
        "take_profit_2": signal.get("take_profit_2", ""),
        "reason": _join(signal.get("reasons", [])),
        "discord_sent": discord_sent,
        "discord_error": discord_error,
    }
    pd.DataFrame([row], columns=ALERT_COLUMNS).to_csv(
        path,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig",
    )
