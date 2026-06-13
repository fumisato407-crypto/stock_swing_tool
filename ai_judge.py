from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from config import AI_VIRTUAL_MODEL, OPENAI_API_KEY, OPENAI_MODEL, get_setting


def _rule_based_comment(signal: Dict[str, Any]) -> str:
    entry_type = signal.get("entry_type", "見送り")
    category = signal.get("category", "触らない")
    risk_notes = signal.get("risk_notes") or []
    wait_condition = signal.get("wait_condition") or signal.get("watch_condition") or "-"
    entry_trigger = signal.get("entry_trigger") or signal.get("buy_condition") or "-"
    metrics = signal.get("raw_metrics") or {}

    if category == "触らない":
        if risk_notes:
            return (
                "今は無理に触るより、見送りが優先です。"
                f"{'、'.join(risk_notes)}ため、短期勢の売りが残っている可能性があります。"
                "前日安値を割る動きや出来高不足が続くなら、反発を取りに行く場面ではありません。"
            )
        return (
            "買いの根拠がまだ薄い形です。反発確認や出来高増加が出るまでは、"
            "焦って買う側ではなく待つ側に回る方が安全です。"
        )

    if category == "監視":
        if entry_type == "ブレイク待ち":
            return (
                "形は悪くありませんが、今は高値を追うより出来高つきの突破を待つ場面です。"
                f"{wait_condition}。{entry_trigger}が出るまでは、先に買った人の利益確定に巻き込まれやすいです。"
                "上ヒゲで失速するなら見送りです。"
            )
        if entry_type == "押し目待ち":
            return (
                "トレンドは見たい銘柄ですが、今すぐ飛びつくより押しを待つ方が拾う側に回れます。"
                f"{wait_condition}。{entry_trigger}が出た時だけ候補にして、浅い押しでの高値掴みは避けます。"
            )
        if entry_type == "反発待ち":
            return (
                "売りが止まるかを確認したい監視です。"
                f"{wait_condition}。{entry_trigger}が出れば、投げ売りを拾う側になれる可能性があります。"
                "前日安値を割るなら、まだ売りが残っているため見送りです。"
            )
        if entry_type == "出来高待ち":
            return (
                "値動きだけで入るにはまだ薄く、出来高の裏付け待ちです。"
                f"{wait_condition}。買い戻しや新規資金が見えないまま入ると、反発が続かないリスクがあります。"
            )
        return (
            "個別の形だけでなく地合いの回復を待ちたい状態です。"
            f"{wait_condition}。条件が整うまでは、無理に拾わず現金を残す方が次のチャンスを取りやすいです。"
        )

    if entry_type == "押し目反発":
        return (
            "直近の下落で短期勢の損切りが出た可能性があります。"
            "25日線付近で売りが止まり、5日線回復や前日高値突破が出るなら、"
            "焦って売った人の玉を拾う側になれます。前日安値を割る場合は見送りです。"
        )
    if entry_type == "ブレイク狙い":
        return (
            "高値突破に資金が乗るかを確認したい形です。"
            "出来高を伴って抜けるなら順張り候補ですが、上ヒゲだけで終わると高値掴みになりやすいです。"
            "ブレイクラインを割り込む場合は撤退目線です。"
        )
    if entry_type == "急落リバ":
        return (
            "急落後に投げ売りを拾うリバ狙いです。"
            "前日安値を割らずに前日高値を超えるなら、売りが出切った可能性を見られます。"
            "ただし戻りが弱い場合は下落継続の初動になりやすいため深追いは避けます。"
        )

    if metrics.get("drawdown_from_20d_high_pct", 0) > -2:
        return (
            "高値圏に近く、買うなら明確な出来高増加と高値突破を待ちたい形です。"
            "中途半端な位置で入ると高値掴みになりやすいため、条件が出るまでは見送りです。"
        )
    return (
        "一部の条件はありますが、買いのタイミングとしてはまだ決め手不足です。"
        "反発確認、出来高増加、前日安値を割らない動きがそろうまで待つのが基本です。"
    )


def _openai_comment(signal: Dict[str, Any]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=8)
    prompt = f"""
日本株の1〜5日スイング候補について、投資助言ではなく分析補助の短いコメントを書いてください。
焦って買う側か、投げ売りを拾う側か、高値掴みリスク、無効条件を自然に含めてください。

銘柄: {signal.get("code")} {signal.get("name")}
分類: {signal.get("category")}
狙い: {signal.get("entry_type")}
スコア: {signal.get("score")}
スコア内訳: {signal.get("score_breakdown")}
RSIなど: {signal.get("raw_metrics")}
リスク: {signal.get("risk_notes")}
監視理由: {signal.get("monitoring_reason")}
待ち条件: {signal.get("wait_condition")}
買い条件: {signal.get("buy_condition")}
無効条件: {signal.get("invalid_conditions")}
買ってはいけない条件: {signal.get("no_buy_conditions")}
"""
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
        input=prompt.strip(),
        max_output_tokens=220,
    )
    return response.output_text.strip()


def generate_hyena_comment(signal: Dict[str, Any]) -> str:
    if OPENAI_API_KEY:
        try:
            return _openai_comment(signal)
        except Exception:
            return _rule_based_comment(signal)
    return _rule_based_comment(signal)


VIRTUAL_DECISIONS = {"virtual_buy", "virtual_watch", "virtual_avoid"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def build_market_snapshot(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Create a point-in-time snapshot for AI paper trading without future data."""
    return {
        "timestamp_basis": signal.get("last_date") or signal.get("last_time") or "",
        "code": signal.get("code", ""),
        "name": signal.get("name", ""),
        "price": signal.get("price", signal.get("current_price")),
        "score": signal.get("score", signal.get("intraday_score", 0)),
        "category": signal.get("category", signal.get("judgement", "")),
        "entry_type": signal.get("entry_type", signal.get("signal_type", "")),
        "entry_zone": signal.get("entry_zone", signal.get("buy_zone", "")),
        "stop_loss": signal.get("stop_loss"),
        "target_1": signal.get("target_1", signal.get("take_profit_1")),
        "target_2": signal.get("target_2", signal.get("take_profit_2")),
        "risk_reward": signal.get("risk_reward", ""),
        "confidence": signal.get("confidence", ""),
        "score_breakdown": signal.get("score_breakdown", {}),
        "positive_reasons": _safe_list(signal.get("positive_reasons", signal.get("reasons", []))),
        "negative_reasons": _safe_list(signal.get("negative_reasons", signal.get("risk_notes", []))),
        "wait_reasons": _safe_list(signal.get("wait_reasons", [])),
        "no_buy_conditions": signal.get("no_buy_conditions", ""),
        "raw_metrics": signal.get("raw_metrics", {}),
    }


def _fallback_virtual_decision(signal: Dict[str, Any]) -> Dict[str, Any]:
    score = int(_num(signal.get("score", signal.get("intraday_score", 0))))
    price = _num(signal.get("price", signal.get("current_price", 0)))
    entry_type = str(signal.get("entry_type", signal.get("signal_type", "仮想押し目")) or "仮想押し目")
    stop_loss = _num(signal.get("stop_loss"), price * 0.97 if price else 0)
    take_profit = _num(signal.get("target_1", signal.get("take_profit_1")), price * 1.04 if price else 0)

    if score >= 70:
        decision = "virtual_buy"
        confidence = "中"
    elif score >= 60:
        decision = "virtual_watch"
        confidence = "低〜中"
    else:
        decision = "virtual_avoid"
        confidence = "低"

    return {
        "decision": decision,
        "entry_type": entry_type,
        "confidence": confidence,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "max_hold_days": 5,
        "reasons": _safe_list(signal.get("positive_reasons", signal.get("reasons", [])))[:5]
        or ["固定スコアと現在のテクニカル条件から仮想判断"],
        "risk_factors": _safe_list(signal.get("negative_reasons", signal.get("risk_notes", [])))[:5]
        or ["出来高不足、地合い悪化、想定ライン割れ"],
        "model_used": "rule_based_fallback",
        "is_ai_generated": False,
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _validate_virtual_decision(raw: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    decision = str(raw.get("decision", fallback["decision"]))
    if decision not in VIRTUAL_DECISIONS:
        decision = fallback["decision"]

    return {
        "decision": decision,
        "entry_type": str(raw.get("entry_type") or fallback["entry_type"]),
        "confidence": str(raw.get("confidence") or fallback["confidence"]),
        "entry_price": _num(raw.get("entry_price"), fallback["entry_price"]),
        "stop_loss": _num(raw.get("stop_loss"), fallback["stop_loss"]),
        "take_profit": _num(raw.get("take_profit"), fallback["take_profit"]),
        "max_hold_days": int(_num(raw.get("max_hold_days"), fallback["max_hold_days"])),
        "reasons": _safe_list(raw.get("reasons")) or fallback["reasons"],
        "risk_factors": _safe_list(raw.get("risk_factors")) or fallback["risk_factors"],
        "model_used": str(raw.get("model_used") or get_setting("AI_VIRTUAL_MODEL", AI_VIRTUAL_MODEL)),
        "is_ai_generated": bool(raw.get("is_ai_generated", True)),
    }


def judge_virtual_trade(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe virtual trading decision. This never places real orders."""
    fallback = _fallback_virtual_decision(signal)
    api_key = get_setting("OPENAI_API_KEY", "")
    if not api_key:
        return fallback

    snapshot = build_market_snapshot(signal)
    prompt = f"""
あなたは日本株のpaper trading / virtual trading専用の検証エンジンです。
実売買推奨、発注、自動売買の指示は絶対にしません。
以下の時点スナップショットだけを使い、未来データを仮定せず、仮想取引判断をJSONだけで返してください。

decisionは必ず virtual_buy / virtual_watch / virtual_avoid のどれかです。
virtual_buyは検証用の仮想ポジション作成に使います。
virtual_watch / virtual_avoid は判断ログとして保存します。

返答JSONのキー:
decision, entry_type, confidence, entry_price, stop_loss, take_profit,
max_hold_days, reasons, risk_factors

スナップショット:
{json.dumps(snapshot, ensure_ascii=False, default=str)}
"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=20)
        response = client.responses.create(
            model=get_setting("AI_VIRTUAL_MODEL", AI_VIRTUAL_MODEL),
            input=prompt.strip(),
            max_output_tokens=700,
        )
        parsed = _extract_json_object(response.output_text)
        parsed["model_used"] = get_setting("AI_VIRTUAL_MODEL", AI_VIRTUAL_MODEL)
        parsed["is_ai_generated"] = True
        return _validate_virtual_decision(parsed, fallback)
    except Exception:
        return fallback
