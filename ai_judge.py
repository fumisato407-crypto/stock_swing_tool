from __future__ import annotations

import os
from typing import Any, Dict

from config import OPENAI_API_KEY, OPENAI_MODEL


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
