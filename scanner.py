from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from ai_judge import generate_hyena_comment
from config import TRADES_PATH, WATCHLIST_PATH
from daily_top_n_filter import build_buy_condition_json, daily_score_from_filter, rank_daily_candidates
from data_fetcher import fetch_price_data, normalize_jp_symbol
from indicators import add_indicators
from intraday_scanner import fetch_intraday_data
from multi_timeframe_rules import evaluate_daily_filter, evaluate_multi_timeframe_signal, evaluate_risk_filter
from notifier import build_notification_text
from scoring import BUY_SCORE_THRESHOLD, WATCH_SCORE_THRESHOLD, score_stock


DEFAULT_DAILY_TOP_N = 10
DEFAULT_MIN_DAILY_SCORE = 70
WATCHLIST_COLUMNS = ["code", "name", "theme", "market", "raw_code", "normalized_symbol"]
CODE_COLUMN_CANDIDATES = ["code", "ticker", "銘柄コード", "コード"]
OPTIONAL_COLUMN_ALIASES = {
    "name": ["name", "銘柄名", "名称"],
    "theme": ["theme", "テーマ"],
    "market": ["market", "市場"],
}
TABLE_COLUMNS = [
    "code",
    "name",
    "price",
    "score",
    "category",
    "entry_type",
    "wait_condition",
    "entry_zone",
    "stop_loss",
    "target_1",
    "target_2",
    "expected_value_label",
    "confidence",
    "risk_reward",
    "risk_reward_label",
    "monitoring_reason",
    "entry_trigger",
    "invalidation_condition",
    "comment",
    "normalized_symbol",
    "error_type",
    "error_message",
    "fetched_rows",
    "last_attempt_at",
]
TRADES_COLUMNS = [
    "date",
    "code",
    "name",
    "score",
    "category",
    "entry_type",
    "entry_low",
    "entry_high",
    "stop_loss",
    "target_1",
    "target_2",
    "close_price",
    "result_1d",
    "result_3d",
    "result_5d",
    "memo",
]


def _apply_multi_timeframe_result(
    signal: Dict[str, Any],
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame,
    intraday_error_type: str = "",
    intraday_error_message: str = "",
    daily_min_ok: int = 3,
    intraday_min_ok: int = 2,
    use_vwap: bool = True,
    use_volume_spike: bool = True,
    use_risk_filter: bool = True,
    max_stop_loss_pct: float = 3.0,
    max_loss_yen_limit: float = 20000.0,
    min_risk_reward: float = 1.2,
    use_daily_top_n: bool = True,
    daily_top_n: int = DEFAULT_DAILY_TOP_N,
    min_daily_score: int = DEFAULT_MIN_DAILY_SCORE,
    max_buy_candidates: int = 1,
) -> Dict[str, Any]:
    mtf = evaluate_multi_timeframe_signal(
        daily_df,
        intraday_df,
        config={
            "daily_min_ok": daily_min_ok,
            "intraday_min_ok": intraday_min_ok,
            "use_vwap": use_vwap,
            "use_volume_spike": use_volume_spike,
        },
    )
    decision = str(mtf.get("decision_category", "avoid"))
    daily_filter = mtf.get("daily_filter", {})
    intraday_entry = mtf.get("intraday_entry", {})
    base_score = int(signal.get("score", 0) or 0)
    risk_filter = evaluate_risk_filter(
        signal,
        shares=100,
        config={
            "enabled": use_risk_filter,
            "max_stop_loss_pct": max_stop_loss_pct,
            "max_loss_yen_limit": max_loss_yen_limit,
            "min_risk_reward": min_risk_reward,
        },
    )
    risk_pass = bool(risk_filter.get("risk_pass"))

    if decision == "buy" and risk_pass:
        category = "買い候補"
        score = max(BUY_SCORE_THRESHOLD, int(mtf.get("score", base_score) or base_score))
        entry_type = str(mtf.get("entry_type") or signal.get("entry_type") or "ブレイク狙い")
        wait_condition = signal.get("wait_condition", "-")
        monitoring_reason = (
            f"日足{daily_filter.get('daily_ok_count', 0)}/4 + "
            f"5分足{intraday_entry.get('intraday_ok_count', 0)}/4 + リスクOK"
        )
    elif decision == "buy" and not risk_pass:
        category = "監視"
        score = min(BUY_SCORE_THRESHOLD - 1, max(WATCH_SCORE_THRESHOLD, int(mtf.get("score", base_score) or base_score)))
        entry_type = "リスク条件NG"
        wait_condition = "損切り幅、最大損失、損益比の改善待ち"
        monitoring_reason = "リスク条件NG: " + "、".join(risk_filter.get("risk_reasons", []))
    elif bool(daily_filter.get("daily_pass")):
        category = "監視"
        score = min(BUY_SCORE_THRESHOLD - 1, max(WATCH_SCORE_THRESHOLD, int(mtf.get("score", base_score) or base_score)))
        entry_type = "5分足エントリー待ち"
        wait_condition = "VWAP上、直近高値突破、押し目反発、出来高急増の達成待ち"
        monitoring_reason = f"日足は{daily_filter.get('daily_ok_count', 0)}/4で通過。5分足は{intraday_entry.get('intraday_ok_count', 0)}/4"
    else:
        category = "触らない"
        score = min(WATCH_SCORE_THRESHOLD - 1, int(mtf.get("score", base_score) or base_score))
        entry_type = "見送り"
        wait_condition = "日足フィルター未達"
        monitoring_reason = f"日足フィルター{daily_filter.get('daily_ok_count', 0)}/4で未達"

    updated = dict(signal)
    updated.update(
        {
            "score": score,
            "category": category,
            "entry_type": entry_type,
            "wait_condition": wait_condition,
            "monitoring_reason": monitoring_reason,
            "multi_timeframe_enabled": True,
            "multi_timeframe_pass": bool(mtf.get("multi_timeframe_pass")),
            "multi_timeframe_score": mtf.get("score"),
            "multi_timeframe_decision": decision,
            "daily_filter": daily_filter,
            "intraday_entry": intraday_entry,
            "daily_ok_count": daily_filter.get("daily_ok_count", 0),
            "daily_total_count": daily_filter.get("daily_total_count", 4),
            "daily_score": daily_filter.get("daily_score", daily_score_from_filter(daily_filter)),
            "intraday_ok_count": intraday_entry.get("intraday_ok_count", 0),
            "intraday_total_count": intraday_entry.get("intraday_total_count", 4),
            "daily_filter_json": daily_filter,
            "intraday_entry_json": intraday_entry,
            "multi_timeframe_detail": mtf.get("detail_json", {}),
            "multi_timeframe_reasons": mtf.get("reasons", []),
            "intraday_error_type": intraday_error_type,
            "intraday_error_message": intraday_error_message,
            "risk_filter": risk_filter,
            "risk_filter_json": risk_filter.get("risk_filter_json", risk_filter),
            "risk_pass": risk_pass,
            "stop_loss_pct": risk_filter.get("stop_loss_pct"),
            "max_loss_yen": risk_filter.get("max_loss_yen"),
            "expected_profit_yen": risk_filter.get("expected_profit_yen"),
            "risk_reward_ratio": risk_filter.get("risk_reward_ratio"),
            "risk_reasons": risk_filter.get("risk_reasons", []),
        }
    )
    for key in ("daily_rank_at_scan", "daily_rank_total", "daily_top_n_pass", "daily_top_n"):
        if key in signal:
            updated[key] = signal.get(key)
    updated["buy_condition_json"] = build_buy_condition_json(
        updated,
        {
            "use_daily_top_n": use_daily_top_n,
            "daily_top_n": updated.get("daily_top_n", daily_top_n),
            "min_daily_score": min_daily_score,
            "min_score": BUY_SCORE_THRESHOLD,
            "daily_min_ok": daily_min_ok,
            "intraday_min_ok": intraday_min_ok,
            "use_vwap": use_vwap,
            "use_volume_spike": use_volume_spike,
            "use_risk_filter": use_risk_filter,
            "max_stop_loss_pct": risk_filter.get("max_stop_loss_pct"),
            "max_loss_yen_limit": risk_filter.get("max_loss_yen_limit"),
            "min_risk_reward": risk_filter.get("min_risk_reward"),
            "max_buy_candidates": max_buy_candidates,
        },
    )
    updated["score_breakdown"] = dict(updated.get("score_breakdown") or {})
    updated["score_breakdown"]["total_score"] = score
    positives = list(updated.get("positive_reasons") or [])
    negatives = list(updated.get("negative_reasons") or [])
    waits = list(updated.get("wait_reasons") or [])
    positives.extend([reason for reason in mtf.get("reasons", []) if reason not in positives])
    if intraday_error_message:
        waits.append(f"5分足未取得: {intraday_error_message}")
    if category == "監視":
        waits.append(wait_condition)
    if category == "触らない":
        negatives.append(monitoring_reason)
    if not risk_pass:
        negatives.extend(risk_filter.get("risk_reasons", []))
    updated["positive_reasons"] = list(dict.fromkeys(positives))
    updated["negative_reasons"] = list(dict.fromkeys(negatives))
    updated["wait_reasons"] = list(dict.fromkeys(waits))
    return updated


def load_watchlist(path: Path = WATCHLIST_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)
    except Exception as exc:
        empty = pd.DataFrame(columns=WATCHLIST_COLUMNS)
        empty.attrs["load_error"] = f"watchlist.csvを読み込めませんでした: {exc}"
        return empty

    df.columns = [str(col).strip().replace("\ufeff", "") for col in df.columns]
    code_col = next((col for col in CODE_COLUMN_CANDIDATES if col in df.columns), "")
    if not code_col:
        empty = pd.DataFrame(columns=WATCHLIST_COLUMNS)
        empty.attrs["load_error"] = "銘柄コード列が見つかりません。code / ticker / 銘柄コード / コード のいずれかを用意してください。"
        return empty

    normalized = df[code_col].map(normalize_jp_symbol)
    df["raw_code"] = df[code_col].astype(str).str.strip()
    df["normalized_symbol"] = normalized
    df["code"] = normalized.str.replace(r"\.T$", "", regex=True)

    for target_col, aliases in OPTIONAL_COLUMN_ALIASES.items():
        if target_col not in df.columns:
            source_col = next((col for col in aliases if col in df.columns), "")
            df[target_col] = df[source_col] if source_col else ""

    for col in WATCHLIST_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    valid_df = df[df["normalized_symbol"].astype(str).str.len() > 0].copy()
    result = valid_df[WATCHLIST_COLUMNS]
    if result.empty and not df.empty:
        result.attrs["load_error"] = "有効な銘柄コードが見つかりません。コードの空欄、NaN、形式崩れを確認してください。"
    return result


def apply_max_buy_candidate_limit(
    signals: List[Dict[str, Any]],
    max_buy_candidates: int = 1,
) -> List[Dict[str, Any]]:
    sorted_signals = sorted([dict(signal) for signal in signals], key=lambda item: item.get("score", -1), reverse=True)
    buy_seen = 0
    limit = max(0, int(max_buy_candidates or 0))
    for signal in sorted_signals:
        if signal.get("category") != "買い候補":
            continue
        buy_seen += 1
        condition = signal.get("buy_condition_json") if isinstance(signal.get("buy_condition_json"), dict) else {}
        condition = dict(condition)
        condition["max_buy_candidates"] = limit
        if buy_seen <= limit:
            condition["max_buy_candidates_pass"] = True
            signal["buy_condition_json"] = condition
            continue
        signal["category"] = "監視"
        signal["score"] = min(BUY_SCORE_THRESHOLD - 1, int(signal.get("score", 0) or 0))
        signal["entry_type"] = "買い候補上限外"
        signal["wait_condition"] = "次回スキャンで上位買い候補入り待ち"
        signal["monitoring_reason"] = f"1回の最大買い候補件数 {limit}件を超過"
        condition["max_buy_candidates_pass"] = False
        signal["buy_condition_json"] = condition
    return sorted_signals


def scan_watchlist(
    records: Iterable[Dict[str, Any]],
    period: str = "6mo",
    use_daily_top_n: bool = True,
    daily_top_n: int = DEFAULT_DAILY_TOP_N,
    min_daily_score: int = DEFAULT_MIN_DAILY_SCORE,
    daily_min_ok: int = 3,
    intraday_min_ok: int = 2,
    use_vwap: bool = True,
    use_volume_spike: bool = True,
    use_risk_filter: bool = True,
    max_stop_loss_pct: float = 3.0,
    max_loss_yen_limit: float = 20000.0,
    min_risk_reward: float = 1.2,
    max_buy_candidates: int = 1,
) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    daily_signals: List[Dict[str, Any]] = []
    for record in records:
        raw_code = record.get("raw_code") or record.get("code", "")
        normalized_symbol = normalize_jp_symbol(record.get("normalized_symbol") or raw_code)
        if not normalized_symbol:
            continue
        code = normalized_symbol.replace(".T", "")
        fetched = fetch_price_data(code, period=period)
        if fetched.error:
            error_message = fetched.error_message or fetched.error or "株価取得に失敗しました。"
            signals.append(
                {
                    "code": code,
                    "name": record.get("name", ""),
                    "theme": record.get("theme", ""),
                    "market": record.get("market", ""),
                    "raw_code": raw_code,
                    "normalized_symbol": fetched.ticker,
                    "error_type": fetched.error_type,
                    "error_message": error_message,
                    "fetched_rows": fetched.fetched_rows,
                    "last_attempt_at": fetched.last_attempt_at,
                    "price": None,
                    "score": 0,
                    "category": "取得失敗",
                    "entry_type": "-",
                    "wait_condition": "-",
                    "entry_zone": "-",
                    "stop_loss": None,
                    "target_1": None,
                    "target_2": None,
                    "expected_value_label": "-",
                    "confidence": "-",
                    "risk_reward": "-",
                    "risk_reward_label": "-",
                    "comment": error_message,
                    "notification_text": f"{code} {record.get('name', '')}: {error_message}",
                    "history": fetched.data,
                    "score_breakdown": {
                        "market_score": 0,
                        "trend_score": 0,
                        "pullback_score": 0,
                        "volume_score": 0,
                        "theme_score": 0,
                        "risk_penalty": 0,
                        "total_score": 0,
                    },
                    "positive_reasons": [],
                    "negative_reasons": [error_message],
                    "wait_reasons": [],
                    "risk_notes": [error_message],
                    "reasons": [],
                    "buy_condition": "-",
                    "entry_trigger": "-",
                    "invalid_conditions": error_message,
                    "invalidation_condition": error_message,
                    "no_buy_conditions": error_message,
                    "monitoring_reason": error_message,
                    "last_date": "-",
                }
            )
            continue

        analyzed = add_indicators(fetched.data)
        scoring_record = dict(record)
        scoring_record["code"] = code
        scoring_record["normalized_symbol"] = fetched.ticker
        signal = score_stock(analyzed, scoring_record)
        daily_filter = evaluate_daily_filter(
            analyzed,
            config={
                "daily_min_ok": daily_min_ok,
                "intraday_min_ok": intraday_min_ok,
                "use_vwap": use_vwap,
                "use_volume_spike": use_volume_spike,
            },
        )
        signal.update(
            {
                "daily_filter": daily_filter,
                "daily_filter_json": daily_filter,
                "daily_ok_count": daily_filter.get("daily_ok_count", 0),
                "daily_total_count": daily_filter.get("daily_total_count", 4),
                "daily_score": daily_filter.get("daily_score", daily_score_from_filter(daily_filter)),
                "history": analyzed.tail(160),
                "normalized_symbol": fetched.ticker,
                "error_type": "",
                "error_message": "",
                "fetched_rows": fetched.fetched_rows,
                "last_attempt_at": fetched.last_attempt_at,
            }
        )
        daily_signals.append(signal)

    ranked_daily_signals = rank_daily_candidates(
        daily_signals,
        top_n=daily_top_n,
        min_daily_score=min_daily_score,
        enabled=use_daily_top_n,
    )

    for signal in ranked_daily_signals:
        code = str(signal.get("code") or "").replace(".T", "")
        if use_daily_top_n and not signal.get("daily_top_n_pass"):
            signal["category"] = "触らない"
            signal["score"] = min(WATCH_SCORE_THRESHOLD - 1, int(signal.get("score", 0) or 0))
            signal["entry_type"] = "日足上位N対象外"
            signal["wait_condition"] = "日足スコア上位N入り待ち"
            signal["monitoring_reason"] = (
                f"日足順位 {signal.get('daily_rank_at_scan', '-')}位 / "
                f"{signal.get('daily_rank_total', '-')}銘柄中、"
                f"日足スコア {signal.get('daily_score', 0)}点"
            )
            signal["buy_condition_json"] = build_buy_condition_json(
                signal,
                {
                    "use_daily_top_n": use_daily_top_n,
                    "daily_top_n": daily_top_n,
                    "min_daily_score": min_daily_score,
                    "min_score": BUY_SCORE_THRESHOLD,
                    "daily_min_ok": daily_min_ok,
                    "intraday_min_ok": intraday_min_ok,
                    "use_vwap": use_vwap,
                    "use_volume_spike": use_volume_spike,
                    "use_risk_filter": use_risk_filter,
                    "max_stop_loss_pct": max_stop_loss_pct,
                    "max_loss_yen_limit": max_loss_yen_limit,
                    "min_risk_reward": min_risk_reward,
                    "max_buy_candidates": max_buy_candidates,
                },
            )
            signal["comment"] = generate_hyena_comment(signal)
            signal["notification_text"] = build_notification_text(signal)
            signals.append(signal)
            continue

        intraday = fetch_intraday_data(code, interval="5m", period="5d")
        signal = _apply_multi_timeframe_result(
            signal,
            signal.get("history") if isinstance(signal.get("history"), pd.DataFrame) else pd.DataFrame(),
            intraday.data if not intraday.error else pd.DataFrame(),
            intraday_error_type=intraday.error_type,
            intraday_error_message=intraday.error_message,
            daily_min_ok=daily_min_ok,
            intraday_min_ok=intraday_min_ok,
            use_vwap=use_vwap,
            use_volume_spike=use_volume_spike,
            use_risk_filter=use_risk_filter,
            max_stop_loss_pct=max_stop_loss_pct,
            max_loss_yen_limit=max_loss_yen_limit,
            min_risk_reward=min_risk_reward,
            use_daily_top_n=use_daily_top_n,
            daily_top_n=daily_top_n,
            min_daily_score=min_daily_score,
            max_buy_candidates=max_buy_candidates,
        )
        signal["comment"] = generate_hyena_comment(signal)
        signal["notification_text"] = build_notification_text(signal)
        signals.append(signal)

    return apply_max_buy_candidate_limit(signals, max_buy_candidates=max_buy_candidates)


def build_signal_table(signals: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        row = {}
        for col in TABLE_COLUMNS:
            value = signal.get(col, "")
            if col == "risk_reward":
                value = _format_table_risk_reward(value)
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _format_table_risk_reward(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value or "-")


def ensure_trades_file(path: Path = TRADES_PATH) -> None:
    if not path.exists():
        pd.DataFrame(columns=TRADES_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def append_trade_candidate(signal: Dict[str, Any], memo: str = "", path: Path = TRADES_PATH) -> None:
    ensure_trades_file(path)
    row = {
        "date": date.today().isoformat(),
        "code": signal.get("code", ""),
        "name": signal.get("name", ""),
        "score": signal.get("score", ""),
        "category": signal.get("category", ""),
        "entry_type": signal.get("entry_type", ""),
        "entry_low": signal.get("entry_zone_low", ""),
        "entry_high": signal.get("entry_zone_high", ""),
        "stop_loss": signal.get("stop_loss", ""),
        "target_1": signal.get("target_1", ""),
        "target_2": signal.get("target_2", ""),
        "close_price": signal.get("price", ""),
        "result_1d": "",
        "result_3d": "",
        "result_5d": "",
        "memo": memo,
    }
    pd.DataFrame([row], columns=TRADES_COLUMNS).to_csv(
        path,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig",
    )
