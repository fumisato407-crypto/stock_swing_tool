from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from auto_virtual_logger import run_rule_based_virtual_logging, select_rule_buy_candidates
from config import BASE_DIR, DEFAULT_PRICE_PERIOD, VIRTUAL_TRADES_DB_PATH
from market_hours import is_market_open_jst, market_status_label, next_market_open_hint, now_jst
from outcome_tracker import update_open_virtual_trade_outcomes
from scanner import load_watchlist, scan_watchlist


LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "market_runner.log"
_STOP_REQUESTED = False


def _handle_stop(signum: int, frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    logging.info("stop_requested signal=%s", signum)


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based virtual trade logger. Paper trading only.")
    parser.add_argument("--interval-minutes", type=int, choices=[1, 3, 5, 10], default=5)
    parser.add_argument("--target", choices=["buy", "buy_watch"], default="buy")
    parser.add_argument("--min-score", type=int, choices=[70, 75, 80], default=70)
    parser.add_argument("--max-candidates", type=int, choices=[1, 3, 5, 10], default=3)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force-market-open", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-openai", choices=["off"], default="off")
    parser.add_argument("--db-path", default=str(VIRTUAL_TRADES_DB_PATH))
    parser.add_argument("--period", default=DEFAULT_PRICE_PERIOD)
    return parser.parse_args()


def _log_event(event: str, payload: Dict[str, Any]) -> None:
    safe_payload = json.dumps(payload, ensure_ascii=False, default=str)
    logging.info("%s %s", event, safe_payload)


def _scan_signals(period: str) -> List[Dict[str, Any]]:
    watchlist = load_watchlist()
    records = watchlist.to_dict("records")
    return scan_watchlist(records, period=period)


def _run_once(args: argparse.Namespace) -> Dict[str, Any]:
    current = now_jst()
    status = market_status_label(current)
    db_path = Path(args.db_path)
    summary: Dict[str, Any] = {
        "timestamp_jst": current.strftime("%Y-%m-%d %H:%M:%S JST"),
        "market_status": status,
        "scan_elapsed_seconds": 0,
        "signals_count": 0,
        "failed_count": 0,
        "selected_count": 0,
        "saved_count": 0,
        "duplicate_count": 0,
        "failed_save_count": 0,
        "error": "",
        "db_path": str(db_path),
        "next_run_hint": next_market_open_hint(current),
        "dry_run": bool(args.dry_run),
        "use_openai": False,
    }

    if not args.force_market_open and not is_market_open_jst(current):
        summary["error"] = "market_closed"
        _log_event("skip", summary)
        return summary

    started = time.perf_counter()
    try:
        logging.info("scan_start")
        signals = _scan_signals(args.period)
        summary["scan_elapsed_seconds"] = round(time.perf_counter() - started, 2)
        summary["signals_count"] = len(signals)
        summary["failed_count"] = sum(1 for signal in signals if signal.get("category") == "取得失敗")
        selected = select_rule_buy_candidates(signals, args.target, args.min_score, args.max_candidates)
        summary["selected_count"] = len(selected)

        if args.dry_run:
            summary["dry_run_candidates"] = [
                {
                    "code": signal.get("code", ""),
                    "name": signal.get("name", ""),
                    "score": signal.get("score", 0),
                    "category": signal.get("category", ""),
                    "entry_type": signal.get("entry_type", ""),
                }
                for signal in selected
            ]
        else:
            log_summary = run_rule_based_virtual_logging(
                signals,
                args.target,
                args.min_score,
                args.max_candidates,
                db_path=db_path,
            )
            summary["saved_count"] = log_summary.get("saved_count", 0)
            summary["duplicate_count"] = log_summary.get("duplicate_count", 0)
            summary["failed_save_count"] = log_summary.get("failed_count", 0)
            summary["error"] = log_summary.get("error", "")
            try:
                outcome_summary = update_open_virtual_trade_outcomes()
                summary["outcome_update"] = outcome_summary
            except Exception as exc:
                summary["outcome_update_error"] = f"{exc.__class__.__name__}: {exc}"
    except Exception as exc:
        summary["error"] = f"{exc.__class__.__name__}: {exc}"

    _log_event("run_once", summary)
    return summary


def main() -> int:
    _setup_logging()
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    args = _parse_args()

    logging.info("market_runner_start log_path=%s db_path=%s use_openai=off", LOG_PATH, args.db_path)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_stop)
    while not _STOP_REQUESTED:
        summary = _run_once(args)
        print(json.dumps(summary, ensure_ascii=False, default=str))

        if args.once:
            break

        sleep_seconds = max(1, int(args.interval_minutes) * 60)
        logging.info(
            "next_run seconds=%s hint=%s",
            sleep_seconds,
            summary.get("next_run_hint") or next_market_open_hint(),
        )
        for _ in range(sleep_seconds):
            if _STOP_REQUESTED:
                break
            time.sleep(1)

    logging.info("market_runner_stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
