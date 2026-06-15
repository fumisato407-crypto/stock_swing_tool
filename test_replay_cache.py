from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from replay_batch import (
    ConditionVariant,
    build_config_hash,
    build_run_summary_export,
    build_trade_detail_export,
    build_condition_variants,
)
from replay_cache import (
    get_cached_technical_score,
    set_cached_technical_score,
    stable_config_hash,
    technical_score_cache_key,
)
from replay_engine import (
    CANDIDATE_MODE_EXISTING,
    CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL,
    CANDIDATE_MODE_TECHNICAL_ONLY,
    evaluate_replay_step,
    leak_check_for_trade,
    run_replay,
)
from historical_scan_replay import HistoricalScanReplayConfig, attach_watchlist_replay_exports


def _daily_df() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=130, freq="D")
    close = [100 + idx * 0.2 for idx in range(len(index))]
    return pd.DataFrame(
        {
            "Open": [value - 0.3 for value in close],
            "High": [value + 1.0 for value in close],
            "Low": [value - 1.0 for value in close],
            "Close": close,
            "Volume": [4_000_000 for _ in close],
        },
        index=index,
    )


def _intraday_df() -> pd.DataFrame:
    index = pd.date_range("2026-05-11 09:00", periods=80, freq="5min")
    close = [130 + idx * 0.05 for idx in range(len(index))]
    return pd.DataFrame(
        {
            "Open": [value - 0.1 for value in close],
            "High": [value + 0.5 for value in close],
            "Low": [value - 0.5 for value in close],
            "Close": close,
            "Volume": [700_000 for _ in close],
        },
        index=index,
    )


class ReplayCacheTest(unittest.TestCase):
    def _existing_pass_signal(self) -> dict:
        return {
            "code": "5803.T",
            "name": "Fujikura",
            "current_price": 100,
            "judgement": "買い検討OK",
            "signal_type": "ブレイク狙い",
            "intraday_score": 80,
            "stop_loss": 98,
            "take_profit_1": 104,
            "reasons": ["mock existing pass"],
        }

    def _technical_score(self, score: int = 80) -> dict:
        return {
            "technical_score_raw": score,
            "technical_penalty_score": 0,
            "technical_score_final": score,
            "technical_judgement": "条件付き買い",
            "confidence": 90,
            "score_breakdown": {"trend": ["mock trend"]},
            "penalty_reasons": [],
            "hard_filter_reason": [],
            "_cache_hit": False,
        }

    def test_existing_plus_runs_technical_after_existing_pass(self) -> None:
        with patch("replay_engine.evaluate_intraday_entry", return_value=self._existing_pass_signal()) as existing_mock:
            with patch("replay_engine._score_technical_for_history", return_value=self._technical_score(80)) as tech_mock:
                signal = evaluate_replay_step(
                    _intraday_df(),
                    _intraday_df().index[-1],
                    {
                        "candidate_generation_mode": CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL,
                        "min_score": 70,
                        "target_rule": "すべて",
                        "use_multi_timeframe": False,
                        "use_risk_filter": False,
                        "technical_min_score": 50,
                        "technical_min_confidence": 0,
                        "technical_config": {"use_hard_filter": False},
                    },
                    symbol="5803.T",
                    daily_df=_daily_df(),
                )
        existing_mock.assert_called_once()
        tech_mock.assert_called_once()
        self.assertTrue(signal["existing_logic_pass"])
        self.assertNotIn("replay_skip_reason", signal)

    def test_existing_plus_rejects_when_technical_score_fails_after_existing_pass(self) -> None:
        with patch("replay_engine.evaluate_intraday_entry", return_value=self._existing_pass_signal()):
            with patch("replay_engine._score_technical_for_history", return_value=self._technical_score(20)) as tech_mock:
                signal = evaluate_replay_step(
                    _intraday_df(),
                    _intraday_df().index[-1],
                    {
                        "candidate_generation_mode": CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL,
                        "min_score": 70,
                        "target_rule": "すべて",
                        "use_multi_timeframe": False,
                        "use_risk_filter": False,
                        "technical_min_score": 50,
                        "technical_min_confidence": 0,
                        "technical_config": {"use_hard_filter": False},
                    },
                    symbol="5803.T",
                    daily_df=_daily_df(),
                )
        tech_mock.assert_called_once()
        self.assertTrue(signal["existing_logic_pass"])
        self.assertEqual(signal["replay_skip_reason"], "technical_score_below_min")

    def test_technical_only_does_not_call_existing_logic(self) -> None:
        with patch("replay_engine.evaluate_intraday_entry") as existing_mock:
            with patch("replay_engine._score_technical_for_history", return_value=self._technical_score(80)):
                signal = evaluate_replay_step(
                    _intraday_df(),
                    _intraday_df().index[-1],
                    {
                        "candidate_generation_mode": CANDIDATE_MODE_TECHNICAL_ONLY,
                        "technical_min_score": 50,
                        "technical_min_confidence": 0,
                        "technical_config": {"use_hard_filter": False},
                    },
                    symbol="5803.T",
                    daily_df=_daily_df(),
                )
        existing_mock.assert_not_called()
        self.assertEqual(signal["candidate_generation_mode"], CANDIDATE_MODE_TECHNICAL_ONLY)
        self.assertNotIn("replay_skip_reason", signal)

    def test_existing_logic_does_not_require_technical_even_when_enabled_flag_is_true(self) -> None:
        with patch("replay_engine.evaluate_intraday_entry", return_value=self._existing_pass_signal()):
            with patch("replay_engine._score_technical_for_history") as tech_mock:
                signal = evaluate_replay_step(
                    _intraday_df(),
                    _intraday_df().index[-1],
                    {
                        "candidate_generation_mode": CANDIDATE_MODE_EXISTING,
                        "use_technical_score": True,
                        "min_score": 70,
                        "target_rule": "すべて",
                        "use_multi_timeframe": False,
                        "use_risk_filter": False,
                        "technical_min_score": 99,
                    },
                    symbol="5803.T",
                    daily_df=_daily_df(),
                )
        tech_mock.assert_not_called()
        self.assertTrue(signal["existing_logic_pass"])
        self.assertNotIn("replay_skip_reason", signal)

    def test_config_hash_changes_when_condition_changes(self) -> None:
        base = {"preset": "standard_swing", "use_hard_filter": True}
        changed = {"preset": "standard_swing", "use_hard_filter": False}
        self.assertNotEqual(stable_config_hash(base), stable_config_hash(changed))

    def test_same_condition_hits_technical_score_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay_cache.sqlite"
            key = technical_score_cache_key("5803.T", "2026-01-01 09:00", "standard_swing", {"use_trend": True})
            self.assertIsNone(get_cached_technical_score(key, path=path))
            set_cached_technical_score(key, {"technical_score_final": 32}, elapsed_seconds=0.5, path=path)
            cached = get_cached_technical_score(key, path=path)
            self.assertIsNotNone(cached)
            self.assertEqual(cached["technical_score_final"], 32)
            self.assertTrue(cached["_cache_hit"])

    def test_leak_check_rejects_future_feature_timestamp(self) -> None:
        result = leak_check_for_trade(
            {
                "decision_time": "2026-01-01 09:00",
                "feature_max_timestamp": "2026-01-01 09:05",
                "outcome_start_timestamp": "2026-01-01 09:10",
            }
        )
        self.assertEqual(result["leak_check_result"], "NG")

    def test_one_position_limit_reduces_duplicate_entries(self) -> None:
        common = {
            "candidate_generation_mode": CANDIDATE_MODE_TECHNICAL_ONLY,
            "use_technical_score": True,
            "technical_min_score": 0,
            "technical_config": {
                "preset_name": "standard_swing",
                "use_trend": True,
                "use_entry_position": True,
                "use_volume": True,
                "use_candle": True,
                "use_breakout": True,
                "use_momentum": True,
                "use_risk_reward": True,
                "use_penalty": False,
                "use_hard_filter": False,
            },
            "max_trades": 5,
            "cooldown_bars": 1,
            "interval": "5m",
            "use_replay_cache": False,
        }
        unrestricted = run_replay("5803.T", _intraday_df(), daily_df=_daily_df(), rule_config=common)
        restricted = run_replay(
            "5803.T",
            _intraday_df(),
            daily_df=_daily_df(),
            rule_config={**common, "one_position_per_symbol": True},
        )
        self.assertGreater(len(unrestricted["trades"]), len(restricted["trades"]))
        self.assertGreater(restricted["summary"]["stage_counts"]["position_open_excluded_count"], 0)

    def test_condition_variant_builder_cross_product(self) -> None:
        variants = build_condition_variants(
            modes=[CANDIDATE_MODE_TECHNICAL_ONLY],
            presets=["standard_swing", "breakout"],
            min_scores=[20, 32],
            hard_filter_options=[True, False],
        )
        self.assertEqual(len(variants), 8)

    def test_trade_detail_export_includes_run_conditions(self) -> None:
        context = {
            "run_id": "run-1",
            "run_datetime": "2026-06-15T09:00:00+09:00",
            "period": "1mo",
            "interval": "5m",
            "assumed_shares": 100,
            "feature_version": "test",
            "app_version": "test",
        }
        trades = [
            {
                "symbol": "5803.T",
                "candidate_mode": CANDIDATE_MODE_TECHNICAL_ONLY,
                "preset": "standard_swing",
                "min_technical_score": 32,
                "config_hash": "hash-1",
            }
        ]
        df = build_trade_detail_export(trades, context)
        self.assertIn("candidate_mode", df.columns)
        self.assertIn("preset", df.columns)
        self.assertIn("min_technical_score", df.columns)
        self.assertIn("config_hash", df.columns)
        self.assertEqual(df.loc[0, "candidate_mode"], CANDIDATE_MODE_TECHNICAL_ONLY)

    def test_run_summary_export_includes_conditions_and_counts(self) -> None:
        context = {
            "run_id": "run-1",
            "run_datetime": "2026-06-15T09:00:00+09:00",
            "period": "1mo",
            "interval": "5m",
            "target_symbols": "5803.T",
        }
        condition_rows = pd.DataFrame(
            [
                {
                    "condition_name": "standard",
                    "candidate_mode": CANDIDATE_MODE_TECHNICAL_ONLY,
                    "preset": "standard_swing",
                    "min_technical_score": 32,
                    "raw_signal_count": 292,
                    "final_trade_count": 24,
                    "settled_trade_count": 23,
                    "csv_export_count": 24,
                    "config_hash": "hash-1",
                }
            ]
        )
        df = build_run_summary_export(condition_rows, context)
        self.assertEqual(int(df.loc[0, "raw_signal_count"]), 292)
        self.assertEqual(int(df.loc[0, "final_trade_count"]), 24)
        self.assertEqual(int(df.loc[0, "csv_export_count"]), 24)
        self.assertEqual(df.loc[0, "preset"], "standard_swing")

    def test_config_hash_is_stable_without_run_id(self) -> None:
        variant = ConditionVariant(name="standard", technical_min_score=32)
        context_a = {"run_id": "run-a", "run_datetime": "2026-06-15T09:00:00+09:00", "period": "1mo"}
        context_b = {"run_id": "run-b", "run_datetime": "2026-06-15T10:00:00+09:00", "period": "1mo"}
        context_c = {"run_id": "run-c", "run_datetime": "2026-06-15T10:00:00+09:00", "period": "3mo"}
        self.assertEqual(build_config_hash(context_a, variant), build_config_hash(context_b, variant))
        self.assertNotEqual(build_config_hash(context_a, variant), build_config_hash(context_c, variant))

    def test_watchlist_replay_export_includes_conditions_and_counts(self) -> None:
        result = {
            "summary": {
                "replay_run_id": "watchlist-run-1",
                "run_datetime": "2026-06-15T09:00:00+09:00",
                "candidate_generation_mode": CANDIDATE_MODE_TECHNICAL_ONLY,
                "technical_preset": "standard_swing",
                "technical_min_score": 32,
                "technical_config": {"use_penalty": True, "use_hard_filter": False},
                "period": "1mo",
                "interval": "5m",
                "shares": 100,
                "cooldown": "30m",
                "one_position_per_symbol": True,
                "target_symbols": "5803.T",
                "stage_counts": {
                    "raw_signal_count": 10,
                    "cooldown_filtered_count": 3,
                    "position_open_excluded_count": 2,
                    "max_trades_excluded_count": 1,
                    "settled_trade_count": 4,
                },
            },
            "trades": [
                {
                    "symbol": "5803.T",
                    "name": "Fujikura",
                    "signal_time": "2026-06-15 09:30",
                    "status": "closed",
                    "exit_price": 105,
                }
                for _ in range(4)
            ],
        }
        enriched = attach_watchlist_replay_exports(result)
        detail = enriched["trade_detail_df"]
        summary = enriched["run_summary_df"]
        self.assertIn("mode", detail.columns)
        self.assertIn("candidate_mode", detail.columns)
        self.assertIn("preset", detail.columns)
        self.assertIn("config_hash", detail.columns)
        self.assertEqual(len(detail), enriched["summary"]["csv_export_count"])
        self.assertEqual(int(summary.loc[0, "raw_signal_count"]), 10)
        self.assertEqual(int(summary.loc[0, "csv_export_count"]), len(detail))

    def test_watchlist_replay_config_hash_changes_when_condition_changes(self) -> None:
        base = {
            "summary": {
                "replay_run_id": "watchlist-run-1",
                "run_datetime": "2026-06-15T09:00:00+09:00",
                "candidate_generation_mode": CANDIDATE_MODE_TECHNICAL_ONLY,
                "technical_preset": "standard_swing",
                "technical_min_score": 32,
                "period": "1mo",
                "interval": "5m",
            },
            "trades": [],
        }
        changed = {
            "summary": {
                **base["summary"],
                "technical_min_score": 36,
            },
            "trades": [],
        }
        self.assertNotEqual(
            attach_watchlist_replay_exports(base)["summary"]["config_hash"],
            attach_watchlist_replay_exports(changed)["summary"]["config_hash"],
        )

    def test_watchlist_replay_config_accepts_cache_and_parallel_settings(self) -> None:
        config = HistoricalScanReplayConfig(use_replay_cache=True, parallel=True, max_workers=2)
        self.assertTrue(config.use_replay_cache)
        self.assertTrue(config.parallel)
        self.assertEqual(config.max_workers, 2)

    def test_watchlist_replay_export_mode_and_new_counts(self) -> None:
        result = {
            "summary": {
                "replay_run_id": "watchlist-run-2",
                "run_datetime": "2026-06-15T09:00:00+09:00",
                "candidate_generation_mode": CANDIDATE_MODE_EXISTING_PLUS_TECHNICAL,
                "technical_preset": "standard_swing",
                "technical_min_score": 32,
                "use_replay_cache": True,
                "parallel": True,
                "max_workers": 2,
                "stage_counts": {
                    "scan_count": 100,
                    "existing_logic_checked_count": 80,
                    "existing_logic_pass_count": 20,
                    "existing_logic_reject_count": 60,
                    "technical_score_checked_count": 20,
                    "technical_score_pass_count": 12,
                    "technical_score_reject_count": 8,
                    "hard_filter_reject_count": 3,
                    "raw_signal_count": 9,
                },
            },
            "trades": [{"symbol": "5803.T", "status": "closed", "exit_price": 100}],
        }
        enriched = attach_watchlist_replay_exports(result)
        detail = enriched["trade_detail_df"]
        summary = enriched["run_summary_df"]
        self.assertEqual(detail.loc[0, "mode"], "watchlist_replay")
        self.assertEqual(int(detail.loc[0, "existing_logic_pass_count"]), 20)
        self.assertEqual(int(summary.loc[0, "technical_score_checked_count"]), 20)


if __name__ == "__main__":
    unittest.main()
