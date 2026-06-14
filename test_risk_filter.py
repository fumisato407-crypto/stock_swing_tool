from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from multi_timeframe_rules import evaluate_risk_filter
from replay_engine import evaluate_replay_step


def _signal(entry: float, stop: float, target: float) -> dict:
    return {
        "entry_price": entry,
        "current_price": entry,
        "stop_loss": stop,
        "take_profit_1": target,
    }


def _history() -> pd.DataFrame:
    index = pd.date_range("2026-01-01 09:00", periods=30, freq="5min")
    return pd.DataFrame(
        {
            "Open": [100.0] * len(index),
            "High": [101.0] * len(index),
            "Low": [99.0] * len(index),
            "Close": [100.0] * len(index),
            "Volume": [1000] * len(index),
        },
        index=index,
    )


def _patched_intraday_signal(entry: float = 12710, stop: float = 11850, target: float = 13000) -> dict:
    return {
        "code": "5802",
        "name": "住友電工",
        "judgement": "買い検討OK",
        "intraday_score": 80,
        "signal_type": "ブレイク狙い",
        "current_price": entry,
        "stop_loss": stop,
        "take_profit_1": target,
        "reasons": ["テスト用買い候補"],
    }


class RiskFilterTest(unittest.TestCase):
    def test_stop_loss_pct_under_three_percent_passes(self) -> None:
        result = evaluate_risk_filter(_signal(10000, 9850, 10300), shares=100)
        self.assertTrue(result["stop_loss_pct_ok"])
        self.assertTrue(result["risk_pass"])

    def test_stop_loss_pct_6_8_percent_fails(self) -> None:
        result = evaluate_risk_filter(_signal(12710, 11850, 13000), shares=100)
        self.assertFalse(result["stop_loss_pct_ok"])
        self.assertFalse(result["risk_pass"])

    def test_max_loss_under_20000_passes(self) -> None:
        result = evaluate_risk_filter(_signal(10000, 9900, 10250), shares=100)
        self.assertTrue(result["max_loss_yen_ok"])
        self.assertTrue(result["risk_pass"])

    def test_max_loss_86000_fails(self) -> None:
        result = evaluate_risk_filter(_signal(12710, 11850, 14000), shares=100)
        self.assertEqual(result["max_loss_yen"], 86000)
        self.assertFalse(result["max_loss_yen_ok"])
        self.assertFalse(result["risk_pass"])

    def test_risk_reward_1_2_or_more_passes(self) -> None:
        result = evaluate_risk_filter(_signal(10000, 9900, 10120), shares=100)
        self.assertGreaterEqual(result["risk_reward_ratio"], 1.2)
        self.assertTrue(result["risk_reward_ok"])
        self.assertTrue(result["risk_pass"])

    def test_risk_reward_under_1_2_fails(self) -> None:
        result = evaluate_risk_filter(_signal(10000, 9900, 10100), shares=100)
        self.assertLess(result["risk_reward_ratio"], 1.2)
        self.assertFalse(result["risk_reward_ok"])
        self.assertFalse(result["risk_pass"])

    def test_daily_and_intraday_ok_but_risk_ng_is_not_buy_candidate(self) -> None:
        with patch("replay_engine.evaluate_intraday_entry", return_value=_patched_intraday_signal()), patch(
            "replay_engine.evaluate_multi_timeframe_signal",
            return_value={
                "decision_category": "buy",
                "score": 80,
                "entry_type": "ブレイク狙い",
                "multi_timeframe_pass": True,
                "daily_filter": {"daily_ok_count": 4, "daily_total_count": 4},
                "intraday_entry": {"intraday_ok_count": 4, "intraday_total_count": 4},
                "reasons": ["日足OK", "5分足OK"],
                "detail_json": {},
            },
        ):
            result = evaluate_replay_step(
                _history(),
                _history().index[-1],
                {
                    "min_score": 70,
                    "use_risk_filter": True,
                    "shares": 100,
                    "max_stop_loss_pct": 3.0,
                    "max_loss_yen_limit": 20000,
                    "min_risk_reward": 1.2,
                },
                symbol="5802",
                name="住友電工",
            )

        self.assertEqual(result["replay_skip_reason"], "risk_filter_ng")
        self.assertFalse(result["risk_pass"])
        self.assertEqual(result["judgement"], "監視強化")

    def test_risk_filter_off_keeps_previous_buy_candidate(self) -> None:
        with patch("replay_engine.evaluate_intraday_entry", return_value=_patched_intraday_signal()), patch(
            "replay_engine.evaluate_multi_timeframe_signal",
            return_value={
                "decision_category": "buy",
                "score": 80,
                "entry_type": "ブレイク狙い",
                "multi_timeframe_pass": True,
                "daily_filter": {"daily_ok_count": 4, "daily_total_count": 4},
                "intraday_entry": {"intraday_ok_count": 4, "intraday_total_count": 4},
                "reasons": ["日足OK", "5分足OK"],
                "detail_json": {},
            },
        ):
            result = evaluate_replay_step(
                _history(),
                _history().index[-1],
                {"min_score": 70, "use_risk_filter": False, "shares": 100},
                symbol="5802",
                name="住友電工",
            )

        self.assertNotIn("replay_skip_reason", result)
        self.assertTrue(result["risk_pass"])
        self.assertEqual(result["judgement"], "買い検討OK")


if __name__ == "__main__":
    unittest.main()
