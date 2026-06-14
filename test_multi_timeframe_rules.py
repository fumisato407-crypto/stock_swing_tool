from __future__ import annotations

import unittest

import pandas as pd

from multi_timeframe_rules import (
    build_replay_daily_context,
    evaluate_daily_filter,
    evaluate_intraday_entry,
    evaluate_multi_timeframe_signal,
)


def _daily_data() -> pd.DataFrame:
    index = pd.date_range("2026-04-01", periods=30, freq="B")
    close = pd.Series(range(100, 130), index=index, dtype=float)
    data = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1000.0,
        },
        index=index,
    )
    data.iloc[-1, data.columns.get_loc("Volume")] = 1500.0
    return data


def _intraday_data() -> pd.DataFrame:
    index = pd.date_range("2026-05-15 09:00", periods=15, freq="5min")
    data = pd.DataFrame(
        {
            "Open": [100.0] * 15,
            "High": [101.0] * 14 + [104.5],
            "Low": [99.0] * 14 + [100.0],
            "Close": [100.0] * 14 + [104.0],
            "Volume": [100.0] * 14 + [200.0],
        },
        index=index,
    )
    return data


class MultiTimeframeRulesTest(unittest.TestCase):
    def test_daily_filter_conditions(self) -> None:
        result = evaluate_daily_filter(_daily_data())
        self.assertTrue(result["ma25_ok"])
        self.assertTrue(result["above_prev_close_ok"])
        self.assertTrue(result["near_5day_high_ok"])
        self.assertTrue(result["daily_volume_increase_ok"])
        self.assertTrue(result["daily_pass"])

    def test_intraday_entry_conditions(self) -> None:
        result = evaluate_intraday_entry(_intraday_data())
        self.assertTrue(result["vwap_ok"])
        self.assertTrue(result["breakout_ok"])
        self.assertTrue(result["pullback_rebound_ok"])
        self.assertTrue(result["intraday_volume_spike_ok"])
        self.assertTrue(result["intraday_pass"])

    def test_multi_timeframe_buy(self) -> None:
        result = evaluate_multi_timeframe_signal(_daily_data(), _intraday_data())
        self.assertEqual(result["decision_category"], "buy")
        self.assertTrue(result["multi_timeframe_pass"])
        self.assertGreaterEqual(result["score"], 70)

    def test_current_time_does_not_use_future_intraday_bar(self) -> None:
        data = _intraday_data()
        current_time = data.index[-2]
        result = evaluate_intraday_entry(data, current_time=current_time)
        self.assertFalse(result["breakout_ok"])
        self.assertFalse(result["intraday_volume_spike_ok"])

    def test_replay_daily_context_uses_only_visible_intraday(self) -> None:
        intraday = _intraday_data()
        prior_daily = _daily_data()
        current_time = intraday.index[-2]
        context = build_replay_daily_context(intraday, current_time, prior_daily_df=prior_daily)
        self.assertLessEqual(context.index.max(), current_time.normalize())
        self.assertEqual(float(context.iloc[-1]["Close"]), 100.0)


if __name__ == "__main__":
    unittest.main()
