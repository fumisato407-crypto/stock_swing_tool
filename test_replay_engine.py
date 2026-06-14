from __future__ import annotations

import unittest

import pandas as pd

from replay_engine import evaluate_replay_trade_outcome


def _future_df(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [row[1] for row in rows],
            "High": [row[2] for row in rows],
            "Low": [row[3] for row in rows],
            "Close": [row[4] for row in rows],
            "Volume": [1000 for _ in rows],
        },
        index=pd.to_datetime([row[0] for row in rows]),
    )


class ReplayOutcomeTest(unittest.TestCase):
    def test_take_profit_uses_entry_to_exit_window(self) -> None:
        trade = {
            "signal_time": "2026-01-01 09:00",
            "entry_price": 100,
            "stop_loss": 97,
            "take_profit": 104,
        }
        future = _future_df([
            ("2026-01-01 09:05", 100, 105, 98, 104),
        ])

        result = evaluate_replay_trade_outcome(trade, future, interval="5m")

        self.assertEqual(result["outcome"], "hit_take_profit")
        self.assertEqual(result["return_pct"], 4.0)
        self.assertEqual(result["max_profit_pct"], 5.0)
        self.assertEqual(result["max_drawdown_pct"], -2.0)

    def test_stop_loss_uses_entry_to_exit_window(self) -> None:
        trade = {
            "signal_time": "2026-01-01 09:00",
            "entry_price": 100,
            "stop_loss": 97,
            "take_profit": 105,
        }
        future = _future_df([
            ("2026-01-01 09:05", 100, 102, 96, 97),
        ])

        result = evaluate_replay_trade_outcome(trade, future, interval="5m")

        self.assertEqual(result["outcome"], "hit_stop_loss")
        self.assertEqual(result["return_pct"], -3.0)
        self.assertEqual(result["max_profit_pct"], 2.0)
        self.assertEqual(result["max_drawdown_pct"], -4.0)

    def test_pre_entry_low_is_not_used_for_drawdown(self) -> None:
        trade = {
            "signal_time": "2026-01-01 09:00",
            "entry_price": 100,
            "stop_loss": 90,
            "take_profit": 110,
        }
        all_rows = _future_df([
            ("2026-01-01 08:55", 100, 101, 50, 100),
            ("2026-01-01 09:05", 100, 102, 98, 101),
        ])
        future = all_rows.loc[all_rows.index > pd.Timestamp(trade["signal_time"])]

        result = evaluate_replay_trade_outcome(trade, future, interval="1d")

        self.assertEqual(result["max_drawdown_pct"], -2.0)


if __name__ == "__main__":
    unittest.main()
