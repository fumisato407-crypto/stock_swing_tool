from __future__ import annotations

import unittest

import pandas as pd

from replay_engine import (
    apply_replay_money_metrics,
    create_replay_trade,
    evaluate_replay_trade_outcome,
    summarize_replay_money,
)


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
    def test_first_bar_stop_loss_exits_immediately(self) -> None:
        trade = {
            "signal_time": "2026-01-01 09:00",
            "entry_price": 100,
            "stop_loss": 97,
            "take_profit": 110,
        }
        future = _future_df([
            ("2026-01-01 09:05", 100, 101, 96, 98),
            ("2026-01-01 09:10", 98, 120, 80, 119),
        ])

        result = evaluate_replay_trade_outcome(trade, future, interval="5m")

        self.assertEqual(result["outcome"], "hit_stop_loss")
        self.assertEqual(result["exit_time"], "2026-01-01 09:05")
        self.assertEqual(result["exit_price"], 97)

    def test_third_bar_take_profit_exits_immediately(self) -> None:
        trade = {
            "signal_time": "2026-01-01 09:00",
            "entry_price": 100,
            "stop_loss": 95,
            "take_profit": 104,
        }
        future = _future_df([
            ("2026-01-01 09:05", 100, 101, 99, 100),
            ("2026-01-01 09:10", 100, 102, 99, 101),
            ("2026-01-01 09:15", 101, 105, 100, 104),
            ("2026-01-01 09:20", 104, 150, 103, 149),
        ])

        result = evaluate_replay_trade_outcome(trade, future, interval="5m")

        self.assertEqual(result["outcome"], "hit_take_profit")
        self.assertEqual(result["exit_time"], "2026-01-01 09:15")
        self.assertEqual(result["exit_price"], 104)

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

    def test_future_after_take_profit_is_not_used(self) -> None:
        trade = {
            "signal_time": "2026-01-01 09:00",
            "entry_price": 100,
            "stop_loss": 95,
            "take_profit": 104,
        }
        future = _future_df([
            ("2026-01-01 09:05", 100, 105, 99, 104),
            ("2026-01-01 09:10", 104, 180, 103, 179),
        ])

        result = evaluate_replay_trade_outcome(trade, future, interval="5m")

        self.assertEqual(result["exit_price"], 104)
        self.assertEqual(result["max_profit_pct"], 5.0)

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

    def test_future_after_stop_loss_is_not_used(self) -> None:
        trade = {
            "signal_time": "2026-01-01 09:00",
            "entry_price": 100,
            "stop_loss": 97,
            "take_profit": 120,
        }
        future = _future_df([
            ("2026-01-01 09:05", 100, 102, 96, 97),
            ("2026-01-01 09:10", 97, 98, 40, 41),
        ])

        result = evaluate_replay_trade_outcome(trade, future, interval="5m")

        self.assertEqual(result["exit_price"], 97)
        self.assertEqual(result["max_drawdown_pct"], -4.0)

    def test_stop_loss_priority_when_stop_and_target_hit_same_bar(self) -> None:
        trade = {
            "signal_time": "2026-01-01 09:00",
            "entry_price": 100,
            "stop_loss": 97,
            "take_profit": 104,
        }
        future = _future_df([
            ("2026-01-01 09:05", 100, 105, 96, 101),
        ])

        result = evaluate_replay_trade_outcome(trade, future, interval="5m")

        self.assertEqual(result["outcome"], "hit_stop_loss")
        self.assertEqual(result["exit_price"], 97)
        self.assertTrue(result["hit_stop_loss"])
        self.assertFalse(result["hit_take_profit"])

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

    def test_watchlist_scan_path_uses_same_exit_logic(self) -> None:
        signal = {
            "current_price": 100,
            "stop_loss": 97,
            "take_profit_1": 104,
            "intraday_score": 75,
            "signal_type": "ブレイク狙い",
        }
        current_bar = pd.Series({"Close": 100})
        trade = create_replay_trade(signal, current_bar, "2026-01-01 09:00")
        future = _future_df([
            ("2026-01-01 09:05", 100, 101, 99, 100),
            ("2026-01-01 09:10", 100, 105, 99, 104),
        ])

        result = evaluate_replay_trade_outcome(trade, future, interval="5m")

        self.assertEqual(result["outcome"], "hit_take_profit")
        self.assertEqual(result["exit_time"], "2026-01-01 09:10")
        self.assertEqual(result["exit_price"], 104)

    def test_profit_yen_for_100_shares(self) -> None:
        trades = [
            {
                "signal_time": "2026-01-01 09:00",
                "status": "closed",
                "entry_price": 7872,
                "exit_price": 7900,
            },
            {
                "signal_time": "2026-01-01 09:05",
                "status": "closed",
                "entry_price": 7802,
                "exit_price": 7800,
            },
        ]

        prepared = apply_replay_money_metrics(trades, shares=100)
        summary = summarize_replay_money(trades, shares=100)

        self.assertEqual(prepared[0]["profit_yen"], 2800)
        self.assertEqual(prepared[1]["profit_yen"], -200)
        self.assertEqual(prepared[0]["required_capital_yen"], 787200)
        self.assertEqual(prepared[1]["cumulative_profit_yen"], 2600)
        self.assertEqual(summary["gross_profit_yen"], 2800)
        self.assertEqual(summary["gross_loss_yen"], -200)
        self.assertEqual(summary["net_profit_yen"], 2600)
        self.assertEqual(summary["profit_loss_ratio"], 14.0)

    def test_open_trade_is_excluded_from_confirmed_profit(self) -> None:
        trades = [
            {
                "signal_time": "2026-01-01 09:00",
                "status": "open",
                "entry_price": 100,
                "exit_price": None,
            },
            {
                "signal_time": "2026-01-01 09:05",
                "status": "closed",
                "entry_price": 100,
                "exit_price": 98,
            },
        ]

        prepared = apply_replay_money_metrics(trades, shares=100)
        summary = summarize_replay_money(trades, shares=100)

        self.assertIsNone(prepared[0]["profit_yen"])
        self.assertEqual(summary["closed_trade_count"], 1)
        self.assertEqual(summary["net_profit_yen"], -200)


if __name__ == "__main__":
    unittest.main()
