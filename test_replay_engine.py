from __future__ import annotations

import unittest

import pandas as pd

from replay_engine import (
    CANDIDATE_MODE_TECHNICAL_ONLY,
    apply_replay_money_metrics,
    create_replay_trade,
    evaluate_replay_trade_outcome,
    run_replay,
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


def _technical_daily_df() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=130, freq="D")
    close = [100 + idx * 0.35 for idx in range(len(index))]
    return pd.DataFrame(
        {
            "Open": [value - 0.4 for value in close],
            "High": [value + 1.2 for value in close],
            "Low": [value - 1.2 for value in close],
            "Close": close,
            "Volume": [5_000_000 + idx * 1000 for idx in range(len(index))],
        },
        index=index,
    )


def _technical_intraday_df() -> pd.DataFrame:
    index = pd.date_range("2026-05-11 09:00", periods=80, freq="5min")
    close = [145 + idx * 0.08 for idx in range(len(index))]
    return pd.DataFrame(
        {
            "Open": [value - 0.1 for value in close],
            "High": [value + 0.5 for value in close],
            "Low": [value - 0.5 for value in close],
            "Close": close,
            "Volume": [800_000 + idx * 500 for idx in range(len(index))],
        },
        index=index,
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

    def test_create_replay_trade_preserves_technical_fields(self) -> None:
        signal = {
            "current_price": 100,
            "stop_loss": 97,
            "take_profit_1": 104,
            "intraday_score": 75,
            "signal_type": "ブレイク狙い",
            "technical_preset": "standard_swing",
            "technical_score_raw": 44,
            "technical_penalty_score": -3,
            "technical_score_final": 41,
            "technical_judgement": "条件付き買い",
            "technical_confidence": 90,
            "technical_score_breakdown": {"trend": ["25日線より上 +4"]},
            "technical_penalty_reasons": ["長い上ヒゲ -5"],
            "technical_hard_filter_reason": [],
        }
        current_bar = pd.Series({"Close": 100})

        trade = create_replay_trade(signal, current_bar, "2026-01-01 09:00")

        self.assertEqual(trade["technical_preset"], "standard_swing")
        self.assertEqual(trade["technical_score_final"], 41)
        self.assertEqual(trade["technical_score_breakdown"]["trend"], ["25日線より上 +4"])

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

    def test_technical_only_mode_bypasses_existing_score_and_creates_trade(self) -> None:
        result = run_replay(
            symbol="5803.T",
            name="テスト",
            df=_technical_intraday_df(),
            daily_df=_technical_daily_df(),
            rule_config={
                "candidate_generation_mode": CANDIDATE_MODE_TECHNICAL_ONLY,
                "min_score": 999,
                "max_trades": 2,
                "cooldown_bars": 6,
                "interval": "5m",
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
                    "use_penalty": True,
                    "use_hard_filter": False,
                },
            },
        )

        self.assertGreater(len(result["trades"]), 0)
        self.assertEqual(result["trades"][0]["rule_name"], "technical_only_replay_rule")
        self.assertEqual(result["trades"][0]["candidate_generation_mode"], CANDIDATE_MODE_TECHNICAL_ONLY)
        self.assertGreater(result["summary"]["stage_counts"]["technical_score_attempt_count"], 0)
        self.assertEqual(result["summary"]["stage_counts"]["final_virtual_buy_count"], len(result["trades"]))


if __name__ == "__main__":
    unittest.main()
