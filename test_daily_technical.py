import unittest
from types import SimpleNamespace

import pandas as pd

import daily_technical as daily


def _sample_daily_frame(rows: int = 140) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="B")
    base = pd.Series(range(rows), index=index).astype(float)
    return pd.DataFrame(
        {
            "Open": 1000 + base * 2,
            "High": 1015 + base * 2,
            "Low": 990 + base * 2,
            "Close": 1008 + base * 2,
            "Volume": 1_200_000 + (base % 8) * 120_000,
        },
        index=index,
    )


class DailyTechnicalTest(unittest.TestCase):
    def test_add_daily_technical_indicators_adds_required_columns(self):
        prepared = daily.add_daily_technical_indicators(_sample_daily_frame())

        for column in ["ma5", "ma25", "ma75", "rsi14", "macd", "atr14", "is_bullish_candle"]:
            self.assertIn(column, prepared.columns)
        self.assertGreater(len(prepared), 120)

    def test_score_daily_technical_item_returns_required_fields(self):
        prepared = daily.add_daily_technical_indicators(_sample_daily_frame())
        result = daily.score_daily_technical_item(
            {
                "code": "6501",
                "name": "TEST",
                "normalized_symbol": "6501.T",
                "data": prepared,
                "last_attempt_at": "2026-06-15 09:00",
            }
        )

        self.assertEqual(result["code"], "6501")
        self.assertGreaterEqual(result["technical_score_final"], 0)
        self.assertLessEqual(result["technical_score_final"], 60)
        self.assertIn("score_breakdown", result)
        self.assertIn("hold_days_hint", result)
        self.assertIn("buy_timing_hint", result)

    def test_fetch_invalid_symbol_records_nonblank_error(self):
        daily_data, errors = daily.fetch_daily_technical_data([{"code": None, "name": "EMPTY"}])

        self.assertEqual(daily_data, {})
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0]["error_type"])
        self.assertTrue(errors[0]["error_message"])
        self.assertEqual(errors[0]["fetch_target"], "daily_technical")

    def test_fetch_empty_dataframe_records_nonblank_error(self):
        original_fetch = daily.fetch_price_data
        try:
            daily.fetch_price_data = lambda *args, **kwargs: SimpleNamespace(
                error=None,
                error_type="",
                error_message="",
                data=pd.DataFrame(),
                last_attempt_at="2026-06-15 09:00",
            )
            daily_data, errors = daily.fetch_daily_technical_data([{"code": "6501", "name": "TEST"}])
        finally:
            daily.fetch_price_data = original_fetch

        self.assertEqual(daily_data, {})
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_type"], "empty_dataframe")
        self.assertTrue(errors[0]["error_message"])

    def test_score_with_missing_indicator_data_does_not_raise(self):
        prepared = daily.add_daily_technical_indicators(_sample_daily_frame(rows=20))
        result = daily.score_daily_technical_item(
            {
                "code": "6501",
                "name": "SHORT",
                "normalized_symbol": "6501.T",
                "data": prepared,
                "last_attempt_at": "2026-06-15 09:00",
            }
        )

        self.assertIn("missing_data", result)
        self.assertTrue(result["missing_data"])
        self.assertIn("データ不足", result["hard_filter_reason"])
        self.assertLess(result["confidence"], 100)

    def test_preset_and_group_switches_disable_sections(self):
        prepared = daily.add_daily_technical_indicators(_sample_daily_frame())
        config = daily.get_technical_config_for_preset("lightweight")
        config["use_volume"] = False
        result = daily.score_daily_technical_item(
            {
                "code": "6501",
                "name": "LIGHT",
                "normalized_symbol": "6501.T",
                "data": prepared,
                "last_attempt_at": "2026-06-15 09:00",
            },
            config=config,
        )

        self.assertEqual(result["technical_preset"], "lightweight")
        self.assertEqual(result["volume_score"], 0)
        self.assertIn("設定で未使用", result["score_breakdown"]["volume"])

    def test_score_technical_item_respects_as_of(self):
        frame = _sample_daily_frame(rows=140)
        as_of = frame.index[100]
        result = daily.score_technical_item(
            frame,
            config=daily.get_default_technical_config(),
            code="6501",
            name="ASOF",
            normalized_symbol="6501.T",
            as_of=as_of,
        )

        self.assertEqual(result["latest_metrics"]["date"], as_of.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    unittest.main()
