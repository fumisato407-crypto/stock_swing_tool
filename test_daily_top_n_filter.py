from __future__ import annotations

import unittest

from daily_top_n_filter import build_buy_condition_json, daily_score_from_filter, rank_daily_candidates


def _candidate(symbol: str, score: int, ok_count: int = 3) -> dict:
    return {
        "symbol": symbol,
        "daily_score": score,
        "daily_ok_count": ok_count,
        "daily_total_count": 4,
        "daily_filter": {"daily_ok_count": ok_count, "daily_total_count": 4},
    }


class DailyTopNFilterTest(unittest.TestCase):
    def test_daily_score_from_filter(self) -> None:
        self.assertEqual(daily_score_from_filter({"daily_ok_count": 3, "daily_total_count": 4}), 75)

    def test_candidates_are_ranked_by_daily_score(self) -> None:
        ranked = rank_daily_candidates([_candidate("A", 75), _candidate("B", 100), _candidate("C", 50)], top_n=2)
        self.assertEqual([item["symbol"] for item in ranked], ["B", "A", "C"])
        self.assertEqual(ranked[0]["daily_rank_at_scan"], 1)

    def test_only_top_n_pass(self) -> None:
        ranked = rank_daily_candidates([_candidate(str(i), 100 - i) for i in range(12)], top_n=10)
        passed = [item for item in ranked if item["daily_top_n_pass"]]
        self.assertEqual(len(passed), 10)
        self.assertFalse(ranked[10]["daily_top_n_pass"])

    def test_min_daily_score_filters_low_score(self) -> None:
        ranked = rank_daily_candidates([_candidate("A", 69), _candidate("B", 80)], top_n=10, min_daily_score=70)
        by_symbol = {item["symbol"]: item for item in ranked}
        self.assertFalse(by_symbol["A"]["daily_top_n_pass"])
        self.assertTrue(by_symbol["B"]["daily_top_n_pass"])

    def test_disabled_keeps_all_candidates(self) -> None:
        ranked = rank_daily_candidates([_candidate(str(i), 10) for i in range(12)], top_n=5, enabled=False)
        self.assertEqual(sum(1 for item in ranked if item["daily_top_n_pass"]), 12)

    def test_buy_condition_json_contains_daily_rank(self) -> None:
        signal = _candidate("A", 75)
        signal.update({"daily_rank_at_scan": 3, "daily_rank_total": 30, "intraday_ok_count": 2, "intraday_total_count": 4})
        result = build_buy_condition_json(signal, {"daily_top_n": 10, "min_daily_score": 70})
        self.assertEqual(result["daily_rank_at_scan"], 3)
        self.assertEqual(result["daily_top_n"], 10)
        self.assertEqual(result["min_daily_score"], 70)


if __name__ == "__main__":
    unittest.main()
