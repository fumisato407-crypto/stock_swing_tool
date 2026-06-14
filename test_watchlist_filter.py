from __future__ import annotations

import unittest

import pandas as pd

from watchlist_filter import filter_watchlist_symbols, normalize_symbol, parse_symbol_input


def _watchlist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"code": "6501", "normalized_symbol": "6501.T", "name": "日立製作所"},
            {"code": "6723", "normalized_symbol": "6723.T", "name": "ルネサス"},
            {"code": "6752", "normalized_symbol": "6752.T", "name": "パナソニック"},
            {"code": "7751", "normalized_symbol": "7751.T", "name": "キヤノン"},
        ]
    )


class WatchlistFilterTest(unittest.TestCase):
    def test_normalize_symbol_adds_suffix(self) -> None:
        self.assertEqual(normalize_symbol("6501"), "6501.T")
        self.assertEqual(normalize_symbol(" 6501.t "), "6501.T")

    def test_parse_comma_separated(self) -> None:
        self.assertEqual(parse_symbol_input("6501,6723.T"), ["6501.T", "6723.T"])

    def test_parse_newline_separated(self) -> None:
        self.assertEqual(parse_symbol_input("6501\n6723"), ["6501.T", "6723.T"])

    def test_parse_space_separated(self) -> None:
        self.assertEqual(parse_symbol_input("6501 6723 6752"), ["6501.T", "6723.T", "6752.T"])

    def test_include_symbols_filters_watchlist(self) -> None:
        result = filter_watchlist_symbols(_watchlist(), include_symbols=["6501.T", "6752.T"])
        self.assertEqual(result["final_count"], 2)
        self.assertEqual(result["filtered_watchlist"]["normalized_symbol"].tolist(), ["6501.T", "6752.T"])

    def test_exclude_symbols_removes_symbols(self) -> None:
        result = filter_watchlist_symbols(_watchlist(), exclude_symbols=["6501.T", "7751.T"])
        self.assertEqual(result["final_count"], 2)
        self.assertNotIn("6501.T", result["filtered_watchlist"]["normalized_symbol"].tolist())
        self.assertIn("6501.T", result["excluded_symbols"])

    def test_exclude_wins_over_include(self) -> None:
        result = filter_watchlist_symbols(
            _watchlist(),
            include_symbols=["6501.T", "6723.T"],
            exclude_symbols=["6501.T"],
        )
        self.assertEqual(result["final_count"], 1)
        self.assertEqual(result["filtered_watchlist"]["normalized_symbol"].tolist(), ["6723.T"])

    def test_missing_symbols_are_reported(self) -> None:
        result = filter_watchlist_symbols(_watchlist(), include_symbols=["6501.T", "9999.T"])
        self.assertEqual(result["final_count"], 1)
        self.assertIn("9999.T", result["missing_symbols"])

    def test_final_zero_count(self) -> None:
        result = filter_watchlist_symbols(_watchlist(), include_symbols=["6501.T"], exclude_symbols=["6501.T"])
        self.assertEqual(result["final_count"], 0)


if __name__ == "__main__":
    unittest.main()
