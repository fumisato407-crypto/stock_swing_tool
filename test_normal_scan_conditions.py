from __future__ import annotations

import unittest

from scanner import apply_max_buy_candidate_limit


def _signal(code: str, score: int, category: str = "買い候補") -> dict:
    return {
        "code": code,
        "score": score,
        "category": category,
        "entry_type": "ブレイク狙い",
        "buy_condition_json": {},
    }


class NormalScanConditionsTest(unittest.TestCase):
    def test_max_buy_candidate_limit_keeps_only_top_n_buy_candidates(self) -> None:
        signals = [_signal("A", 90), _signal("B", 80), _signal("C", 70)]
        result = apply_max_buy_candidate_limit(signals, max_buy_candidates=1)

        buy = [item for item in result if item["category"] == "買い候補"]
        watch = [item for item in result if item["category"] == "監視"]

        self.assertEqual([item["code"] for item in buy], ["A"])
        self.assertEqual([item["code"] for item in watch], ["B", "C"])
        self.assertTrue(result[0]["buy_condition_json"]["max_buy_candidates_pass"])
        self.assertFalse(result[1]["buy_condition_json"]["max_buy_candidates_pass"])
        self.assertEqual(result[1]["entry_type"], "買い候補上限外")
        self.assertLess(result[1]["score"], 70)

    def test_non_buy_candidates_are_not_counted_against_limit(self) -> None:
        signals = [_signal("A", 95, "監視"), _signal("B", 90), _signal("C", 80)]
        result = apply_max_buy_candidate_limit(signals, max_buy_candidates=1)
        by_code = {item["code"]: item for item in result}

        self.assertEqual(by_code["A"]["category"], "監視")
        self.assertEqual(by_code["B"]["category"], "買い候補")
        self.assertEqual(by_code["C"]["category"], "監視")

    def test_limit_zero_demotes_all_buy_candidates(self) -> None:
        result = apply_max_buy_candidate_limit([_signal("A", 90), _signal("B", 80)], max_buy_candidates=0)

        self.assertEqual([item["category"] for item in result], ["監視", "監視"])
        self.assertFalse(result[0]["buy_condition_json"]["max_buy_candidates_pass"])


if __name__ == "__main__":
    unittest.main()
