# -*- coding: utf-8 -*-
"""stance_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from stance_analysis import (CATEGORY_WORDS, decile_table, density_per_100,
                             domain_standardized_series, length_bin_codes,
                             standardize_weights, word_counts)


class TestWordCounts(unittest.TestCase):
    def test_per_word_counts(self):
        counts = word_counts("可能明天应该可以，可能吧")
        self.assertEqual(counts["hedge"]["可能"], 2)
        self.assertEqual(counts["hedge"]["应该"], 1)
        self.assertEqual(counts["_words"]["可能"], 2)

    def test_buting_not_counted_as_boost(self):
        """「不一定」只计入 hedge 的「不一定」，不会同时计入 boost 的「一定」。"""
        counts = word_counts("不一定，一定")
        self.assertEqual(counts["hedge"]["不一定"], 1)
        self.assertEqual(counts["_words"].get("一定", 0), 1)
        self.assertEqual(counts["boost"].get("一定", 0), 1)
        counts2 = word_counts("不一定")
        self.assertEqual(counts2["boost"].get("一定", 0), 0)

    def test_category_isolation(self):
        counts = word_counts("姐妹们真的超级必须集美")
        self.assertEqual(counts["vocative"]["姐妹们"], 1)
        self.assertEqual(counts["vocative"]["集美"], 1)
        self.assertEqual(counts["intens"]["真的"], 1)
        self.assertEqual(counts["intens"]["超级"], 1)
        self.assertEqual(counts["boost"]["必须"], 1)

    def test_all_category_words_roundtrip(self):
        text = "".join(w for k in CATEGORY_WORDS for w in CATEGORY_WORDS[k][1])
        counts = word_counts(text)
        for k in CATEGORY_WORDS:
            for w in CATEGORY_WORDS[k][1]:
                self.assertEqual(counts[k].get(w, 0), 1, msg=f"{k}:{w}")

    def test_no_match_returns_empty_dicts(self):
        counts = word_counts("今天天气不错")
        for k in CATEGORY_WORDS:
            self.assertEqual(counts[k], {})


class TestDensity(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(density_per_100(1, 50), 2.0)
        self.assertAlmostEqual(density_per_100(3, 0), 0.0)


class TestDecileTable(unittest.TestCase):
    def test_monotone_signal(self):
        scores = np.linspace(0, 1, 1000)
        values = scores * 2.0
        rows = decile_table(values, scores)
        self.assertEqual(len(rows), 10)
        means = [r["mean"] for r in rows]
        self.assertEqual(means, sorted(means))
        self.assertEqual(sum(r["n"] for r in rows), 1000)
        self.assertLessEqual(rows[0]["ci_low"], rows[0]["mean"])

    def test_constant_scores(self):
        rows = decile_table([1.0, 2.0], [0.5, 0.5])
        self.assertEqual(rows, [])


class TestLengthBins(unittest.TestCase):
    def test_boundaries(self):
        codes = length_bin_codes([49, 50, 149, 150, 799, 800, 10_000])
        np.testing.assert_array_equal(codes, [0, 0, 0, 1, 3, 4, 4])


class TestStandardizeWeights(unittest.TestCase):
    def test_normalize(self):
        w = standardize_weights({"a": 2.0, "b": 2.0})
        self.assertAlmostEqual(w["a"], 0.5)
        self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_all_zero_equal_weights(self):
        w = standardize_weights({"a": 0.0, "b": 0.0})
        self.assertAlmostEqual(w["a"], 0.5)
        self.assertAlmostEqual(w["b"], 0.5)


class TestDomainStandardizedSeries(unittest.TestCase):
    def test_weights_follow_domain_shares(self):
        df = pd.DataFrame({
            "quarter": ["Q1", "Q1", "Q1", "Q2", "Q2", "Q2"],
            "domain": ["a", "a", "b", "a", "a", "b"],
            "v": [1.0, 1.0, 5.0, 3.0, 3.0, 7.0],
        })
        s = domain_standardized_series(df, "v", "quarter")
        # 领域记录占比 a=4/6, b=2/6；Q1 = (1*2/3 + 5*1/3)
        self.assertAlmostEqual(s["Q1"], 1.0 * (2 / 3) + 5.0 * (1 / 3))
        self.assertAlmostEqual(s["Q2"], 3.0 * (2 / 3) + 7.0 * (1 / 3))

    def test_missing_domain_renormalizes(self):
        df = pd.DataFrame({
            "quarter": ["Q1", "Q1", "Q1", "Q2", "Q2"],
            "domain": ["a", "a", "b", "a", "a"],
            "v": [1.0, 1.0, 5.0, 3.0, 3.0],
        })
        s = domain_standardized_series(df, "v", "quarter")
        self.assertAlmostEqual(s["Q2"], 3.0)


if __name__ == "__main__":
    unittest.main()
