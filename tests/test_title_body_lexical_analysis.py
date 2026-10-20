# -*- coding: utf-8 -*-
"""title_body_lexical_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from title_body_lexical_analysis import (domain_standardized_block,
                                         excerpt_sensitivity_block,
                                         length_control_block, mean_ci,
                                         subset_block, title_bin_codes)


class TestTitleBinCodes(unittest.TestCase):
    def test_boundaries(self):
        got = title_bin_codes([1, 5, 6, 9, 10, 13, 14, 19, 20, 100])
        np.testing.assert_array_equal(got, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_invalid_values_are_negative(self):
        got = title_bin_codes([0, -1, float("nan")])
        np.testing.assert_array_equal(got, [-1, -1, -1])


class TestMeanCi(unittest.TestCase):
    def test_mean_and_symmetric_interval(self):
        out = mean_ci([1.0, 2.0, 3.0])
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["mean"], 2.0)
        self.assertAlmostEqual(out["ci_low"], 2.0 - 1.96 / np.sqrt(3))
        self.assertAlmostEqual(out["ci_high"], 2.0 + 1.96 / np.sqrt(3))

    def test_empty(self):
        out = mean_ci([np.nan])
        self.assertEqual(out["n"], 0)
        self.assertTrue(np.isnan(out["mean"]))


def make_scan(coverage, domains=None):
    n = len(coverage)
    return {
        "n_total": n,
        "body_len": np.full(n, 100, dtype=np.int32),
        "n_title": np.full(n, 3, dtype=np.int32),
        "coverage": np.asarray(coverage, dtype=float),
        "jaccard": np.asarray(coverage, dtype=float) / 10,
        "excerpt": np.zeros(n, dtype=np.int8),
        "domain": domains or ["A"] * n,
    }


class TestSubsetBlock(unittest.TestCase):
    def setUp(self):
        self.scan = {
            "n_total": 4,
            "body_len": np.array([100, 40, 100, 100], dtype=np.int32),
            "n_title": np.array([3, 3, 0, 2], dtype=np.int32),
            "coverage": np.array([0.5, 0.5, np.nan, 1.0]),
            "jaccard": np.array([0.1, 0.1, np.nan, 0.2]),
            "excerpt": np.array([1, 0, 0, 0], dtype=np.int8),
        }

    def test_filters_short_bodies_and_empty_titles(self):
        out = subset_block(self.scan, np.arange(4))
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["dropped_short_body"], 1)
        self.assertEqual(out["dropped_no_title_tokens"], 1)
        self.assertAlmostEqual(out["coverage"]["mean"], 0.75)
        self.assertAlmostEqual(out["coverage_median"], 0.75)
        self.assertAlmostEqual(out["coverage_high_share"], 1.0)
        self.assertAlmostEqual(out["excerpt"]["share"], 0.5)
        self.assertAlmostEqual(out["title_tokens_mean"], 2.5)

    def test_ci_is_bounded(self):
        out = subset_block(self.scan, np.arange(4))
        lo, hi = out["coverage_high_ci"]
        self.assertLessEqual(lo, out["coverage_high_share"])
        self.assertGreaterEqual(hi, out["coverage_high_share"])


class TestLengthControlBlock(unittest.TestCase):
    def test_same_length_bins_recover_mean_gap(self):
        scans = {
            "human": make_scan([0.8, 0.6, 0.7, 0.9]),
            "aigc": make_scan([0.2, 0.3, 0.2, 0.3]),
            "exploring": make_scan([0.6, 0.7, 0.6, 0.7]),
        }
        keeps = {name: np.arange(4) for name in scans}
        out = length_control_block(scans, keeps)
        self.assertAlmostEqual(out["title_matched"]["aigc"]["coverage"]["gap"], 0.5)
        self.assertAlmostEqual(out["body_matched"]["aigc"]["coverage"]["gap"], 0.5)
        self.assertAlmostEqual(out["title_matched"]["exploring"]["coverage"]["gap"], 0.1)
        self.assertGreater(out["title_matched"]["aigc"]["coverage"]["se"], 0.0)
        self.assertEqual(out["body_bins"]["human"][0]["n"], 4)
        self.assertAlmostEqual(out["body_bins"]["human"][0]["coverage"], 0.75)


class TestExcerptSensitivityBlock(unittest.TestCase):
    def test_excludes_excerpt_titles(self):
        human = make_scan([1.0, 0.6, 0.8, 0.5])
        human["excerpt"] = np.array([1, 0, 0, 0], dtype=np.int8)
        scans = {
            "human": human,
            "aigc": make_scan([0.2, 0.3, 0.4, 0.3]),
            "exploring": make_scan([0.3, 0.4, 0.5, 0.4]),
        }
        keeps = {name: np.arange(4) for name in scans}
        out = excerpt_sensitivity_block(scans, keeps)
        self.assertEqual(out["means"]["human"]["n"], 3)
        self.assertAlmostEqual(out["means"]["human"]["coverage_mean"], 0.6333333333)
        self.assertAlmostEqual(
            out["title_matched"]["aigc"]["coverage"]["gap"], 1 / 3, places=5)


class TestDomainStandardizedBlock(unittest.TestCase):
    def test_uses_human_domain_weights(self):
        scans = {
            "human": make_scan([0.9, 0.9, 0.1, 0.1], ["A", "A", "B", "B"]),
            "aigc": make_scan([0.8, 0.8, 0.2, 0.2], ["A", "A", "B", "B"]),
            "exploring": make_scan([0.8, 0.8, 0.2, 0.2], ["A", "A", "B", "B"]),
        }
        keeps = {name: np.arange(4) for name in scans}
        out = domain_standardized_block(scans, keeps)
        self.assertAlmostEqual(out["human"]["coverage"], 0.5)
        self.assertAlmostEqual(out["aigc"]["coverage"], 0.5)
        self.assertAlmostEqual(out["exploring"]["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
