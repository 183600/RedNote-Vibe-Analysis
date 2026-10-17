# -*- coding: utf-8 -*-
"""author_style_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from author_style_analysis import (bootstrap_mean_ci, bootstrap_wls_slope,
                                   cluster_bootstrap_slope, decompose_slope,
                                   demean_group, demean_two_way,
                                   factorize_codes, ols_slope,
                                   paired_style_deltas, top_author_shares,
                                   variance_decomposition)


class TestFactorize(unittest.TestCase):
    def test_codes_and_count(self):
        codes, k = factorize_codes(["b", "a", "b", "c"])
        self.assertEqual(k, 3)
        self.assertEqual(codes.tolist(), [1, 0, 1, 2])


class TestDemean(unittest.TestCase):
    def test_group_means_removed(self):
        x = np.array([1.0, 3.0, 10.0, 14.0])
        codes = np.array([0, 0, 1, 1])
        out = demean_group(x, codes)
        self.assertAlmostEqual(out[0], -1.0)
        self.assertAlmostEqual(out[1], 1.0)
        self.assertAlmostEqual(out[2], -2.0)
        self.assertAlmostEqual(out[3], 2.0)

    def test_weighted_group_means_removed(self):
        x = np.array([0.0, 2.0, 10.0])
        codes = np.array([0, 0, 1])
        w = np.array([1.0, 3.0, 2.0])
        out = demean_group(x, codes, w)
        self.assertAlmostEqual(out[0], -1.5)
        self.assertAlmostEqual(out[1], 0.5)
        self.assertAlmostEqual(out[2], 0.0)

    def test_two_way_removes_both_factors(self):
        codes_a = np.array([0, 0, 1, 1, 2, 2])
        codes_b = np.array([0, 1, 0, 1, 0, 1])
        x = np.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
        out = demean_two_way(x, codes_a, codes_b)
        means_a = np.bincount(codes_a, weights=out) / np.bincount(codes_a)
        means_b = np.bincount(codes_b, weights=out) / np.bincount(codes_b)
        for m in (means_a, means_b):
            for v in m:
                self.assertAlmostEqual(v, 0.0, places=8)


class TestOlsSlope(unittest.TestCase):
    def test_simple_slope(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 3.0, 5.0, 7.0])
        self.assertAlmostEqual(ols_slope(x, y), 2.0)

    def test_offset_slope_matches_centered(self):
        x = np.array([10.0, 11.0, 12.0])
        y = np.array([100.0, 102.0, 104.0])
        self.assertAlmostEqual(ols_slope(x, y), 2.0)

    def test_weighted_slope(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        w = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(ols_slope(x, y, w), 1.0)


class TestDecompose(unittest.TestCase):
    def test_identity_holds(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 5, 200)
        x = rng.normal(size=200) + codes
        y = -0.5 * x + rng.normal(size=200)
        dec = decompose_slope(x, y, codes)
        combined = (dec["between_weight"] * dec["between"]
                    + (1 - dec["between_weight"]) * dec["within"])
        self.assertAlmostEqual(dec["pooled"], combined, places=10)

    def test_pure_between_variation(self):
        codes = np.repeat(np.arange(4), 3)
        x = np.repeat([1.0, 2.0, 3.0, 4.0], 3)
        y = 2.0 * x
        dec = decompose_slope(x, y, codes)
        # 组内无离差：between_weight=1，组内斜率未定义（NaN 或 0）
        self.assertAlmostEqual(dec["between_weight"], 1.0)
        self.assertTrue(np.isnan(dec["within"]) or dec["within"] == 0.0)
        self.assertAlmostEqual(dec["between"], 2.0, places=8)
        self.assertAlmostEqual(dec["pooled"], 2.0, places=8)

    def test_pure_within_variation(self):
        codes = np.repeat(np.arange(2), 3)
        x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        y = 3.0 * x
        dec = decompose_slope(x, y, codes)
        self.assertAlmostEqual(dec["within"], 3.0, places=8)
        self.assertAlmostEqual(dec["between_weight"], 0.0, places=8)


class TestVarianceDecomposition(unittest.TestCase):
    def test_icc_one_when_groups_constant(self):
        codes = np.repeat(np.arange(4), 5)
        x = np.repeat([0.0, 1.0, 2.0, 3.0], 5)
        v = variance_decomposition(x, codes)
        self.assertAlmostEqual(v["icc"], 1.0)
        self.assertAlmostEqual(v["sd_within"], 0.0)

    def test_icc_low_for_iid(self):
        rng = np.random.default_rng(42)
        codes = rng.integers(0, 100, 5000)
        x = rng.normal(size=5000)
        v = variance_decomposition(x, codes)
        self.assertLess(v["icc"], 0.05)

    def test_single_group_nan(self):
        v = variance_decomposition([1.0, 2.0], ["a", "a"])
        self.assertTrue(np.isnan(v["icc"]))


class TestPairedDeltas(unittest.TestCase):
    def test_only_authors_with_both_halves(self):
        ids = ["a", "a", "b", "c", "c"]
        p = [0.1, 0.9, 0.1, 0.2, 0.8]
        y = [1.0, 2.0, 9.0, 0.0, 1.0]
        d = paired_style_deltas(ids, p, y)
        self.assertEqual(d.size, 2)
        self.assertAlmostEqual(d[0], 1.0)
        self.assertAlmostEqual(d[1], 1.0)

    def test_no_author_with_both_halves(self):
        d = paired_style_deltas(["a", "a"], [0.1, 0.2], [1.0, 2.0])
        self.assertEqual(d.size, 0)


class TestTopAuthorShares(unittest.TestCase):
    def test_known_shares(self):
        counts = np.array([10.0, 5.0, 3.0, 2.0])
        indicator = np.array([5.0, 1.0, 0.0, 0.0])
        out = top_author_shares(counts, indicator, frac=0.25)
        self.assertEqual(out["top_n"], 1)
        self.assertAlmostEqual(out["note_share"], 10.0 / 20.0)
        self.assertAlmostEqual(out["hit_share"], 5.0 / 6.0)


class TestBootstrap(unittest.TestCase):
    def test_mean_ci_contains_mean(self):
        rng = np.random.default_rng(1)
        x = rng.normal(5.0, 1.0, 500)
        out = bootstrap_mean_ci(x, n_boot=500, seed=7)
        self.assertLess(out["ci_low"], out["mean"])
        self.assertGreater(out["ci_high"], out["mean"])

    def test_wls_slope_recovers_known_slope(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=400)
        y = -1.5 * x + rng.normal(0, 0.3, 400)
        w = rng.integers(1, 5, 400).astype(float)
        out = bootstrap_wls_slope(x, y, w, n_boot=200, seed=3)
        self.assertAlmostEqual(out["slope"], -1.5, delta=0.15)
        self.assertLess(out["ci_low"], out["slope"])
        self.assertGreater(out["ci_high"], out["slope"])

    def test_cluster_bootstrap_degenerate_groups_nan(self):
        # 每组一条观测：无组内变异，点估计 NaN 且不产生噪声 CI
        out = cluster_bootstrap_slope([1.0, 2.0, 3.0], [2.0, 4.0, 6.0],
                                      ["a", "b", "c"], n_boot=20, seed=1)
        self.assertTrue(np.isnan(out["slope"]))
        self.assertEqual(out["n_boot"], 0)

    def test_cluster_bootstrap_within_zero(self):
        # 组内 y 与 x 无关、组间趋势显著 -> 人内斜率远小于总体斜率且 CI 覆盖 0
        rng = np.random.default_rng(3)
        n_groups, per = 100, 20
        codes = np.repeat(np.arange(n_groups), per)
        x = codes + rng.normal(0, 1.0, n_groups * per)
        y = 2.0 * codes + rng.normal(0, 1.0, n_groups * per)
        out = cluster_bootstrap_slope(x, y, codes, n_boot=100, seed=5)
        pooled = ols_slope(x, y)
        self.assertGreater(pooled, 1.0)
        self.assertLess(abs(out["slope"]), abs(pooled) / 10)
        self.assertLess(out["ci_low"], 0.0)
        self.assertGreater(out["ci_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
