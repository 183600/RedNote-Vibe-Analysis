# -*- coding: utf-8 -*-
"""title_body_congruence_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from title_body_congruence_analysis import (half_year_shares,
                                            model_congruence_table,
                                            quadrant_labels, quadrant_shares,
                                            quarterly_title_body_rho,
                                            share_with_ci)


class TestQuadrantLabels(unittest.TestCase):
    def test_boundaries_and_coding(self):
        title = np.array([0.1, 0.9, 0.1, 0.9, 0.5])
        body = np.array([0.1, 0.1, 0.9, 0.9, 0.5])
        labels = quadrant_labels(title, body, 0.5, 0.5)
        np.testing.assert_array_equal(labels, [0, 1, 2, 3, 3])

    def test_threshold_is_inclusive(self):
        labels = quadrant_labels([0.5], [0.5], 0.5, 0.5)
        np.testing.assert_array_equal(labels, [3])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            quadrant_labels([0.1, 0.2], [0.1], 0.5, 0.5)


class TestQuadrantShares(unittest.TestCase):
    def test_shares(self):
        shares = quadrant_shares(np.array([0, 0, 1, 2, 3, 3, 3, 0]))
        self.assertEqual(shares["n"], 8)
        self.assertAlmostEqual(shares["Q1_双低"], 0.375)
        self.assertAlmostEqual(shares["Q2_标题端高"], 0.125)
        self.assertAlmostEqual(shares["Q3_正文端高"], 0.125)
        self.assertAlmostEqual(shares["Q4_双高"], 0.375)
        total = sum(v for k, v in shares.items() if k.startswith("Q"))
        self.assertAlmostEqual(total, 1.0)

    def test_empty(self):
        shares = quadrant_shares(np.array([], dtype=int))
        self.assertEqual(shares["n"], 0)
        self.assertEqual(shares["Q4_双高"], 0.0)


class TestShareWithCi(unittest.TestCase):
    def test_bounds(self):
        out = share_with_ci([True] * 70 + [False] * 30)
        self.assertEqual(out["n"], 100)
        self.assertAlmostEqual(out["share"], 0.7)
        lo, hi = out["ci"]
        self.assertLess(lo, 0.7)
        self.assertGreater(hi, 0.7)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)


class TestHalfYearShares(unittest.TestCase):
    def test_buckets(self):
        labels = np.array([0, 3, 2])
        months = ["202301", "202307", "202401"]
        rows = half_year_shares(labels, months)
        self.assertEqual([r["half"] for r in rows],
                         ["2023H1", "2023H2", "2024H1"])
        self.assertEqual(rows[0]["n"], 1)
        self.assertAlmostEqual(rows[0]["Q1"]["share"], 1.0)
        self.assertAlmostEqual(rows[1]["Q4"]["share"], 1.0)
        self.assertAlmostEqual(rows[2]["Q3"]["share"], 1.0)

    def test_invalid_months_skipped(self):
        rows = half_year_shares([0, 0], ["", "202301"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n"], 1)


class TestQuarterlyRho(unittest.TestCase):
    def test_direction_and_keys(self):
        title = [0.1, 0.2, 0.3, 0.4]
        body = [0.1, 0.3, 0.2, 0.9]
        months = ["202301", "202301", "202304", "202304"]
        rows = quarterly_title_body_rho(title, body, months)
        self.assertEqual([r["quarter"] for r in rows], ["2023Q1", "2023Q2"])
        self.assertEqual(rows[0]["n"], 2)
        self.assertAlmostEqual(rows[0]["rho"], 1.0)
        self.assertAlmostEqual(rows[1]["rho"], 1.0)


class TestModelCongruence(unittest.TestCase):
    def test_monotone_relation(self):
        aigc = pd.DataFrame({
            "model_family": ["gpt"] * 4,
            "model": ["m1", "m2", "m3", "m4"],
            "title_p_ai": [0.1, 0.2, 0.3, 0.4],
            "body_p_ai": [0.2, 0.4, 0.6, 0.8],
        })
        table, stats = model_congruence_table(aigc)
        self.assertEqual(stats["n_models"], 4)
        self.assertAlmostEqual(stats["rho"], 1.0)
        self.assertEqual(table["n"].tolist(), [1, 1, 1, 1])

    def test_averages(self):
        aigc = pd.DataFrame({
            "model_family": ["gpt", "gpt"],
            "model": ["m1", "m1"],
            "title_p_ai": [0.2, 0.4],
            "body_p_ai": [0.5, 0.9],
        })
        table, _ = model_congruence_table(aigc)
        self.assertAlmostEqual(float(table["title_mean"].iloc[0]), 0.3)
        self.assertAlmostEqual(float(table["body_mean"].iloc[0]), 0.7)


if __name__ == "__main__":
    unittest.main()
