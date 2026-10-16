# -*- coding: utf-8 -*-
"""engagement_structure_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from engagement_structure_analysis import (decile_table, within_cell_rho,
                                           trend_across_deciles)


def make_df(n_months=2, n_domains=2, per_cell=150, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_months):
        for d in range(n_domains):
            for _ in range(per_cell):
                p = rng.uniform(0, 1)
                likes = rng.integers(0, 5)
                rows.append({
                    "month": f"20230{m + 1}",
                    "domain": f"d{d}",
                    "p_ai": p,
                    "likes": likes,
                    "collections": np.sqrt(likes),
                    "comments": rng.integers(0, 3),
                })
    return pd.DataFrame(rows)


class TestDecileTable(unittest.TestCase):
    def test_columns_and_shares(self):
        df = make_df()
        df["decile"] = pd.qcut(df["p_ai"], 10, labels=False, duplicates="drop")
        t = decile_table(df)
        self.assertEqual(len(t), 10)
        for col in ("likes_zero_share", "likes_gini", "coll_per_like",
                    "comm_per_like", "likes_top10_share"):
            self.assertIn(col, t.columns)
        self.assertTrue(((t["likes_zero_share"] >= 0)
                         & (t["likes_zero_share"] <= 1)).all())
        self.assertTrue(((t["likes_gini"] >= 0)
                         & (t["likes_gini"] <= 1)).all())

    def test_zero_share_ci_contains_point_estimate(self):
        df = make_df()
        df["decile"] = pd.qcut(df["p_ai"], 10, labels=False, duplicates="drop")
        t = decile_table(df)
        for d, r in t.iterrows():
            lo, hi = r["likes_zero_ci"]
            self.assertLessEqual(lo, r["likes_zero_share"])
            self.assertGreaterEqual(hi, r["likes_zero_share"])


class TestWithinCellRho(unittest.TestCase):
    def test_positive_when_metric_increases_with_p_ai(self):
        df = make_df()
        df["likes"] = (df["p_ai"] * 100).astype(int)
        cells = within_cell_rho(df)
        self.assertEqual(cells["likes"]["n_cells"], 4)
        self.assertGreater(cells["likes"]["weighted_rho"], 0.99)
        self.assertAlmostEqual(cells["likes"]["neg_cell_share"], 0.0)

    def test_skips_small_cells(self):
        df = make_df(per_cell=50)
        cells = within_cell_rho(df)
        self.assertEqual(cells["likes"]["n_cells"], 0)
        self.assertTrue(np.isnan(cells["likes"]["weighted_rho"]))


class TestTrend(unittest.TestCase):
    def test_monotone_trend_detected(self):
        t = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5]},
                         index=[0, 1, 2, 3, 4])
        tr = trend_across_deciles(t, "x")
        self.assertAlmostEqual(tr["rho"], 1.0)
        self.assertLess(tr["p"], 0.05)


if __name__ == "__main__":
    unittest.main()
