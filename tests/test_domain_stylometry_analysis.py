# -*- coding: utf-8 -*-
"""domain_stylometry_analysis 的纯函数单元测试（合成数据，不读研究数据）。"""
import json
import math
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from domain_stylometry_analysis import (HI, MONTH_FROM, auc_or_nan,
                                        dedup_sensitivity, domain_penalty,
                                        family_share_table, halfyear_of,
                                        json_safe, rate_summary, safe_spearman)


class TestPeriods(unittest.TestCase):
    def test_halfyear(self):
        self.assertEqual(halfyear_of("202301"), "2023H1")
        self.assertEqual(halfyear_of("202306"), "2023H1")
        self.assertEqual(halfyear_of("202307"), "2023H2")
        self.assertEqual(halfyear_of("202512"), "2025H2")
        self.assertIsNone(halfyear_of(None))
        self.assertIsNone(halfyear_of(""))
        self.assertIsNone(halfyear_of("2023"))
        self.assertIsNone(halfyear_of("2023ab"))


class TestRateSummary(unittest.TestCase):
    def test_mixed_values(self):
        r = rate_summary([0.95, 0.80, 0.20])
        self.assertEqual(r["n"], 3)
        self.assertAlmostEqual(r["mean"], 0.65, places=4)
        self.assertAlmostEqual(r["hi_share"], 1 / 3, places=4)
        self.assertLessEqual(r["hi_ci_low"], r["hi_share"])
        self.assertGreaterEqual(r["hi_ci_high"], r["hi_share"])

    def test_threshold_is_inclusive(self):
        self.assertEqual(rate_summary([HI])["hi_share"], 1.0)
        self.assertEqual(rate_summary([HI - 1e-9])["hi_share"], 0.0)

    def test_empty(self):
        r = rate_summary([])
        self.assertEqual(r["n"], 0)
        self.assertTrue(math.isnan(r["hi_share"]))
        self.assertTrue(math.isnan(r["mean"]))

    def test_nan_is_dropped(self):
        r = rate_summary([0.2, np.nan, 0.95])
        self.assertEqual(r["n"], 2)
        self.assertAlmostEqual(r["hi_share"], 0.5)


class TestMetrics(unittest.TestCase):
    def test_auc_perfect(self):
        self.assertEqual(auc_or_nan([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_auc_single_class(self):
        self.assertTrue(math.isnan(auc_or_nan([1, 1], [0.2, 0.8])))

    def test_spearman(self):
        rho, p = safe_spearman([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertAlmostEqual(rho, 1.0)
        self.assertLess(p, 0.05)
        rho, p = safe_spearman([1, 1, 1], [1, 2, 3])
        self.assertTrue(math.isnan(rho))
        self.assertTrue(math.isnan(p))


class TestFamilyTable(unittest.TestCase):
    def test_shares_and_test(self):
        rows = []
        for d in ("健康", "穿搭"):
            for fam, n in (("gpt", 3), ("deepseek", 1)):
                rows += [{"domain": d, "model_family": fam}] * n
        out = family_share_table(pd.DataFrame(rows), min_n=1)
        self.assertEqual(set(out["domains"]), {"健康", "穿搭"})
        for d in out["domains"]:
            self.assertAlmostEqual(sum(out["domains"][d]["shares"].values()), 1.0)
            self.assertEqual(out["domains"][d]["n"], 4)
        self.assertEqual(out["dof"], (2 - 1) * (2 - 1))
        self.assertTrue(math.isfinite(out["chi2"]))
        self.assertGreaterEqual(out["p"], 0.0)
        self.assertLessEqual(out["p"], 1.0)

    def test_too_few_domains(self):
        df = pd.DataFrame([{"domain": "健康", "model_family": "gpt"}] * 2)
        out = family_share_table(df, min_n=1)
        self.assertEqual(set(out["domains"]), {"健康"})
        self.assertTrue(math.isnan(out["chi2"]))


class TestDomainPenalty(unittest.TestCase):
    def _frame(self):
        rng = np.random.default_rng(7)
        months = np.repeat(["202301", "202302", "202303"], 80)
        p_ai = rng.uniform(0, 1, size=months.size)
        length = rng.uniform(100, 800, size=months.size)
        tags = rng.integers(0, 8, size=months.size)
        likes = np.maximum(0, 200 * (1 - 0.2 * p_ai) + rng.normal(0, 20, size=months.size))
        return pd.DataFrame({"month": months, "p_ai": p_ai,
                             "content_len": length, "n_hashtags": tags,
                             "likes": likes})

    def test_fit_contract(self):
        out = domain_penalty(self._frame(), "likes", min_month_n=50, n_boot=20)
        self.assertEqual(out["n_obs"], 240)
        self.assertEqual(out["n_months"], 3)
        self.assertTrue(math.isfinite(out["beta"]))
        self.assertLessEqual(out["ci_low"], out["beta"])
        self.assertGreaterEqual(out["ci_high"], out["beta"])
        self.assertTrue(math.isfinite(out["within_month_rho"]))
        self.assertGreaterEqual(out["neg_month_share"], 0.0)
        self.assertLessEqual(out["neg_month_share"], 1.0)

    def test_too_small_cell(self):
        df = self._frame().iloc[:40]
        out = domain_penalty(df, "likes", min_month_n=50, n_boot=5)
        self.assertEqual(out["n_obs"], 0)
        self.assertTrue(math.isnan(out["beta"]))

    def test_constants(self):
        self.assertEqual(MONTH_FROM, "202301")
        self.assertAlmostEqual(HI, 0.9)


class TestDedupSensitivity(unittest.TestCase):
    def test_missing_file_is_reported(self):
        out = dedup_sensitivity(pd.DataFrame({"p_ai": [0.1], "domain": ["健康"]}),
                                path="/nonexistent/exploring.jsonl")
        self.assertFalse(out["available"])


class TestJsonSafe(unittest.TestCase):
    def test_nested_non_finite_to_none(self):
        obj = {"a": float("nan"), "b": [1.0, float("inf")],
               "c": {"d": float("-inf")}, "e": "ok"}
        clean = json_safe(obj)
        self.assertIsNone(clean["a"])
        self.assertEqual(clean["b"], [1.0, None])
        self.assertIsNone(clean["c"]["d"])
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            json.dump(clean, f, ensure_ascii=False)
            path = f.name
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), clean)
        pathlib.Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
