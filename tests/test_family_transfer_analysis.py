# -*- coding: utf-8 -*-
"""family_transfer_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from family_transfer_analysis import (cell_residuals, composition_shares,
                                      composition_with_ci, confidence_by_band,
                                      cv_family_predictions, cv_metrics,
                                      family_design, family_residual_table,
                                      fe_ols_raw, half_of, make_family_pipeline,
                                      quarter_of, score_exploring)


def make_family_df(n_per_class=40, seed=0):
    """三类线性可分的合成家族数据（两个特征）。"""
    rng = np.random.default_rng(seed)
    centers = {"gpt": (0.0, 0.0), "gemini": (4.0, 0.0), "glm": (0.0, 4.0)}
    rows = []
    for fam, (c1, c2) in centers.items():
        rows.append(pd.DataFrame({
            "f1": rng.normal(c1, 0.5, n_per_class),
            "f2": rng.normal(c2, 0.5, n_per_class),
            "model_family": fam,
        }))
    return pd.concat(rows, ignore_index=True)


class TestCvFamilyPredictions(unittest.TestCase):
    def test_separable_data_high_accuracy(self):
        df = make_family_df()
        pred, proba = cv_family_predictions(df, ["f1", "f2"])
        self.assertEqual(pred.shape, (len(df),))
        self.assertEqual(proba.shape, (len(df), 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        metrics = cv_metrics(df["model_family"].values, pred,
                             sorted(df["model_family"].unique()))
        self.assertGreater(metrics["accuracy"], 0.95)
        self.assertGreater(metrics["macro_f1"], 0.95)
        cm = np.asarray(metrics["confusion_row_normalized"])
        np.testing.assert_allclose(cm.sum(axis=1), 1.0, atol=1e-9)


class TestScoreExploring(unittest.TestCase):
    def test_columns_and_argmax_consistency(self):
        df = make_family_df()
        model = make_family_pipeline().fit(df[["f1", "f2"]].values,
                                           df["model_family"].values)
        mode = model.classes_[0]
        exploring = df.iloc[:12][["f1", "f2"]].copy()
        scored = score_exploring(model, exploring, ["f1", "f2"])
        pf_cols = [c for c in scored.columns if c.startswith("pf_")]
        self.assertEqual(len(pf_cols), 3)
        np.testing.assert_allclose(scored[pf_cols].sum(axis=1), 1.0, atol=1e-6)
        # fam_pred 与最大后验列一致；fam_conf 等于最大的 pf 值
        best = scored[pf_cols].idxmax(axis=1).str.replace("pf_", "", regex=False)
        self.assertTrue((scored["fam_pred"] == best).all())
        np.testing.assert_allclose(scored["fam_conf"], scored[pf_cols].max(axis=1))
        self.assertEqual(mode, sorted(df["model_family"].unique())[0])


class TestComposition(unittest.TestCase):
    def test_counts_and_shares(self):
        labels = ["gpt", "gpt", "gemini", "glm", "glm", "glm"]
        comp = composition_shares(labels)
        self.assertEqual(comp["n"], 6)
        self.assertEqual(comp["counts"]["glm"], 3)
        self.assertAlmostEqual(comp["shares"]["glm"], 0.5)
        self.assertAlmostEqual(sum(comp["shares"].values()), 1.0)

    def test_wilson_ci_bounds(self):
        comp = composition_with_ci(["gpt"] * 80 + ["gemini"] * 20)
        lo, hi = comp["ci"]["gpt"]
        self.assertLess(lo, comp["shares"]["gpt"])
        self.assertGreater(hi, comp["shares"]["gpt"])
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_empty_input(self):
        comp = composition_shares([])
        self.assertEqual(comp["n"], 0)
        self.assertEqual(comp["shares"]["gpt"], 0.0)


class TestConfidenceByBand(unittest.TestCase):
    def test_band_assignment_and_shares(self):
        p_ai = np.array([0.1, 0.2, 0.5, 0.7, 0.95, 0.99])
        conf = np.array([0.4, 0.6, 0.5, 0.7, 0.8, 0.9])
        fam = np.array(["glm", "glm", "gpt", "gpt", "gemini", "gemini"])
        bands = confidence_by_band(p_ai, conf, fam)
        by_label = {b["band"]: b for b in bands}
        self.assertEqual(by_label["<0.4"]["n"], 2)
        self.assertEqual(by_label[">=0.9"]["n"], 2)
        self.assertAlmostEqual(by_label["<0.4"]["conf_mean"], 0.5)
        self.assertEqual(by_label[">=0.9"]["family_shares"]["gemini"], 1.0)
        self.assertEqual(by_label["0.4–0.6"]["family_shares"]["gpt"], 1.0)


class TestTimeBuckets(unittest.TestCase):
    def test_half_and_quarter(self):
        self.assertEqual(half_of("202401"), "2024H1")
        self.assertEqual(half_of("202407"), "2024H2")
        self.assertEqual(quarter_of("202401"), "2024Q1")
        self.assertEqual(quarter_of("202412"), "2024Q4")
        self.assertIsNone(half_of("2024"))
        self.assertIsNone(quarter_of(""))

    def test_half_rejects_non_numeric(self):
        self.assertIsNone(half_of("abcdef"))
        self.assertIsNone(half_of("2024"))


class TestCellHelpers(unittest.TestCase):
    def test_residual_cell_means_zero(self):
        cells = np.array(["a", "a", "b", "b", "b"])
        values = np.array([1.0, 3.0, -2.0, 4.0, 7.0])
        resid = cell_residuals(values, cells)
        np.testing.assert_allclose(resid[cells == "a"].mean(), 0)
        np.testing.assert_allclose(resid[cells == "b"].mean(), 0)

    def test_family_residual_table(self):
        resid = np.array([1.0, 2.0, -1.0, -2.0])
        fam = np.array(["gpt", "gpt", "glm", "glm"])
        table = family_residual_table(resid, fam)
        self.assertAlmostEqual(table["gpt"]["mean"], 1.5)
        self.assertAlmostEqual(table["glm"]["mean"], -1.5)
        self.assertEqual(table["gemini"]["n"], 0)


class TestFeOlsRaw(unittest.TestCase):
    def test_recovers_family_contrast(self):
        """构造 y = 0.5 * is_gpt + 单元截距，检验族哑变量系数约为 0.5。"""
        rng = np.random.default_rng(3)
        cells, fam, y = [], [], []
        for c in range(4):
            n = 120
            is_gpt = rng.random(n) < 0.5
            cells.extend([c] * n)
            fam.extend(np.where(is_gpt, "gpt", "glm"))
            y.extend(0.5 * is_gpt + 10.0 * c + rng.normal(0, 0.01, n))
        fam = np.asarray(fam)
        X = np.column_stack([(fam == "gpt").astype(float)])
        fit = fe_ols_raw(X, np.asarray(y), np.asarray(cells), min_cell_n=100)
        self.assertEqual(fit["n_obs"], len(y))
        self.assertEqual(fit["n_cells"], 4)
        self.assertAlmostEqual(float(fit["coef"][0]), 0.5, places=2)

    def test_small_cells_dropped(self):
        cells = np.array([0] * 50 + [1] * 150)
        X = np.column_stack([np.linspace(0, 1, 200)])
        y = np.arange(200.0)
        fit = fe_ols_raw(X, y, cells, min_cell_n=100)
        self.assertEqual(fit["n_obs"], 150)
        self.assertEqual(fit["n_cells"], 1)


class TestFamilyDesign(unittest.TestCase):
    def test_columns(self):
        df = pd.DataFrame({
            "fam_pred": ["gpt", "glm", "gemini"],
            "content_len": [0, 9, 99],
            "p_ai": [0.1, 0.5, 0.9],
            "n_hashtags": [1, 2, 3],
        })
        X, names = family_design(df)
        self.assertEqual(X.shape, (3, 5 + 3))  # 5 个家族哑变量（去参照）+ 3 个控制
        self.assertEqual(names[:5], ["anthropic", "deepseek", "gemini", "gpt", "qwen"])
        self.assertIn("log10(1+正文长度)", names)
        self.assertEqual(X[0, names.index("gpt")], 1.0)
        self.assertEqual(X[0, names.index("gemini")], 0.0)


if __name__ == "__main__":
    unittest.main()
