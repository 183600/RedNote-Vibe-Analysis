# -*- coding: utf-8 -*-
"""engagement_drivers_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from engagement_drivers_analysis import (cell_keep_mask, standardize,
                                         demean_within, fe_ols,
                                         cluster_bootstrap_fe_ols,
                                         univariate_table)


def make_panel(n_cells=6, per_cell=120, seed=0):
    """构造包含单元固定效应的合成面板数据。

    x1 在单元内有变异且与 y 线性相关；x2 纯噪声；单元截距只影响水平。
    """
    rng = np.random.default_rng(seed)
    cells, x1, x2, y = [], [], [], []
    for c in range(n_cells):
        x1c = rng.normal(0, 1, per_cell)
        x2c = rng.normal(0, 1, per_cell)
        yc = 3.0 * x1c + 0.0 * x2c + 10.0 * c  # 单元截距 c 被固定效应吸收
        cells.extend([c] * per_cell)
        x1.extend(x1c)
        x2.extend(x2c)
        y.extend(yc)
    return (np.asarray(cells), np.column_stack([x1, x2]), np.asarray(y))


class TestCellKeepMask(unittest.TestCase):
    def test_masks_small_cells(self):
        codes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        mask = cell_keep_mask(codes, min_cell_n=3)
        self.assertEqual(mask.tolist(), [True] * 3 + [False] * 2 + [True] * 4)


class TestStandardize(unittest.TestCase):
    def test_mean_zero_std_one(self):
        rng = np.random.default_rng(1)
        x = rng.normal(5, 3, (200, 3))
        z, mean, scale = standardize(x)
        np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-12)
        np.testing.assert_allclose(mean, x.mean(axis=0))
        np.testing.assert_allclose(scale, x.std(axis=0))

    def test_constant_column_scale_one(self):
        x = np.column_stack([np.ones(10), np.arange(10.0)])
        z, _, scale = standardize(x)
        self.assertEqual(scale[0], 1.0)
        np.testing.assert_allclose(z[:, 0], 0)


class TestDemeanWithin(unittest.TestCase):
    def test_cell_means_become_zero(self):
        codes = np.array([0, 0, 1, 1, 1])
        v = np.array([1.0, 3.0, -2.0, 4.0, 7.0])
        out = demean_within(v, codes)
        self.assertEqual(out.ndim, 1)
        np.testing.assert_allclose(out[codes == 0].mean(), 0)
        np.testing.assert_allclose(out[codes == 1].mean(), 0)
        np.testing.assert_allclose(out, v - np.array([2, 2, 3, 3, 3]))

    def test_two_dimensional(self):
        codes = np.array([0, 0, 1, 1])
        v = np.arange(8.0).reshape(4, 2)
        out = demean_within(v, codes)
        self.assertEqual(out.shape, (4, 2))
        for c in (0, 1):
            np.testing.assert_allclose(out[codes == c].mean(axis=0), 0)


class TestFeOls(unittest.TestCase):
    def test_recovers_within_cell_effect(self):
        cells, X, y = make_panel(seed=0)
        fit = fe_ols(X, y, cells, min_cell_n=100)
        self.assertEqual(fit["n_obs"], len(y))
        self.assertEqual(fit["n_cells"], 6)
        sd_x1 = X[:, 0].std()
        self.assertAlmostEqual(fit["coef"][0], 3.0 * sd_x1, places=6)
        self.assertAlmostEqual(fit["coef"][1], 0.0, places=6)

    def test_small_cells_dropped(self):
        cells = np.array([0] * 50 + [1] * 150)
        X = np.column_stack([np.linspace(0, 1, 200), np.linspace(1, 0, 200)])
        y = np.arange(200.0)
        fit = fe_ols(X, y, cells, min_cell_n=100)
        self.assertEqual(fit["n_obs"], 150)
        self.assertEqual(fit["n_cells"], 1)


class TestClusterBootstrap(unittest.TestCase):
    def test_shape_and_ci_contains_estimate(self):
        cells, X, y = make_panel(n_cells=4, per_cell=120, seed=2)
        fit = fe_ols(X, y, cells, min_cell_n=100)
        boots = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                         fit["cell_codes"], n_boot=25, seed=7)
        self.assertEqual(boots.shape, (25, 2))
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        # x2 的真实系数恰为 0（合成数据里 y 只依赖 x1），点估计与自举
        # 系数都在 1e-15 量级的浮点噪声内；用绝对容差比较，避免把纯
        # 浮点尾差误判为「CI 未覆盖点估计」。
        self.assertTrue((lo - 1e-9 <= fit["coef"]).all())
        self.assertTrue((fit["coef"] <= hi + 1e-9).all())


class TestUnivariateTable(unittest.TestCase):
    def test_positive_relation_detected(self):
        cells, X, y = make_panel(seed=3)
        columns = {"x": {"x1": X[:, 0], "x2": X[:, 1]},
                   "targets": {"likes": y}}
        table = univariate_table(["x1", "x2"], columns, cells, min_cell_n=100)
        self.assertGreater(table["x1"]["likes"]["weighted_rho"], 0.99)
        self.assertLess(abs(table["x2"]["likes"]["weighted_rho"]), 0.2)
        self.assertEqual(table["x1"]["likes"]["n_cells"], 6)


if __name__ == "__main__":
    unittest.main()
