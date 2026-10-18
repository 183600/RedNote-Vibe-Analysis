# -*- coding: utf-8 -*-
"""axis_synthesis_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from axis_synthesis_analysis import (ALL_CATEGORIES, AXES, AXIS_INDICES,
                                     AXIS_ORDER, CAT_INDEX, CATEGORY_AXIS,
                                     CATEGORY_LABEL, composite_density,
                                     composite_z, category_vector, fe_r2,
                                     log2_ratio, mean_ci, stack_hits,
                                     within_cell_rank_z)


class TestCategoryVector(unittest.TestCase):
    def test_counts_all_five_axes(self):
        v = category_vector("姐妹们，这个真的只要100元！因为评论区在问？")
        self.assertEqual(v[CAT_INDEX["vocative"]], 1)   # 语气情态
        self.assertEqual(v[CAT_INDEX["intens"]], 1)
        self.assertEqual(v[CAT_INDEX["unit"]], 1)       # 具体性
        self.assertEqual(v[CAT_INDEX["comment"]], 1)    # 互动引导
        self.assertEqual(v[CAT_INDEX["comma"]], 1)      # 标点排版
        self.assertEqual(v[CAT_INDEX["excl"]], 1)
        self.assertEqual(v[CAT_INDEX["ques"]], 1)
        self.assertEqual(v[CAT_INDEX["causal"]], 1)     # 篇章连接
        for c in ("hedge", "boost", "num", "vague", "place", "follow", "urge",
                  "period", "contrast", "exemplify"):
            self.assertEqual(v[CAT_INDEX[c]], 0, msg=c)

    def test_empty_and_neutral_text(self):
        self.assertEqual(int(category_vector("").sum()), 0)
        self.assertEqual(int(category_vector("今天天气不错").sum()), 0)


class TestAxisDefinitions(unittest.TestCase):
    def test_partition_is_complete_and_unique(self):
        self.assertEqual(len(ALL_CATEGORIES), 30)
        self.assertEqual(len(set(ALL_CATEGORIES)), 30)
        self.assertEqual(set(ALL_CATEGORIES), set(CATEGORY_AXIS))
        self.assertEqual(set(ALL_CATEGORIES), set(CATEGORY_LABEL))
        self.assertEqual(len(set(CATEGORY_LABEL.values())), 30)
        for axis, _, cats, labels in AXES:
            self.assertEqual(AXIS_INDICES[axis], [CAT_INDEX[c] for c in cats])
            for c in cats:
                self.assertEqual(CATEGORY_AXIS[c], axis)
                self.assertIn(c, labels)
        self.assertEqual(AXIS_ORDER, [a[0] for a in AXES])


class TestWithinCellRankZ(unittest.TestCase):
    def test_offset_invariance_and_rank_order(self):
        groups = [np.array([0, 1, 2]), np.array([3, 4, 5])]
        z = within_cell_rank_z([1.0, 2.0, 3.0, 101.0, 102.0, 103.0], groups)
        np.testing.assert_allclose(z[:3], z[3:])
        self.assertGreater(z[2], z[0])

    def test_constant_and_uncovered_cells(self):
        groups = [np.array([0, 1]), np.array([2, 3])]
        z = within_cell_rank_z([5.0, 5.0, 1.0, 2.0], groups)
        self.assertTrue(np.isnan(z[0]) and np.isnan(z[1]))
        self.assertFalse(np.isnan(z[2]) or np.isnan(z[3]))
        z2 = within_cell_rank_z([1.0, 2.0, 3.0], [np.array([0, 1])])
        self.assertTrue(np.isnan(z2[2]))

    def test_ties_get_average_rank(self):
        groups = [np.array([0, 1, 2, 3])]
        z = within_cell_rank_z([1.0, 1.0, 2.0, 2.0], groups)
        self.assertAlmostEqual(z[0], z[1])
        self.assertAlmostEqual(z[2], z[3])
        self.assertLess(z[0], z[2])


class TestComposite(unittest.TestCase):
    def test_composite_z_means_within_axis_and_skips_nan(self):
        z = np.zeros((2, 30))
        for c in ("hedge", "boost", "intens", "vocative"):
            z[:, CAT_INDEX[c]] = 2.0
        z[0, CAT_INDEX["hedge"]] = np.nan
        comp = composite_z(z)
        self.assertEqual(list(comp.columns), AXIS_ORDER)
        # 第一行 hedge 缺失：剩 3 个 2.0，均值仍为 2.0
        self.assertAlmostEqual(comp.loc[0, "stance"], 2.0)
        self.assertAlmostEqual(comp.loc[0, "cta"], 0.0)
        # 某轴全部缺失 -> NaN
        z[:, [CAT_INDEX["num"], CAT_INDEX["unit"], CAT_INDEX["vague"],
              CAT_INDEX["place"]]] = np.nan
        comp2 = composite_z(z)
        self.assertTrue(np.isnan(comp2.loc[0, "concreteness"]))

    def test_composite_density_sums_axis_columns(self):
        d = np.zeros((3, 30))
        d[:, AXIS_INDICES["stance"]] = 1.0
        d[:, AXIS_INDICES["typography"]] = 2.0
        comp = composite_density(d)
        self.assertEqual(comp.shape, (3, 5))
        col = {a: i for i, a in enumerate(AXIS_ORDER)}
        np.testing.assert_allclose(comp[:, col["stance"]], 4.0)
        np.testing.assert_allclose(comp[:, col["typography"]], 24.0)
        np.testing.assert_allclose(comp[:, col["cta"]], 0.0)


class TestStackHits(unittest.TestCase):
    def test_hits_and_valid(self):
        z = np.asarray([[1.0, -1.0, 0.0, np.nan, 1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0]])
        signs = [1, -1, 1, 1, 1]
        hits, n_valid = stack_hits(z, signs)
        np.testing.assert_array_equal(hits, [3, 1])
        np.testing.assert_array_equal(n_valid, [4, 5])

    def test_zero_z_is_not_a_hit(self):
        hits, n_valid = stack_hits([[0.0, 0.0]], [1, -1])
        np.testing.assert_array_equal(hits, [0])
        np.testing.assert_array_equal(n_valid, [2])


class TestFeR2(unittest.TestCase):
    def test_exact_fit(self):
        X = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        coef = np.asarray([2.0, -1.0])
        fit = {"X_demeaned": X, "y_demeaned": X @ coef, "coef": coef}
        self.assertAlmostEqual(fe_r2(fit), 1.0)

    def test_zero_variance_y(self):
        fit = {"X_demeaned": np.zeros((3, 1)),
               "y_demeaned": np.zeros(3), "coef": np.zeros(1)}
        self.assertTrue(np.isnan(fe_r2(fit)))


class TestHelpers(unittest.TestCase):
    def test_log2_ratio(self):
        self.assertAlmostEqual(log2_ratio(0.05, 0.05), 0.0)
        self.assertAlmostEqual(log2_ratio(0.0, 0.0), 0.0)
        self.assertGreater(log2_ratio(1.0, 0.05), 0.0)
        self.assertLess(log2_ratio(0.0, 1.0), 0.0)

    def test_mean_ci_constant(self):
        rec = mean_ci([3.0, 3.0, 3.0])
        self.assertAlmostEqual(rec["mean"], 3.0)
        self.assertAlmostEqual(rec["ci_low"], 3.0)
        self.assertAlmostEqual(rec["ci_high"], 3.0)
        self.assertEqual(rec["n"], 3)

    def test_mean_ci_empty(self):
        rec = mean_ci([])
        self.assertTrue(np.isnan(rec["mean"]))
        self.assertEqual(rec["n"], 0)


if __name__ == "__main__":
    unittest.main()
