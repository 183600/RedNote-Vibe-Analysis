# -*- coding: utf-8 -*-
"""analysis_utils 的单元测试（stdlib unittest，无需额外依赖）。"""
import json
import pathlib
import random
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from analysis_utils import (post_text, iter_jsonl, load_posts, detector_score,
                            cliffs_delta, cliffs_delta_from_u, wilson_ci,
                            bootstrap_auc, setup_chinese_fonts)


class TestJsonlHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "sample.jsonl"
        records = [
            {"note_title": "标题一", "note_content": "正文一", "likes": 1},
            {"note_title": "标题二", "desc": "正文二", "likes": 2},
            {"note_title": "标题三", "likes": 3},
        ]
        self.path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
            + "\n\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_iter_jsonl_skips_blank_lines(self):
        recs = list(iter_jsonl(self.path))
        self.assertEqual(len(recs), 3)
        self.assertEqual(recs[0]["likes"], 1)
        self.assertEqual(recs[2]["note_title"], "标题三")

    def test_post_text_field_fallbacks(self):
        recs = list(iter_jsonl(self.path))
        self.assertTrue(post_text(recs[0]).endswith("正文一"))
        self.assertIn("正文二", post_text(recs[1]))
        self.assertEqual(post_text(recs[2], with_title=False), "")
        self.assertNotIn("标题三", post_text(recs[2], with_title=False))

    def test_load_posts_sampling_matches_legacy_algorithm(self):
        texts = [post_text(r) for r in iter_jsonl(self.path)]
        got = load_posts(str(self.path), sample=2, seed=42)
        random.seed(42)
        expected = random.sample(texts, 2)
        self.assertEqual(got, expected)
        self.assertEqual(load_posts(str(self.path), sample=2, seed=42), got)


class TestStatsHelpers(unittest.TestCase):
    def test_cliffs_delta_extremes(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0, 8.0])
        delta, p = cliffs_delta(a, b)
        self.assertAlmostEqual(delta, -1.0)
        self.assertLess(p, 0.05)
        delta, _ = cliffs_delta(b, a)
        self.assertAlmostEqual(delta, 1.0)
        delta, _ = cliffs_delta(a, a)
        self.assertAlmostEqual(delta, 0.0)
        self.assertAlmostEqual(cliffs_delta_from_u(8, 4, 4), 0.0)

    def test_wilson_ci_bounds(self):
        lo, hi = wilson_ci(0, 100)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLess(hi, 0.05)
        lo, hi = wilson_ci(100, 100)
        self.assertAlmostEqual(hi, 1.0)
        lo, hi = wilson_ci(65, 100)
        self.assertLess(lo, 0.65)
        self.assertGreater(hi, 0.65)

    def test_bootstrap_auc_perfect_ranking(self):
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        res = bootstrap_auc(y, proba, n_boot=50, seed=7)
        self.assertAlmostEqual(res["auc"], 1.0)
        self.assertAlmostEqual(res["ci_low"], 1.0)
        self.assertAlmostEqual(res["ci_high"], 1.0)


class TestDetectorScore(unittest.TestCase):
    def test_matches_manual_computation(self):
        model = {
            "robust_features": ["a", "b"],
            "robust_scaler_mean": [1.0, 2.0],
            "robust_scaler_scale": [2.0, 4.0],
            "robust_coef": [0.5, -1.0],
            "robust_intercept": 0.25,
        }
        df = {"a": [3.0], "b": [6.0]}
        import pandas as pd
        got = detector_score(pd.DataFrame(df), model, robust=True)[0]
        z = np.array([(3.0 - 1.0) / 2.0, (6.0 - 2.0) / 4.0])
        expected = 1.0 / (1.0 + np.exp(-(z @ np.array([0.5, -1.0]) + 0.25)))
        self.assertAlmostEqual(float(got), float(expected))


class TestFontSetup(unittest.TestCase):
    def test_returns_name_or_none_without_crashing(self):
        name = setup_chinese_fonts()
        self.assertTrue(name is None or isinstance(name, str))


if __name__ == "__main__":
    unittest.main()
