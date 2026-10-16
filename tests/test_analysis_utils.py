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

from analysis_utils import (post_text, iter_jsonl, load_posts, load_records,
                            to_int, engagement_of, parse_local_time, hour_of,
                            weekday_of, detector_score,
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

    def test_load_records_matches_load_posts_sampling(self):
        recs_full = load_records(str(self.path))
        self.assertEqual(len(recs_full), 3)
        recs = load_records(str(self.path), sample=2, seed=42)
        self.assertEqual([post_text(r) for r in recs],
                         load_posts(str(self.path), sample=2, seed=42))
        self.assertEqual(recs, load_records(str(self.path), sample=2, seed=42))


class TestRecordHelpers(unittest.TestCase):
    def test_to_int_mixed_types(self):
        self.assertEqual(to_int("123"), 123)
        self.assertEqual(to_int(5), 5)
        self.assertEqual(to_int(None), 0)
        self.assertEqual(to_int("abc", default=-1), -1)

    def test_engagement_of_both_schemas(self):
        human_like = {"liked_count": "10", "collected_count": "3",
                      "comments_count": "1"}
        aigc_like = {"likes": 7, "collections": 2, "comments": 0}
        self.assertEqual(engagement_of(human_like), (10, 3, 1))
        self.assertEqual(engagement_of(aigc_like), (7, 2, 0))
        self.assertEqual(engagement_of({}), (0, 0, 0))

    def test_parse_local_time(self):
        dt = parse_local_time("2022091308")
        self.assertEqual((dt.year, dt.month, dt.day, dt.hour), (2022, 9, 13, 8))
        self.assertEqual(parse_local_time("20220913").day, 13)
        self.assertIsNone(parse_local_time(""))
        self.assertIsNone(parse_local_time("not-a-time"))
        self.assertIsNone(parse_local_time("2022130100"))

    def test_hour_and_weekday(self):
        self.assertEqual(hour_of("2022091308"), 8)
        self.assertEqual(weekday_of("2022091308"), 1)  # 2022-09-13 是周二
        self.assertIsNone(hour_of(None))
        self.assertIsNone(weekday_of(""))


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

    def test_init_plot_style_and_save_fig(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from analysis_utils import init_plot_style, save_fig

        name = init_plot_style()
        self.assertTrue(name is None or isinstance(name, str))
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "sub" / "fig.png"
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot([0, 1], [0, 1])
            save_fig(fig, str(out))
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
