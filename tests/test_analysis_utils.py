# -*- coding: utf-8 -*-
"""analysis_utils 的单元测试（stdlib unittest，无需额外依赖）。"""
import json
import pathlib
import random
import sys
import tempfile
import unittest
from functools import partial
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from analysis_utils import (post_text, content_of, clean_body, body_cache,
                            shannon_entropy, topk_coverage,
                            iter_jsonl, load_posts, load_records,
                            to_int, engagement_of, parse_local_time, hour_of,
                            weekday_of, detector_score,
                            cliffs_delta, cliffs_delta_from_u, wilson_ci,
                            bootstrap_auc, setup_chinese_fonts,
                            dataset_path, title_of, compact_text,
                            split_sentences, burstiness, mtld,
                            prefix_ngram_stats, bootstrap_median_ci,
                            DATASETS, linear_score, prior_corrected_proba,
                            reliability_bins, expected_calibration_error,
                            LENGTH_BINS, decile_table, length_bin_codes,
                            standardize_weights, domain_standardized_series,
                            ngram_repeat_share, length_matched_mean_gap,
                            length_residualize, run_parallel)


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


class TestTextHelpers(unittest.TestCase):
    def test_content_of_field_fallbacks(self):
        self.assertEqual(content_of({"note_content": "a"}), "a")
        self.assertEqual(content_of({"desc": "b"}), "b")
        self.assertEqual(content_of({"note_content": "", "desc": "b"}), "b")
        self.assertEqual(content_of({}), "")

    def test_clean_body_matches_paper_features_pipeline(self):
        raw = "[话题]前文#标签甲# 中间[话题]后文 #标签乙#"
        body = clean_body(raw)
        self.assertNotIn("[话题]", body)
        self.assertNotIn("#标签甲", body)
        self.assertNotIn("#标签乙", body)
        self.assertIn("前文", body)
        self.assertIn("后文", body)
        self.assertEqual(clean_body(None), "")

    def test_shannon_entropy_known_values(self):
        self.assertAlmostEqual(shannon_entropy([1, 1]), 1.0)
        self.assertAlmostEqual(shannon_entropy({"a": 3}), 0.0)
        self.assertEqual(shannon_entropy([]), 0.0)
        h = shannon_entropy([5, 1, 1], miller_madow=True)
        self.assertGreater(h, shannon_entropy([5, 1, 1]))

    def test_topk_coverage(self):
        self.assertAlmostEqual(topk_coverage([5, 3, 2], k=1), 0.5)
        self.assertAlmostEqual(topk_coverage([5, 3, 2], k=3), 1.0)
        self.assertEqual(topk_coverage([], k=5), 0.0)


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

    def test_gini_coefficient(self):
        from analysis_utils import gini_coefficient
        self.assertAlmostEqual(gini_coefficient([1, 1, 1, 1]), 0.0)
        self.assertAlmostEqual(gini_coefficient([0, 0, 0, 10]), 0.75)
        self.assertEqual(gini_coefficient([]), 0.0)
        self.assertEqual(gini_coefficient([0, 0]), 0.0)
        self.assertGreater(gini_coefficient([1, 1, 1, 100]), gini_coefficient([1, 2, 3, 4]))

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


class TestDatasetRegistry(unittest.TestCase):
    def test_dataset_path_known_and_unknown(self):
        self.assertTrue(dataset_path("human", base="/x/").endswith(DATASETS["human"]))
        self.assertTrue(dataset_path("exploring", base="").endswith(".jsonl"))
        with self.assertRaises(ValueError):
            dataset_path("nope")

    def test_title_of(self):
        self.assertEqual(title_of({"note_title": "标题"}), "标题")
        self.assertEqual(title_of({"note_title": ""}), "")
        self.assertEqual(title_of({}), "")


class TestSentenceHelpers(unittest.TestCase):
    def test_split_sentences_boundaries(self):
        text = "第一句。第二句！第三句？第四句\n第五句；第六句…第七句"
        got = split_sentences(text)
        self.assertEqual(got, ["第一句", "第二句", "第三句", "第四句",
                               "第五句", "第六句", "第七句"])
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("。。。"), [])
        self.assertEqual(split_sentences(None), [])

    def test_burstiness(self):
        self.assertAlmostEqual(burstiness([5, 5, 5, 5]), 0.0)
        cv = burstiness([2, 8])
        self.assertAlmostEqual(cv, 0.6)
        self.assertEqual(burstiness([3]), 0.0)
        self.assertEqual(burstiness([]), 0.0)
        self.assertEqual(burstiness([0, 0]), 0.0)

    def test_compact_text(self):
        self.assertEqual(compact_text(" a \n b\tc "), "abc")
        self.assertEqual(compact_text(None), "")

    def test_mtld_bounds_and_monotonicity(self):
        repeated = ["的"] * 200
        diverse = [f"词{i}" for i in range(200)]
        self.assertGreater(mtld(diverse), mtld(repeated))
        self.assertGreater(mtld(diverse), 100)
        self.assertEqual(mtld([]), 0.0)
        self.assertEqual(mtld(["a"]), 0.0)
        with self.assertRaises(ValueError):
            mtld(["a", "b"], threshold=1.0)

    def test_prefix_ngram_stats(self):
        texts = ["姐妹们冲呀", "姐妹们冲鸭", "完全不同啊"]
        stats = prefix_ngram_stats(texts, n=4, k=1)
        self.assertEqual(stats["n_texts"], 3)
        self.assertAlmostEqual(stats["coverage_topk"], 2 / 3)
        self.assertEqual(stats["top"][0][0], "姐妹们冲")
        empty = prefix_ngram_stats(["短"], n=4)
        self.assertEqual(empty["n_texts"], 0)
        self.assertEqual(empty["coverage_topk"], 0.0)


class TestBootstrapMedianCI(unittest.TestCase):
    def test_ci_contains_median_and_degenerate_case(self):
        res = bootstrap_median_ci([1, 2, 3, 4, 5], n_boot=200, seed=1)
        self.assertEqual(res["median"], 3.0)
        self.assertLessEqual(res["ci_low"], 3.0)
        self.assertGreaterEqual(res["ci_high"], 3.0)
        self.assertEqual(res["n"], 5)
        empty = bootstrap_median_ci([], n_boot=10)
        self.assertTrue(np.isnan(empty["median"]))
        self.assertEqual(empty["n"], 0)

    def test_linear_score_matches_detector_score(self):
        import pandas as pd
        model = {
            "robust_features": ["a"],
            "robust_scaler_mean": [0.0],
            "robust_scaler_scale": [1.0],
            "robust_coef": [2.0],
            "robust_intercept": 0.0,
        }
        df = pd.DataFrame({"a": [1.0]})
        got = linear_score(df, ["a"], [0.0], [1.0], [2.0], 0.0)
        self.assertAlmostEqual(float(got[0]), float(detector_score(df, model)[0]))


class TestCalibrationHelpers(unittest.TestCase):
    def test_prior_corrected_proba_shifts_down_for_imbalanced_negatives(self):
        p = np.array([0.5])
        corr = prior_corrected_proba(p, n_neg=900, n_pos=100)
        self.assertAlmostEqual(float(corr[0]), 0.1, places=6)
        # 先验均衡时不应改变数值
        same = prior_corrected_proba(p, n_neg=100, n_pos=100)
        self.assertAlmostEqual(float(same[0]), 0.5, places=9)

    def test_reliability_bins_counts_and_means(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.05, 0.15, 0.85, 0.95])
        r = reliability_bins(p, y, bins=10)
        self.assertEqual(sum(r["count"]), 4)
        self.assertEqual(r["count"][0], 1)
        self.assertEqual(r["count"][1], 1)
        self.assertEqual(r["count"][8], 1)
        self.assertEqual(r["count"][9], 1)
        self.assertAlmostEqual(r["mean_true"][8], 1.0)
        self.assertAlmostEqual(r["mean_true"][9], 1.0)
        self.assertTrue(np.isnan(r["mean_pred"][4]))

    def test_ece_perfect_calibration_and_extreme_miscalibration(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        p_perfect = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
        self.assertLess(expected_calibration_error(p_perfect, y), 0.11)
        p_wrong = 1.0 - p_perfect
        self.assertGreater(expected_calibration_error(p_wrong, y), 0.7)

    def test_reliability_bins_validates_input(self):
        with self.assertRaises(ValueError):
            reliability_bins([], [], bins=10)
        with self.assertRaises(ValueError):
            reliability_bins([0.5], [0, 1], bins=10)

    def test_threshold_metrics(self):
        from analysis_utils import threshold_metrics
        y = np.array([0, 0, 1, 1])
        p = np.array([0.2, 0.6, 0.8, 0.4])
        rows = threshold_metrics(y, p, thresholds=(0.5, 0.9), with_ci=False)
        self.assertEqual(rows[0]["threshold"], 0.5)
        self.assertAlmostEqual(rows[0]["precision"], 0.5)
        self.assertAlmostEqual(rows[0]["recall"], 0.5)
        self.assertAlmostEqual(rows[0]["flagged_neg_share"], 0.5)
        self.assertAlmostEqual(rows[1]["precision"], 0.0)
        self.assertAlmostEqual(rows[1]["recall"], 0.0)
        self.assertEqual(len(threshold_metrics(y, p, with_ci=True)[0]["precision_ci"]), 2)


class TestWriteCsvGzIfChanged(unittest.TestCase):
    def test_skips_identical_and_writes_when_changed(self):
        import os
        import pandas as pd
        from analysis_utils import write_csv_gz_if_changed

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scores.csv.gz")
            df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
            self.assertTrue(write_csv_gz_if_changed(df, path))
            first_bytes = pathlib.Path(path).read_bytes()
            self.assertFalse(write_csv_gz_if_changed(df.copy(), path))
            self.assertEqual(pathlib.Path(path).read_bytes(), first_bytes)
            changed = df.copy()
            changed.loc[0, "a"] = 9
            self.assertTrue(write_csv_gz_if_changed(changed, path))
            back = pd.read_csv(path)
            self.assertEqual(back["a"].tolist(), [9, 2])
            self.assertEqual(back["b"].tolist(), ["x", "y"])


class TestFontSetup(unittest.TestCase):
    def test_returns_name_or_none_without_crashing(self):
        import os
        old = os.environ.get("REDNOTE_FONT_DOWNLOAD")
        os.environ["REDNOTE_FONT_DOWNLOAD"] = "0"
        try:
            name = setup_chinese_fonts(allow_download=False)
        finally:
            if old is None:
                os.environ.pop("REDNOTE_FONT_DOWNLOAD", None)
            else:
                os.environ["REDNOTE_FONT_DOWNLOAD"] = old
        self.assertTrue(name is None or isinstance(name, str))

    def test_discover_cjk_font_name_matching(self):
        from analysis_utils import discover_cjk_font

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            only = [str(base)]
            (base / "DejaVuSans.ttf").write_bytes(b"x")
            self.assertIsNone(discover_cjk_font(dirs=only))
            nested = base / "nested"
            nested.mkdir()
            expected = nested / "NotoSansSC-Regular.otf"
            expected.write_bytes(b"x")
            self.assertEqual(discover_cjk_font(dirs=only), str(expected))
            (base / "wqy-microhei.ttc").write_bytes(b"x")
            self.assertEqual(discover_cjk_font(dirs=only),
                             str(base / "wqy-microhei.ttc"))

    def test_discover_emoji_font_skips_color_fonts(self):
        from analysis_utils import discover_emoji_font

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            only = [str(base)]
            color = base / "NotoColorEmoji.ttf"
            color.write_bytes(b"x")
            self.assertIsNone(discover_emoji_font(dirs=only))
            mono = base / "NotoEmoji-VariableFont_wght.ttf"
            mono.write_bytes(b"x")
            self.assertEqual(discover_emoji_font(dirs=only), str(mono))

    def test_font_search_dirs_env_priority(self):
        import os
        from analysis_utils import _font_search_dirs

        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("REDNOTE_FONT_DIR")
            os.environ["REDNOTE_FONT_DIR"] = tmp
            try:
                dirs = _font_search_dirs("extra-dir-not-exists")
                self.assertEqual(dirs[0], tmp)
            finally:
                if old is None:
                    os.environ.pop("REDNOTE_FONT_DIR", None)
                else:
                    os.environ["REDNOTE_FONT_DIR"] = old

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


class TestHashtagHelpers(unittest.TestCase):
    def test_extract_hashtags_plain_and_huati_format(self):
        from analysis_utils import extract_hashtags

        self.assertEqual(extract_hashtags("#减肥  #打卡"),
                         ["减肥", "打卡"])
        self.assertEqual(extract_hashtags("#穿搭[话题]# #春日[话题]#"),
                         ["穿搭", "春日"])
        self.assertEqual(extract_hashtags("正文没有标签"), [])
        self.assertEqual(extract_hashtags(None), [])

    def test_extract_hashtags_strips_bom_and_skips_junk(self):
        from analysis_utils import extract_hashtags

        self.assertEqual(extract_hashtags("\ufeff#TVB[话题]#\ufeff"),
                         ["TVB"])
        self.assertEqual(extract_hashtags("#\ufeff \ufeff#我是追剧党[话题]#"),
                         ["我是追剧党"])
        self.assertEqual(extract_hashtags("##人生不会完蛋的[话题]#"),
                         ["人生不会完蛋的"])
        self.assertEqual(extract_hashtags("#话题 #"), [])

    def test_extract_hashtags_keeps_original_case_and_length_cap(self):
        from analysis_utils import extract_hashtags

        self.assertEqual(extract_hashtags("#MBTI #mbti"), ["MBTI", "mbti"])
        long_tag = "字" * 40
        self.assertEqual(extract_hashtags("#" + long_tag),
                         [long_tag[:30]])

    def test_quarter_of(self):
        from analysis_utils import quarter_of

        self.assertEqual(quarter_of("202301"), "2023Q1")
        self.assertEqual(quarter_of("202412"), "2024Q4")
        self.assertIsNone(quarter_of("2023"))
        self.assertIsNone(quarter_of(""))
        self.assertIsNone(quarter_of(None))
        self.assertIsNone(quarter_of("2023ab"))

    def test_dedup_indices_keeps_best_per_id(self):
        from analysis_utils import dedup_indices

        ids = ["a", "a", "b", None, "b"]
        meta = np.array([1, 5, 2, 0, 4])
        self.assertEqual(dedup_indices(ids, meta), [1, 3, 4])
        self.assertEqual(dedup_indices([None, None], np.array([0, 0])), [0, 1])


class TestWithinCellSpearman(unittest.TestCase):
    @staticmethod
    def _make():
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(size=150), rng.normal(size=150)])
        y = np.concatenate([x[:150] + rng.normal(scale=0.01, size=150),
                            5 - x[150:] + rng.normal(scale=0.01, size=150)])
        codes = np.array(["a"] * 150 + ["b"] * 150)
        return x, y, codes

    def test_positive_and_negative_cells(self):
        from analysis_utils import within_cell_spearman

        x, y, codes = self._make()
        stat = within_cell_spearman(x, y, codes, min_cell_n=100)
        self.assertEqual(stat["n_cells"], 2)
        self.assertAlmostEqual(stat["weighted_rho"], 0.0, delta=0.05)
        self.assertAlmostEqual(stat["neg_cell_share"], 0.5)

    def test_single_positive_cell(self):
        from analysis_utils import within_cell_spearman

        x, y, codes = self._make()
        stat = within_cell_spearman(x[:150], y[:150], codes[:150],
                                    min_cell_n=100)
        self.assertEqual(stat["n_cells"], 1)
        self.assertGreater(stat["weighted_rho"], 0.99)
        self.assertEqual(stat["neg_cell_share"], 0.0)

    def test_small_cells_skipped(self):
        from analysis_utils import within_cell_spearman

        x, y, _ = self._make()
        stat = within_cell_spearman(x, y, np.zeros(300, dtype=int),
                                    min_cell_n=1000)
        self.assertEqual(stat["n_cells"], 0)
        self.assertTrue(np.isnan(stat["weighted_rho"]))
        self.assertTrue(np.isnan(stat["neg_cell_share"]))

    def test_constant_cell_skipped(self):
        from analysis_utils import within_cell_spearman

        x = np.zeros(150)
        y = np.arange(150, dtype=float)
        stat = within_cell_spearman(x, y, np.zeros(150), min_cell_n=100)
        self.assertEqual(stat["n_cells"], 0)

    def test_length_mismatch_raises(self):
        from analysis_utils import within_cell_spearman

        with self.assertRaises(ValueError):
            within_cell_spearman([1, 2], [1, 2, 3], [0, 0])


class TestLengthBinCodes(unittest.TestCase):
    def test_boundaries(self):
        codes = length_bin_codes([49, 50, 149, 150, 799, 800, 10_000])
        np.testing.assert_array_equal(codes, [0, 0, 0, 1, 3, 4, 4])

    def test_default_bins_cover_all_lengths(self):
        codes = length_bin_codes([50, 999_999_999])
        self.assertEqual(codes[0], 0)
        self.assertEqual(codes[1], len(LENGTH_BINS) - 1)


class TestNgramRepeatShare(unittest.TestCase):
    def test_no_repeat_is_zero(self):
        self.assertEqual(ngram_repeat_share("abcdef", n=3), 0.0)

    def test_known_repeat_ratio(self):
        # "abababab" 的 2-gram 共 7 个，唯一 2 个 -> 1 - 2/7
        self.assertAlmostEqual(ngram_repeat_share("abababab", n=2), 5 / 7)

    def test_constant_text_all_repeat(self):
        self.assertAlmostEqual(ngram_repeat_share("x" * 100, n=6), 1 - 1 / 95)

    def test_undefined_short_text(self):
        self.assertTrue(np.isnan(ngram_repeat_share("abcd", n=6)))
        self.assertTrue(np.isnan(ngram_repeat_share("", n=6)))
        self.assertTrue(np.isnan(ngram_repeat_share(None, n=6)))


class TestLengthMatchedMeanGap(unittest.TestCase):
    def test_constant_shift_exact(self):
        lens = np.full(400, 100)
        a = np.full(200, 3.0)
        b = np.full(200, 1.0)
        rec = length_matched_mean_gap(a, lens[:200], b, lens[:200])
        self.assertAlmostEqual(rec["gap"], 2.0)
        self.assertEqual(rec["used_bins"], 1)
        self.assertEqual(rec["n_a"], 200)

    def test_missing_bin_renormalizes(self):
        # a 侧只有 100 字分箱，b 侧只有 400 字分箱：没有共同分箱
        rec = length_matched_mean_gap([1.0], [100], [2.0], [400])
        self.assertTrue(np.isnan(rec["gap"]))
        self.assertEqual(rec["used_bins"], 0)

    def test_weights_follow_side_b(self):
        # b 侧 90% 在 100 字分箱、10% 在 400 字分箱；a 侧两侧均有
        a = np.array([1.0, 5.0])
        la = np.array([100, 400])
        b = np.array([0.0] * 9 + [0.0])
        lb = np.array([100] * 9 + [400])
        rec = length_matched_mean_gap(a, la, b, lb)
        self.assertAlmostEqual(rec["gap"], 0.9 * 1.0 + 0.1 * 5.0)


class TestLengthResidualize(unittest.TestCase):
    def test_residuals_are_zero_within_bins(self):
        lens = np.array([100.0, 120.0, 200.0, 250.0])
        vals = np.array([1.0, 3.0, -1.0, 1.0])
        res = length_residualize(vals, lens)
        np.testing.assert_allclose(res, [-1.0, 1.0, -1.0, 1.0])

    def test_nan_preserved(self):
        res = length_residualize([1.0, np.nan], [100, 100])
        self.assertAlmostEqual(res[0], 0.0)
        self.assertTrue(np.isnan(res[1]))

    def test_reference_sample_used(self):
        # 参考样本在 100 字分箱均值为 10，待标准化值减 10
        res = length_residualize([12.0], [100], ref_values=[9.0, 11.0],
                                 ref_lens=[100, 100])
        self.assertAlmostEqual(res[0], 2.0)


class TestDecileTable(unittest.TestCase):
    def test_monotone_signal(self):
        scores = np.linspace(0, 1, 1000)
        rows = decile_table(scores * 2.0, scores)
        self.assertEqual(len(rows), 10)
        means = [r["mean"] for r in rows]
        self.assertEqual(means, sorted(means))
        self.assertEqual(sum(r["n"] for r in rows), 1000)
        self.assertLessEqual(rows[0]["ci_low"], rows[0]["mean"])

    def test_constant_scores_returns_empty(self):
        self.assertEqual(decile_table([1.0, 2.0], [0.5, 0.5]), [])


class TestStandardizeWeights(unittest.TestCase):
    def test_normalize(self):
        w = standardize_weights({"a": 2.0, "b": 2.0})
        self.assertAlmostEqual(w["a"], 0.5)
        self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_all_zero_equal_weights(self):
        w = standardize_weights({"a": 0.0, "b": 0.0})
        self.assertAlmostEqual(w["a"], 0.5)
        self.assertAlmostEqual(w["b"], 0.5)


class TestDomainStandardizedSeries(unittest.TestCase):
    def test_weights_follow_domain_shares(self):
        df = pd.DataFrame({
            "quarter": ["Q1", "Q1", "Q1", "Q2", "Q2", "Q2"],
            "domain": ["a", "a", "b", "a", "a", "b"],
            "v": [1.0, 1.0, 5.0, 3.0, 3.0, 7.0],
        })
        s = domain_standardized_series(df, "v", "quarter")
        self.assertAlmostEqual(s["Q1"], 1.0 * (2 / 3) + 5.0 * (1 / 3))
        self.assertAlmostEqual(s["Q2"], 3.0 * (2 / 3) + 7.0 * (1 / 3))

    def test_missing_domain_renormalizes(self):
        df = pd.DataFrame({
            "quarter": ["Q1", "Q1", "Q1", "Q2", "Q2"],
            "domain": ["a", "a", "b", "a", "a"],
            "v": [1.0, 1.0, 5.0, 3.0, 3.0],
        })
        s = domain_standardized_series(df, "v", "quarter")
        self.assertAlmostEqual(s["Q2"], 3.0)


def _double(x: int) -> int:
    return 2 * x


class TestRunParallel(unittest.TestCase):
    def test_results_in_input_order(self):
        tasks = [partial(_double, i) for i in range(7)]
        self.assertEqual(run_parallel(tasks), [0, 2, 4, 6, 8, 10, 12])

    def test_serial_fallback_matches(self):
        tasks = [partial(_double, i) for i in range(4)]
        with mock.patch("analysis_utils.make_pool", return_value=None):
            self.assertEqual(run_parallel(tasks), [0, 2, 4, 6])


class TestBodyCache(unittest.TestCase):
    def test_duplicate_inputs_reuse_result(self):
        calls = []

        def fn(t):
            calls.append(t)
            return {"v": len(t)}

        cached = body_cache(fn)
        a = cached("aa")
        b = cached("aa")
        c = cached("bbb")
        self.assertEqual(a, b)
        self.assertIs(a, b)
        self.assertEqual(c, {"v": 3})
        self.assertEqual(calls, ["aa", "bbb"])

    def test_disables_on_low_hit_rate(self):
        calls = []

        def fn(t):
            calls.append(t)
            return t.upper()

        cached = body_cache(fn, probe=4, min_hit=0.5)
        for i in range(4):
            self.assertEqual(cached(f"t{i}"), f"T{i}")
        self.assertEqual(len(calls), 4)
        # 命中率不足后缓存关闭：重复输入重算但结果不变
        self.assertEqual(cached("t0"), "T0")
        self.assertEqual(len(calls), 5)

    def test_stays_enabled_on_duplicate_heavy_input(self):
        calls = []

        def fn(t):
            calls.append(t)
            return len(t)

        cached = body_cache(fn, probe=4, min_hit=0.5)
        for _ in range(6):
            self.assertEqual(cached("aaa"), 3)
        self.assertEqual(len(calls), 1)

    def test_small_input_keeps_caching(self):
        calls = []

        def fn(t):
            calls.append(t)
            return t

        cached = body_cache(fn, probe=100, min_hit=0.5)
        self.assertEqual(cached("x"), "x")
        self.assertEqual(cached("x"), "x")
        self.assertEqual(cached("y"), "y")
        self.assertEqual(calls, ["x", "y"])


if __name__ == "__main__":
    unittest.main()
