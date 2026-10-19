# -*- coding: utf-8 -*-
"""redundancy_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from redundancy_analysis import (DUP_SIM_THRESHOLD, GRAM_N, METRIC_ORDER,
                                 SENT_MIN_CHARS, char_ngram_set, jaccard,
                                 mean_ci, redundancy_metrics,
                                 sentence_repetition)


class TestCharNgramSet(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(char_ngram_set("abcd", 3), {"abc", "bcd"})

    def test_short_returns_empty(self):
        self.assertEqual(char_ngram_set("ab", 3), frozenset())

    def test_jaccard(self):
        a = char_ngram_set("abcdef", 3)
        self.assertAlmostEqual(jaccard(a, a), 1.0)
        self.assertAlmostEqual(jaccard(a, char_ngram_set("xyz", 3)), 0.0)
        self.assertAlmostEqual(jaccard(frozenset(), a), 0.0)


class TestSentenceRepetition(unittest.TestCase):
    def test_identical_sentences(self):
        rec = sentence_repetition("今天天气真好呀。今天天气真好呀。")
        self.assertEqual(rec["n_sent"], 2)
        self.assertAlmostEqual(rec["max_sim"], 1.0)
        self.assertAlmostEqual(rec["dup_sent"], 1.0)
        self.assertAlmostEqual(rec["open_rep"], 1.0)

    def test_distinct_sentences(self):
        rec = sentence_repetition("今天天气真好呀。完全不同的另一段话。")
        self.assertEqual(rec["n_sent"], 2)
        self.assertAlmostEqual(rec["max_sim"], 0.0)
        self.assertAlmostEqual(rec["dup_sent"], 0.0)
        self.assertAlmostEqual(rec["open_rep"], 0.0)

    def test_short_sentences_filtered(self):
        rec = sentence_repetition("好。" * 10)
        self.assertEqual(rec["n_sent"], 0)
        for k in ("max_sim", "dup_sent", "open_rep"):
            self.assertTrue(np.isnan(rec[k]), msg=k)

    def test_single_sentence_undefined(self):
        rec = sentence_repetition("只有一句比较长的话在这里。")
        self.assertEqual(rec["n_sent"], 1)
        self.assertTrue(np.isnan(rec["max_sim"]))

    def test_min_chars_boundary(self):
        s = "一" * SENT_MIN_CHARS
        self.assertEqual(sentence_repetition(f"{s}。{s}。")["n_sent"], 2)
        short = "一" * (SENT_MIN_CHARS - 1)
        self.assertEqual(sentence_repetition(f"{short}。{short}。")["n_sent"], 0)

    def test_open_rep_partial(self):
        rec = sentence_repetition(
            "我们今天去公园。我们今天看电影。完全不同内容。")
        self.assertEqual(rec["n_sent"], 3)
        self.assertAlmostEqual(rec["open_rep"], 2 / 3)

    def test_threshold_is_half(self):
        self.assertEqual(DUP_SIM_THRESHOLD, 0.5)


class TestRedundancyMetrics(unittest.TestCase):
    def test_metric_keys(self):
        rec = redundancy_metrics("任意一段正文内容。")
        self.assertEqual(set(rec), set(METRIC_ORDER) | {"n_sent"})

    def test_repeated_body_has_positive_gram6(self):
        body = "这是一段会重复的话。" * 3
        self.assertGreater(redundancy_metrics(body)["gram6"], 0.0)

    def test_short_body_gram6_nan(self):
        self.assertTrue(np.isnan(redundancy_metrics("短")["gram6"]))

    def test_gram6_deterministic(self):
        body = "abcdef" * 20
        self.assertEqual(redundancy_metrics(body)["gram6"],
                         redundancy_metrics(body)["gram6"])

    def test_gram_n_constant(self):
        self.assertEqual(GRAM_N, 6)


class TestMeanCI(unittest.TestCase):
    def test_basic(self):
        m, lo, hi = mean_ci([1.0, 2.0, 3.0])
        self.assertAlmostEqual(m, 2.0)
        self.assertLess(lo, m)
        self.assertGreater(hi, m)

    def test_empty_is_nan(self):
        for v in mean_ci([]):
            self.assertTrue(np.isnan(v))

    def test_filters_nan(self):
        m, _, _ = mean_ci([1.0, np.nan, 3.0])
        self.assertAlmostEqual(m, 2.0)


if __name__ == "__main__":
    unittest.main()
