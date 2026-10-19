# -*- coding: utf-8 -*-
"""affect_analysis 的单元测试（合成文本，无需研究数据）。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from affect_analysis import (ALL_WORDS, CATEGORY_ORDER, CATEGORY_WORDS,
                             NEGATIVE_PARTS, POSITIVE_PARTS, SCORE_ORDER,
                             WORD_CATEGORY, _counts_for, density_per_100,
                             rates_block, word_counts)


class TestWordCounts(unittest.TestCase):
    def test_five_emotions(self):
        counts = word_counts("开心、喜欢、焦虑、难过、生气")
        self.assertEqual(counts["joy"]["开心"], 1)
        self.assertEqual(counts["affection"]["喜欢"], 1)
        self.assertEqual(counts["anxiety"]["焦虑"], 1)
        self.assertEqual(counts["sadness"]["难过"], 1)
        self.assertEqual(counts["anger"]["生气"], 1)

    def test_longest_match_avoids_double_counting(self):
        counts = word_counts("爱了可爱的你，甜蜜")
        self.assertEqual(counts["joy"]["爱了"], 1)
        self.assertEqual(counts["joy"]["可爱"], 1)
        self.assertEqual(counts["affection"].get("爱", 0), 0)
        self.assertEqual(counts["affection"]["甜蜜"], 1)
        self.assertEqual(counts["affection"].get("甜", 0), 0)

    def test_all_words_roundtrip(self):
        text = "".join(ALL_WORDS)
        counts = word_counts(text)
        for k in CATEGORY_ORDER:
            for w in CATEGORY_WORDS[k][1]:
                self.assertEqual(counts[k].get(w, 0), 1, msg=f"{k}:{w}")

    def test_no_match(self):
        counts = word_counts("天气晴朗")
        for k in CATEGORY_ORDER:
            self.assertEqual(counts[k], {})


class TestLexiconIntegrity(unittest.TestCase):
    def test_words_are_unique_and_mapped(self):
        flat = [w for k in CATEGORY_ORDER for w in CATEGORY_WORDS[k][1]]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(flat), set(WORD_CATEGORY))
        self.assertEqual(len(flat), len(ALL_WORDS))

    def test_composites_partition_categories(self):
        self.assertEqual(set(POSITIVE_PARTS) | set(NEGATIVE_PARTS),
                         set(CATEGORY_ORDER))
        self.assertFalse(set(POSITIVE_PARTS) & set(NEGATIVE_PARTS))
        self.assertEqual(SCORE_ORDER,
                         CATEGORY_ORDER + ["positive", "negative", "net"])


class TestCountsFor(unittest.TestCase):
    def test_composites_and_net(self):
        scan = {
            "cat_counts": {
                "joy": np.asarray([2, 0], dtype=np.int32),
                "affection": np.asarray([1, 0], dtype=np.int32),
                "anxiety": np.asarray([0, 3], dtype=np.int32),
                "sadness": np.asarray([1, 0], dtype=np.int32),
                "anger": np.asarray([0, 1], dtype=np.int32),
            },
            "positive_counts": np.asarray([3, 0], dtype=np.int32),
            "negative_counts": np.asarray([1, 4], dtype=np.int32),
        }
        np.testing.assert_array_equal(_counts_for(scan, "positive"), [3, 0])
        np.testing.assert_array_equal(_counts_for(scan, "negative"), [1, 4])
        np.testing.assert_array_equal(_counts_for(scan, "net"), [2, -4])


class TestRatesBlock(unittest.TestCase):
    def test_density_and_share(self):
        scan = {
            "body_len": np.asarray([100, 100, 20], dtype=np.int32),
            "cat_counts": {k: np.asarray([0, 0, 0], dtype=np.int32)
                           for k in CATEGORY_ORDER},
            "positive_counts": np.asarray([2, 0, 5], dtype=np.int32),
            "negative_counts": np.asarray([0, 1, 0], dtype=np.int32),
            "word_counts": {w: np.asarray([0, 0, 0], dtype=np.int32)
                            for w in ALL_WORDS},
        }
        scan["cat_counts"]["joy"] = np.asarray([2, 0, 1], dtype=np.int32)
        scan["word_counts"]["开心"] = np.asarray([2, 0, 1], dtype=np.int32)
        out = rates_block(scan, np.asarray([0, 1, 2]), min_len=50)
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["chars"], 200)
        self.assertAlmostEqual(out["categories"]["positive"]["density100"], 1.0)
        self.assertAlmostEqual(out["categories"]["net"]["density100"], 0.5)
        self.assertAlmostEqual(out["words"]["开心"]["per10k_chars"], 100.0)


class TestDensity(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(density_per_100(1, 50), 2.0)
        self.assertAlmostEqual(density_per_100(3, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
