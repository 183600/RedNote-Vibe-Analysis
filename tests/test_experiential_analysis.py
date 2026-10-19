# -*- coding: utf-8 -*-
"""experiential_analysis 的单元测试（合成文本，无需研究数据）。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiential_analysis import (ALL_WORDS, CATEGORY_ORDER, CATEGORY_WORDS,
                                   COMPOSITE, GROUNDING_PARTS, WORD_CATEGORY,
                                   density_per_100, rates_block, word_counts)


class TestWordCounts(unittest.TestCase):
    def test_longest_match_avoids_double_counting(self):
        counts = word_counts("我们自己")
        self.assertEqual(counts["self"]["我们"], 1)
        self.assertEqual(counts["self"]["自己"], 1)
        self.assertEqual(counts["self"].get("我", 0), 0)

    def test_nested_relation_terms(self):
        counts = word_counts("男朋友和女朋友都是朋友")
        self.assertEqual(counts["relation"]["男朋友"], 1)
        self.assertEqual(counts["relation"]["女朋友"], 1)
        self.assertEqual(counts["relation"]["朋友"], 1)

    def test_category_isolation(self):
        counts = word_counts("亲测好用，昨天闺蜜说味道不错")
        self.assertEqual(counts["experience"]["亲测"], 1)
        self.assertEqual(counts["time"]["昨天"], 1)
        self.assertEqual(counts["relation"]["闺蜜"], 1)
        self.assertEqual(counts["sensory"]["味道"], 1)

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
        self.assertEqual(counts["_words"], {})


class TestLexiconIntegrity(unittest.TestCase):
    def test_words_are_unique_and_mapped(self):
        flat = [w for k in CATEGORY_ORDER for w in CATEGORY_WORDS[k][1]]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(flat), set(WORD_CATEGORY))
        self.assertEqual(len(flat), len(ALL_WORDS))

    def test_composite_excludes_self_reference(self):
        self.assertNotIn("self", GROUNDING_PARTS)
        self.assertEqual(set(GROUNDING_PARTS),
                         set(CATEGORY_ORDER) - {"self"})
        self.assertEqual(COMPOSITE, "grounding")


class TestRatesBlock(unittest.TestCase):
    def test_density_and_share(self):
        scan = {
            "body_len": np.asarray([100, 100, 20], dtype=np.int32),
            "cat_counts": {k: np.asarray([0, 0, 0], dtype=np.int32)
                           for k in CATEGORY_ORDER},
            "grounding_counts": np.asarray([2, 0, 5], dtype=np.int32),
            "word_counts": {w: np.asarray([0, 0, 0], dtype=np.int32)
                            for w in ALL_WORDS},
        }
        scan["cat_counts"]["self"] = np.asarray([2, 0, 1], dtype=np.int32)
        scan["cat_counts"]["experience"] = np.asarray([1, 0, 0], dtype=np.int32)
        scan["cat_counts"]["time"] = np.asarray([1, 0, 0], dtype=np.int32)
        scan["word_counts"]["我"] = np.asarray([2, 0, 1], dtype=np.int32)
        out = rates_block(scan, np.asarray([0, 1, 2]), min_len=50)
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["chars"], 200)
        self.assertAlmostEqual(out["categories"]["self"]["density100"], 1.0)
        self.assertAlmostEqual(out["categories"][COMPOSITE]["share"], 0.5)
        self.assertAlmostEqual(out["words"]["我"]["per10k_chars"], 100.0)


class TestDensity(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(density_per_100(1, 50), 2.0)
        self.assertAlmostEqual(density_per_100(3, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
