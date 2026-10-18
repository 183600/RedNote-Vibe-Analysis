# -*- coding: utf-8 -*-
"""discourse_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from discourse_analysis import (ALL_WORDS, CATEGORY_ORDER, SPOKEN_WORDS,
                                WORD_CATEGORY, WORD_REGISTER, WRITTEN_WORDS,
                                category_richness, density_per_100,
                                register_counts, word_counts)


class TestWordCounts(unittest.TestCase):
    def test_basic_category_counts(self):
        counts = word_counts("因为天气很好，所以出门了，因此心情不错")
        self.assertEqual(counts["causal"], {"因为": 1, "所以": 1, "因此": 1})

    def test_zhisuoyi_not_double_counted(self):
        """「之所以」不能被拆成「所以」再加「之所以」。"""
        counts = word_counts("我之所以这么说")
        self.assertEqual(counts["causal"], {"之所以": 1})
        self.assertEqual(counts["_words"].get("所以", 0), 0)
        self.assertEqual(counts["_words"]["之所以"], 1)

    def test_summary_nested_words(self):
        counts = word_counts("总的来说，简单来说，也是如此")
        self.assertEqual(counts["summary"], {"总的来说": 1, "简单来说": 1})

    def test_sequence_and_addition(self):
        counts = word_counts("首先看配料，其次看价格，另外还要注意保质期")
        self.assertEqual(counts["sequence"], {"首先": 1, "其次": 1, "另外": 1})
        counts = word_counts("它不仅好看，而且便宜，甚至还有赠品")
        self.assertEqual(counts["addition"], {"不仅": 1, "而且": 1, "甚至": 1})

    def test_no_match_returns_zeros(self):
        counts = word_counts("今天天气不错，风很温柔")
        for k in CATEGORY_ORDER:
            self.assertEqual(counts[k], {})
        self.assertEqual(counts["_words"], {})

    def test_category_equals_word_sum(self):
        text = ("因为喜欢，所以买了；但是有点贵，不过质量好；"
                "首先开箱，然后拍照，最后发笔记；比如配色，例如材质；总之很值")
        counts = word_counts(text)
        for k in CATEGORY_ORDER:
            self.assertEqual(counts[k] and sum(counts[k].values()) or 0,
                             sum(n for w, n in counts["_words"].items()
                                 if WORD_CATEGORY[w] == k), msg=k)

    def test_register_counts_cover_all_words(self):
        counts = word_counts("因为所以但是然后不过其实比如而且，然而因此")
        regs = register_counts(counts)
        self.assertEqual(regs["spoken"], 8)
        self.assertEqual(regs["written"], 2)
        self.assertEqual(regs["spoken"] + regs["written"],
                         sum(counts["_words"].values()))


class TestHelpers(unittest.TestCase):
    def test_category_richness(self):
        self.assertEqual(category_richness(word_counts("因为所以")), 1)
        self.assertEqual(category_richness(
            word_counts("因为，但是，首先，比如，总之")), 5)
        self.assertEqual(category_richness(word_counts("天气不错")), 0)

    def test_density(self):
        self.assertAlmostEqual(density_per_100(2, 100), 2.0)
        self.assertAlmostEqual(density_per_100(3, 0), 0.0)


class TestLexiconDefinition(unittest.TestCase):
    def test_words_are_unique_and_categorized(self):
        self.assertEqual(len(ALL_WORDS), len(set(ALL_WORDS)))
        self.assertEqual(set(WORD_CATEGORY.values()), set(CATEGORY_ORDER))

    def test_registers_cover_every_word(self):
        self.assertEqual(set(WORD_REGISTER), set(ALL_WORDS))
        self.assertEqual({w for w in ALL_WORDS if WORD_REGISTER[w] == "spoken"},
                         set(SPOKEN_WORDS))
        self.assertEqual({w for w in ALL_WORDS if WORD_REGISTER[w] == "written"},
                         set(WRITTEN_WORDS))
        self.assertTrue(SPOKEN_WORDS and WRITTEN_WORDS)

    def test_expected_spoken_words(self):
        self.assertEqual(set(SPOKEN_WORDS),
                         {"因为", "所以", "但是", "然后", "不过", "其实",
                          "比如", "而且"})


if __name__ == "__main__":
    unittest.main()
