# -*- coding: utf-8 -*-
"""词频 Top-N 选取的排序口径回归测试（不触碰研究数据）。

覆盖 legacy 分析脚本中「按频次降序、同频按词升序」的 Top-N 选取：
- more_analysis / additional_analysis 的 analyze_unique_words
- complete_analysis / word_freq_analysis 的 compute_correlation
实现改用 (-count, word) 元组直接比较后，结果必须与原先 key=(-count, word)
的 heapq.nsmallest 完全一致（含并列频次时按词升序的次序）。
"""
import heapq
import pathlib
import sys
import unittest
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import additional_analysis as AA
import complete_analysis as CA
import more_analysis as MA
import word_freq_analysis as WF


def legacy_top(counter, n):
    """历史口径：heapq.nsmallest(n, counter.items(), key=(-count, word))。"""
    return heapq.nsmallest(n, counter.items(), key=lambda x: (-x[1], x[0]))


def make_freqs():
    freq_a = Counter({"的": 100, "你好": 50, "世界": 50, "苹果": 33,
                      "香蕉": 10, "橘子": 5, "猫": 5})
    freq_b = Counter({"的": 80, "你好": 50, "世界": 20, "西瓜": 33,
                      "香蕉": 10, "狗": 7, "橘子": 5})
    return freq_a, freq_b


class TestUniqueWordsOrdering(unittest.TestCase):
    def test_more_analysis_matches_legacy(self):
        freq_a, freq_b = make_freqs()
        freq_c = Counter({"的": 60, "独立词": 3})
        freqs = [freq_a, freq_b, freq_c]
        files = ["a", "b", "c"]
        result = MA.analyze_unique_words(freqs, files)

        expected_ab = legacy_top(
            Counter({w: c for w, c in freq_a.items() if w not in freq_b}), 30)
        self.assertEqual(result["a_unique_in_b"], expected_ab)
        expected_ba = legacy_top(
            Counter({w: c for w, c in freq_b.items() if w not in freq_a}), 30)
        self.assertEqual(result["b_unique_in_a"], expected_ba)
        # common_words 只用三个数据集的交集（这里只有「的」），频次为三者之和
        common3 = set(freq_a) & set(freq_b) & set(freq_c)
        combined3 = Counter({w: freq_a[w] + freq_b[w] + freq_c[w]
                             for w in common3})
        self.assertEqual(result["common_words"], legacy_top(combined3, 50))

    def test_additional_analysis_matches_legacy(self):
        freq_a, freq_b = make_freqs()
        result = AA.analyze_unique_words([freq_a, freq_b], ["a", "b"])
        self.assertEqual(
            result["a_unique_in_b"],
            legacy_top(Counter({w: c for w, c in freq_a.items()
                                if w not in freq_b}), 20))
        self.assertEqual(
            result["b_unique_in_a"],
            legacy_top(Counter({w: c for w, c in freq_b.items()
                                if w not in freq_a}), 20))

    def test_more_common_words_tie_break(self):
        freq_a, freq_b = make_freqs()
        result = MA.analyze_unique_words([freq_a, freq_b], ["a", "b"])
        # common_words 只统计交集词的合并频次；并列频次仍按词升序
        common = set(freq_a) & set(freq_b)
        combined = Counter({w: freq_a[w] + freq_b[w] for w in common})
        self.assertEqual(result["common_words"], legacy_top(combined, 50))


class TestComputeCorrelationOrdering(unittest.TestCase):
    def test_complete_analysis_union_branch(self):
        freq_a, freq_b = make_freqs()
        _, all_words, _, _ = CA.compute_correlation(freq_a, freq_b)
        combined = Counter(freq_a)
        combined.update(freq_b)
        self.assertLess(len(set(freq_a) & set(freq_b)), 10)
        self.assertEqual(all_words, [w for w, _ in legacy_top(combined, 100)])

    def test_word_freq_analysis_union_branch(self):
        freq_a, freq_b = make_freqs()
        _, all_words, _, _ = WF.compute_correlation(freq_a, freq_b)
        combined = Counter(freq_a)
        combined.update(freq_b)
        self.assertEqual(all_words, [w for w, _ in legacy_top(combined, 100)])

    def test_complete_analysis_common_branch(self):
        freq_a = Counter({"w%02d" % i: i for i in range(15)})
        freq_b = Counter({"w%02d" % i: 15 - i for i in range(15)})
        _, all_words, _, _ = CA.compute_correlation(freq_a, freq_b)
        common = set(freq_a) & set(freq_b)
        self.assertGreaterEqual(len(common), 10)
        combined = Counter({w: freq_a[w] + freq_b[w] for w in common})
        self.assertEqual(all_words, [w for w, _ in legacy_top(combined, 100)])


if __name__ == "__main__":
    unittest.main()
