# -*- coding: utf-8 -*-
"""concreteness_analysis 的单元测试（合成文本，无需研究数据）。"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from concreteness_analysis import (CITIES, CATEGORY_ORDER, MASTER_RE, PLACE_RE,
                                   PROVINCES, VAGUE_WORDS, classify,
                                   concrete_count, density_per_100,
                                   length_matched_gap, quarter_of, unit_token_of)


class TestUnitTokenOf(unittest.TestCase):
    def test_strips_digits_and_space(self):
        self.assertEqual(unit_token_of("50元"), "元")
        self.assertEqual(unit_token_of("3.5 kg"), "kg")
        self.assertEqual(unit_token_of("30分钟"), "分钟")
        self.assertEqual(unit_token_of("¥199"), "¥")

    def test_pure_number_falls_back(self):
        self.assertEqual(unit_token_of("123"), "符号")


class TestClassify(unittest.TestCase):
    def test_category_priority(self):
        counts, subtypes, units, _ = classify("这件50元，涨了3.5%，重5kg，跑了30分钟，第3天")
        # 50元/3.5%/5kg/30分钟 归入 unit，第3天 的 3天 也归入 unit
        self.assertEqual(counts["num"], 0)
        self.assertEqual(counts["unit"], 5)
        self.assertEqual(subtypes["money"], 1)
        self.assertEqual(subtypes["percent"], 1)
        self.assertEqual(subtypes["measure"], 1)
        self.assertEqual(subtypes["time"], 2)
        self.assertEqual(units["元"], 1)
        self.assertEqual(units["kg"], 1)

    def test_bare_numbers_counted_as_num(self):
        counts, _, _, _ = classify("2024年发布了3个版本和150条笔记")
        # 2024年 是带单位时间，3、150 是裸数字
        self.assertEqual(counts["num"], 2)
        self.assertEqual(counts["unit"], 1)

    def test_percent_chinese_numeral(self):
        counts, subtypes, _, _ = classify("成功率百分之九十九")
        self.assertEqual(subtypes["percent"], 1)
        self.assertEqual(counts["num"], 0)

    def test_time_clock_pattern(self):
        counts, subtypes, _, _ = classify("约在10:30见面")
        self.assertEqual(subtypes["time"], 1)
        self.assertEqual(counts["unit"], 1)

    def test_vague_words(self):
        counts, _, _, _ = classify("很多好吃的，一些朋友，各种玩法，或多或少都有")
        self.assertEqual(counts["vague"], 4)

    def test_vague_words_all_matchable(self):
        for w in VAGUE_WORDS:
            counts, _, _, _ = classify(w)
            self.assertEqual(counts["vague"], 1, msg=w)

    def test_place_gazetteer_all_matchable(self):
        for name in set(PROVINCES) | set(CITIES):
            self.assertEqual(len(PLACE_RE.findall(name)), 1, msg=name)

    def test_place_tokens(self):
        counts, _, _, places = classify("从北京飞到三亚，再去杭州")
        self.assertEqual(counts["place"], 3)
        self.assertEqual(places["杭州"], 1)

    def test_concrete_composite(self):
        counts, _, _, _ = classify("50元 3 北京 很多")
        self.assertEqual(concrete_count(counts), 3)

    def test_no_match(self):
        counts, subtypes, units, places = classify("今天天气不错")
        self.assertEqual(counts, {k: 0 for k in CATEGORY_ORDER})
        self.assertEqual(sum(subtypes.values()), 0)
        self.assertEqual(len(units), 0)
        self.assertEqual(len(places), 0)

    def test_master_re_finds_all_non_overlapping(self):
        self.assertEqual(len(MASTER_RE.findall("12.5 和 3.5%")), 2)


class TestDensity(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(density_per_100(1, 50), 2.0)
        self.assertAlmostEqual(density_per_100(3, 0), 0.0)


class TestLengthMatchedGap(unittest.TestCase):
    def test_identical_samples_zero_gap(self):
        g = length_matched_gap([10, 20], [100, 100], [10, 20], [100, 100],
                               [1.0, 1.0])
        self.assertAlmostEqual(g["gap"], 0.0)
        # 解析 SE 含二项方差项，样本相同也不为 0，只要比 gap 小即可
        self.assertGreater(g["se"], 0.0)
        self.assertEqual(g["used_bins"], 2)

    def test_known_weighted_gap(self):
        # bin0: 0.5 vs 0.25, bin1: 0.2 vs 0.1；权重各半
        g = length_matched_gap([50, 20], [100, 100], [25, 10], [100, 100],
                               [1.0, 1.0])
        self.assertAlmostEqual(g["gap"], 0.5 * 0.25 + 0.5 * 0.1)

    def test_empty_bin_renormalizes(self):
        g = length_matched_gap([50, 0], [100, 0], [25, 0], [100, 0], [1.0, 1.0])
        self.assertEqual(g["used_bins"], 1)
        self.assertAlmostEqual(g["gap"], 0.25)

    def test_all_bins_missing(self):
        g = length_matched_gap([0, 0], [0, 0], [0, 0], [0, 0], [1.0, 1.0])
        self.assertTrue(g["gap"] != g["gap"])  # NaN


class TestQuarterOf(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(quarter_of("202301"), "2023Q1")
        self.assertEqual(quarter_of("202412"), "2024Q4")
        self.assertEqual(quarter_of("202506"), "2025Q2")

    def test_invalid(self):
        self.assertIsNone(quarter_of(None))
        self.assertIsNone(quarter_of(""))
        self.assertIsNone(quarter_of("2023"))
        self.assertIsNone(quarter_of("abcdef"))


if __name__ == "__main__":
    unittest.main()
