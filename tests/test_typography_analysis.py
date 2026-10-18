# -*- coding: utf-8 -*-
"""typography_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from typography_analysis import (CATEGORY_DEFS, CATEGORY_LABELS, CATEGORY_ORDER,
                                 MARKER_CATEGORY, MARKER_DISPLAY, LAYOUT_KEYS,
                                 density_per_100, layout_metrics,
                                 punct_type_count, typography_counts)


class TestTypographyCounts(unittest.TestCase):
    def test_full_and_half_width_variants(self):
        counts = typography_counts("好！太棒了! 真的吗？really?")
        self.assertEqual(counts["excl"], 2)
        self.assertEqual(counts["ques"], 2)

    def test_ellipsis_forms(self):
        counts = typography_counts("嗯…再想想。。。或者...")
        self.assertEqual(counts["ellip"], 3)
        self.assertEqual(counts["_markers"]["…"], 1)
        self.assertEqual(counts["_markers"]["。。。"], 1)
        self.assertEqual(counts["_markers"]["..."], 1)

    def test_period_counts_inside_ellipsis(self):
        """省略号中的句号同时计入句号（与 paper_features 口径一致）。"""
        counts = typography_counts("。。。")
        self.assertEqual(counts["period"], 3)
        self.assertEqual(counts["ellip"], 1)

    def test_paired_quotes_and_brackets(self):
        counts = typography_counts("“你好”「世界」【总结】(a)")
        self.assertEqual(counts["quote"], 4)     # “ ” 「 」
        self.assertEqual(counts["bracket"], 4)   # 【 】 ( )
        self.assertEqual(counts["_markers"]["“"], 1)
        self.assertEqual(counts["_markers"]["】"], 1)

    def test_regular_markers(self):
        counts = typography_counts("首先，其次、然后。再者：最后；完")
        self.assertEqual(counts["comma"], 1)
        self.assertEqual(counts["pause"], 1)
        self.assertEqual(counts["period"], 1)
        self.assertEqual(counts["colon"], 1)
        self.assertEqual(counts["semcolon"], 1)

    def test_tilde_and_dash(self):
        counts = typography_counts("好开心～开心~ " + "—" * 3 + " －")
        self.assertEqual(counts["tilde"], 2)
        self.assertEqual(counts["dash"], 4)      # — ×3 + － ×1

    def test_no_markers(self):
        counts = typography_counts("今天天气很好")
        for k in CATEGORY_ORDER:
            self.assertEqual(counts[k], 0, msg=k)
        self.assertEqual(counts["_markers"], {})

    def test_category_equals_marker_sum(self):
        text = "姐妹们！速看？…真的很～棒，收藏、点赞。最重要：“码住”；冲鸭——"
        counts = typography_counts(text)
        for k in CATEGORY_ORDER:
            total = sum(n for w, n in counts["_markers"].items()
                        if MARKER_CATEGORY[w] == k)
            self.assertEqual(counts[k], total, msg=k)

    def test_punct_type_count(self):
        self.assertEqual(punct_type_count(typography_counts("好！")), 1)
        self.assertEqual(punct_type_count(typography_counts("好！真的吗？")), 2)
        self.assertEqual(punct_type_count(typography_counts("今天天气很好")), 0)


class TestLayoutMetrics(unittest.TestCase):
    def test_line_count_and_avg_len(self):
        lay = layout_metrics("第一行\n\n第二行  \n第三行")
        self.assertEqual(lay["n_lines"], 3)
        self.assertAlmostEqual(lay["avg_line_len"], len("第一行\n\n第二行  \n第三行") / 3)
        self.assertEqual(lay["single_line"], 0)

    def test_single_line(self):
        lay = layout_metrics("只有一行")
        self.assertEqual(lay["n_lines"], 1)
        self.assertEqual(lay["single_line"], 1)
        self.assertAlmostEqual(lay["avg_line_len"], 4.0)

    def test_empty_body(self):
        lay = layout_metrics("")
        self.assertEqual(lay["n_lines"], 0)
        self.assertEqual(lay["single_line"], 1)
        self.assertAlmostEqual(lay["avg_line_len"], 0.0)

    def test_short_line_ratio(self):
        lay = layout_metrics("短\n" + "这是一条很长的行足够超过十五个字符了吧")
        self.assertAlmostEqual(lay["short_line_ratio"], 0.5)

    def test_list_and_colon_lines(self):
        body = "1. 第一步：准备材料\n2. 第二步\n普通叙述行"
        lay = layout_metrics(body)
        self.assertAlmostEqual(lay["list_line_ratio"], 2 / 3)
        self.assertAlmostEqual(lay["colon_line_ratio"], 1 / 3)

    def test_metrics_keys(self):
        lay = layout_metrics("任意正文")
        for k in LAYOUT_KEYS:
            if k == "punct_types":
                continue
            self.assertIn(k, lay)


class TestHelpers(unittest.TestCase):
    def test_density(self):
        self.assertAlmostEqual(density_per_100(2, 100), 2.0)
        self.assertAlmostEqual(density_per_100(3, 0), 0.0)


class TestLexiconDefinition(unittest.TestCase):
    def test_labels_complete(self):
        for k in CATEGORY_ORDER:
            self.assertEqual(set(CATEGORY_DEFS), set(CATEGORY_ORDER))
            self.assertTrue(CATEGORY_LABELS[k])

    def test_markers_unique_and_mapped(self):
        self.assertEqual(len(MARKER_DISPLAY), len(set(MARKER_DISPLAY)))
        self.assertEqual(set(MARKER_CATEGORY.values()), set(CATEGORY_ORDER))

    def test_paper_features_shared_categories(self):
        """与 paper_features 同源的类别必须包含对应全角字符。"""
        for k, ch in (("excl", "！"), ("ques", "？"), ("comma", "，"),
                      ("pause", "、"), ("period", "。"), ("tilde", "～")):
            self.assertIn(ch, MARKER_DISPLAY, msg=k)


if __name__ == "__main__":
    unittest.main()
