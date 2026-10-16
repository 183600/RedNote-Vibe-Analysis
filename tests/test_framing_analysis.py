# -*- coding: utf-8 -*-
"""framing_analysis 纯函数单元测试（不触碰研究数据文件）。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from framing_analysis import (compact, dedup_indices, ending_class,
                              specific_templates, top_templates)


class TestCompact(unittest.TestCase):
    def test_removes_hashtags_whitespace_and_bom(self):
        self.assertEqual(compact("#护肤# 姐妹们！\u200b\ufeff"), "护肤姐妹们！")
        self.assertEqual(compact("  a\n\tb  "), "ab")
        self.assertEqual(compact(""), "")


class TestEndingClass(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(ending_class(""), "纯标签/空正文")
        self.assertEqual(ending_class("好开心[飞吻R]"), "小红书表情[xxR]")
        self.assertEqual(ending_class("太棒了！"), "感叹号")
        self.assertEqual(ending_class("真的吗？"), "问号")
        self.assertEqual(ending_class("爱了爱了～"), "波浪号")
        self.assertEqual(ending_class("结束了。"), "句号")
        self.assertEqual(ending_class("谢谢大家，"), "逗号")
        self.assertEqual(ending_class("转给@小明"), "@提及")
        self.assertEqual(ending_class("普通结尾"), "文字结尾")
        self.assertEqual(ending_class("笑死😊"), "Unicode emoji")


class TestDedupIndices(unittest.TestCase):
    def test_keeps_best_meta_per_note(self):
        ids = ["a", "a", "b", None, "b"]
        meta = np.array([1, 5, 2, 0, 4])
        kept = dedup_indices(ids, meta)
        self.assertEqual(kept, [1, 3, 4])

    def test_no_note_id_keeps_all(self):
        ids = [None, None]
        meta = np.array([0, 0])
        self.assertEqual(dedup_indices(ids, meta), [0, 1])


class TestTemplates(unittest.TestCase):
    def test_specific_templates_direction_and_threshold(self):
        a_cnt = {"ai词": 20, "共同": 10, "少见": 2}
        h_cnt = {"ai词": 1, "共同": 100, "人类词": 200}
        n_a, n_h = 1000, 1000
        a_spec = specific_templates(a_cnt, n_a, h_cnt, n_h,
                                    reverse=False, min_count=5, k=10)
        words = [t["word"] for t in a_spec]
        self.assertEqual(words[0], "ai词")
        self.assertIn("共同", words)
        self.assertNotIn("少见", words)
        h_spec = specific_templates(a_cnt, n_a, h_cnt, n_h,
                                    reverse=True, min_count=50, k=10)
        self.assertNotIn("少见", [t["word"] for t in h_spec])
        self.assertEqual(h_spec[0]["word"], "人类词")

    def test_top_templates_sorted_by_rate(self):
        tops = top_templates({"a": 30, "b": 10, "c": 5}, n_a=1000,
                             min_count=10, k=10)
        self.assertEqual([t["word"] for t in tops], ["a", "b"])

    def test_specific_templates_tie_order_is_canonical(self):
        counts = {"甲": 5, "乙": 5, "丙": 5}
        got = [t["word"] for t in specific_templates(counts, 1000, counts, 1000,
                                                     reverse=False, min_count=5,
                                                     k=10)]
        self.assertEqual(got, sorted(counts))


if __name__ == "__main__":
    unittest.main()
