# -*- coding: utf-8 -*-
"""cta_analysis 的单元测试（合成数据，无需研究数据）。"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cta_analysis import (CATEGORY_DEFS, CATEGORY_ORDER, PATTERN_CATEGORY,
                          PATTERN_DISPLAY, any_density, cta_counts,
                          density_per_100, has_any)


class TestCtaCounts(unittest.TestCase):
    def test_basic_category_counts(self):
        counts = cta_counts("姐妹们评论区告诉我，记得点赞收藏哦～")
        self.assertEqual(counts["comment"], 2)   # 评论区 + 告诉我
        self.assertEqual(counts["reaction"], 2)  # 点赞 + 收藏
        self.assertEqual(counts["follow"], 0)
        self.assertEqual(counts["urge"], 0)

    def test_multi_pattern_same_category(self):
        counts = cta_counts("在评论区交流，评论区见")
        self.assertEqual(counts["comment"], 3)   # 评论区 ×2 + 交流
        self.assertEqual(counts["_patterns"]["评论区"], 2)
        self.assertEqual(counts["_patterns"]["交流"], 1)

    def test_liuyan_qu_longest_first(self):
        """「留言区」按最长优先计为一次，不拆成「留言」。"""
        counts = cta_counts("留言区见")
        self.assertEqual(counts["comment"], 1)
        self.assertEqual(counts["_patterns"].get("留言区", 0), 1)
        self.assertEqual(counts["_patterns"].get("留言", 0), 0)

    def test_shoucang_guard(self):
        """「收藏」排除收藏夹/收藏品/收藏家这类名词。"""
        counts = cta_counts("收藏夹里收藏了收藏品，收藏家也收藏")
        self.assertEqual(counts["_patterns"].get("收藏", 0), 2)
        self.assertEqual(counts["reaction"], 2)

    def test_liaoliao_guard(self):
        """「聊聊」排除「聊聊天」（叙事而非邀约）。"""
        self.assertEqual(cta_counts("聊聊天")["comment"], 0)
        self.assertEqual(cta_counts("聊聊吧")["comment"], 1)

    def test_cross_category_patterns(self):
        counts = cta_counts("关注我，分享一下，分享给朋友")
        self.assertEqual(counts["follow"], 2)    # 关注我 + 分享给
        self.assertEqual(counts["comment"], 1)   # 分享一下
        self.assertEqual(counts["_patterns"]["分享给"], 1)
        self.assertEqual(counts["_patterns"]["分享一下"], 1)

    def test_urge_markers(self):
        counts = cta_counts("快冲！冲鸭！别错过！")
        self.assertEqual(counts["urge"], 3)
        for w in ("快冲", "冲鸭", "别错过"):
            self.assertEqual(counts["_patterns"][w], 1)

    def test_category_equals_pattern_sum(self):
        text = "评论区聊聊吧，快收藏，关注一下，快冲别错过"
        counts = cta_counts(text)
        for k in CATEGORY_ORDER:
            total = sum(n for w, n in counts["_patterns"].items()
                        if PATTERN_CATEGORY[w] == k)
            self.assertEqual(counts[k], total, msg=k)

    def test_no_match_returns_zeros(self):
        counts = cta_counts("今天天气不错，风很温柔")
        for k in CATEGORY_ORDER:
            self.assertEqual(counts[k], 0)
        self.assertEqual(counts["_patterns"], {})


class TestHelpers(unittest.TestCase):
    def test_has_any(self):
        self.assertEqual(has_any(cta_counts("评论区见")), 1)
        self.assertEqual(has_any(cta_counts("天气不错")), 0)

    def test_density(self):
        self.assertAlmostEqual(density_per_100(2, 100), 2.0)
        self.assertAlmostEqual(density_per_100(3, 0), 0.0)

    def test_any_density_sums_categories(self):
        counts = cta_counts("评论区告诉我，点赞收藏")
        self.assertAlmostEqual(any_density(counts, 50),
                               (counts["comment"] + counts["reaction"]) * 2.0)


class TestLexiconDefinition(unittest.TestCase):
    def test_patterns_are_unique_and_ordered(self):
        self.assertEqual(len(PATTERN_DISPLAY), len(set(PATTERN_DISPLAY)))
        self.assertEqual(set(PATTERN_CATEGORY.values()), set(CATEGORY_ORDER))

    def test_every_category_has_label_and_patterns(self):
        for k in CATEGORY_ORDER:
            label, patterns = CATEGORY_DEFS[k]
            self.assertTrue(label)
            self.assertTrue(patterns)


if __name__ == "__main__":
    unittest.main()
