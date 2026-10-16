# -*- coding: utf-8 -*-
"""unique_comparison 用户层面指标单元测试（不触碰研究数据文件）。"""
import pathlib
import sys
import unittest
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unique_comparison import analyze_user_behavior_detailed


class TestUserBehaviorDenominators(unittest.TestCase):
    def test_metrics_use_attributed_posts_only(self):
        aggs = {
            "Exploring": {
                "has_user_id": True,
                "total": 10,
                "user_counts": Counter({"u1": 3, "u2": 1}),
            }
        }
        res = analyze_user_behavior_detailed(aggs)["Exploring"]
        self.assertEqual(res["posts_per_user"], 2.0)
        self.assertEqual(res["top_user_post_share"], 75.0)

    def test_dataset_without_user_id_is_na(self):
        aggs = {
            "Human": {
                "has_user_id": False,
                "total": 5,
                "user_counts": Counter(),
            }
        }
        self.assertIsNone(analyze_user_behavior_detailed(aggs)["Human"])


if __name__ == "__main__":
    unittest.main()
