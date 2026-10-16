# -*- coding: utf-8 -*-
"""diversity_analysis 纯函数单元测试（不触碰研究数据文件）。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from diversity_analysis import (doc_metrics, heaps_curve, heaps_beta,
                                MIN_SENTENCES_BURST)


class TestDocMetrics(unittest.TestCase):
    def test_counts_and_burstiness(self):
        body = "第一句。第二句！第三句？"
        m = doc_metrics(body)
        self.assertEqual(m["n_sent"], 3)
        self.assertEqual(m["sent_lens"], [3, 3, 3])
        self.assertAlmostEqual(m["burst"], 0.0)
        self.assertEqual(m["body_len"], len(body))

    def test_burstiness_none_for_short_posts(self):
        m = doc_metrics("只有一句。")
        self.assertEqual(m["n_sent"], 1)
        self.assertIsNone(m["burst"])
        self.assertEqual(m["body_len"], 5)

    def test_three_sentences_meets_threshold(self):
        m = doc_metrics("短。中等长度的一句。这是一个明显更长的句子用来拉开差距。")
        self.assertEqual(MIN_SENTENCES_BURST, 3)
        self.assertIsNotNone(m["burst"])
        self.assertGreater(m["burst"], 0.0)


class TestHeaps(unittest.TestCase):
    def test_curve_accumulates_types(self):
        docs = [["a", "b", "a"], ["b", "c"]]
        tokens, types = heaps_curve(docs)
        self.assertEqual(tokens.tolist(), [3.0, 5.0])
        self.assertEqual(types.tolist(), [2.0, 3.0])

    def test_budget_truncates(self):
        docs = [["w%d" % i for i in range(10)] for _ in range(5)]
        tokens, types = heaps_curve(docs, max_tokens=25)
        self.assertEqual(tokens[-1], 25.0)
        self.assertEqual(types[-1], 10.0)
        self.assertLessEqual(len(tokens), 5)

    def test_beta_recovers_power_law(self):
        rng = np.random.default_rng(0)
        vocab = [f"w{i}" for i in range(30_000)]
        docs = [list(rng.choice(vocab, size=100)) for _ in range(200)]
        tokens, types = heaps_curve(docs)
        beta = heaps_beta(tokens, types, min_tokens=100)
        self.assertTrue(0.5 < beta < 1.0)

    def test_empty_and_degenerate(self):
        tokens, types = heaps_curve([])
        self.assertEqual(len(tokens), 0)
        self.assertTrue(np.isnan(heaps_beta(tokens, types)))


if __name__ == "__main__":
    unittest.main()
