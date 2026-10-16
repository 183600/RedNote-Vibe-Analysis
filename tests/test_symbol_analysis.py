# -*- coding: utf-8 -*-
"""symbol_analysis 纯函数单元测试（不触碰研究数据文件）。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from symbol_analysis import (BIN_LABELS, bin_share, clean_emojis,
                             cliff_share_test, decile_gradient, emoji_structure,
                             note_tag_counts, tag_ai_ness, tag_engagement,
                             tag_structure)


class TestEmojiClean(unittest.TestCase):
    def test_drops_variation_selectors_and_zwj(self):
        self.assertEqual(clean_emojis(["✨", "\ufe0f", "✅", "\ufe0e", "❤",
                                       "\u200d"]),
                         ["✨", "✅", "❤"])
        self.assertEqual(clean_emojis([]), [])


class TestBinShare(unittest.TestCase):
    def test_each_bin_once(self):
        counts = np.array([0, 1, 2, 3, 5, 10], dtype=np.int64)
        shares = bin_share(counts)
        for lab in BIN_LABELS:
            self.assertAlmostEqual(shares[lab], 100.0 / 6, places=6)

    def test_edges_are_inclusive_lower(self):
        counts = np.array([4, 9, 11], dtype=np.int64)
        shares = bin_share(counts)
        self.assertAlmostEqual(shares["3–4"], 100.0 / 3, places=6)
        self.assertAlmostEqual(shares["5–9"], 100.0 / 3, places=6)
        self.assertAlmostEqual(shares["10+"], 100.0 / 3, places=6)


class TestNoteTagCounts(unittest.TestCase):
    def test_dedups_within_note(self):
        counts = note_tag_counts([["a", "a", "b"], ["b"], []])
        self.assertEqual(counts, {"a": 1, "b": 2})


class TestTagStructure(unittest.TestCase):
    def test_basic_metrics(self):
        notes = [["a", "b"], ["a"], [], ["c"]]
        s = tag_structure(notes)
        self.assertEqual(s["n_notes"], 4)
        self.assertEqual(s["n_tagged"], 3)
        self.assertAlmostEqual(s["tagged_share"], 0.75)
        self.assertEqual(s["tags_total"], 4)
        self.assertAlmostEqual(s["tags_per_note_mean"], 1.0)
        self.assertEqual(s["unique_tags"], 3)
        self.assertAlmostEqual(s["unique_tag_ratio"], 0.75)
        self.assertAlmostEqual(s["singleton_tag_share"], 2 / 3)
        self.assertIn("bins_pct", s)

    def test_empty_input(self):
        s = tag_structure([])
        self.assertEqual(s["n_notes"], 0)
        self.assertEqual(s["tagged_share"], 0.0)
        self.assertEqual(s["unique_tags"], 0)


class TestEmojiStructure(unittest.TestCase):
    def test_rates_and_top(self):
        s = emoji_structure([["✨", "✨"], ["✅"], []], [[], ["[赞R]"], []])
        self.assertAlmostEqual(s["emoji_share"], 2 / 3)
        self.assertAlmostEqual(s["emoji_per_note_mean"], 1.0)
        self.assertEqual(s["emoji_distinct"], 2)
        self.assertAlmostEqual(s["xhs_share"], 1 / 3)
        self.assertEqual(s["top_emoji"][0][0], "✨")
        self.assertEqual(s["top_emoji"][0][1], 2)
        self.assertAlmostEqual(s["top_emoji"][0][2], 666.67, places=2)


class TestCliffShareTest(unittest.TestCase):
    def test_all_true_returns_neutral(self):
        values = np.array([0.1, 0.2, 0.3])
        out = cliff_share_test(np.array([True, True, True]), values)
        self.assertEqual(out["delta"], 0.0)
        self.assertEqual(out["p"], 1.0)

    def test_clear_separation(self):
        values = np.array([0.1] * 8 + [0.9] * 8)
        flag = np.array([False] * 8 + [True] * 8)
        out = cliff_share_test(flag, values)
        self.assertAlmostEqual(out["delta"], 1.0)
        self.assertLess(out["p"], 0.05)

    def test_fisher_needs_enough_observations(self):
        values = np.array([0.1, 0.2, 0.8, 0.9])
        out = cliff_share_test(np.array([False, False, True, True]), values)
        self.assertAlmostEqual(out["delta"], 1.0)
        self.assertGreaterEqual(out["p"], 0.05)


class TestDecileGradient(unittest.TestCase):
    def test_shapes_and_hits(self):
        p = np.linspace(0, 1, 200)
        notes = [["ai_tag"] if i % 2 == 0 else ["hum_tag"] for i in range(200)]
        lens = np.full(200, 100)
        g = decile_gradient(p, notes, {"ai_tag"}, {"hum_tag"}, {"ai_tag"}, lens)
        self.assertEqual(g["n"], 200)
        self.assertEqual(len(g["deciles"]), 10)
        self.assertEqual(sum(d["n"] for d in g["deciles"]), 200)
        total_ai = sum(d["hit_aigc_tag"] * d["n"] for d in g["deciles"])
        self.assertAlmostEqual(total_ai, 100, places=0)
        self.assertIn("spearman_tags_nonpure", g)
        self.assertIn("within_length_rho", g)


class TestTagAiNess(unittest.TestCase):
    def test_month_residual_and_threshold(self):
        p = np.array([0.2, 0.4, 0.8, 1.0])
        months = np.array(["202401", "202401", "202402", "202402"])
        tags = [["a"], ["b"], ["a", "b"], ["a", "b"]]
        out = tag_ai_ness(p, months, tags, min_notes=3)
        rows = {r["tag"]: r for r in out["top_aigc"] + out["top_human"]}
        self.assertEqual(rows["a"]["n"], 3)
        self.assertEqual(rows["b"]["n"], 3)
        self.assertAlmostEqual(rows["a"]["mean_p"], (0.2 + 0.8 + 1.0) / 3,
                               places=3)
        self.assertEqual(out["min_notes"], 3)

    def test_min_notes_filters(self):
        p = np.array([0.5])
        out = tag_ai_ness(p, np.array(["202401"]), [["rare"]], min_notes=5)
        self.assertEqual(out["n_tags_reported"], 0)


class TestTagEngagement(unittest.TestCase):
    def test_bins_and_cells(self):
        n = 400
        tags_n = np.where(np.arange(n) % 5 == 0, 10, 0)
        likes = np.where(np.arange(n) % 5 == 0, 100, 0).astype(np.int64)
        colls = np.zeros(n, dtype=np.int64)
        months = np.array(["202401"] * n)
        domains = np.array(["心理"] * n)
        out = tag_engagement(tags_n, likes, colls, months, domains)
        rows = {b["bin"]: b for b in out["bins"]}
        self.assertEqual(rows["0"]["n"], 320)
        self.assertEqual(rows["10+"]["n"], 80)
        self.assertAlmostEqual(rows["10+"]["likes_median"], 100.0)
        self.assertAlmostEqual(rows["10+"]["zero_like_share"], 0.0)
        self.assertAlmostEqual(rows["0"]["zero_like_share"], 1.0)
        self.assertEqual(out["within_cell"]["n_cells"], 1)


if __name__ == "__main__":
    unittest.main()
