# -*- coding: utf-8 -*-
"""near_duplicate_analysis 纯函数单元测试（不触碰研究数据文件）。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from near_duplicate_analysis import (MIN_SHINGLES, bottom_k,
                                     candidate_pair_keys,
                                     candidate_pair_keys_flat,
                                     connected_components, decile_rates,
                                     jaccard_sorted, recall_check,
                                     select_verify_pairs, shingle_hashes,
                                     verified_pairs, within_max_j)


def long_text(seed: int, words: int = 60) -> str:
    return " ".join(f"w{seed}_{i}" for i in range(words))


class TestShingleHashes(unittest.TestCase):
    def test_deterministic_and_sorted_unique(self):
        h = shingle_hashes("同一个句子反复出现。同一个句子反复出现。")
        self.assertGreater(h.size, 0)
        self.assertTrue(np.all(np.diff(h) > 0))  # 严格递增 = 已排序去重
        np.testing.assert_array_equal(h, shingle_hashes(
            "同一个句子反复出现。同一个句子反复出现。"))

    def test_short_text_empty(self):
        self.assertEqual(shingle_hashes("啊").size, 0)
        self.assertEqual(shingle_hashes("").size, 0)

    def test_single_char_change_affects_hashes(self):
        a = shingle_hashes(long_text(1))
        b = shingle_hashes(long_text(1) + "x")
        self.assertFalse(np.array_equal(a, b))
        self.assertGreater(np.intersect1d(a, b).size, 0)


class TestBottomK(unittest.TestCase):
    def test_slice_and_passthrough(self):
        h = np.arange(10, dtype=np.uint64)
        np.testing.assert_array_equal(bottom_k(h, k=4), np.arange(4))
        np.testing.assert_array_equal(bottom_k(h, k=100), h)


class TestJaccard(unittest.TestCase):
    def test_basic_cases(self):
        a = np.array([1, 2, 3], dtype=np.uint64)
        self.assertEqual(jaccard_sorted(a, a.copy()), 1.0)
        self.assertEqual(jaccard_sorted(a, np.array([9], dtype=np.uint64)), 0.0)
        self.assertAlmostEqual(
            jaccard_sorted(a, np.array([2, 3, 4, 5], dtype=np.uint64)),
            2 / 5)

    def test_empty_returns_zero(self):
        self.assertEqual(jaccard_sorted(
            np.empty(0, dtype=np.uint64), np.array([1], dtype=np.uint64)), 0.0)


class TestCandidateGeneration(unittest.TestCase):
    def _sketch(self, hashes):
        arr = np.zeros(4, dtype=np.uint64)
        arr[:len(hashes)] = hashes
        return arr

    def test_shared_hash_yields_one_pair(self):
        s0 = self._sketch([10, 20])
        s1 = self._sketch([10, 30])
        s2 = self._sketch([99, 98])
        keys, counts = candidate_pair_keys(np.stack([s0, s1, s2]), 3)
        self.assertEqual(keys.tolist(), [0 * 3 + 1])
        self.assertEqual(counts.tolist(), [1])

    def test_too_common_hash_is_dropped(self):
        docs = [self._sketch([7, 40 + i]) for i in range(3)]
        keys, counts = candidate_pair_keys(np.stack(docs), 3, cap=2)
        self.assertEqual(keys.size, 0)

    def test_zero_slots_do_not_link(self):
        s0 = self._sketch([1])
        s1 = self._sketch([2])
        keys, _ = candidate_pair_keys(np.stack([s0, s1]), 2)
        self.assertEqual(keys.size, 0)

    def test_flat_entry_matches_matrix_entry(self):
        s0 = self._sketch([10, 20])
        s1 = self._sketch([10, 30])
        s2 = self._sketch([99, 98])
        mat = np.stack([s0, s1, s2])
        want_keys, want_counts = candidate_pair_keys(mat, 3)
        flat = mat.ravel()
        nz = np.flatnonzero(flat)
        got_keys, got_counts = candidate_pair_keys_flat(
            flat[nz], (nz // mat.shape[1]).astype(np.int32), 3)
        self.assertEqual(got_keys.tolist(), want_keys.tolist())
        self.assertEqual(got_counts.tolist(), want_counts.tolist())

    def test_zero_hash_filtered_in_flat_entry(self):
        hv = np.array([0, 5, 5], dtype=np.uint64)
        ids = np.array([0, 1, 2], dtype=np.int32)
        keys, counts = candidate_pair_keys_flat(hv, ids, 3)
        self.assertEqual(keys.tolist(), [1 * 3 + 2])
        self.assertEqual(counts.tolist(), [1])


class TestVerifyFilter(unittest.TestCase):
    def test_keeps_multi_key_and_short_docs(self):
        keys = np.array([0 * 3 + 1, 0 * 3 + 2, 1 * 3 + 2], dtype=np.int64)
        counts = np.array([1, 1, 2], dtype=np.int64)
        lengths = np.array([200, 200, 5], dtype=np.int64)
        kept = select_verify_pairs(keys, counts, lengths, 3, k=128)
        self.assertEqual(kept.tolist(), [2, 5])  # 单键长-长对(0,1)丢弃


class TestVerifiedPairs(unittest.TestCase):
    def test_identical_docs_found(self):
        body = long_text(7)
        sets = [shingle_hashes(body), shingle_hashes(body),
                shingle_hashes(long_text(8))]
        sketches = np.zeros((3, 128), dtype=np.uint64)
        for i, s in enumerate(sets):
            sketches[i, :min(s.size, 128)] = bottom_k(s)
        keys, counts = candidate_pair_keys(sketches, 3)
        pairs = verified_pairs(keys, counts, sets)
        self.assertEqual([(round(j, 3), a, b) for j, a, b in pairs],
                         [(1.0, 0, 1)])


class TestWithinMaxJ(unittest.TestCase):
    def test_cross_dataset_pairs_ignored(self):
        ds = np.array(["human", "aigc", "human"])
        pairs = [(0.9, 0, 1), (0.8, 0, 2)]
        mj = within_max_j(ds, pairs, 3)
        np.testing.assert_allclose(mj, [0.8, 0.0, 0.8])


class TestConnectedComponents(unittest.TestCase):
    def test_merges_and_sorts_by_size(self):
        comps = connected_components(6, [(0, 1), (1, 2), (3, 4)])
        self.assertEqual(comps[0], [0, 1, 2])
        self.assertEqual(comps[1], [3, 4])

    def test_singletons_dropped(self):
        self.assertEqual(connected_components(3, []), [])


class TestDecileRates(unittest.TestCase):
    def test_constant_values_have_no_deciles(self):
        dec = decile_rates(np.ones(50), np.zeros(50, dtype=bool))
        self.assertEqual(len(dec), 0)
        self.assertIn("share_pct", dec.columns)

    def test_monotone_indicator(self):
        vals = np.arange(100, dtype=float)
        ind = vals >= 50
        dec = decile_rates(vals, ind)
        self.assertEqual(len(dec), 10)
        self.assertEqual(dec.iloc[0]["share_pct"], 0.0)
        self.assertEqual(dec.iloc[-1]["share_pct"], 100.0)


class TestRecallCheck(unittest.TestCase):
    def test_full_sample_recall(self):
        sets = [shingle_hashes(long_text(1)), shingle_hashes(long_text(1)),
                shingle_hashes(long_text(2))]
        pairs = [(1.0, 0, 1)]
        out = recall_check(sets, pairs, n_sample=3, seed=0, min_j=0.5)
        self.assertEqual(out["true_pairs"], 1)
        self.assertEqual(out["found_pairs"], 1)
        self.assertEqual(out["recall_pct"], 100.0)

    def test_short_docs_are_excluded(self):
        short = shingle_hashes("#  #  #")
        self.assertLess(short.size, MIN_SHINGLES)
        out = recall_check([short, short], [], n_sample=2, seed=0)
        self.assertEqual(out["true_pairs"], 0)

    def test_missing_pairs_lower_recall(self):
        sets = [shingle_hashes(long_text(1)), shingle_hashes(long_text(1))]
        out = recall_check(sets, [], n_sample=2, seed=0, min_j=0.5)
        self.assertEqual(out["true_pairs"], 1)
        self.assertEqual(out["found_pairs"], 0)
        self.assertEqual(out["recall_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
