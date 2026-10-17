# -*- coding: utf-8 -*-
"""深潜分析之九：话题标签与表情符号的「符号生态」。

既有分析把标签与表情压缩成 `paper_features.py` 里的三个计数字段
（n_hashtags / emoji100 / xhsemo100），只作为检测器特征使用，没有回答：
「AI 味」在符号层面长什么样？哪些话题标签与表情是 AI 的指纹？
符号用量是否也随 P(AI) 上升、并在后 LLM 时代扩散？

四个维度（全量扫描，口径与论文管线一致）：

1. 跨子集标签结构：每帖标签数分布、标签词表规模、集中度（Top-20 覆盖、
   Miller–Madow 校正熵）与标签长度。
2. 标签词汇差异：AIGC 专属 / 人类专属标签（每帖发生率之差）、AIGC 高频标签，
   以及探索集内标签级「AI 味」（每个标签的帖子平均 P(AI) 与其月份残差）。
3. 探索集梯度与扩散：每帖标签数、AI 专属标签命中率在 P(AI) 十分位上的梯度，
   季度扩散轨迹与人类干净窗口（≤2022.06）/ AIGC 生成集基线。
4. 表情符号生态：Unicode emoji 与小红书 [xxR] 表情的使用率、多样度与
   Top 表情跨子集对比、探索集 P(AI) 梯度；另附标签数与互动的关系
   （探索集 2023.01 起，月份×领域单元内秩相关）。

口径说明：
- 正文清洗复用 `analysis_utils.clean_body`（与 paper_features.py 一致）；
- 标签提取复用 `analysis_utils.extract_hashtags`（兼容 `#标签[话题]#` 格式、
  BOM/零宽字符残留与 `##` 双重井号）；
- 表情计数正则与 `paper_features.py` 相同（EMOJI / XHS_EMO），在标签剥离后的
  正文上计数，并额外剔除变体选择符（U+FE0F/U+FE0E）与 ZWJ（它们在原正则里
  会与基字符重复计数）；
- 人类/探索集按 note_id 去重（保留元数据最完整的一条，规则同 deep_dive），
  AIGC 无 note_id 逐记录处理；
- 探索集 p_ai 按行序对齐 `paper_exploring_scores.csv.gz`，用 content_len
  逐条校验对齐（不一致时在结果中记录 mismatch 数）。

输出：
  insights/symbol_hashtag.png   标签结构 / 词表集中度 / 专属标签 / 标签级 AI 味
  insights/symbol_gradient.png  标签梯度 / 季度扩散 / 标签数×互动
  insights/symbol_emoji.png     表情使用率 / Top 表情差异 / P(AI) 梯度
  symbol_results.json           全部数值结果
  symbol_results.md             结果摘要
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from analysis_utils import (BASE, INSIGHTS, PALETTE, init_plot_style, save_fig,
                            cliffs_delta, clean_body, content_of,
                            engagement_of, extract_hashtags, iter_jsonl,
                            parse_local_time, quarter_of, rate_diff_ranking,
                            rate_top, save_json, shannon_entropy, topk_coverage,
                            wilson_ci, within_cell_spearman)
from paper_features import EMOJI, XHS_EMO

DATASETS = {
    "human": BASE + "training_set_human_fixed.jsonl",
    "aigc": BASE + "training_set_aigc.jsonl",
    "exploring": BASE + "exploring_set_fixed.jsonl",
}
LABELS = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
          "exploring": "探索集(后LLM)"}
SCORES_PATH = BASE + "paper_exploring_scores.csv.gz"

BIN_EDGES = (1, 2, 3, 5, 10)
BIN_LABELS = ("0", "1", "2", "3–4", "5–9", "10+")
TAG_TOPK = 20
TAG_SPEC_K = 12          # 专属标签展示条数
MIN_A_COUNT = 15         # AI 专属标签的 AIGC 最小出现篇数
MIN_H_COUNT = 50         # 人类专属标签的人类最小出现篇数
TAG_AI_MIN_NOTES = 100   # 标签级 AI 味的最小出现篇数
MIN_QUARTER_N = 300      # 季度趋势最少样本
MIN_CELL_N = 100         # 月份×领域单元最少样本
N_DECILES = 10
MIN_BODY_LEN = 8         # 清洗后正文短于此长度视为「纯标签/空正文」帖
ENG_MONTH_FROM = "202301"
# 变体选择符（FE0E/FE0F）与零宽连接符（ZWJ）在 paper_features 的 EMOJI
# 类里会随基字符一起被单独匹配，导致同一个 emoji 计两次；符号分析剔除。
_EMOJI_SKIP = frozenset("\ufe0e\ufe0f\u200d")


# --------------------------------------------------------------------------
# 扫描与结构统计
# --------------------------------------------------------------------------
def clean_emojis(chars: list[str]) -> list[str]:
    """剔除变体选择符（U+FE0F/FE0E）与 ZWJ；paper_features 的 EMOJI 正则会把
    它们与基字符一起单独匹配（如 `✨️` 计成两个字符），符号分析需去重。"""
    return [c for c in chars if c not in _EMOJI_SKIP]


def scan(path: str, content_len_ref: np.ndarray | None = None) -> dict:
    """单次流式扫描：逐记录收集标签、表情、时间与互动，并即时按 note_id 去重。

    去重规则与 `dedup_indices` 完全一致（同名笔记保留 meta 分数最高的一条、
    并列取最早一条；无 note_id 的记录全部保留；最终按原始行序输出），只是把
    去重提前到扫描过程中：被更完整记录覆盖的标签/表情列表即时释放，不再为
    全部原始记录驻留内存。返回值中的逐帖列表/数组均按去重后的顺序给出。
    """
    best: dict = {}
    rows: list = []
    n_raw = mismatch = 0
    month_pool: dict = {}
    for rec in iter_jsonl(path):
        content = content_of(rec)
        body = clean_body(content)
        if content_len_ref is not None and len(body) != int(content_len_ref[n_raw]):
            mismatch += 1
        lk, cl, cm = engagement_of(rec)
        dt = parse_local_time(rec.get("local_time"))
        has_meta = (bool(rec.get("user_id"))
                    and rec.get("domain") not in (None, "", "未知"))
        meta = (int(has_meta) + int(bool(lk)) + int(bool(cl)) + int(bool(cm))
                + int(dt is not None)
                + int(bool(content) and not content.isspace()))
        if dt is None:
            month = ""
        else:
            ym = (dt.year, dt.month)
            month = month_pool.get(ym)
            if month is None:
                month = month_pool[ym] = f"{dt.year:04d}{dt.month:02d}"
        nid = rec.get("note_id")
        if nid is None:
            rows.append((n_raw, extract_hashtags(content),
                         clean_emojis(EMOJI.findall(body)), XHS_EMO.findall(body),
                         month, meta, len(body), lk, cl, cm))
        else:
            cur = best.get(nid)
            # 仅当该 note_id 首次出现或被更完整的记录覆盖时才做标签/表情抽取，
            # 探索集一半以上原始记录会被去重丢弃，这里省掉它们的无用抽取。
            if cur is None or meta > cur[5]:
                best[nid] = (n_raw, extract_hashtags(content),
                             clean_emojis(EMOJI.findall(body)), XHS_EMO.findall(body),
                             month, meta, len(body), lk, cl, cm)
        n_raw += 1
    rows.extend(best.values())
    rows.sort(key=lambda r: r[0])
    return {
        "path": path, "n_raw": n_raw, "mismatch": mismatch,
        "kept_indices": np.asarray([r[0] for r in rows], dtype=np.int64),
        "tags": [r[1] for r in rows],
        "emojis": [r[2] for r in rows],
        "xhs": [r[3] for r in rows],
        "months": [r[4] for r in rows],
        "content_lens": np.asarray([r[6] for r in rows], dtype=np.int64),
        "likes": np.asarray([r[7] for r in rows], dtype=np.int64),
        "colls": np.asarray([r[8] for r in rows], dtype=np.int64),
        "comms": np.asarray([r[9] for r in rows], dtype=np.int64),
    }


def note_tag_counts(note_tags: list[list[str]]) -> Counter:
    """每篇最多计一次的标签 Counter（同帖内重复标签只计一次）。"""
    counts: Counter = Counter()
    for ts in note_tags:
        counts.update(set(ts))
    return counts


def bin_share(counts: np.ndarray) -> dict:
    """每帖标签数落入 0 / 1 / 2 / 3–4 / 5–9 / 10+ 的篇数占比（%）。"""
    idx = np.searchsorted(np.asarray(BIN_EDGES), counts, side="right")
    total = counts.size
    if total == 0:
        return {lab: 0.0 for lab in BIN_LABELS}
    return {lab: float((idx == i).sum() / total * 100)
            for i, lab in enumerate(BIN_LABELS)}


def tag_structure(note_tags: list[list[str]],
                  tag_counts: Counter | None = None) -> dict:
    """一个子集的标签结构指标（输入为去重后的逐笔记标签列表）。

    `tag_counts` 可传入已算好的每篇计一次标签 Counter，避免调用方重复统计。
    """
    counts = np.asarray([len(t) for t in note_tags], dtype=np.int64)
    if tag_counts is None:
        tag_counts = note_tag_counts(note_tags)
    n_notes = counts.size
    tagged = int((counts > 0).sum())
    total = int(counts.sum())
    uniq = len(tag_counts)
    singles = sum(1 for c in tag_counts.values() if c == 1)
    len_sum = sum(len(t) * c for t, c in tag_counts.items())
    return {
        "n_notes": int(n_notes),
        "n_tagged": tagged,
        "tagged_share": tagged / n_notes if n_notes else 0.0,
        "tags_total": total,
        "tags_per_note_mean": float(counts.mean()) if n_notes else 0.0,
        "tags_per_note_median": float(np.median(counts)) if n_notes else 0.0,
        "tags_per_note_p90": float(np.percentile(counts, 90)) if n_notes else 0.0,
        "tags_per_tagged_mean": total / tagged if tagged else 0.0,
        "unique_tags": uniq,
        "unique_tag_ratio": uniq / total if total else 0.0,
        "singleton_tag_share": singles / uniq if uniq else 0.0,
        "top20_coverage": topk_coverage(tag_counts, TAG_TOPK),
        "entropy_mm": shannon_entropy(tag_counts, miller_madow=True),
        "tag_len_mean": len_sum / total if total else 0.0,
        "bins_pct": bin_share(counts),
    }


def emoji_structure(note_emoji: list[list[str]],
                    note_xhs: list[list[str]],
                    vocab: Counter | None = None,
                    xhs_vocab: Counter | None = None) -> dict:
    """一个子集的 Unicode / 小红书表情结构指标（输入为去重后的逐笔记列表）。

    `vocab` / `xhs_vocab` 可传入已算好的全量词频 Counter（调用方后续复用），
    避免同一份列表被多次展开统计。
    """
    uni = np.asarray([len(e) for e in note_emoji], dtype=np.int64)
    xhs = np.asarray([len(e) for e in note_xhs], dtype=np.int64)
    if vocab is None:
        vocab = Counter(e for lst in note_emoji for e in lst)
    if xhs_vocab is None:
        xhs_vocab = Counter(e for lst in note_xhs for e in lst)
    n = uni.size
    return {
        "n_notes": int(n),
        "emoji_share": float((uni > 0).mean()) if n else 0.0,
        "emoji_per_note_mean": float(uni.mean()) if n else 0.0,
        "emoji_per_note_median": float(np.median(uni)) if n else 0.0,
        "emoji_distinct": len(vocab),
        "xhs_share": float((xhs > 0).mean()) if n else 0.0,
        "xhs_per_note_mean": float(xhs.mean()) if n else 0.0,
        "xhs_distinct": len(xhs_vocab),
        "top_emoji": [[e, int(c), round(c / n * 1000, 2)]
                      for e, c in vocab.most_common(15)] if n else [],
        "top_xhs": [[e, int(c), round(c / n * 1000, 2)]
                    for e, c in xhs_vocab.most_common(15)] if n else [],
    }


# --------------------------------------------------------------------------
# 梯度 / 扩散 / 标签级 AI 味 / 互动
# --------------------------------------------------------------------------
def cliff_share_test(indicator: np.ndarray, values: np.ndarray) -> dict:
    """命中/未命中两组在 values 上的 Cliff's δ 与 Mann–Whitney p。"""
    if indicator.any() and (~indicator).any():
        delta, p = cliffs_delta(values[indicator], values[~indicator])
    else:
        delta, p = 0.0, 1.0
    return {"delta": round(float(delta), 3), "p": float(p)}


def decile_gradient(p: np.ndarray, note_tags: list[list[str]],
                    ai_set: set, h_set: set, top_set: set,
                    body_lens: np.ndarray, *,
                    tag_sets: list[set] | None = None,
                    tags_n: np.ndarray | None = None,
                    decile: np.ndarray | None = None,
                    hit_a: np.ndarray | None = None,
                    hit_h: np.ndarray | None = None,
                    hit_top: np.ndarray | None = None) -> dict:
    """P(AI) 十分位上的标签用量与专属标签命中率梯度。

    `body_lens` 为清洗后正文长度：用于报告各十分位「纯标签/空正文」帖占比，
    并给出剔除这类帖子后的稳健性检验（它们会同时拉低正文长度与 P(AI)）。
    `tag_sets` / `tags_n` / `decile` / `hit_*` 可传入调用方已算好的等价中间量，
    避免重复。
    """
    n = p.size
    if decile is None:
        order = np.argsort(p, kind="stable")
        decile = np.empty(n, dtype=np.int64)
        decile[order] = np.minimum(np.arange(n) * N_DECILES // n, N_DECILES - 1)
    if tags_n is None:
        tags_n = np.asarray([len(t) for t in note_tags], dtype=np.int64)
    if tag_sets is None:
        tag_sets = [set(t) for t in note_tags]
    if hit_a is None:
        hit_a = np.asarray([bool(s & ai_set) for s in tag_sets])
    if hit_h is None:
        hit_h = np.asarray([bool(s & h_set) for s in tag_sets])
    if hit_top is None:
        hit_top = np.asarray([bool(s & top_set) for s in tag_sets])
    short = body_lens < MIN_BODY_LEN
    rows = []
    for d in range(N_DECILES):
        m = decile == d
        n_m = int(m.sum())
        la, ha = wilson_ci(int(hit_a[m].sum()), n_m)
        lh, hh = wilson_ci(int(hit_h[m].sum()), n_m)
        lt, ht = wilson_ci(int(hit_top[m].sum()), n_m)
        rows.append({
            "decile": d + 1, "n": n_m,
            "mean_p": round(float(p[m].mean()), 4),
            "tags_mean": round(float(tags_n[m].mean()), 3),
            "tagged_share": round(float((tags_n[m] > 0).mean()), 4),
            "ge5_share": round(float((tags_n[m] >= 5).mean()), 4),
            "short_body_share": round(float(short[m].mean()), 4),
            "hit_aigc_tag": round(float(hit_a[m].mean()), 4),
            "hit_aigc_tag_ci": [round(float(la), 4), round(float(ha), 4)],
            "hit_human_tag": round(float(hit_h[m].mean()), 4),
            "hit_human_tag_ci": [round(float(lh), 4), round(float(hh), 4)],
            "hit_top_tag": round(float(hit_top[m].mean()), 4),
            "hit_top_tag_ci": [round(float(lt), 4), round(float(ht), 4)],
        })
    rho, pval = spearmanr(p, tags_n)
    rho_tagged, p_tagged = spearmanr(p, (tags_n > 0).astype(float))
    keep = ~short
    rho_np, p_np = spearmanr(p[keep], tags_n[keep])
    # 控制正文长度：按长度十分位分层后的加权单元内 ρ（标签数 vs P(AI)）
    len_bins = np.quantile(body_lens, np.linspace(0, 1, N_DECILES + 1)[1:-1])
    lb = np.searchsorted(len_bins, body_lens, side="right")
    rhos, ns = [], []
    for d in range(N_DECILES):
        m = lb == d
        if int(m.sum()) < MIN_CELL_N or np.unique(tags_n[m]).size < 2:
            continue
        r, _ = spearmanr(p[m], tags_n[m])
        if not np.isnan(r):
            rhos.append(float(r))
            ns.append(int(m.sum()))
    return {
        "n": n,
        "spearman_tags": {"rho": round(float(rho), 3), "p": float(pval)},
        "spearman_tagged": {"rho": round(float(rho_tagged), 3),
                            "p": float(p_tagged)},
        "spearman_tags_nonpure": {"rho": round(float(rho_np), 3),
                                  "p": float(p_np),
                                  "n": int(keep.sum())},
        "within_length_rho": {
            "weighted_rho": round(float(np.average(rhos, weights=ns)), 4)
            if rhos else float("nan"),
            "n_strata": len(rhos)},
        "cliff_aigc_tag": cliff_share_test(hit_a, p),
        "cliff_human_tag": cliff_share_test(hit_h, p),
        "cliff_top_tag": cliff_share_test(hit_top, p),
        "deciles": rows,
    }


def quarterly_diffusion(months: np.ndarray, hit_a: np.ndarray,
                        hit_top: np.ndarray, hit_h: np.ndarray) -> dict:
    """探索集内 AI/人类专属标签命中率的季度轨迹与趋势。"""
    quarters = np.asarray([quarter_of(m) for m in months], dtype=object)
    out: dict = {}
    for q in sorted({x for x in quarters if x is not None}):
        m = quarters == q
        n_q = int(m.sum())
        if n_q < MIN_QUARTER_N:
            continue
        ka = int(hit_a[m].sum())
        la, ha = wilson_ci(ka, n_q)
        out[q] = {
            "n": n_q,
            "share_aigc_tag": round(ka / n_q, 4),
            "ci_aigc_tag": [round(float(la), 4), round(float(ha), 4)],
            "share_top_tag": round(float(hit_top[m].mean()), 4),
            "share_human_tag": round(float(hit_h[m].mean()), 4),
        }

    def trend(key: str) -> dict:
        qs = sorted(out)
        if len(qs) < 3:
            return {}
        rho, pval = spearmanr(range(len(qs)), [out[q][key] for q in qs])
        return {"rho": round(float(rho), 3), "p": float(pval)}

    return {"quarters": out, "trend_aigc_tag": trend("share_aigc_tag"),
            "trend_top_tag": trend("share_top_tag"),
            "trend_human_tag": trend("share_human_tag")}


def tag_ai_ness(p: np.ndarray, months: np.ndarray, note_tags: list[list[str]],
                min_notes: int = TAG_AI_MIN_NOTES) -> dict:
    """标签级「AI 味」：每个标签的帖子平均 P(AI)，并扣除月份均值得到残差。

    月份残差用于抑制发布时间构成差异（如某标签集中在 AI 高发月份）。
    仅报告出现篇数 >= min_notes 的标签。
    """
    month_sum: dict = {}
    month_n: Counter = Counter()
    for pv, m in zip(p, months):
        if m:
            month_sum[m] = month_sum.get(m, 0.0) + pv
            month_n[m] += 1
    month_mean = {m: month_sum[m] / month_n[m] for m in month_sum}
    overall = float(p.mean())
    tag_sum: dict = {}
    tag_resid: dict = {}
    tag_n: Counter = Counter()
    for pv, m, ts in zip(p, months, note_tags):
        base = month_mean.get(m, overall)
        for t in set(ts):
            tag_sum[t] = tag_sum.get(t, 0.0) + pv
            tag_resid[t] = tag_resid.get(t, 0.0) + (pv - base)
            tag_n[t] += 1
    rows = [{"tag": t, "n": int(c), "mean_p": round(tag_sum[t] / c, 4),
             "mean_resid": round(tag_resid[t] / c, 4)}
            for t, c in tag_n.items() if c >= min_notes]
    rows.sort(key=lambda r: -r["mean_resid"])
    return {
        "min_notes": min_notes, "overall_mean_p": round(overall, 4),
        "n_tags_reported": len(rows),
        "top_aigc": rows[:15], "top_human": rows[::-1][:15],
    }


def tag_engagement(tags_n: np.ndarray, likes: np.ndarray, colls: np.ndarray,
                   months: np.ndarray, domains: np.ndarray) -> dict:
    """探索集（2023.01 起）标签数与互动的关系：分箱中位数 + 单元内秩相关。"""
    mask = np.asarray([m >= ENG_MONTH_FROM for m in months])
    tn, lk, cl = tags_n[mask], likes[mask], colls[mask]
    mo, dm = months[mask], domains[mask]
    bins = np.searchsorted(np.asarray(BIN_EDGES), tn, side="right")
    rows = []
    for i, lab in enumerate(BIN_LABELS):
        m = bins == i
        n_b = int(m.sum())
        if not n_b:
            continue
        zero = int((lk[m] == 0).sum())
        lo, hi = wilson_ci(zero, n_b)
        rows.append({
            "bin": lab, "n": n_b,
            "likes_median": float(np.median(lk[m])),
            "zero_like_share": zero / n_b,
            "zero_like_ci": [round(float(lo), 4), round(float(hi), 4)],
            "coll_per_like": float(np.median(cl[m] / (lk[m] + 1))),
        })
    log_likes = np.log10(1.0 + lk)
    # 单元内秩相关统一走 analysis_utils.within_cell_spearman：一次稳定排序
    # 后按单元切片，避免原先「每个单元都对全量数组做两次布尔比较」的
    # O(单元数 × 样本数) 扫描；口径（≥MIN_CELL_N、常量单元跳过）完全一致。
    cell_codes = np.asarray([f"{a}\x00{b}" for a, b in zip(mo, dm)],
                            dtype=object)
    stat = within_cell_spearman(tn, log_likes, cell_codes,
                                min_cell_n=MIN_CELL_N)
    cells = {"weighted_rho": round(float(stat["weighted_rho"]), 4),
             "neg_cell_share": round(float(stat["neg_cell_share"]), 4),
             "n_cells": stat["n_cells"]}
    return {"bins": rows, "within_cell": cells}


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_hashtag(structs: dict, specific: dict, ai_ness: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    keys = ("human", "aigc", "exploring")
    x = np.arange(len(BIN_LABELS))
    w = 0.27

    ax = axes[0, 0]
    for j, k in enumerate(keys):
        vals = [structs[k]["bins_pct"][lab] for lab in BIN_LABELS]
        ax.bar(x + (j - 1) * w, vals, width=w, color=PALETTE[k],
               alpha=0.85, label=LABELS[k])
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("每帖话题标签数")
    ax.set_ylabel("帖子占比 (%)")
    ax.set_title("每帖标签数：随语料时代上升（探索集 92% 帖带标签）")
    ax.legend(fontsize=8.4)

    ax = axes[0, 1]
    xs = np.arange(len(keys))
    uniq = [structs[k]["unique_tag_ratio"] * 100 for k in keys]
    sing = [structs[k]["singleton_tag_share"] * 100 for k in keys]
    w2 = 0.36
    ax.bar(xs - w2 / 2, uniq, width=w2, color=[PALETTE[k] for k in keys],
           alpha=0.85, label="唯一标签/全部标签用量")
    ax.bar(xs + w2 / 2, sing, width=w2, color="#999999", alpha=0.7,
           label="只出现一次的标签/唯一标签")
    for i, (u, s) in enumerate(zip(uniq, sing)):
        ax.annotate(f"{u:.0f}%", (i - w2 / 2, u), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8.2)
        ax.annotate(f"{s:.0f}%", (i + w2 / 2, s), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[k] for k in keys])
    ax.set_ylabel("占比 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("标签词表：AIGC 生成大量「一次性」话题词")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    tops = specific["aigc"][:8][::-1]
    bots = specific["human"][:8]
    labels = [t["word"] for t in tops] + [t["word"] for t in bots]
    diffs = ([t["rate_diff_ppt"] for t in tops]
             + [-t["rate_diff_ppt"] for t in bots])
    colors = [PALETTE["aigc"]] * len(tops) + [PALETTE["human"]] * len(bots)
    ypos = np.arange(len(labels))
    ax.barh(ypos, diffs, color=colors, alpha=0.85, height=0.68)
    for y, d in zip(ypos, diffs):
        off = 4 if d >= 0 else -4
        ax.annotate(f"{d:+.1f}pp", (d, y), textcoords="offset points",
                    xytext=(off, 0), va="center", fontsize=7.6,
                    ha="left" if d >= 0 else "right")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.axvline(0, color="#666666", lw=0.8)
    ax.set_xlabel("每帖发生率之差（AIGC − 人类，百分点）")
    ax.set_title("专属话题标签：AI 侧（上） vs 人类侧（下）\n"
                 "（AIGC 侧多为生成器自造话题词，含时代/话题混杂）",
                 fontsize=10.5)
    ax.grid(axis="y", visible=False)

    ax = axes[1, 1]
    top8 = ai_ness["top_aigc"][:8][::-1]
    bot8 = ai_ness["top_human"][:8]
    labels = [f"#{t['tag']}" for t in top8] + [f"#{t['tag']}" for t in bot8]
    vals = [t["mean_resid"] for t in top8] + [t["mean_resid"] for t in bot8]
    colors = [PALETTE["aigc"]] * len(top8) + [PALETTE["human"]] * len(bot8)
    ypos = np.arange(len(labels))
    ax.barh(ypos, vals, color=colors, alpha=0.85, height=0.68)
    ax.axvline(0, color="#666666", lw=0.8)
    meta_top = {t["tag"]: t for t in top8 + bot8}
    for y, (lab, v) in enumerate(zip(labels, vals)):
        t = meta_top[lab[1:]]
        off = 4 if v >= 0 else -4
        ax.annotate(f"P̄={t['mean_p']:.2f} n={t['n']:,}", (v, y),
                    textcoords="offset points", xytext=(off, 0), va="center",
                    fontsize=7.2, ha="left" if v >= 0 else "right")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xlabel("P(AI) 的月份内残差均值（>0 = 比同月平均更像 AI）")
    ax.set_title(f"标签级 AI 味（探索集，出现 ≥{ai_ness['min_notes']} 篇）")
    ax.grid(axis="y", visible=False)

    fig.suptitle("话题标签生态：标签随时代普及，却不是高 P(AI) 帖的特征",
                 fontsize=13.5, y=1.0)
    save_fig(fig, INSIGHTS + "symbol_hashtag.png")


def plot_gradient(grad: dict, diff: dict, eng: dict,
                  baselines: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    dec = grad["deciles"]
    x = np.arange(len(dec))

    ax = axes[0, 0]
    tags_mean = [d["tags_mean"] for d in dec]
    tagged = [d["tagged_share"] * 100 for d in dec]
    ax.bar(x, tags_mean, color=PALETTE["aigc"], alpha=0.72, width=0.62)
    for i, v in enumerate(tags_mean):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7.6)
    ax.set_ylabel("平均每帖标签数", color=PALETTE["aigc"])
    ax.set_xlabel("AI 风格得分十分位（0=最像人 … 9=最像AI）")
    ax2 = ax.twinx()
    ax2.plot(x, tagged, "o-", color="#333333", ms=4, lw=1.4,
             label="有标签帖占比")
    ax2.set_ylabel("有标签帖占比 (%)")
    ax2.set_ylim(min(tagged) - 3, max(tagged) + 3)
    ax2.grid(False)
    ax2.legend(fontsize=8, loc="lower right")
    ax.set_title(f"标签用量在最高 P(AI) 端回落（ρ="
                 f"{grad['spearman_tags']['rho']:+.2f}）")
    npure = grad["spearman_tags_nonpure"]
    wl = grad["within_length_rho"]
    ax.text(0.02, 0.96,
            f"剔除纯标签帖后 ρ={npure['rho']:+.2f}（n={npure['n']:,}）\n"
            f"控制正文长度分层后加权 ρ={wl['weighted_rho']:+.2f}"
            f"（{wl['n_strata']} 层）",
            transform=ax.transAxes, fontsize=8.2, va="top")

    ax = axes[0, 1]
    for key, base, color, label in (
            ("hit_aigc_tag", "hit_aigc_tag_ci", PALETTE["aigc"], "AI 专属标签"),
            ("hit_top_tag", "hit_top_tag_ci", "#DD8452", "AIGC 高频标签 Top-50"),
            ("hit_human_tag", "hit_human_tag_ci", PALETTE["human"], "人类专属标签")):
        ys = np.array([d[key] * 100 for d in dec])
        lo = np.array([d[base][0] * 100 for d in dec])
        hi = np.array([d[base][1] * 100 for d in dec])
        ax.errorbar(x, ys, yerr=[ys - lo, hi - ys], marker="o", ms=4,
                    lw=1.4, capsize=2.6, color=color, label=label)
    ax.set_xlabel("AI 风格得分十分位")
    ax.set_ylabel("标签命中帖子占比 (%)")
    ax.set_title("专属标签命中率随 P(AI) 下降：\n标签不是稳定的 AI 风格信号",
                 fontsize=10.8)
    ax.legend(fontsize=8.2)
    ax.text(0.03, 0.95,
            f"AI 专属 δ={grad['cliff_aigc_tag']['delta']:+.2f}"
            f"（p={grad['cliff_aigc_tag']['p']:.0e}）\n"
            f"人类专属 δ={grad['cliff_human_tag']['delta']:+.2f}"
            f"（p={grad['cliff_human_tag']['p']:.0e}）",
            transform=ax.transAxes, fontsize=8.4, va="top")

    ax = axes[1, 0]
    qs = sorted(diff["quarters"])
    xs = np.arange(len(qs))
    ys = np.array([diff["quarters"][q]["share_aigc_tag"] * 100 for q in qs])
    lo = np.array([diff["quarters"][q]["ci_aigc_tag"][0] * 100 for q in qs])
    hi = np.array([diff["quarters"][q]["ci_aigc_tag"][1] * 100 for q in qs])
    top = np.array([diff["quarters"][q]["share_top_tag"] * 100 for q in qs])
    hum = np.array([diff["quarters"][q]["share_human_tag"] * 100 for q in qs])
    ax.plot(xs, top, "s--", color="#DD8452", lw=1.3, ms=3.8,
            label="AIGC 高频标签 Top-50")
    ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], marker="o", ms=4.2, lw=1.6,
                capsize=3, color=PALETTE["aigc"], label="AI 专属标签（探索集）")
    ax.plot(xs, hum, "^-", color=PALETTE["human"], lw=1.3, ms=3.8,
            label="人类专属标签（探索集）")
    ax.axhline(baselines["human_clean_aigc_tag"] * 100, color=PALETTE["human"],
               ls=":", lw=1.2, label="人类干净窗口基线(≤2022.06)")
    ax.axhline(baselines["aigc_aigc_tag"] * 100, color=PALETTE["aigc"],
               ls=":", lw=1.2, label="AIGC 生成集参考")
    ax.set_xticks(xs)
    ax.set_xticklabels(qs, rotation=45, fontsize=8)
    ax.set_ylabel("标签命中帖子占比 (%)")
    ax.set_ylim(0, max(list(top) + list(ys) + [baselines["aigc_aigc_tag"] * 100])
                * 1.25)
    tr = diff.get("trend_aigc_tag", {})
    ax.set_title(f"探索集 AI 标签的季度轨迹"
                 f"（ρ={tr.get('rho', float('nan')):+.2f}）")
    ax.legend(fontsize=7.8, loc="upper right")

    ax = axes[1, 1]
    bins = eng["bins"]
    bx = np.arange(len(bins))
    likes_med = [b["likes_median"] for b in bins]
    zero = [b["zero_like_share"] * 100 for b in bins]
    ax.bar(bx, likes_med, color=PALETTE["exploring"], alpha=0.75, width=0.58)
    for i, v in enumerate(likes_med):
        ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(bx)
    ax.set_xticklabels([b["bin"] for b in bins])
    ax.set_xlabel("每帖标签数（探索集 2023.01 起）")
    ax.set_ylabel("点赞中位数")
    ax2 = ax.twinx()
    lo = np.array([b["zero_like_ci"][0] for b in bins]) * 100
    hi = np.array([b["zero_like_ci"][1] for b in bins]) * 100
    ax2.errorbar(bx, zero, yerr=[np.array(zero) - lo, hi - np.array(zero)],
                 marker="o", ms=4, lw=1.3, capsize=3, color=PALETTE["aigc"],
                 label="零点赞占比 (95% CI)")
    ax2.set_ylabel("零点赞帖子占比 (%)", color=PALETTE["aigc"])
    ax2.grid(False)
    ax2.legend(fontsize=8, loc="upper left")
    cell = eng["within_cell"]
    ax.set_title("标签越多互动越好？"
                 f"（月份×领域单元内 ρ={cell['weighted_rho']:+.3f}，"
                 f"{cell['neg_cell_share']:.0%} 单元为负）", fontsize=10.5)
    for i, b in enumerate(bins):
        ax.annotate(f"n={b['n']:,}", (i, 0), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7.2, color="#444444")

    fig.suptitle("标签梯度、扩散与互动：AI 生成集的标签习惯未成为「AI 味」信号",
                 fontsize=13.5, y=1.0)
    save_fig(fig, INSIGHTS + "symbol_gradient.png")


def plot_emoji(structs: dict, grad_emoji: dict, emoji_diff: dict,
               xhs_diff: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    keys = ("human", "aigc", "exploring")
    xs = np.arange(len(keys))

    ax = axes[0, 0]
    uni = [structs[k]["emoji_share"] * 100 for k in keys]
    xhs = [structs[k]["xhs_share"] * 100 for k in keys]
    w = 0.36
    ax.bar(xs - w / 2, uni, width=w, color=PALETTE["aigc"], alpha=0.85,
           label="含 Unicode emoji 的帖子")
    ax.bar(xs + w / 2, xhs, width=w, color="#DD8452", alpha=0.85,
           label="含小红书表情 [xxR] 的帖子")
    for i, (u, x_) in enumerate(zip(uni, xhs)):
        ax.annotate(f"{u:.1f}%", (i - w / 2, u), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
        ax.annotate(f"{x_:.1f}%", (i + w / 2, x_), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    means = [structs[k]["emoji_per_note_mean"] for k in keys]
    xmeans = [structs[k]["xhs_per_note_mean"] for k in keys]
    ax.text(0.02, 0.96,
            "平均每帖：Unicode emoji "
            + " / ".join(f"{v:.2f}" for v in means)
            + "\n　　　　　[xxR] 表情 "
            + " / ".join(f"{v:.2f}" for v in xmeans),
            transform=ax.transAxes, fontsize=8.2, va="top")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[k] for k in keys])
    ax.set_ylabel("帖子占比 (%)")
    ax.set_ylim(0, max(uni + xhs) * 1.28)
    ax.set_title("表情使用率：AIGC 偏 Unicode emoji，人类更用 [xxR] 表情")
    ax.legend(fontsize=8.2)

    ax = axes[0, 1]
    tops = emoji_diff["human"][:7] + emoji_diff["aigc"][:7]
    labels = [t["word"] for t in tops]
    h_rate = [t["human_per1k"] for t in tops]
    a_rate = [t["aigc_per1k"] for t in tops]
    ypos = np.arange(len(labels))
    for y, h, a in zip(ypos, h_rate, a_rate):
        ax.plot([h, a], [y, y], color="#CCCCCC", lw=1.1, zorder=1)
    ax.scatter(h_rate, ypos, s=40, color=PALETTE["human"], zorder=3,
               label="人类")
    ax.scatter(a_rate, ypos, s=40, color=PALETTE["aigc"], zorder=3,
               label="AIGC")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("每千帖出现次数（对数轴）")
    ax.set_title("差异最大的 Unicode emoji（上：人类偏好 / 下：AI 偏好）")
    ax.legend(fontsize=8.4)
    ax.grid(axis="y", visible=False)

    ax = axes[1, 0]
    tops = xhs_diff["human"][:7] + xhs_diff["aigc"][:7]
    labels = [t["word"] for t in tops]
    h_rate = [t["human_per1k"] for t in tops]
    a_rate = [t["aigc_per1k"] for t in tops]
    ypos = np.arange(len(labels))
    for y, h, a in zip(ypos, h_rate, a_rate):
        ax.plot([h, a], [y, y], color="#CCCCCC", lw=1.1, zorder=1)
    ax.scatter(h_rate, ypos, s=40, color=PALETTE["human"], zorder=3,
               label="人类")
    ax.scatter(a_rate, ypos, s=40, color=PALETTE["aigc"], zorder=3,
               label="AIGC")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("每千帖出现次数（对数轴）")
    ax.set_title("差异最大的 [xxR] 表情（人类偏好 vs AI 偏好）")
    ax.legend(fontsize=8.4)
    ax.grid(axis="y", visible=False)

    ax = axes[1, 1]
    dec = grad_emoji["deciles"]
    x = np.arange(len(dec))
    means = [d["emoji_mean"] for d in dec]
    shares = [d["emoji_share"] * 100 for d in dec]
    xshares = [d["xhs_share"] * 100 for d in dec]
    ax.plot(x, shares, "o-", color=PALETTE["aigc"], ms=4.2, lw=1.5,
            label="含 Unicode emoji 帖占比")
    ax.plot(x, xshares, "s-", color="#DD8452", ms=4.0, lw=1.5,
            label="含 [xxR] 表情帖占比")
    ax.set_xlabel("AI 风格得分十分位（0=最像人 … 9=最像AI）")
    ax.set_ylabel("帖子占比 (%)")
    ax2 = ax.twinx()
    ax2.plot(x, means, "^--", color="#333333", ms=4.2, lw=1.3,
             label="平均 emoji 数/帖")
    ax2.set_ylabel("平均 Unicode emoji 数/帖")
    ax2.set_ylim(0, max(means) * 1.35)
    ax2.grid(False)
    ln1, lb1 = ax.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(ln1 + ln2, lb1 + lb2, fontsize=8, loc="upper left")
    rho = grad_emoji["spearman_emoji_share"]
    ax.set_title("探索集表情使用的 P(AI) 梯度：中段最高、两端回落"
                 f"（emoji 帖占比 ρ={rho['rho']:+.2f}）", fontsize=10.5)

    fig.suptitle("表情符号生态：人类偏好清单符号，AI 偏好情绪 emoji",
                 fontsize=13.5, y=1.0)
    save_fig(fig, INSIGHTS + "symbol_emoji.png")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    init_plot_style()
    scores = pd.read_csv(SCORES_PATH,
                         usecols=["p_ai", "month", "domain", "content_len"])
    p_ai = scores["p_ai"].to_numpy()
    months_scores = scores["month"].astype(str).to_numpy()
    domains_scores = scores["domain"].fillna("").astype(str).to_numpy()

    scans: dict[str, dict] = {}
    for key, path in DATASETS.items():
        ref = scores["content_len"].to_numpy() if key == "exploring" else None
        scans[key] = scan(path, content_len_ref=ref)
        print(f"[scan] {key}: {scans[key]['n_raw']} 条记录，"
              f"对齐 mismatch={scans[key]['mismatch']}")
    if scans["exploring"]["n_raw"] != len(scores):
        raise SystemExit("探索集记录数与 paper_exploring_scores 行数不一致，终止")

    kept = {k: s["kept_indices"] for k, s in scans.items()}
    note_tags = {k: s["tags"] for k, s in scans.items()}
    note_emojis = {k: s["emojis"] for k, s in scans.items()}
    note_xhs = {k: s["xhs"] for k, s in scans.items()}
    note_months = {k: s["months"] for k, s in scans.items()}
    for k in scans:
        print(f"[dedup] {k}: {scans[k]['n_raw']} 记录 -> "
              f"{len(kept[k])} 篇笔记")

    # ---- 维度 1：标签结构 ----
    # 每篇计一次的标签 Counter 只统计一次：结构指标、专属标签、基线共用。
    tag_counts_all = {k: note_tag_counts(note_tags[k]) for k in scans}
    structs = {k: tag_structure(note_tags[k], tag_counts_all[k])
               for k in scans}
    for k in scans:
        s = structs[k]
        print(f"[tags] {k}: 每帖 {s['tags_per_note_mean']:.2f} 个，"
              f"有标签 {s['tagged_share']:.1%}，unique={s['unique_tags']:,}"
              f"（{s['singleton_tag_share']:.0%} 只出现一次），"
              f"Top20={s['top20_coverage']:.1%}，H={s['entropy_mm']:.2f} bit")

    # ---- 维度 2：专属标签与标签级 AI 味 ----
    h_counts = tag_counts_all["human"]
    a_counts = tag_counts_all["aigc"]
    n_h = len(note_tags["human"])
    n_a = len(note_tags["aigc"])
    specific = {
        "aigc": rate_diff_ranking(a_counts, n_a, h_counts, n_h,
                                  reverse=False, min_count=MIN_A_COUNT,
                                  k=TAG_SPEC_K),
        "human": rate_diff_ranking(a_counts, n_a, h_counts, n_h,
                                   reverse=True, min_count=MIN_H_COUNT,
                                   k=TAG_SPEC_K),
        "aigc_top": rate_top(a_counts, n_a, min_count=MIN_A_COUNT, k=50),
    }
    ai_tag_set = {t["word"] for t in specific["aigc"]}
    top_tag_set = {t["word"] for t in specific["aigc_top"]}
    h_tag_set = {t["word"] for t in specific["human"]}
    print(f"[specific] AI 专属 Top5: "
          + "、".join(t["word"] for t in specific["aigc"][:5]))
    print(f"[specific] 人类专属 Top5: "
          + "、".join(t["word"] for t in specific["human"][:5]))

    sel_e = kept["exploring"]
    p_e = p_ai[sel_e]
    m_e = months_scores[sel_e]
    d_e = domains_scores[sel_e]
    ai_ness = tag_ai_ness(p_e, m_e, note_tags["exploring"])
    print(f"[tag AI-ness] 报告 {ai_ness['n_tags_reported']} 个标签；"
          f"最像 AI: " + "、".join("#" + t["tag"] for t in ai_ness["top_aigc"][:3]))

    # ---- 维度 3：梯度与扩散 ----
    # 逐帖标签集合 / 标签数 / P(AI) 十分位只算一次，供梯度、扩散、
    # 表情梯度与互动分析复用。
    body_lens_e = scans["exploring"]["content_lens"]
    tag_sets_e = [set(t) for t in note_tags["exploring"]]
    tags_n_e = np.asarray([len(t) for t in note_tags["exploring"]],
                          dtype=np.int64)
    order_e = np.argsort(p_e, kind="stable")
    decile_e = np.empty(p_e.size, dtype=np.int64)
    decile_e[order_e] = np.minimum(
        np.arange(p_e.size) * N_DECILES // p_e.size, N_DECILES - 1)
    # 命中数组只算一次：梯度与季度扩散共用（避免重复的集合求交）。
    hit_a = np.asarray([bool(s & ai_tag_set) for s in tag_sets_e])
    hit_top = np.asarray([bool(s & top_tag_set) for s in tag_sets_e])
    hit_h = np.asarray([bool(s & h_tag_set) for s in tag_sets_e])
    grad = decile_gradient(p_e, note_tags["exploring"], ai_tag_set, h_tag_set,
                           top_tag_set, body_lens_e, tag_sets=tag_sets_e,
                           tags_n=tags_n_e, decile=decile_e,
                           hit_a=hit_a, hit_h=hit_h, hit_top=hit_top)
    diff = quarterly_diffusion(m_e, hit_a, hit_top, hit_h)
    h_tags = note_tags["human"]
    h_months = note_months["human"]
    h_tag_sets = [set(ts) for ts in h_tags]
    clean_sets = [s for s, m in zip(h_tag_sets, h_months) if m and m <= "202206"]
    baselines = {
        "human_all_aigc_tag": round(
            sum(1 for s in h_tag_sets if s & ai_tag_set) / max(n_h, 1), 4),
        "human_clean_aigc_tag": round(
            sum(1 for s in clean_sets if s & ai_tag_set)
            / max(len(clean_sets), 1), 4),
        "aigc_aigc_tag": round(
            sum(1 for ts in note_tags["aigc"] if set(ts) & ai_tag_set)
            / max(n_a, 1), 4),
    }
    b_months = [m for m in h_months if m]
    b_clean = [m for m in b_months if m <= "202206"]
    print(f"[gradient] ρ(tags, P(AI))={grad['spearman_tags']['rho']:+.3f}，"
          f"剔除纯标签帖后 ρ={grad['spearman_tags_nonpure']['rho']:+.3f}，"
          f"控制正文长度后 ρ={grad['within_length_rho']['weighted_rho']:+.3f}；"
          f"AI 专属标签命中 {grad['deciles'][0]['hit_aigc_tag']:.1%}（最低分位）"
          f"→ {grad['deciles'][-1]['hit_aigc_tag']:.1%}（最高分位）")
    print(f"[diffusion] 探索集 AI 专属标签 {baselines['aigc_aigc_tag']:.1%} (AIGC) "
          f"/ {diff['quarters'][sorted(diff['quarters'])[-1]]['share_aigc_tag']:.1%} "
          f"(最新季度) / 人类干净窗口 {baselines['human_clean_aigc_tag']:.1%}")

    # ---- 维度 4：表情生态 ----
    # 表情词频 Counter 只展开一次：结构指标与专属表情对比共用。
    uni_vocab = {k: Counter(e for lst in note_emojis[k] for e in lst)
                 for k in scans}
    xhs_vocab = {k: Counter(e for lst in note_xhs[k] for e in lst)
                 for k in scans}
    emoji_structs = {k: emoji_structure(note_emojis[k], note_xhs[k],
                                        uni_vocab[k], xhs_vocab[k])
                     for k in scans}
    e_uni, h_uni, a_uni = (uni_vocab[k] for k in ("exploring", "human", "aigc"))
    e_xhs, h_xhs, a_xhs = (xhs_vocab[k] for k in ("exploring", "human", "aigc"))
    emoji_diff = {
        "human": rate_diff_ranking(a_uni, len(note_emojis["aigc"]),
                                   h_uni, len(note_emojis["human"]),
                                   reverse=True, min_count=20, k=7),
        "aigc": rate_diff_ranking(a_uni, len(note_emojis["aigc"]),
                                  h_uni, len(note_emojis["human"]),
                                  reverse=False, min_count=20, k=7),
        "exploring_top": [[e, int(c), round(c / len(sel_e) * 1000, 2)]
                          for e, c in e_uni.most_common(15)],
    }
    xhs_diff = {
        "human": rate_diff_ranking(a_xhs, len(note_xhs["aigc"]),
                                   h_xhs, len(note_xhs["human"]),
                                   reverse=True, min_count=20, k=7),
        "aigc": rate_diff_ranking(a_xhs, len(note_xhs["aigc"]),
                                  h_xhs, len(note_xhs["human"]),
                                  reverse=False, min_count=20, k=7),
        "exploring_top": [[e, int(c), round(c / len(sel_e) * 1000, 2)]
                          for e, c in e_xhs.most_common(15)],
    }
    uni_n = np.asarray([len(e) for e in note_emojis["exploring"]])
    xhs_n = np.asarray([len(e) for e in note_xhs["exploring"]])
    grad_emoji = {"deciles": []}
    for d in range(N_DECILES):
        m = decile_e == d
        grad_emoji["deciles"].append({
            "decile": d + 1, "n": int(m.sum()),
            "emoji_mean": round(float(uni_n[m].mean()), 3),
            "emoji_share": round(float((uni_n[m] > 0).mean()), 4),
            "xhs_share": round(float((xhs_n[m] > 0).mean()), 4),
        })
    rho_share, p_share = spearmanr(p_e, (uni_n > 0).astype(float))
    rho_n, p_n = spearmanr(p_e, uni_n)
    grad_emoji["spearman_emoji_share"] = {"rho": round(float(rho_share), 3),
                                          "p": float(p_share)}
    grad_emoji["spearman_emoji_mean"] = {"rho": round(float(rho_n), 3),
                                         "p": float(p_n)}
    print(f"[emoji] exploring 含 emoji {emoji_structs['exploring']['emoji_share']:.1%}"
          f" / [xxR] {emoji_structs['exploring']['xhs_share']:.1%}；"
          f"human 含 emoji {emoji_structs['human']['emoji_share']:.1%}"
          f" / [xxR] {emoji_structs['human']['xhs_share']:.1%}")

    # ---- 维度 5：标签×互动 ----
    eng = tag_engagement(
        tags_n_e,
        scans["exploring"]["likes"],
        scans["exploring"]["colls"],
        m_e, d_e)
    print(f"[engagement] 单元内 ρ(标签数, log点赞)="
          f"{eng['within_cell']['weighted_rho']:+.3f} "
          f"(负单元 {eng['within_cell']['neg_cell_share']:.0%}, "
          f"{eng['within_cell']['n_cells']} 个单元)")

    out = {
        "meta": {
            "tag_topk": TAG_TOPK, "bin_edges": list(BIN_EDGES),
            "min_aigc_count": MIN_A_COUNT, "min_human_count": MIN_H_COUNT,
            "tag_ai_min_notes": TAG_AI_MIN_NOTES,
            "min_quarter_n": MIN_QUARTER_N, "min_cell_n": MIN_CELL_N,
            "min_body_len": MIN_BODY_LEN,
            "eng_month_from": ENG_MONTH_FROM,
            "exploring_content_len_mismatch": scans["exploring"]["mismatch"],
            "note": "人类/探索集按 note_id 去重（保留元数据最完整一条）；"
                    "AIGC 无 note_id 逐记录；p_ai 按行序对齐 "
                    "paper_exploring_scores.csv.gz；标签与表情口径见脚本 docstring。",
            "n_records": {k: scans[k]["n_raw"] for k in scans},
            "n_notes": {k: int(len(kept[k])) for k in scans},
        },
        "hashtag_structure": {k: structs[k] for k in scans},
        "hashtag_specific": specific,
        "tag_ai_ness": ai_ness,
        "gradient": {"tags": grad, "emoji": grad_emoji},
        "quarterly": {"diffusion": diff, "baselines": baselines,
                      "human_months": {"n": len(b_months),
                                       "clean_window_n": len(b_clean)}},
        "emoji_structure": emoji_structs,
        "emoji_specific": {"unicode": emoji_diff, "xhs": xhs_diff},
        "tag_engagement": eng,
    }
    save_json(out, BASE + "symbol_results.json")
    write_markdown(out)
    plot_hashtag(structs, specific, ai_ness)
    plot_gradient(grad, diff, eng, baselines)
    plot_emoji(emoji_structs, grad_emoji, emoji_diff, xhs_diff)
    print("完成：symbol_results.json / symbol_results.md / "
          "insights/symbol_hashtag.png / symbol_gradient.png / symbol_emoji.png")


def write_markdown(out: dict) -> None:
    st = out["hashtag_structure"]
    sp = out["hashtag_specific"]
    g = out["gradient"]["tags"]
    eg = out["gradient"]["emoji"]
    q = out["quarterly"]["diffusion"]
    bl = out["quarterly"]["baselines"]
    em = out["emoji_structure"]
    eng = out["tag_engagement"]
    lines = [
        "# 话题标签与表情符号生态（Symbol）",
        "",
        "> 由 `symbol_analysis.py` 全量运行生成；标签口径与 `paper_features.py` 的",
        "> `n_hashtags` 同源（兼容 `#标签[话题]#` 与 BOM 残留），表情正则与其",
        "> `EMOJI` / `XHS_EMO` 完全相同；人类/探索集按 note_id 去重，AIGC 逐记录；",
        "> 探索集 p_ai 按行序对齐 `paper_exploring_scores.csv.gz`（content_len 校验",
        f"> mismatch={out['meta']['exploring_content_len_mismatch']}）。",
        "",
        "## 1 标签结构",
        "",
        "| 子集 | 笔记数 | 有标签占比 | 每帖标签数(均/中位) | 标签数 ≥5 占比 | 唯一标签 | 唯一率 | 仅出现一次占比 | Top-20 覆盖 | 熵(bit) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, lab in LABELS.items():
        s = st[k]
        lines.append(
            f"| {lab} | {s['n_notes']:,} | {s['tagged_share']:.1%} | "
            f"{s['tags_per_note_mean']:.2f} / {s['tags_per_note_median']:.0f} | "
            f"{s['bins_pct']['5–9'] + s['bins_pct']['10+']:.1f}% | "
            f"{s['unique_tags']:,} | {s['unique_tag_ratio']:.1%} | "
            f"{s['singleton_tag_share']:.1%} | {s['top20_coverage']:.1%} | "
            f"{s['entropy_mm']:.2f} |")
    lines += [
        "",
        "> 结构要点：带标签比例与每帖标签数随语料时代上升（人类 54.9% / 2.96 个"
        " → AIGC 71.2% / 3.94 个 → 探索集 91.9% / 6.61 个）；AIGC 的标签词表"
        "最「一次性」——65.6% 的标签用量是唯一标签（人类 36.4%），更像生成器"
        "临场自造的话题词，而人类/探索集复用社区约定俗成的标签。",
        "",
        "## 2 专属标签（每帖发生率之差）",
        "",
        "**AI 专属标签 Top 10：** " + "、".join(
            f"`#{t['word']}`（+{t['rate_diff_ppt']:.1f}pp）"
            for t in sp["aigc"][:10]),
        "",
        "**AIGC 高频标签 Top 10（按生成集发生率）：** " + "、".join(
            f"`#{t['word']}`（{t['aigc_per1k']:.0f}/千帖）"
            for t in sp["aigc_top"][:10]),
        "",
        "**人类专属标签 Top 10：** " + "、".join(
            f"`#{t['word']}`（{t['rate_diff_ppt']:.1f}pp）"
            for t in sp["human"][:10]),
        "",
        "> 注意：AIGC 生成集与人类训练集存在时代与话题构成差异（生成提示覆盖的"
        "领域 vs 前 LLM 社区语料），此处差异只能解释为「生成器的标签偏好」，"
        "不能单独视为风格信号——下一节的探索集内检验正说明它没有迁移。",
        "",
        "## 3 标签级 AI 味（探索集，出现 ≥"
        f"{out['meta']['tag_ai_min_notes']} 篇，按月份残差排序）",
        "",
        "**最像 AI 的标签：** " + "、".join(
            f"`#{t['tag']}`（P̄={t['mean_p']:.2f}，n={t['n']:,}）"
            for t in out["tag_ai_ness"]["top_aigc"][:8]),
        "",
        "**最像人类的标签：** " + "、".join(
            f"`#{t['tag']}`（P̄={t['mean_p']:.2f}，n={t['n']:,}）"
            for t in out["tag_ai_ness"]["top_human"][:8]),
        "",
        "## 4 探索集梯度与季度扩散",
        "",
        f"- 每帖标签数在高 P(AI) 端显著回落：Spearman ρ="
        f"{g['spearman_tags']['rho']:+.3f}（p={g['spearman_tags']['p']:.1e}）；"
        f"最像人十分位平均 {g['deciles'][0]['tags_mean']:.2f} 个 vs 最像 AI "
        f"十分位 {g['deciles'][-1]['tags_mean']:.2f} 个。两端都有「纯标签帖」"
        f"（正文 < {out['meta']['min_body_len']} 字）扰动：最低分位 "
        f"{g['deciles'][0]['short_body_share']:.1%}、最高分位 "
        f"{g['deciles'][-1]['short_body_share']:.1%}。",
        f"- 稳健性：剔除纯标签帖后 ρ="
        f"{g['spearman_tags_nonpure']['rho']:+.3f}"
        f"（n={g['spearman_tags_nonpure']['n']:,}）；控制正文长度分层后加权 ρ="
        f"{g['within_length_rho']['weighted_rho']:+.3f}"
        f"（{g['within_length_rho']['n_strata']} 层）——负向关系不是长度构成"
        f"造成的假象。",
        f"- AI 专属标签命中率 {g['deciles'][0]['hit_aigc_tag']:.1%} → "
        f"{g['deciles'][-1]['hit_aigc_tag']:.1%}（δ="
        f"{g['cliff_aigc_tag']['delta']:+.3f}）；人类专属标签 "
        f"{g['deciles'][0]['hit_human_tag']:.1%} → "
        f"{g['deciles'][-1]['hit_human_tag']:.1%}（δ="
        f"{g['cliff_human_tag']['delta']:+.3f}）：两类标签都随 P(AI) 下降，"
        f"说明高 P(AI) 帖整体少用标签，而非在标签上「更像生成集」。",
        f"- 使用率基线：AIGC 生成集 {bl['aigc_aigc_tag']:.1%}，人类集 "
        f"{bl['human_all_aigc_tag']:.1%}（干净窗口 ≤2022.06："
        f"{bl['human_clean_aigc_tag']:.1%}）。",
        f"- 季度趋势：AI 专属标签 ρ={q['trend_aigc_tag'].get('rho', float('nan')):+.3f}"
        f"（p={q['trend_aigc_tag'].get('p', float('nan')):.2f}，不显著），"
        f"AIGC 高频 Top-50 ρ={q['trend_top_tag'].get('rho', float('nan')):+.3f}"
        f"（p={q['trend_top_tag'].get('p', float('nan')):.2f}）；"
        f"人类专属标签 ρ={q['trend_human_tag'].get('rho', float('nan')):+.3f}"
        f"（p={q['trend_human_tag'].get('p', float('nan')):.2f}）——标签层面"
        f"没有出现词汇层面那样的 AI 扩散。",
        "",
        "## 5 表情生态",
        "",
        "| 子集 | 含 Unicode emoji | 含 [xxR] 表情 | emoji/帖(均) | [xxR]/帖(均) |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, lab in LABELS.items():
        s = em[k]
        lines.append(
            f"| {lab} | {s['emoji_share']:.1%} | {s['xhs_share']:.1%} | "
            f"{s['emoji_per_note_mean']:.2f} | {s['xhs_per_note_mean']:.2f} |")
    lines += [
        "",
        f"- 探索集 P(AI) 越高的帖子越常含 Unicode emoji（emoji 帖占比 "
        f"ρ={eg['spearman_emoji_share']['rho']:+.3f}，p="
        f"{eg['spearman_emoji_share']['p']:.1e}），每帖 emoji 数 "
        f"ρ={eg['spearman_emoji_mean']['rho']:+.3f}。",
        f"- 人类最偏好的 Unicode emoji：" + "、".join(
            t["word"] for t in out["emoji_specific"]["unicode"]["human"][:5])
        + "；AI 最偏好：" + "、".join(
            t["word"] for t in out["emoji_specific"]["unicode"]["aigc"][:5])
        + "。",
        f"- 人类最偏好的 [xxR] 表情：" + "、".join(
            t["word"] for t in out["emoji_specific"]["xhs"]["human"][:5])
        + "；AI 最偏好：" + "、".join(
            t["word"] for t in out["emoji_specific"]["xhs"]["aigc"][:5])
        + "。",
        "",
        "## 6 标签数与互动（探索集 2023.01 起）",
        "",
        "| 每帖标签数 | n | 点赞中位数 | 零点赞占比 | 收藏/点赞中位 |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in eng["bins"]:
        lines.append(
            f"| {b['bin']} | {b['n']:,} | {b['likes_median']:.0f} | "
            f"{b['zero_like_share']:.1%} | {b['coll_per_like']:.3f} |")
    cell = eng["within_cell"]
    lines += [
        "",
        f"- 控制月份×领域后（{cell['n_cells']} 个单元），标签数与 log 点赞的单元内"
        f"加权 ρ={cell['weighted_rho']:+.3f}，仅 {cell['neg_cell_share']:.0%} "
        f"的单元为负——标签越多互动略好（与 P(AI) 的负向「AI 味惩罚」方向相反），"
        f"点赞中位数从 0 标签的 {eng['bins'][0]['likes_median']:.0f} 整体上升至 "
        f"10+ 标签的 {eng['bins'][-1]['likes_median']:.0f}（中间 1–2 标签略有回落）。",
        "",
        "图表见 `insights/`：`symbol_hashtag.png`、`symbol_gradient.png`、"
        "`symbol_emoji.png`。",
    ]
    with open(BASE + "symbol_results.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
