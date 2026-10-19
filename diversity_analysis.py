# -*- coding: utf-8 -*-
"""文本多样性与节奏深潜：句长「起伏」（burstiness）与词汇长尾结构。

既有维度已覆盖平均句长（deep_dive_analysis）、独特字/二元组比率（paper_features），
但缺少两个风格计量学的经典指标：

1. 句长 burstiness（每帖句长的变异系数 CV）：人类写作长短句交错，模板化文本
   句长均匀——平均句长相同但 CV 可以完全不同，是「读起来像 AI」的直接量化；
2. 词汇多样性 MTLD（Measure of Textual Lexical Diversity）与 Heaps 定律指数：
   独特字比率受文本长度强烈影响，MTLD 对长度稳健；Heaps 定律刻画词汇长尾的
   增长速度，描述「文本越长还能带来多少新词」。

口径说明：
- 人类/探索集按 note_id 去重（保留元数据最完整的一条，规则同 deep_dive_analysis.py），
  AIGC 无 note_id 逐记录处理；
- 正文清洗复用 analysis_utils.clean_body（与 paper_features.py 一致）；
- burstiness 与句长统计为全量计算（仅要求每帖 ≥3 个句子）；MTLD 与 Heaps 定律
  需要分词，采用固定种子（seed=42）的分层随机抽样（每子集 1.2 万帖）以避免
  全量 jieba 的长时间开销；
- 差异检验统一 Mann-Whitney U + Cliff's delta，中位数附 bootstrap 95% CI。

输出：
  insights/diversity_burstiness.png   句长分布 + burstiness ECDF
  insights/diversity_lexical.png      MTLD 分布 + Heaps 定律曲线
  diversity_results.json              全部数值结果
  diversity_results.md                结果摘要
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter

import jieba

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from analysis_utils import (BASE, INSIGHTS, PALETTE, init_plot_style,
                            save_fig, save_json, bootstrap_median_ci,
                            cliffs_delta_from_u, clean_body, content_of,
                            engagement_of, load_records_with_total,
                            make_pool, close_pool,
                            mtld, parse_local_time, split_sentences, burstiness,
                            DATASET_LABELS)

DATASET_PATHS = {
    "human": BASE + "training_set_human_fixed.jsonl",
    "aigc": BASE + "training_set_aigc.jsonl",
    "exploring": BASE + "exploring_set_fixed.jsonl",
}
MIN_SENTENCES_BURST = 3     # burstiness 稳定性要求
MIN_TOKENS_MTLD = 50        # MTLD 稳定性要求
SAMPLE_PER_DATASET = 12_000  # 分词抽样规模（每子集）
SAMPLE_SEED = 42
BOOT = 1000
BOOT_SEED = 42
TOKEN_KEEP = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
HEAPS_BUDGET = 1_000_000     # 各子集统一 token 预算，保证 Heaps 曲线跨度可比


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def doc_metrics(body: str) -> dict:
    """单帖句法指标：句子长度列表、句数、burstiness（句数不足时为 None）。"""
    lens = [len(s) for s in split_sentences(body)]
    return {
        "body_len": len(body),
        "sent_lens": lens,
        "n_sent": len(lens),
        "burst": burstiness(lens) if len(lens) >= MIN_SENTENCES_BURST else None,
    }


jieba.setLogLevel(60)


def tokenize(text: str) -> list[str]:
    """jieba 分词并过滤纯标点/空白 token；拉丁词统一小写。"""
    out = []
    for tok in jieba.lcut(text or ""):
        # 先按未 strip 的 token 判保留（TOKEN_KEEP 命中即可保证 strip 后非空），
        # 只对保留的 token 做 strip/lower，省去对丢弃 token 的字符串变换。
        if tok and TOKEN_KEEP.search(tok):
            out.append(tok.strip().lower())
    return out


_TOK_SEP = "\x00"


def tokenize_encode(body: str) -> tuple:
    """`tokenize` 的进程池任务版：顺带在 worker 内算好 MTLD 并紧凑编码。

    返回 (tokens 编码, token 数, mtld 或 None)：token 列表连接成单个
    字符串回传（父进程按 `_TOK_SEP` 还原），避免逐个 token 小对象的
    pickle/IPC 开销；MTLD 只在达到 `MIN_TOKENS_MTLD` 时计算（与父进程
    原先的门槛一致）。token 本身含分隔符（理论上的 \\u0000）时回退传输
    原列表，保证任何输入下与 `tokenize` 结果逐位一致。
    """
    toks = tokenize(body)
    joined = _TOK_SEP.join(toks)
    if joined.count(_TOK_SEP) != len(toks) - 1:
        joined = toks
    mt = mtld(toks) if len(toks) >= MIN_TOKENS_MTLD else None
    return joined, len(toks), mt


def heaps_curve(doc_tokens: list[list[str]],
                max_tokens: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Heaps 定律累积曲线：逐文档累计 token 数与词型数。

    max_tokens 可在相同 token 预算处截断，保证不同子集的曲线跨度可比。
    """
    seen: set[str] = set()
    tokens, types = [], []
    n = 0
    for doc in doc_tokens:
        for tok in doc:
            if max_tokens is not None and n >= max_tokens:
                break
            seen.add(tok)
            n += 1
        tokens.append(n)
        types.append(len(seen))
        if max_tokens is not None and n >= max_tokens:
            break
    return np.asarray(tokens, dtype=float), np.asarray(types, dtype=float)


def heaps_beta(tokens: np.ndarray, types: np.ndarray,
               min_tokens: int = 500) -> float:
    """对 log-log 曲线做线性拟合返回斜率 β（vocabulary growth exponent）。"""
    mask = (tokens >= min_tokens) & (tokens > 0) & (types > 0)
    if mask.sum() < 2:
        return float("nan")
    slope = np.polyfit(np.log(tokens[mask]), np.log(types[mask]), 1)[0]
    return float(slope)


# --------------------------------------------------------------------------
# 全量扫描与抽样
# --------------------------------------------------------------------------
def scan_sentence_stats(path: str) -> dict:
    """全量流式扫描：按 note_id 去重后收集每帖句长指标（无 note_id 全部保留）。

    扫描时顺带记录每条记录的字节偏移（`offsets`），供后续按索引抽样时直接
    seek 定位、免去对同一文件再做两次全量解析。
    """
    slot_of: dict = {}
    metrics: list[dict] = []
    offsets: list[int] = []
    # 去重后保留记录的字节偏移（与 metrics 一一对应）：后续词法抽样必须从
    # 这批偏移里抽，否则会按「快照条数」而不是「笔记数」加权，且可能抽到
    # 已被句法扫描丢弃的重复快照，两个分析口径不一致。
    dedup_offsets: list[int] = []
    with open(path, "rb") as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            if line.isspace():
                continue
            rec = json.loads(line)
            offsets.append(start)
            body = clean_body(content_of(rec))
            likes, colls, comms = engagement_of(rec)
            dt = parse_local_time(rec.get("local_time"))
            has_meta = bool(rec.get("user_id")) and rec.get("domain") not in (None, "", "未知")
            score = (int(has_meta) + int(bool(likes)) + int(bool(colls))
                     + int(bool(comms)) + int(dt is not None) + int(bool(body)))
            nid = rec.get("note_id")
            if nid is None:
                m = doc_metrics(body)
                m["score"] = score
                metrics.append(m)
                dedup_offsets.append(start)
                continue
            slot = slot_of.get(nid)
            # 只有确定要保留这份记录时才做句法指标（探索集约一半的重复
            # 记录会在去重时被丢弃，省掉它们的 split/burstiness 计算）。
            if slot is None:
                m = doc_metrics(body)
                m["score"] = score
                slot_of[nid] = len(metrics)
                metrics.append(m)
                dedup_offsets.append(start)
            elif score > metrics[slot]["score"]:
                m = doc_metrics(body)
                m["score"] = score
                metrics[slot] = m
                dedup_offsets[slot] = start
    sent_lens = [l for m in metrics for l in m["sent_lens"]]
    bursts = [m["burst"] for m in metrics if m["burst"] is not None]
    # 平均句长：sum/len 与 np.mean 对整数列表逐位一致，但省掉 15 万次数组构造。
    mean_sents = [sum(m["sent_lens"]) / len(m["sent_lens"])
                  for m in metrics if m["sent_lens"]]
    n_sents = [m["n_sent"] for m in metrics if m["sent_lens"]]
    return {
        "n_reps": len(metrics),
        "sent_lens": np.asarray(sent_lens, dtype=np.int64),
        "burst": np.asarray(bursts, dtype=float),
        "mean_sent": np.asarray(mean_sents, dtype=float),
        "n_sent": np.asarray(n_sents, dtype=np.int64),
        "offsets": offsets,
        "dedup_offsets": dedup_offsets,
    }


def sample_records_at_offsets(path: str, offsets: list[int], sample: int,
                              seed: int) -> list[dict]:
    """按 `random.Random(seed).sample(range(n), sample)` 的索引顺序解析抽样记录。

    与 `load_records_with_total(path, sample=..., seed=...)` 的返回列表完全一致
    （同样的抽样索引、同样的顺序），但借助预记录的字节偏移只解析被抽中的行，
    跳过了计数与逐行解析两次全文件扫描。
    """
    n = len(offsets)
    if sample is None or n <= sample:
        idx = range(n)
    else:
        idx = random.Random(seed).sample(range(n), sample)
    out: list[dict] = []
    with open(path, "rb") as f:
        for i in idx:
            f.seek(offsets[i])
            out.append(json.loads(f.readline()))
    return out


def sample_lexical(path: str, pool=None, offsets=None) -> dict:
    """固定种子抽样并分词，单遍流式计算 MTLD 与 Heaps 曲线。

    与"先保留全部抽样帖 token 列表、再统一计算"的旧实现结果完全一致，
    但逐帖算完即释放 token，内存占用从 O(抽样帖数) 降为 O(批次)；
    Heaps 曲线在遍历过程中增量累积（含 HEAPS_BUDGET 截断语义）。
    传入 pool 时按批并行分词，逐帖结果与串行完全一致且顺序不变。
    """
    if offsets is not None:
        recs = sample_records_at_offsets(path, offsets,
                                         SAMPLE_PER_DATASET, SAMPLE_SEED)
        total = len(offsets)
    else:
        recs, total = load_records_with_total(path, sample=SAMPLE_PER_DATASET,
                                              seed=SAMPLE_SEED)
    seen_ids: set = set()
    bodies: list[str] = []
    for rec in recs:
        nid = rec.get("note_id")
        if nid is not None:
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
        bodies.append(clean_body(content_of(rec)))
    del recs, seen_ids  # 正文已提取，抽样记录与去重集合不再需要

    # 每个保留的 token 至少含 1 个字（TOKEN_KEEP 要求），故正文长度不足
    # MIN_TOKENS_MTLD 时 token 数必然不足，跳过其分词与统计不影响结果。
    bodies = [b for b in bodies if len(b) >= MIN_TOKENS_MTLD]

    mtld_values: list[float] = []
    seen: set[str] = set()
    heap_tokens: list[int] = []
    heap_types: list[int] = []
    n_tokens = 0
    batch = 2048
    # 重复正文（同一文本多帖）分词结果相同，缓存其 token 列表复用；
    # 仅缓存确实重复出现的正文，避免内存随抽样规模增长。
    body_counts = Counter(bodies)
    tok_cache: dict[str, tuple] = {}
    for start in range(0, len(bodies), batch):
        chunk = bodies[start:start + batch]
        need: list[str] = []
        need_index: dict[str, int] = {}
        for b in chunk:
            if b in tok_cache or b in need_index:
                continue
            need_index[b] = len(need)
            need.append(b)
        tok_lists = (pool.map(tokenize_encode, need) if pool is not None
                     else [tokenize_encode(t) for t in need])
        for b, i in need_index.items():
            if body_counts[b] > 1:
                tok_cache[b] = tok_lists[i]
        for b in chunk:
            enc = tok_cache[b] if b in tok_cache else tok_lists[need_index[b]]
            joined, ntok, mt = enc
            if ntok < MIN_TOKENS_MTLD:
                continue
            mtld_values.append(mt)
            if n_tokens < HEAPS_BUDGET:
                toks = joined.split(_TOK_SEP) if isinstance(joined, str) else joined
                for tok in toks:
                    if n_tokens >= HEAPS_BUDGET:
                        break
                    seen.add(tok)
                    n_tokens += 1
                heap_tokens.append(n_tokens)
                heap_types.append(len(seen))
    return {
        "mtld": np.asarray(mtld_values, dtype=float),
        "tokens": np.asarray(heap_tokens, dtype=float),
        "types": np.asarray(heap_types, dtype=float),
        "total_records": total,
        "n_docs": len(mtld_values),
        "n_docs_in_curve": len(heap_tokens),
        "n_tokens": n_tokens,
    }


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_burstiness(stats: dict, bursts: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    keys = ("human", "aigc", "exploring")
    ax = axes[0]
    bins = np.arange(0, 181, 4)
    for k in keys:
        lens = np.clip(stats[k]["sent_lens"], 0, 179)
        w = np.ones(len(lens)) / len(lens)
        ax.hist(lens, bins=bins, weights=w, histtype="step", lw=1.8,
                color=PALETTE[k], label=DATASET_LABELS[k])
    ax.set_xlim(0, 180)
    ax.set_xlabel("单句长度（字，≥180 截断）")
    ax.set_ylabel("句子占比")
    ax.set_title("句长分布")
    ax.legend()

    ax = axes[1]
    for idx, k in enumerate(keys):
        v = np.sort(stats[k]["burst"])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=1.8,
                color=PALETTE[k], label=DATASET_LABELS[k])
        med = bursts[k]["median"]
        ax.axvline(med, color=PALETTE[k], ls=":", lw=1.1, alpha=0.8)
        ax.annotate(f"{med:.2f}", xy=(med + 0.03, 0.44 + idx * 0.07),
                    fontsize=8.5, color=PALETTE[k])
    ax.set_xlim(0, 2.5)
    ax.set_xlabel("每帖句长变异系数 CV（越大越有起伏）")
    ax.set_ylabel("累计占比")
    ax.set_title("句子节奏的起伏（burstiness）")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, INSIGHTS + "diversity_burstiness.png")


def plot_lexical(mtld_stats: dict, heaps: dict, mtld_dist: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    keys = ("human", "aigc", "exploring")
    ax = axes[0]
    data = [mtld_dist[k] for k in keys]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, k in zip(parts["bodies"], keys):
        body.set_facecolor(PALETTE[k])
        body.set_alpha(0.55)
    for i, k in enumerate(keys, start=1):
        med = mtld_stats[k]["median"]
        ci = mtld_stats[k]
        ax.plot([i], [med], "o", color="black", ms=4)
        ax.vlines(i, ci["ci_low"], ci["ci_high"], color="black", lw=1.6)
        ax.annotate(f"{med:.0f}", xy=(i, med), xytext=(i + 0.12, med + 4),
                    fontsize=9)
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels([DATASET_LABELS[k] for k in keys])
    ax.set_ylabel("每帖 MTLD（越大词汇越丰富）")
    ax.set_title("词汇多样性（点=中位数，线=bootstrap 95%CI）")

    ax = axes[1]
    for k in keys:
        tokens, types = heaps[k]["curve"]
        mask = tokens >= 1
        ax.plot(tokens[mask], types[mask], lw=1.8, color=PALETTE[k],
                label=f"{DATASET_LABELS[k]} β={heaps[k]['beta']:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("累计 token 数（log）")
    ax.set_ylabel("累计词型数（log）")
    ax.set_title("Heaps 定律：词汇长尾的增长速度")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save_fig(fig, INSIGHTS + "diversity_lexical.png")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def _boot_median_ci(payload) -> dict:
    """`bootstrap_median_ci` 的进程池任务包装（供分段并行调用）。

    bootstrap 只依赖固定种子的 numpy RNG，在任何进程里结果都逐位相同。
    """
    values, n_boot, seed = payload
    return bootstrap_median_ci(values, n_boot=n_boot, seed=seed)


def main() -> None:
    init_plot_style()

    stats, bursts = {}, {}
    pool = make_pool(preload=jieba.initialize)
    try:
        # 三个子集的全量句法扫描互不依赖，复用同一个进程池并行执行；
        # 逐子集返回结果与串行调用完全一致、子集顺序不变。该池随后继续
        # 用于 MTLD/Heaps 的抽样分词，不额外增加进程开销。
        names = list(DATASET_PATHS)
        scans = (pool.map(scan_sentence_stats,
                          [DATASET_PATHS[n] for n in names])
                 if pool is not None
                 else [scan_sentence_stats(DATASET_PATHS[n]) for n in names])
        for name, s in zip(names, scans):
            stats[name] = s
        # burst 的中位数 CI 三个子集互不依赖：扫描结束后进程池空闲，
        # 复用同一进程池并行；固定种子使结果与串行逐位一致。
        boot_results = (pool.map(_boot_median_ci,
                                 [(stats[n]["burst"], BOOT, BOOT_SEED)
                                  for n in names])
                        if pool is not None
                        else [bootstrap_median_ci(stats[n]["burst"], n_boot=BOOT,
                                                  seed=BOOT_SEED) for n in names])
        for name, b in zip(names, boot_results):
            bursts[name] = b
            print(f"{name}: {stats[name]['n_reps']} 篇（去重后），burstiness 中位数 "
                  f"{bursts[name]['median']:.3f}")

        mtld_dist, mtld_stats, heaps = {}, {}, {}
        mtld_samples = []
        for name, path in DATASET_PATHS.items():
            # 词法抽样只在 note_id 去重后的记录上进行，与句法扫描口径一致。
            smp = sample_lexical(path, pool,
                                 offsets=stats[name].pop("dedup_offsets"))
            vals = smp["mtld"]
            mtld_dist[name] = vals
            mtld_samples.append(vals)
            tokens, types = smp["tokens"], smp["types"]
            heaps[name] = {"curve": (tokens, types), "beta": heaps_beta(tokens, types),
                           "n_docs": smp["n_docs"],
                           "n_docs_in_curve": smp["n_docs_in_curve"],
                           "n_tokens": smp["n_tokens"]}
        # MTLD 的中位数 CI 同样互相独立，且只需固定规模样本；一次并行算完。
        mtld_results = (pool.map(_boot_median_ci,
                                 [(v, BOOT, BOOT_SEED) for v in mtld_samples])
                        if pool is not None
                        else [bootstrap_median_ci(v, n_boot=BOOT, seed=BOOT_SEED)
                              for v in mtld_samples])
        for name, b in zip(DATASET_PATHS, mtld_results):
            mtld_stats[name] = b
            print(f"{name}: MTLD 中位数 {mtld_stats[name]['median']:.1f}, "
                  f"Heaps β={heaps[name]['beta']:.3f}")
    finally:
        close_pool(pool)

    # 差异检验（human vs 其他）
    def compare(a, b) -> dict:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return {"cliffs_delta": round(cliffs_delta_from_u(u, len(a), len(b)), 3),
                "p": float(p)}

    comparisons = {}
    for other in ("aigc", "exploring"):
        comparisons[f"burst_human_vs_{other}"] = compare(stats["human"]["burst"],
                                                         stats[other]["burst"])
        comparisons[f"mtld_human_vs_{other}"] = compare(mtld_dist["human"],
                                                        mtld_dist[other])

    out = {
        "burstiness": {
            k: {"n": bursts[k]["n"], "median": round(bursts[k]["median"], 4),
                "ci": [round(bursts[k]["ci_low"], 4), round(bursts[k]["ci_high"], 4)]}
            for k in bursts
        },
        "sentence_stats": {
            k: {"n_reps": stats[k]["n_reps"],
                "sent_len_median": float(np.median(stats[k]["sent_lens"])),
                "mean_sent_len_median": float(np.median(stats[k]["mean_sent"])),
                "sentences_per_note_median": float(np.median(stats[k]["n_sent"]))}
            for k in stats
        },
        "mtld": {
            k: {"n_docs": int(len(mtld_dist[k])),
                "median": round(mtld_stats[k]["median"], 2),
                "ci": [round(mtld_stats[k]["ci_low"], 2),
                       round(mtld_stats[k]["ci_high"], 2)]}
            for k in mtld_stats
        },
        "heaps": {k: {"n_docs": heaps[k]["n_docs"],
                      "n_docs_in_curve": heaps[k]["n_docs_in_curve"],
                      "n_tokens": heaps[k]["n_tokens"],
                      "beta": round(heaps[k]["beta"], 4)} for k in heaps},
        "comparisons": comparisons,
        "sample": {"per_dataset": SAMPLE_PER_DATASET, "seed": SAMPLE_SEED,
                   "heaps_token_budget": HEAPS_BUDGET,
                   "min_sentences_for_burst": MIN_SENTENCES_BURST,
                   "min_tokens_for_mtld": MIN_TOKENS_MTLD},
    }
    save_json(out, BASE + "diversity_results.json")

    plot_burstiness(stats, bursts)
    plot_lexical(mtld_stats, heaps, mtld_dist)
    write_markdown(out)
    print("已保存 diversity_results.json / diversity_results.md 与 insights/diversity_*.png")


def write_markdown(out: dict) -> None:
    b = out["burstiness"]
    s = out["sentence_stats"]
    m = out["mtld"]
    h = out["heaps"]
    c = out["comparisons"]
    lines = [
        "# 文本多样性与节奏深潜结果（Diversity & Burstiness）",
        "",
        "> 由 `diversity_analysis.py` 生成；人类/探索集按 note_id 去重（规则同",
        "> `deep_dive_analysis.py`），AIGC 逐记录；句法指标全量计算，MTLD/Heaps",
        f"> 基于每子集 {out['sample']['per_dataset']:,} 帖的固定种子（seed="
        f"{out['sample']['seed']}）随机抽样与 jieba 分词。",
        "",
        "## 1 句子节奏指标（按 note_id 去重后）",
        "",
        "| 子集 | 篇数 | 单句长度中位数 | 平均句长中位数 | 每帖句数中位数 | burstiness 中位数(95%CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in ("human", "aigc", "exploring"):
        lines.append(
            f"| {DATASET_LABELS[k]} | {s[k]['n_reps']:,} | "
            f"{s[k]['sent_len_median']:.0f} | {s[k]['mean_sent_len_median']:.1f} | "
            f"{s[k]['sentences_per_note_median']:.0f} | "
            f"{b[k]['median']:.3f} ({b[k]['ci'][0]:.3f}–{b[k]['ci'][1]:.3f}) |")
    dc = c["burst_human_vs_aigc"]
    de = c["burst_human_vs_exploring"]
    lines += [
        "",
        f"人类帖的句长起伏显著更强：human vs AIGC Cliff's δ={dc['cliffs_delta']:+.3f} "
        f"(p={dc['p']:.1e})，human vs 探索集 δ={de['cliffs_delta']:+.3f} (p={de['p']:.1e})。"
        "平均句长之外，长短句交错程度同样是稳定的风格信号。",
        "",
        "## 2 词汇多样性（MTLD）与 Heaps 定律（抽样）",
        "",
        f"| 子集 | 参与帖子 | MTLD 中位数(95%CI) | Heaps β（前 "
        f"{out['sample']['heaps_token_budget'] // 1000}k token） | 曲线 token 数 |",
        "|---|---:|---|---:|---:|",
    ]
    for k in ("human", "aigc", "exploring"):
        lines.append(f"| {DATASET_LABELS[k]} | {m[k]['n_docs']:,} | "
                     f"{m[k]['median']:.1f} ({m[k]['ci'][0]:.1f}–{m[k]['ci'][1]:.1f}) | "
                     f"{h[k]['beta']:.3f} | {h[k]['n_tokens']:,} |")
    mc = c["mtld_human_vs_aigc"]
    me = c["mtld_human_vs_exploring"]
    lines += [
        "",
        f"MTLD 差异：human vs AIGC δ={mc['cliffs_delta']:+.3f} (p={mc['p']:.1e})，"
        f"human vs 探索集 δ={me['cliffs_delta']:+.3f} (p={me['p']:.1e})。"
        "AIGC 每帖用词更丰富（MTLD 更高、独特字比率更高），但跨帖的词汇增长"
        "斜率更低（Heaps β 最小）——单帖内「词汇量大」与跨帖「模板复用多」并存，"
        "与 framing_analysis 的模板集中度结论一致。",
        "",
        "> 口径说明：Heaps 曲线按各子集统一的 token 预算截断后拟合，跨度可比；",
        "> MTLD 为逐帖指标（对文本长度稳健），比较的是「每帖」词汇丰富度，",
        "> 与 `paper_features` 的独特字/二元组比率结论方向一致（AIGC 更丰富）。",
        "",
        "图表见 `insights/`：`diversity_burstiness.png`、`diversity_lexical.png`。",
    ]
    with open(BASE + "diversity_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
