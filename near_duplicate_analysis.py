# -*- coding: utf-8 -*-
"""深潜分析之十：近重复内容与模板复用（字符 5-gram Jaccard）。

既有重复内容分析（`deep_dive_analysis.py`）只覆盖「完全重复」：探索集未去重时
68.11% 的记录正文重复、去重到 note_id 后降到 1.52%。但逐字节一致的副本只是
模板复用的一部分——同一模板的改写、微调、换词版本仍然是重复劳动，却无法被
精确匹配捕捉。本脚本对三个子集做全量「近重复」扫描，回答：近重复有多少、
集中在哪些子集/模型家族/时段、是否与「AI 味」得分相关、是否存在跨子集的
模板迁移（如生成集与人类集的改写关系）。

方法（可复算、只用 numpy，全无外部依赖）：

1. 正文清洗复用 `analysis_utils.clean_body`（与 paper_features.py 一致），
   对清洗后正文做**字符 5-gram 的 64 位向量化滚动哈希**（多项式 + splitmix
   终混，逐字符窗口由 numpy 一次算出）；每篇文档取排序去重哈希集合。
   清洗后不足 `MIN_SHINGLES=20` 个 unique 5-gram 的「空正文/纯标签」帖
   不参与匹配（探索集中大量正文只剩 `#` 与空白的记录，若参与会以空集
   互相完全匹配、污染所有比率），单独计数报出。
2. 每篇文档再取最小的 K=128 个哈希值作为 **bottom-k 概要**（locality
   sensitive：高 Jaccard 的文档大概率共享多个 bottom-k 哈希）。
3. 候选对生成：把全局文档频率 df 在 [2, 50] 的概要哈希当作「链接键」
   （df=1 的哈希不可能成对；df>50 的哈希是常见短语、无鉴别力且会产生组合
   爆炸），同一链接键下的文档两两成为候选对；只对「共享 ≥2 个链接键」或
   「共享 1 个键但任一侧 unique 5-gram 数 ≤ K（概要即完整集合）」的候选对
   做精确校验。
4. 精确校验：对候选对用完整 5-gram 集合（排序去重 uint64 数组）计算
   J = |A∩B| / |A∪B|（64 位哈希空间，理论碰撞概率可忽略；集合本身也已
   去重，同帖内重复片段不影响 J）。J ≥ 0.5 记为近重复对，另报告 0.7/0.9。
5. 召回自检：随机抽 `RECALL_SAMPLE=1000` 篇文档做暴力基准——对全库逐篇计算
   「与样本文档共享的 5-gram 数」，取出所有真实 J ≥ 0.5 的配对，检验第 3 步
   候选生成是否漏检；召回率与样本量写入 meta，保证方法透明度。

口径与对齐：
- 人类/探索集按 note_id 去重（保留元数据最完整的一条，规则同
  `symbol_analysis.py`；AIGC 无 note_id 逐记录处理）；
- 探索集 p_ai 按行序对齐 `paper_exploring_scores.csv.gz`（逐条 content_len
  校验，不一致计入 mismatch）；AIGC 无时间字段，季度/十分位只统计可解析的文档；
- 清洗后不足 `MIN_SHINGLES=20` 个 unique 5-gram（约 24 字符正文）的
  「空正文/纯标签」帖不参与匹配，按子集单独计数报出。

输出：
  insights/near_duplicate.png    max-J 生存曲线 / 阈值占比 / 跨子集矩阵 /
                                  P(AI) 梯度 / 季度趋势 / 重复簇规模
  near_duplicate_results.json    全部数值结果
  near_duplicate_results.md      结果摘要
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from analysis_utils import (BASE, INSIGHTS, PALETTE, init_plot_style, save_fig,
                            save_json, wilson_ci, clean_body, content_of,
                            engagement_of, iter_jsonl, parse_local_time)

DATASETS = {
    "human": BASE + "training_set_human_fixed.jsonl",
    "aigc": BASE + "training_set_aigc.jsonl",
    "exploring": BASE + "exploring_set_fixed.jsonl",
}
LABELS = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
          "exploring": "探索集(后LLM)"}
SCORES_PATH = BASE + "paper_exploring_scores.csv.gz"

SHINGLE_N = 5           # 字符 n-gram 宽度
SKETCH_K = 128          # bottom-k 概要长度
DF_CAP = 50             # 链接键的最大文档频率（超过视为常见短语）
MIN_SHINGLES = 20       # 参与匹配的最少 unique 5-gram 数（≈ 24 字符正文）
THRESHOLDS = (0.5, 0.7, 0.9)
MIN_J = THRESHOLDS[0]   # 校验阶段保留的最低 Jaccard
GRADIENT_J = 0.7        # 梯度/季度趋势使用的阈值
N_DECILES = 10
QUARTER_MIN_N = 300     # 季度曲线最少样本
RECALL_SAMPLE = 1000    # 召回自检的文档数
RECALL_SEED = 42
MAX_CROSS_EXAMPLES = 10

_MIX1 = np.uint64(0x9E3779B185EBCA87)
_MIX2 = np.uint64(0xC2B2AE3D27D4EB4F)


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def shingle_hashes(text: str, n: int = SHINGLE_N) -> np.ndarray:
    """字符 n-gram 的 64 位滚动哈希（返回排序去重后的 uint64 数组）。

    连续 n 个 Unicode 码点组成一个窗口；窗口哈希为多项式滚动哈希再做
    splitmix 风格终混，全部由 numpy 向量化计算。文本短于 n 时返回空数组。
    """
    if not text or len(text) < n:
        return np.empty(0, dtype=np.uint64)
    codes = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    m = codes.size - n + 1
    h = codes[:m].copy()
    for j in range(1, n):
        h *= _MIX1
        h += codes[j:j + m]
    h ^= h >> np.uint64(30)
    h *= _MIX2
    h ^= h >> np.uint64(27)
    h *= _MIX1
    h ^= h >> np.uint64(31)
    return np.unique(h)


def bottom_k(hashes: np.ndarray, k: int = SKETCH_K) -> np.ndarray:
    """取最小值最小的 k 个哈希（输入需为排序去重数组，即直接切片）。"""
    return hashes[:k] if hashes.size > k else hashes


def jaccard_sorted(a: np.ndarray, b: np.ndarray) -> float:
    """两个「排序去重」uint64 集合数组的精确 Jaccard；任一为空返回 0。"""
    la, lb = a.size, b.size
    if la == 0 or lb == 0:
        return 0.0
    if la > lb:
        a, b, la, lb = b, a, lb, la
    pos = np.searchsorted(a, b)
    np.clip(pos, 0, la - 1, out=pos)
    inter = int(np.count_nonzero(a[pos] == b))
    return inter / (la + lb - inter)


def candidate_pair_keys(sketches: np.ndarray, n_docs: int,
                        cap: int = DF_CAP) -> tuple[np.ndarray, np.ndarray]:
    """bottom-k 概要 + 文档频率过滤生成候选对。

    返回 (keys, counts)：key = i * n_docs + j（i < j）编码的候选对（int64，
    升序去重）与每个候选对共享的链接键个数。sketch 中 0 视为空槽（哈希值
    恰好为 0 的概率可忽略，且 0 只会让该槽失去链接能力，不影响正确性）。

    全 numpy 实现：把 (文档, 概要哈希) 展平后稳定排序，按哈希值分组
    （组内文档下标仍升序），对 df ∈ [2, cap] 的组用预计算三角索引批量
    生成 pair key。生成的多重集与逐对 Python 循环完全一致，因此
    `np.unique(..., return_counts=True)` 返回的 keys/counts 不变。
    """
    flat = sketches.ravel()
    idx = np.flatnonzero(flat)
    if idx.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    # C 顺序展平：flat_idx = doc * K + 槽位，故 doc = flat_idx // K，
    # 与 np.repeat(arange(n_docs), K)[idx] 完全一致但省下 13M 元素的临时数组。
    docs = (idx // sketches.shape[1]).astype(np.int64)
    hv = flat[idx]
    order = np.argsort(hv, kind="stable")
    docs, hv = docs[order], hv[order]
    starts = np.concatenate(([0], np.flatnonzero(hv[1:] != hv[:-1]) + 1))
    ends = np.concatenate((starts[1:], [hv.size]))
    df = ends - starts
    keep = np.flatnonzero((df >= 2) & (df <= cap))
    if keep.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    tri = {int(m): np.triu_indices(int(m), 1)
           for m in np.unique(df[keep])}
    parts = []
    for g in keep.tolist():
        start, end = int(starts[g]), int(ends[g])
        grp = docs[start:end]
        tri_i, tri_j = tri[end - start]
        parts.append(grp[tri_i] * n_docs + grp[tri_j])
    keys_arr = np.concatenate(parts)
    return np.unique(keys_arr, return_counts=True)


def select_verify_pairs(keys: np.ndarray, counts: np.ndarray,
                        lengths: np.ndarray, n_docs: int,
                        k: int = SKETCH_K) -> np.ndarray:
    """筛选值得精确校验的候选对（多次独立链接证据或短文档单键证据）。

    过滤逻辑本身不产生漏检风险：被丢弃的只是「共享 1 个链接键、且两侧文档
    都长于 K 个 5-gram」的弱候选——此时 1 个共享键只是 300+ 个 shingle 里的
    1 个，几乎不可能达到 J ≥ 0.5（召回自检在真实数据上复核这一点）。
    """
    if keys.size == 0:
        return keys
    a = (keys // n_docs).astype(np.int64)
    b = (keys % n_docs).astype(np.int64)
    keep = (counts >= 2) | ((counts == 1)
                            & (np.minimum(lengths[a], lengths[b]) <= k))
    return keys[keep]


def verified_pairs(keys: np.ndarray, counts: np.ndarray, sets: list[np.ndarray],
                   min_j: float = MIN_J, k: int = SKETCH_K,
                   lengths: np.ndarray | None = None):
    """候选对精确校验：返回 J ≥ min_j 的 (相似度, i, j) 列表（降序）。

    候选对的「共享链接键数 ≥ 2 或短文档单键」筛选见 `select_verify_pairs`；
    其余候选对不校验（召回自检会在真实数据上复核漏检率）。
    `lengths` 可传入已算好的各文档 unique 5-gram 数以避免重复计算。
    """
    n_docs = len(sets)
    if lengths is None:
        lengths = np.fromiter((s.size for s in sets), dtype=np.int64,
                              count=n_docs)
    sel = select_verify_pairs(keys, counts, lengths, n_docs, k)
    if sel.size:
        la = lengths[sel // n_docs]
        lb = lengths[sel % n_docs]
        min_len = np.minimum(la, lb)
        # J ≤ min_len / (la + lb - min_len)：达不到 min_j 的对直接跳过
        # （数学上不可能命中，结果与逐对精确校验完全一致）。
        sel = sel[(1.0 + min_j) * min_len >= min_j * (la + lb)]
    pairs: list[tuple[float, int, int]] = []
    for key in sel.tolist():
        i, j = divmod(key, n_docs)
        sim = jaccard_sorted(sets[i], sets[j])
        if sim >= min_j:
            pairs.append((sim, i, j))
    pairs.sort(reverse=True)
    return pairs


def within_max_j(dataset: np.ndarray, pairs, n_docs: int) -> np.ndarray:
    """每篇文档在「同子集内」的最大近重复 J（跨子集配对标为 0，不计入）。"""
    out = np.zeros(n_docs)
    for j, a, b in pairs:
        if dataset[a] == dataset[b]:
            if j > out[a]:
                out[a] = j
            if j > out[b]:
                out[b] = j
    return out


def connected_components(n_docs: int, edges) -> list[list[int]]:
    """并查集求连通分量，返回按大小降序的节点列表（单点不返回）。"""
    parent = list(range(n_docs))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    nodes: set[int] = set()
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
        nodes.add(a)
        nodes.add(b)
    comp: dict[int, list[int]] = defaultdict(list)
    for x in nodes:
        comp[find(x)].append(x)
    return sorted((sorted(v) for v in comp.values()), key=len, reverse=True)


def decile_rates(values: np.ndarray, indicator: np.ndarray,
                 n_deciles: int = N_DECILES) -> pd.DataFrame:
    """按 values 的十分位统计 indicator 占比与 Wilson 95% 置信区间。"""
    df = pd.DataFrame({"v": values, "hit": indicator.astype(bool)})
    df["decile"] = pd.qcut(df["v"], n_deciles, labels=False, duplicates="drop")
    rows = []
    for d, g in df.groupby("decile"):
        k = int(g["hit"].sum())
        lo, hi = wilson_ci(k, len(g))
        rows.append({
            "decile": int(d), "n": len(g), "hits": k,
            "share_pct": round(float(g["hit"].mean()) * 100, 3),
            "ci_low_pct": round(lo * 100, 3), "ci_high_pct": round(hi * 100, 3),
            "v_lo": float(g["v"].min()), "v_hi": float(g["v"].max()),
        })
    cols = ["decile", "n", "hits", "share_pct", "ci_low_pct", "ci_high_pct",
            "v_lo", "v_hi"]
    return pd.DataFrame(rows, columns=cols).set_index("decile")


# --------------------------------------------------------------------------
# 扫描与对齐
# --------------------------------------------------------------------------
def scan_dataset(key: str, path: str) -> dict:
    """流式扫描单数据集：清洗正文、按 note_id 去重、收集辅助字段。

    去重规则与 `symbol_analysis.scan` 完全一致：同名笔记保留 meta 分数最高
    （并列取最早）的一条；无 note_id 的记录全部保留；最终按原始行序输出。
    返回 docs 列表 (raw_idx, body, aux) 与原始记录数；aux 为
    (model_family, model, month)（AIGC 有模型名、无时间；人类两者皆无）。
    """
    best: dict = {}
    raw: list = []
    n_raw = 0
    month_pool: dict = {}
    for i, rec in enumerate(iter_jsonl(path)):
        n_raw += 1
        body = clean_body(content_of(rec))
        likes, colls, comms = engagement_of(rec)
        dt = parse_local_time(rec.get("local_time"))
        has_meta = (bool(rec.get("user_id"))
                    and rec.get("domain") not in (None, "", "未知"))
        meta = (int(has_meta) + int(bool(likes)) + int(bool(colls))
                + int(bool(comms)) + int(dt is not None) + int(bool(body)))
        if dt is None:
            month = ""
        else:
            ym = (dt.year, dt.month)
            month = month_pool.get(ym)
            if month is None:
                month = month_pool[ym] = f"{dt.year:04d}{dt.month:02d}"
        aux = (rec.get("model_family") or "", rec.get("model") or "", month)
        nid = rec.get("note_id")
        if nid is None:
            raw.append((i, body, aux))
        else:
            cur = best.get(nid)
            if cur is None or meta > cur[0]:
                best[nid] = (meta, i, body, aux)
    rows = [(i, body, aux) for _, i, body, aux in best.values()] + raw
    rows.sort()
    return {"path": path, "n_raw": n_raw, "docs": rows}


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def make_figure(out: dict, within_j: np.ndarray, dataset: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    ds_names = ("human", "aigc", "exploring")
    colors = [PALETTE[n] for n in ds_names]

    ax = axes[0][0]
    grid = np.linspace(0.5, 1.0, 101)
    for name, col in zip(ds_names, colors):
        vals = within_j[dataset == name]
        surv = [(vals >= t).mean() * 100 for t in grid]
        ax.plot(grid, surv, color=col, lw=1.8, label=LABELS[name])
    ax.set_yscale("log")
    ax.set_ylim(0.02, 100)
    ax.set_xlabel("Jaccard 相似度阈值 J")
    ax.set_ylabel("max-J ≥ J 的文档占比 (%，log)")
    ax.set_title("各子集内「最近邻」相似度生存曲线")
    ax.legend(fontsize=8.5)

    ax = axes[0][1]
    width = 0.26
    x = np.arange(len(ds_names))
    for t_i, thr in enumerate(THRESHOLDS):
        shares = [out["within"][n]["twin_share_pct"][str(thr)] for n in ds_names]
        bars = ax.bar(x + (t_i - 1) * width, shares, width,
                      color=["#B0B7C6", "#8172B3", "#4C72B0"][t_i],
                      label=f"J ≥ {thr}")
        for xi, v in zip(x + (t_i - 1) * width, shares):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in ds_names])
    ax.set_ylabel("有近重复孪生的文档占比 (%)")
    ax.set_title("近重复率（子集内，按阈值）")
    ax.legend(fontsize=8.5)

    ax = axes[0][2]
    mat = np.zeros((3, 3))
    for i, ni in enumerate(ds_names):
        for j, nj in enumerate(ds_names):
            key = f"{ni}-{nj}" if ni <= nj else f"{nj}-{ni}"
            mat[i, j] = out["cross"]["pairs"]["0.5"].get(key, 0)
    im = ax.imshow(np.log10(mat + 1), cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center",
                    fontsize=9, color="black" if mat[i, j] < 300 else "white")
    ax.set_xticks(range(3), [LABELS[n] for n in ds_names], fontsize=8.5)
    ax.set_yticks(range(3), [LABELS[n] for n in ds_names], fontsize=8.5)
    ax.set_title("J ≥ 0.5 近重复对计数（含子集内）")
    ax.grid(False)

    ax = axes[1][0]
    table = out["exploring_gradient"]
    dec = [r["decile"] for r in table]
    share = [r["share_pct"] for r in table]
    lo = [r["share_pct"] - r["ci_low_pct"] for r in table]
    hi = [r["ci_high_pct"] - r["share_pct"] for r in table]
    ax.errorbar(dec, share, yerr=[lo, hi], fmt="o-", color=PALETTE["exploring"],
                capsize=3, lw=1.6, ms=5)
    base = out["within"]["exploring"]["twin_share_pct"][str(GRADIENT_J)]
    ax.axhline(base, color="#8172B3", ls="--", lw=1.2,
               label=f"总体 {base:.2f}%")
    ax.set_xticks(dec, [f"D{d + 1}" for d in dec])
    ax.set_xlabel("P(AI) 十分位（D1 最像人 → D10 最像 AI）")
    ax.set_ylabel(f"有近重复孪生 (J ≥ {GRADIENT_J}) 的文档占比 (%)")
    sp = out["exploring_gradient_spearman"]
    ax.set_title(f"探索集 P(AI) 梯度（Spearman ρ={sp['rho']:.3f}, p={sp['p']:.3f}）")
    ax.legend(fontsize=8.5)

    ax = axes[1][1]
    q = out["quarterly"]
    xs = np.arange(len(q))
    for thr, col, marker in (("0.5", "#B0B7C6", "s"),
                             ("0.7", PALETTE["exploring"], "o")):
        ax.plot(xs, [r[f"share_{thr}_pct"] for r in q], marker=marker, color=col,
                lw=1.6, ms=4, label=f"J ≥ {thr}")
    step = max(1, len(q) // 8)
    ax.set_xticks(xs[::step], [q[i]["quarter"] for i in range(0, len(q), step)],
                  rotation=45, fontsize=8)
    spq = out["quarterly_spearman"]["0.7"]
    ax.set_ylabel("有近重复孪生的文档占比 (%)")
    ax.set_title(f"探索集季度趋势（J ≥ 0.7: ρ={spq['rho']:.3f}, p={spq['p']:.3f}）")
    ax.legend(fontsize=8.5)

    ax = axes[1][2]
    hist = out["clusters"]["size_hist"]
    labs = list(hist.keys())
    vals = [hist[k] for k in labs]
    ax.bar(labs, vals, color=PALETTE["aigc"])
    for xi, v in enumerate(vals):
        ax.text(xi, v + max(vals) * 0.01, f"{v}", ha="center", va="bottom",
                fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("近重复簇规模（J ≥ 0.7 连通分量）")
    ax.set_ylabel("簇数量 (log)")
    ax.set_title(f"近重复簇规模分布（共 {out['clusters']['n_components']} 簇）")

    fig.suptitle("近重复内容与模板复用：字符 5-gram Jaccard 全量扫描"
                 "（人类/探索集按 note_id 去重）", fontsize=13, y=1.0)
    save_fig(fig, INSIGHTS + "near_duplicate.png")


def md_summary(out: dict) -> str:
    w = out["within"]
    exc = out["meta"]["excluded_short_by_dataset"]
    lines = [
        "# 近重复内容与模板复用（字符 5-gram Jaccard）",
        "",
        f"> 方法：{SHINGLE_N}-gram 滚动哈希 + bottom-k（K={SKETCH_K}）候选生成 + "
        f"完整集合精确 Jaccard；链接键 df ∈ [2, {DF_CAP}]。",
        f"> 人类/探索集按 note_id 去重；探索集 p_ai 按行序对齐 "
        f"`paper_exploring_scores.csv.gz`（content_len 校验 "
        f"mismatch={out['meta']['exploring_content_len_mismatch']}）。",
        f"> 清洗后不足 {out['meta']['min_shingles']} 个 unique 5-gram 的"
        f"「空正文/纯标签」帖（human {exc['human']:,}、aigc {exc['aigc']:,}、"
        f"exploring {exc['exploring']:,}，合计 {out['meta']['n_docs_excluded_short']:,} 篇）"
        "不参与匹配：其正文只剩 `#` 与空白，若参与会以空集互相完全匹配、污染比率。",
        f"> 召回自检（{out['meta']['recall']['n_sample']} 篇抽样暴力基准，"
        f"J ≥ 0.5）：真实对 {out['meta']['recall']['true_pairs']}，"
        f"候选生成找到 {out['meta']['recall']['found_pairs']}，"
        f"召回率 {out['meta']['recall']['recall_pct']}%。",
        "",
        "## 1 子集内近重复率",
        "",
        "| 子集 | 文档数 | J ≥ 0.5 | J ≥ 0.7 | J ≥ 0.9 | J = 1（同 shingle 集） | 子集内 max-J 中位数 |",
        "|------|-------:|--------:|--------:|--------:|----------------------:|--------------------:|",
    ]
    for name in ("human", "aigc", "exploring"):
        r = w[name]
        lines.append(
            f"| {LABELS[name]} | {r['n_docs']:,} | "
            f"{r['twin_share_pct']['0.5']:.2f}% | {r['twin_share_pct']['0.7']:.2f}% | "
            f"{r['twin_share_pct']['0.9']:.2f}% | {r['twin_share_pct']['1.0']:.2f}% | "
            f"{r['max_j_quantiles']['0.5']:.3f} |")
    lines += [
        "",
        f"- 人类 {w['human']['n_docs']:,} 篇（原始 {out['meta']['datasets']['human']['n_raw']:,} 条，"
        f"note_id 去重后）中 {w['human']['twin_share_pct']['0.7']:.2f}% 存在 J ≥ 0.7 的孪生；",
        f"- AIGC {w['aigc']['n_docs']:,} 篇中 {w['aigc']['twin_share_pct']['0.7']:.2f}%；",
        f"- 探索集 {w['exploring']['n_docs']:,} 篇中 {w['exploring']['twin_share_pct']['0.7']:.2f}%"
        f"（J ≥ 0.9 仍有 {w['exploring']['twin_share_pct']['0.9']:.2f}%，说明近重复以"
        f"「近似改写」为主：同 shingle 集的整帖重复仅 "
        f"{w['exploring']['twin_share_pct']['1.0']:.2f}%）。",
        "",
        "## 2 跨子集近重复（模板迁移）",
        "",
        "| 组合 | J ≥ 0.5 对数 | J ≥ 0.7 对数 | 一侧文档占比（J ≥ 0.5） |",
        "|------|------------:|------------:|------------------------:|",
    ]
    for combo, r in out["cross"]["combos"].items():
        lines.append(f"| {combo} | {r['pairs_0.5']} | {r['pairs_0.7']} | "
                     f"{r['doc_share_0.5_pct']:.2f}% |")
    lines += [
        "",
        f"生成集与人类集之间存在 {out['cross']['combos']['aigc-human']['pairs_0.5']} 对 J ≥ 0.5、"
        f"{out['cross']['combos']['aigc-human']['pairs_0.7']} 对 J ≥ 0.7 的近重复对，"
        "且多为「同模板换日期/换词」的改写（见 JSON examples），提示生成集的构造"
        "可能以真实人类帖为底本；生成集与探索集之间只有少量（"
        f"{out['cross']['combos']['aigc-exploring']['pairs_0.5']} 对 ≥ 0.5）。",
        "",
        "## 3 探索集 P(AI) 梯度",
        "",
        "| 十分位 | P(AI) 范围 | 文档数 | 近重复占比 (J ≥ 0.7) |",
        "|--------|-----------|-------:|---------------------:|",
    ]
    for r in out["exploring_gradient"]:
        lines.append(f"| D{r['decile'] + 1} | {r['v_lo']:.3f}–{r['v_hi']:.3f} | "
                     f"{r['n']:,} | {r['share_pct']:.2f}% |")
    sp = out["exploring_gradient_spearman"]
    lines += [
        "",
        f"Spearman（十分位 × 占比）ρ={sp['rho']:.3f}（p={sp['p']:.3f}）："
        "近重复率沿 P(AI) 十分位基本持平，说明「近重复/模板复用」与检测器的 "
        "AI 味风格分基本正交——高 AI 味并不主要来自复制粘贴或套模板。",
        "",
        "## 4 AIGC 家族近重复率",
        "",
        "| 家族 | 文档数 | 子集内近重复占比 (J ≥ 0.7) | 与人类集跨子集占比 (J ≥ 0.5) |",
        "|------|-------:|---------------------------:|------------------------------:|",
    ]
    for fam, r in out["aigc_family"].items():
        lines.append(f"| {fam} | {r['n']:,} | {r['twin_share_pct']:.2f}% | "
                     f"{r['cross_human_share_pct']:.2f}% |")
    top_fam = max(out["aigc_family"].items(),
                  key=lambda kv: kv[1]["cross_human_share_pct"])
    lines += [
        "",
        f"生成集内部无自重复（子集内 0.00%）；跨子集近重复集中在 {top_fam[0]}"
        f"（{top_fam[1]['cross_human_share_pct']:.2f}% 的帖子与人类帖 J ≥ 0.5），"
        "其余家族接近 0。",
        "",
        "## 5 季度趋势（探索集）",
        "",
        "| 季度 | 文档数 | J ≥ 0.5 | J ≥ 0.7 |",
        "|------|-------:|--------:|--------:|",
    ]
    for r in out["quarterly"]:
        lines.append(f"| {r['quarter']} | {r['n']:,} | {r['share_0.5_pct']:.2f}% | "
                     f"{r['share_0.7_pct']:.2f}% |")
    spq = out["quarterly_spearman"]["0.7"]
    lines += [
        "",
        f"季度趋势 Spearman ρ={spq['rho']:.3f}（p={spq['p']:.3f}）。",
        "",
        "## 6 近重复簇结构",
        "",
        f"J ≥ 0.7 连通分量共 {out['clusters']['n_components']} 个，最大簇规模 "
        f"{out['clusters']['largest'][0]['size']}（{out['clusters']['largest'][0]['sample'][:40]}…）。",
        "",
        "| 簇规模 | 簇数 |",
        "|-------:|-----:|",
    ]
    for size, cnt in out["clusters"]["size_hist"].items():
        lines.append(f"| {size} | {cnt} |")
    lines += [
        "",
        "图表：`insights/near_duplicate.png`。",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 召回自检与主流程
# --------------------------------------------------------------------------
def recall_check(sets: list[np.ndarray], pairs, n_sample: int = RECALL_SAMPLE,
                 seed: int = RECALL_SEED, min_j: float = MIN_J) -> dict:
    """采样暴力基准，估计候选生成对真实 J ≥ min_j 对的召回率。

    随机抽 n_sample 篇文档，对全库逐篇计算与其共享的 5-gram 数（对样本
    shingle 建反向索引 + one-hot 位掩码累加），得到所有涉及样本文档的真实
    J ≥ min_j 对；再与主流程验证出的对求交集。位掩码把单篇文档的匹配
    向量化，全量扫描只增加一次 O(全库 shingle 数) 的 numpy 搜索，代价可控。
    """
    n_docs = len(sets)
    if n_docs == 0:
        return {"n_sample": 0, "true_pairs": 0, "found_pairs": 0,
                "recall_pct": 0.0}
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(n_docs, size=min(n_sample, n_docs),
                                replace=False))
    sizes = np.asarray([sets[i].size for i in sample], dtype=np.int64)
    nonempty = sample[sizes >= MIN_SHINGLES]
    sizes = sizes[sizes >= MIN_SHINGLES]
    if nonempty.size == 0:
        return {"n_sample": 0, "true_pairs": 0, "found_pairs": 0,
                "recall_pct": 0.0}

    offs = np.cumsum(np.concatenate(([0], sizes)))
    uniq_h, inverse = np.unique(np.concatenate([sets[i] for i in nonempty]),
                                return_inverse=True)
    mask = np.zeros((uniq_h.size, nonempty.size), dtype=np.uint8)
    for s_pos in range(nonempty.size):
        mask[inverse[offs[s_pos]:offs[s_pos + 1]], s_pos] = 1

    true_pairs: set[tuple[int, int]] = set()
    for d in range(n_docs):
        h = sets[d]
        if h.size < MIN_SHINGLES:
            continue
        pos = np.searchsorted(uniq_h, h)
        np.clip(pos, 0, uniq_h.size - 1, out=pos)
        match = uniq_h[pos] == h
        if not match.any():
            continue
        shared = mask[pos[match]].sum(axis=0)
        nz = np.flatnonzero(shared)
        if nz.size == 0:
            continue
        inter = shared[nz].astype(np.int64)
        j = inter / (h.size + sizes[nz] - inter)
        for s_pos in nz[j >= min_j]:
            s = int(nonempty[s_pos])
            if s == d:
                continue
            true_pairs.add((min(d, s), max(d, s)))

    in_sample = set(int(x) for x in sample)
    found = {(min(a, b), max(a, b)) for _, a, b in pairs
             if a in in_sample or b in in_sample}
    hit = len(true_pairs & found)
    return {
        "n_sample": int(nonempty.size),
        "min_j": min_j,
        "true_pairs": len(true_pairs),
        "found_pairs": hit,
        "recall_pct": round(hit / len(true_pairs) * 100, 3) if true_pairs else 100.0,
    }


def collect_stats(docs: list, dataset: np.ndarray, within_j: np.ndarray,
                  pairs, n_shingles: np.ndarray, scores: pd.DataFrame) -> dict:
    """docs 每项为 (dataset, raw_idx, body_snippet[:60], body_len, aux)。"""
    ds_names = ("human", "aigc", "exploring")
    n_docs = len(docs)
    lengths = np.fromiter((d[3] for d in docs), dtype=np.int64, count=n_docs)

    within: dict = {}
    for name in ds_names:
        m = dataset == name
        vals = within_j[m]
        within[name] = {
            "n_docs": int(m.sum()),
            "twin_share_pct": {
                str(t): round(float((vals >= t).mean()) * 100, 3)
                for t in (*THRESHOLDS, 1.0)},
            "max_j_quantiles": {
                "0.5": round(float(np.median(vals)), 4),
                "0.9": round(float(np.quantile(vals, 0.9)), 4),
                "0.99": round(float(np.quantile(vals, 0.99)), 4)},
        }

    # 跨子集（含子集内对角线）对数与「涉及近重复对的文档占比」
    combos: dict = {}
    for a_name in ds_names:
        for b_name in ds_names:
            if a_name > b_name:
                continue
            combos[f"{a_name}-{b_name}"] = {
                "pairs_0.5": 0, "pairs_0.7": 0, "doc_share_0.5_pct": 0.0}
    pair_counts: dict[str, dict[str, int]] = {
        str(t): {k: 0 for k in combos} for t in THRESHOLDS}
    touched: dict = {n: set() for n in ds_names}          # 子集内近重复文档
    cross_touched: dict = {n: set() for n in ds_names}    # 跨子集近重复文档
    for j, a, b in pairs:
        na, nb = sorted((str(dataset[a]), str(dataset[b])))
        key = f"{na}-{nb}"
        for t in THRESHOLDS:
            if j >= t:
                pair_counts[str(t)][key] += 1
        if j >= 0.5:
            if na == nb:
                touched[na].update((a, b))
            else:
                cross_touched[na].add(a)
                cross_touched[nb].add(b)
    for key, rec in combos.items():
        na, nb = key.split("-")
        rec["pairs_0.5"] = pair_counts["0.5"][key]
        rec["pairs_0.7"] = pair_counts["0.7"][key]
        if na == nb:
            share = len(touched[na]) / within[na]["n_docs"] * 100
        else:
            denom = within[na]["n_docs"] + within[nb]["n_docs"]
            share = ((len(cross_touched[na]) + len(cross_touched[nb]))
                     / denom * 100) if denom else 0.0
        rec["doc_share_0.5_pct"] = round(share, 3)

    # 探索集 p_ai 梯度
    exp_idx = np.flatnonzero(dataset == "exploring")
    raw_idx = np.asarray([docs[i][1] for i in exp_idx], dtype=np.int64)
    p_ai = np.full(exp_idx.size, np.nan)
    clen_all = scores["content_len"].to_numpy(dtype=np.int64)
    p_ai_all = scores["p_ai"].to_numpy(dtype=float)
    ok = raw_idx < p_ai_all.size
    p_ai[ok] = p_ai_all[raw_idx[ok]]
    mismatch = int((~ok).sum()) + int(
        (lengths[exp_idx[ok]] != clen_all[raw_idx[ok]]).sum())
    valid = ~np.isnan(p_ai)
    dec = decile_rates(p_ai[valid], within_j[exp_idx][valid] >= GRADIENT_J)
    rho, p_val = spearmanr(dec.index.to_numpy(), dec["share_pct"].to_numpy())

    # AIGC 家族：子集内近重复 + 与人类集跨子集近重复
    fam = np.asarray([d[4][0] for d in docs])
    aigc_family = {}
    a_m = dataset == "aigc"
    cross_human: set[int] = set()
    for j, a, b in pairs:
        da, db = str(dataset[a]), str(dataset[b])
        if j >= 0.5 and {da, db} == {"aigc", "human"}:
            cross_human.add(a if da == "aigc" else b)
    for f in sorted(set(fam[a_m])):
        m = a_m & (fam == f)
        aigc_family[str(f)] = {
            "n": int(m.sum()),
            "twin_share_pct": round(float((within_j[m] >= GRADIENT_J).mean()) * 100, 3),
            "cross_human_share_pct": round(
                float(sum(1 for i in cross_human if fam[i] == f)) / max(int(m.sum()), 1) * 100, 3)}

    # 季度趋势
    quarters = [d[4][2] for d in docs]
    qmap: dict[str, list[int]] = defaultdict(list)
    for i in exp_idx.tolist():
        mm = quarters[i]
        if len(mm) >= 6 and mm[:6].isdigit():
            q = f"{mm[:4]}Q{(int(mm[4:6]) - 1) // 3 + 1}"
            qmap[q].append(i)
    quarterly = []
    for q in sorted(qmap):
        idx = np.asarray(qmap[q])
        if idx.size < QUARTER_MIN_N:
            continue
        quarterly.append({
            "quarter": q, "n": int(idx.size),
            "share_0.5_pct": round(float((within_j[idx] >= 0.5).mean()) * 100, 3),
            "share_0.7_pct": round(float((within_j[idx] >= 0.7).mean()) * 100, 3)})
    if len(quarterly) >= 3:
        rho_q, p_q = spearmanr(range(len(quarterly)),
                               [r["share_0.7_pct"] for r in quarterly])
        quarterly_spearman = {"0.7": {"rho": round(float(rho_q), 3),
                                      "p": float(p_q)}}
    else:
        quarterly_spearman = {"0.7": {"rho": 0.0, "p": 1.0}}

    # 近重复簇（J ≥ 0.7）
    edges = [(a, b) for j, a, b in pairs if j >= GRADIENT_J]
    comps = connected_components(n_docs, edges)
    hist_bins = [(2, 2), (3, 3), (4, 4), (5, 6), (7, 10), (11, 20),
                 (21, 10 ** 9)]
    size_hist = {}
    for lo, hi in hist_bins:
        label = f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi < 10 ** 9
                                          else f"{lo}+")
        size_hist[label] = sum(1 for c in comps if lo <= len(c) <= hi)
    largest = []
    for comp in comps[:8]:
        ds_count = {n: int((dataset[comp] == n).sum()) for n in ds_names}
        months = defaultdict(int)
        for i in comp:
            if quarters[i]:
                months[quarters[i]] += 1
        largest.append({
            "size": len(comp),
            "datasets": {k: v for k, v in ds_count.items() if v},
            "months": dict(sorted(months.items())[:6]),
            "sample": docs[comp[0]][2]})

    # 示例对
    def example(j: float, a: int, b: int) -> dict:
        return {
            "jaccard": round(float(j), 4),
            "a": {"dataset": str(dataset[a]), "n_shingles": int(n_shingles[a]),
                  "snippet": docs[a][2]},
            "b": {"dataset": str(dataset[b]), "n_shingles": int(n_shingles[b]),
                  "snippet": docs[b][2]}}

    top_pairs = [example(j, a, b) for j, a, b in pairs[:10]]
    cross_examples = [example(j, a, b) for j, a, b in pairs
                      if dataset[a] != dataset[b]][:MAX_CROSS_EXAMPLES]

    return {
        "within": within,
        "cross": {"pairs": pair_counts, "combos": combos},
        "exploring_gradient": dec.reset_index().to_dict("records"),
        "exploring_gradient_spearman": {"rho": round(float(rho), 3),
                                        "p": float(p_val)},
        "aigc_family": aigc_family,
        "quarterly": quarterly,
        "quarterly_spearman": quarterly_spearman,
        "clusters": {"n_components": len(comps), "size_hist": size_hist,
                     "largest": largest},
        "examples": {"top_pairs": top_pairs, "cross_pairs": cross_examples},
        "_exploring_mismatch": mismatch,
    }


def main() -> int:
    init_plot_style()
    scans = {name: scan_dataset(name, path) for name, path in DATASETS.items()}
    scan_meta = {name: {"n_raw": scans[name]["n_raw"],
                        "n_docs": len(scans[name]["docs"])} for name in scans}
    all_docs = [(name, raw_i, body, aux)
                for name in ("human", "aigc", "exploring")
                for raw_i, body, aux in scans[name]["docs"]]
    del scans  # 全量正文只保留 all_docs 一份引用

    all_sets = [shingle_hashes(d[2]) for d in all_docs]
    keep = [s.size >= MIN_SHINGLES for s in all_sets]
    excluded = {name: 0 for name in scan_meta}
    for d, k in zip(all_docs, keep):
        if not k:
            excluded[d[0]] += 1
    # 下游只用到正文前 60 字符与长度：换成短片段后释放全量正文，峰值内存大幅下降
    docs = [(d[0], d[1], d[2][:60], len(d[2]), d[3])
            for d, k in zip(all_docs, keep) if k]
    del all_docs
    sets = [s for s, k in zip(all_sets, keep) if k]
    del all_sets
    dataset = np.asarray([d[0] for d in docs])
    n_docs = len(docs)
    # 校验用「unique 5-gram 数」（原实现口径）；正文长度由 collect_stats 自行取
    shingle_counts = np.fromiter((s.size for s in sets), dtype=np.int64,
                                 count=n_docs)
    print(f"[近重复] 参与匹配 {n_docs:,} 篇（清洗后不足 {MIN_SHINGLES} 个 "
          f"5-gram、按「空正文/纯标签」剔除 {sum(excluded.values()):,} 篇）")

    sketches = np.zeros((n_docs, SKETCH_K), dtype=np.uint64)
    for idx, s in enumerate(sets):
        sketches[idx, :min(s.size, SKETCH_K)] = bottom_k(s)

    keys, counts = candidate_pair_keys(sketches, n_docs)
    del sketches
    n_verified = int(select_verify_pairs(keys, counts, shingle_counts,
                                         n_docs).size)
    print(f"[近重复] 候选对 {keys.size:,}（校验子集 {n_verified:,}）")
    pairs = verified_pairs(keys, counts, sets, lengths=shingle_counts)
    n_candidate_pairs = int(keys.size)
    del keys, counts
    within_j = within_max_j(dataset, pairs, n_docs)
    print(f"[近重复] J ≥ {MIN_J} 的对 {len(pairs):,}；召回自检中 ...")
    recall = recall_check(sets, pairs)
    print(f"[近重复] 召回自检: {recall['found_pairs']}/{recall['true_pairs']} "
          f"= {recall['recall_pct']}%")

    scores = pd.read_csv(SCORES_PATH, usecols=["p_ai", "content_len"])
    out = collect_stats(docs, dataset, within_j, pairs, shingle_counts, scores)
    mismatch = out.pop("_exploring_mismatch")
    out["meta"] = {
        "shingle_chars": SHINGLE_N, "sketch_k": SKETCH_K, "df_cap": DF_CAP,
        "min_shingles": MIN_SHINGLES,
        "thresholds": list(THRESHOLDS), "n_docs": n_docs,
        "n_docs_excluded_short": sum(excluded.values()),
        "excluded_short_by_dataset": excluded,
        "n_candidate_pairs": n_candidate_pairs,
        "n_verified_pairs": n_verified,
        "n_pairs_min_j": len(pairs),
        "datasets": scan_meta,
        "exploring_content_len_mismatch": mismatch,
        "recall": recall,
    }

    save_json(out, BASE + "near_duplicate_results.json")
    with open(BASE + "near_duplicate_results.md", "w", encoding="utf-8") as f:
        f.write(md_summary(out))
    make_figure(out, within_j, dataset)
    print("[近重复] 输出: insights/near_duplicate.png、"
          "near_duplicate_results.{json,md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
