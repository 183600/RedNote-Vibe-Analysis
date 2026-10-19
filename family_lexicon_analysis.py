# -*- coding: utf-8 -*-
"""模型家族词表指纹：17 个 LLM 各自偏爱哪些词。

paper_model_fingerprint.py 只量化了「与人类基线的数值距离」（表情、标点、
长度等 20 个特征），无法回答"这个模型到底爱说什么词"。本脚本在 AIGC 子集
内部做家族/单模型 one-vs-rest 的 informative Dirichlet log-odds（口径与
paper_lexicon.py 完全一致，同一套清洗/分词/先验强度），得到：

1. 每个厂商家族（GPT/Gemini/DeepSeek/Claude/GLM/Qwen）相对其他家族的
   高区分度词表（Top-25），以及词 × 家族的 z 分数热力图；
2. 每个具体模型（17 个）的 Top-12 特征词——补充 §5.2 风格指纹的语言学证据；
3. 共享"AI 味词表"（paper_features.AI_WORDS）在各家族中的使用密度
   （每万字符，按清洗后正文子串计数），与人类基线对照。

输出：
  insights/family_lexicon.png       家族词表热力图 + AI 味词使用密度
  family_lexicon_results.json       全部数值结果
  family_lexicon_results.md         结果摘要（词表表格）
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jieba

from analysis_utils import (BASE, INSIGHTS, content_of, init_plot_style,
                            save_fig, save_json, post_text,
                            make_pool, close_pool)
# AIGC 语料经共享派生缓存读取（只用 PROJECT_KEYS 内字段）；缓存缺失时
# 自动逐位回退到逐行 iter_jsonl，分词与统计口径不变。
from derived_cache import iter_cached
from paper_lexicon import clean, count_clean_words
from paper_features import AI_WORDS

FAMILY_NAME = {"gpt": "GPT", "gemini": "Gemini", "deepseek": "DeepSeek",
               "anthropic": "Claude", "glm": "GLM", "qwen": "Qwen"}
FAMILY_COLOR = {"GPT": "#4C72B0", "Gemini": "#C44E52", "DeepSeek": "#55A868",
                "Claude": "#8172B3", "GLM": "#DD8452", "Qwen": "#64B5CD"}
TOP_FAMILY = 25
TOP_MODEL = 12
TOP_HEATMAP = 8
MIN_COUNT_FAMILY = 15
MIN_COUNT_MODEL = 8
MIN_COUNT_SHARED = 20

NAMES = list(FAMILY_NAME.values())
FAMILY_KEY = {v: k for k, v in FAMILY_NAME.items()}


def rate_per10k(cnt: Counter, n: int, w: str) -> float:
    return round(cnt[w] / n * 1e4, 2) if n else 0.0


def _finalize_weighting(cell: dict[tuple[str, str], Counter],
                        cell_n: Counter, group_n: Counter,
                        dom_n: Counter) -> tuple[dict[str, Counter],
                                                 dict[str, Counter]]:
    """把 (组 × 领域) 单元计数合并为原始计数与领域重加权计数。

    家族/模型的用词差异可能被话题构成（如某家族多写美食）混杂；这里用
    后分层权重 w = 全语料领域占比 / 组内领域占比 对每个 (组, 领域) 的计数
    加权后再合并，使各组在同一领域分布下比较（与 paper_drift 的领域调整同思路）。
    """
    total = float(sum(dom_n.values()))
    raw: dict[str, Counter] = {}
    weighted: dict[str, Counter] = {}
    for (g, d), cnt in cell.items():
        raw.setdefault(g, Counter()).update(cnt)
        share_group = cell_n[(g, d)] / group_n[g]
        share_all = dom_n[d] / total
        w = share_all / share_group if share_group > 0 else 0.0
        wc = weighted.setdefault(g, Counter())
        for word, c in cnt.items():
            wc[word] += c * w
    return raw, weighted


def _doc_meta(doc: tuple[str, str]) -> tuple[float, Counter]:
    """worker：（清洗后「标题+正文」, 清洗后正文）→（单帖 AI 味词密度, 词频）。

    词频口径与 paper_lexicon 一致（清洗后标题+正文）；AI 味词密度口径与
    paper_features 的 ai_lex100 一致——只数清洗后正文子串，按 max(len,1)
    归一为每万字符（正文先 strip，与 paper_features.extract 的正文口径一致），
    因此可与人类基线的文档宏平均直接对照。
    """
    full, body = doc
    body = body.strip()
    hits = sum(body.count(w) for w in AI_WORDS)
    return (hits / max(len(body), 1) * 1e4, count_clean_words(full))


def collect_group_stats(records, pool=None, keys: tuple[str, ...] = (
        "model_family", "model")) -> tuple[
        dict[str, dict[str, Counter]], dict[str, dict[str, Counter]],
        Counter, Counter, int]:
    """单次遍历记录：一次清洗 + 一次分词，同时聚合多个分组维度的词频。

    `records` 可以是列表或任意可迭代（如 iter_jsonl 生成器），按批处理避免
    一次性持有全部记录。返回 (raw, weighted, ai_dens, ai_docs, n_records)：
    raw/weighted 按分组键索引到 {组: Counter}（口径与原先逐维度独立扫描完全
    一致）；ai_dens/ai_docs 按 `model_family` 累计「单帖 AI 味词密度之和」
    与帖数（正文口径，与 paper_features.ai_lex100 同源）。传入 pool 时按批
    并行分词，逐帖结果与串行完全一致。
    """
    cell: dict[str, dict[tuple[str, str], Counter]] = {key: {} for key in keys}
    cell_n: dict[str, Counter] = {key: Counter() for key in keys}
    group_n: dict[str, Counter] = {key: Counter() for key in keys}
    dom_n: dict[str, Counter] = {key: Counter() for key in keys}
    ai_dens: Counter = Counter()
    ai_docs: Counter = Counter()
    n_records = 0

    batch = 1024
    kept: list = []

    def flush() -> None:
        docs = [(clean(post_text(rec)), clean(content_of(rec)))
                for rec, _ in kept]
        metas = (pool.map(_doc_meta, docs) if pool is not None
                 else [_doc_meta(t) for t in docs])
        for (rec, by_key), (dens, tokens) in zip(kept, metas):
            fam = by_key.get("model_family") or ""
            if fam:
                ai_dens[fam] += dens
                ai_docs[fam] += 1
            d = rec.get("domain") or "未知"
            for key in keys:
                g = by_key[key]
                if not g:
                    continue
                k = (g, d)
                if k not in cell[key]:
                    cell[key][k] = Counter()
                cell[key][k].update(tokens)
                cell_n[key][k] += 1
                group_n[key][g] += 1
                dom_n[key][d] += 1
        kept.clear()

    for rec in records:
        n_records += 1
        by_key = {key: (rec.get(key) or "") for key in keys}
        if any(by_key.values()):
            kept.append((rec, by_key))
        if len(kept) >= batch:
            flush()
    if kept:
        flush()

    raw, weighted = {}, {}
    for key in keys:
        raw[key], weighted[key] = _finalize_weighting(
            cell[key], cell_n[key], group_n[key], dom_n[key])
    return raw, weighted, ai_dens, ai_docs, n_records


def one_vs_rest_all(counts: dict[str, Counter],
                    min_count: int) -> dict[str, dict[str, float]]:
    """一次算出所有分组的 one-vs-rest log-odds（口径同 paper_lexicon.log_odds）。

    原实现每个分组都要重建「其他组」Counter 并在整张并集词表上做 Python 级
    math.log 循环（模型分组时数万词 × 17 组）；这里把各分组计数对齐到同一
    全局词表、用 numpy 向量化求和与过滤，数值与逐词标量计算等价（仅浮点
    结合顺序在约 1e-12 相对量级上的差异，不影响按两位小数报告的结果）。
    """
    groups = list(counts)
    total: Counter = Counter()
    for g in groups:
        total.update(counts[g])
    vocab = list(total)
    if not vocab:
        return {g: {} for g in groups}
    pos = {w: i for i, w in enumerate(vocab)}
    n_total = sum(total.values())
    arrays = {}
    for g in groups:
        arr = np.zeros(len(vocab))
        for w, c in counts[g].items():
            arr[pos[w]] = c
        arrays[g] = arr
    tot = np.zeros(len(vocab))
    for g in groups:
        tot += arrays[g]

    alpha0 = 500.0
    log = np.log
    sqrt = np.sqrt
    out: dict[str, dict[str, float]] = {}
    for g in groups:
        n_a = sum(counts[g].values())
        n_b = n_total - n_a
        n_p = n_a + n_b
        ca = arrays[g]
        cb = tot - ca
        c = ca + cb
        idx = np.flatnonzero(c >= min_count)
        if idx.size == 0:
            out[g] = {}
            continue
        aa = ca[idx]
        bb = cb[idx]
        aw = alpha0 * (c[idx] / n_p)
        na0 = n_a + alpha0
        nb0 = n_b + alpha0
        la = log((aa + aw) / (na0 - aa - aw))
        lb = log((bb + aw) / (nb0 - bb - aw))
        z = (la - lb) / sqrt(1.0 / (aa + aw) + 1.0 / (bb + aw))
        out[g] = dict(zip([vocab[i] for i in idx], z.tolist()))
    return out


def word_rates(counts: dict[str, Counter], total: dict[str, int],
               words: list[str]) -> dict[str, dict[str, float]]:
    return {g: {w: rate_per10k(counts[g], total[g], w) for w in words}
            for g in counts}


def plot_family(fam_z: dict[str, dict[str, float]], fam_cnt: dict[str, Counter],
                fam_total: dict[str, int], ai_density: dict[str, float],
                human_density: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.4),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    # ---- 左：词 × 家族 z 分数热力图（每个家族 Top-8 词的并集去重）----
    ax = axes[0]
    words: list[str] = []
    fam_sorted = [f for f in NAMES if f in fam_z]
    seen: set[str] = set()
    for fam in fam_sorted:
        top = sorted(fam_z[fam].items(), key=lambda x: -x[1])[:TOP_HEATMAP]
        for w, _ in top:
            if w not in seen:
                seen.add(w)
                words.append(w)
    mat = np.zeros((len(words), len(fam_sorted)))
    for j, fam in enumerate(fam_sorted):
        for i, w in enumerate(words):
            mat[i, j] = fam_z[fam].get(w, 0.0)
    vmax = float(np.percentile(np.abs(mat), 99))
    vmax = max(vmax, 2.0)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(fam_sorted)))
    ax.set_xticklabels(fam_sorted, fontsize=10)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=6.5)
    ax.set_title(f"家族特征词 log-odds z 分数（每家族 Top-{TOP_HEATMAP} 词并集）")
    fig.colorbar(im, ax=ax, shrink=0.6, label="z（红=该家族偏好）")

    # ---- 右：AI 味词使用密度 ----
    ax = axes[1]
    fams = sorted(ai_density, key=ai_density.get)
    vals = [ai_density[f] for f in fams]
    ax.barh(fams, vals, color=[FAMILY_COLOR.get(f, "#888888") for f in fams])
    ax.axvline(human_density, color="#555555", ls="--", lw=1.2,
               label=f"人类基线 {human_density:.2f}")
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("AI 味词密度（每万字符，清洗后正文子串计数）")
    ax.set_title(f"共享 AI 味词表（{len(AI_WORDS)} 词）的家族使用密度")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    save_fig(fig, INSIGHTS + "family_lexicon.png")


def main() -> None:
    init_plot_style()
    pool = make_pool(preload=jieba.initialize)
    try:
        raw, weighted, ai_dens, ai_docs, n_records = collect_group_stats(
            iter_cached(BASE + "training_set_aigc.jsonl"), pool)
    finally:
        close_pool(pool)
    print(f"AIGC 记录数: {n_records}")
    fam_cnt, fam_wcnt = raw["model_family"], weighted["model_family"]
    model_cnt, model_wcnt = raw["model"], weighted["model"]
    fam_total = {g: sum(c.values()) for g, c in fam_cnt.items()}
    print("家族词元数:", {FAMILY_NAME.get(g, g): v for g, v in fam_total.items()})

    fam_z = {FAMILY_NAME.get(g, g): z for g, z in
             one_vs_rest_all(fam_wcnt, MIN_COUNT_FAMILY).items()}
    model_z = one_vs_rest_all(model_wcnt, MIN_COUNT_MODEL)

    family_top = {
        fam: [{"word": w, "z": round(s, 2),
               "per10k_family": rate_per10k(fam_cnt[FAMILY_KEY[fam]],
                                            fam_total[FAMILY_KEY[fam]], w)}
              for w, s in sorted(z.items(), key=lambda x: -x[1])[:TOP_FAMILY]]
        for fam, z in fam_z.items()
    }
    model_top = {
        m: [{"word": w, "z": round(s, 2)} for w, s in
            sorted(z.items(), key=lambda x: -x[1])[:TOP_MODEL]]
        for m, z in model_z.items()
    }

    # ---- 共享 AI 味词的家族密度（正文口径，按帖宏平均，同 paper_features）----
    ai_density = {}
    for g in ai_dens:
        ai_density[FAMILY_NAME.get(g, g)] = round(
            ai_dens[g] / max(ai_docs[g], 1), 3)

    import pandas as pd
    # paper_features 的 ai_lex100 是"每帖每百字命中数"（清洗后正文口径），
    # 文档宏平均 ×100 换算成"每万字符"，与上面的家族密度完全同口径
    human_density = float(pd.read_csv(
        BASE + "paper_features_human.csv.gz", usecols=["ai_lex100"])["ai_lex100"].mean()) * 100.0
    print(f"AI 味词密度（每万字符）: human={human_density:.2f}", ai_density)

    # 家族词表是否被话题构成混杂：家族 × 领域占比（%）
    aigc_feats = pd.read_csv(BASE + "paper_features_aigc.csv.gz",
                             usecols=["model_family", "domain"])
    ct = pd.crosstab(aigc_feats["model_family"], aigc_feats["domain"],
                     normalize="index") * 100
    domain_share = {FAMILY_NAME.get(f, f): {d: round(float(v), 2)
                                            for d, v in row.items()}
                    for f, row in ct.iterrows()}
    max_dev = float((ct.max(axis=0) - ct.min(axis=0)).max())

    plot_family(fam_z, fam_cnt, fam_total, ai_density, human_density)

    out = {
        "n_records": n_records,
        "family_token_counts": {FAMILY_NAME.get(g, g): v
                                for g, v in fam_total.items()},
        "family_top_words": family_top,
        "model_top_words": model_top,
        "ai_word_density_per10k": ai_density,
        "human_ai_word_density_per10k": round(human_density, 3),
        "family_domain_share_pct": domain_share,
        "max_domain_share_deviation_pp": round(max_dev, 2),
        "method": {"lexicon": "informative Dirichlet log-odds (Monroe et al. 2008)",
                   "tokenizer": "jieba",
                   "domain_reweighted": True,
                   "reweight_note": "各组 (family/model) 的领域构成按后分层权重"
                                    "重加权到全语料领域分布后再做 log-odds",
                   "top_family": TOP_FAMILY,
                   "top_model": TOP_MODEL, "min_count_family": MIN_COUNT_FAMILY,
                   "min_count_model": MIN_COUNT_MODEL},
    }
    save_json(out, BASE + "family_lexicon_results.json")
    write_markdown(out)
    print("已保存 family_lexicon_results.json/.md 与 insights/family_lexicon.png")


def write_markdown(out: dict) -> None:
    lines = [
        "# 模型家族词表指纹（AIGC 子集内部 one-vs-rest log-odds）",
        "",
        f"方法口径与 `paper_lexicon.py` 一致（jieba 分词、informative Dirichlet "
        f"log-odds、先验强度 500），并额外按后分层权重把各家族/各模型的领域构成"
        f"重加权到全语料分布（消除话题构成混杂）；仅在 AIGC 子集内部对比，"
        f"n={out['n_records']}，因此这些词刻画的是**模型之间的口癖差异**，"
        f"而非人机差异。作为参照，各家族原始领域构成的最大占比差约 "
        f"{out['max_domain_share_deviation_pp']:.1f} 个百分点。",
        "",
        "## 各家族 Top-10 特征词（z 分数，越大越专属该家族）",
        "",
    ]
    for fam, words in out["family_top_words"].items():
        top = "、".join(f"{d['word']}({d['z']:.1f})" for d in words[:10])
        lines.append(f"- **{fam}**：{top}")
    lines += [
        "",
        "## 共享 AI 味词表的家族使用密度（每万字符）",
        "",
        f"人类基线 {out['human_ai_word_density_per10k']:.2f}（`paper_features` 口径）。",
        "",
        "| 家族 | 每万字符 |",
        "|------|---------:|",
    ]
    for fam, v in sorted(out["ai_word_density_per10k"].items(),
                          key=lambda x: -x[1]):
        lines.append(f"| {fam} | {v:.2f} |")
    lines += [
        "",
        "## 17 个模型 Top-6 特征词",
        "",
    ]
    for model, words in sorted(out["model_top_words"].items()):
        top = "、".join(f"{d['word']}({d['z']:.1f})" for d in words[:6])
        lines.append(f"- **{model}**：{top}")
    lines.append("")
    with open(BASE + "family_lexicon_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
