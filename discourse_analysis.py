# -*- coding: utf-8 -*-
"""深潜分析之十九：篇章连接层（因果 / 转折 / 递进 / 序列 / 举例 / 总结 + 口语-书面语域）。

既有分析覆盖了词汇表、句子节奏、框架模板、标题、符号、标点排版与互动结构，
但把「AI 味」落到**篇章结构**层（句子之间用什么逻辑连接词、语域偏口语还是
偏书面）的系统分析还缺失。本脚本用六个先验定义的连接词类别与一个先验的
口语/书面语域划分（不依赖任何监督信号）回答三组问题：

1. 跨子集使用率：三类语料如何组织句间逻辑？生成集是否更爱显式书面连接词，
   人类语料是否更依赖口语连接词？帖均覆盖几个逻辑类别（连接丰富度）？
2. 探索集内的 P(AI) 梯度：哪些连接类别/连接词在野外真正区分「像人 / 像 AI」，
   梯度是否稳健（去重 / 长度分层）？
3. 互动回报：在控制 P(AI)、正文长度与标签数后，各连接类别与点赞、收藏、
   评论的偏效应方向与大小（月份×领域固定效应 + 整群 bootstrap CI）。

口径说明：
- 正文清洗复用 analysis_utils.clean_body（与 paper_features.py 一致）；
- 人类/探索集按 note_id 去重（保留元数据最完整一条，规则同 deep_dive_analysis），
  AIGC 无 note_id，逐记录处理；
- 探索集 p_ai 按行序对齐 paper_exploring_scores.csv.gz，脚本内用 content_len
  逐条校验（不一致时直接报错退出）；
- 连接词密度统一按「正文每百字」计；正文不足 MIN_BODY_LEN 字的帖子
  （纯标签/空正文等）不参与统计，数量单独报告；
- 互动回归与 engagement_drivers_analysis.py 完全同协议（特征先 z-score
  再单元内去均值、单元 ≥100 条、整群 bootstrap 300 次），额外加入六个
  连接类别密度特征，覆盖点赞/收藏/评论三端。

词表（先验定义，不做数据驱动的筛选）：
- 因果解释 causal：因为 所以 因此 由于 导致 从而 于是 之所以
- 转折让步 contrast：但是 不过 然而 虽然 尽管 其实 反而
- 递进并列 addition：而且 并且 不仅 甚至 同时 以及 此外
- 序列结构 sequence：首先 其次 然后 最后 另外 接下来 另一方面
- 举例说明 exemplify：比如 例如 譬如
- 总结收束 summary：总之 总的来说 综上所述 简单来说 换句话说
- 类别计数为「正文中出现次数（非重叠匹配）」；`所以` 只计入不以「之所以」
  形式出现的位置（扣除「之所以」）。
- 语域为每个连接词的先验标签：口语 spoken（因为 所以 但是 然后 不过 其实
  比如 而且）vs 书面 written（其余 29 词），用于刻画「口语连接 → 书面连接」
  的语域迁移。

输出：
  insights/discourse_lexicon.png   跨子集类别密度 + 逐词 AIGC−人类使用差
  insights/discourse_gradient.png  探索集 P(AI) 梯度 / 逐词单元内 ρ / 互动偏效应 / 家族指纹
  insights/discourse_register.png  口语/书面语域：跨子集密度 / P(AI) 梯度 / 季度趋势
  discourse_results.json           全部数值结果
  discourse_results.md             结果摘要
"""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from analysis_utils import (BASE, INSIGHTS, PALETTE, init_plot_style, save_fig,
                            save_json, cliffs_delta, cell_rank_cache, cell_groups,
                            clean_body, content_of, decile_table,
                            dedup_indices, domain_standardized_series,
                            engagement_of, iter_jsonl, length_bin_codes,
                            parse_local_time, run_parallel, within_cell_spearman)
from engagement_drivers_analysis import (cluster_bootstrap_fe_ols, fe_ols,
                                         prepare_fe)

DATASETS = {
    "human": BASE + "training_set_human_fixed.jsonl",
    "aigc": BASE + "training_set_aigc.jsonl",
    "exploring": BASE + "exploring_set_fixed.jsonl",
}
SCORES_PATH = BASE + "paper_exploring_scores.csv.gz"
FEATURES_PATH = BASE + "paper_features_exploring.csv.gz"

MIN_BODY_LEN = 50        # 连接词密度在更短正文（纯标签/空正文）上无意义
MIN_CELL_N = 100         # 单元内统计的最小单元样本量（与论文口径一致）
MONTH_FROM = "202301"    # 互动分析起始月份（与 paper_engagement 一致）
N_BOOT = 300             # 整群 bootstrap 次数
BOOT_SEED = 42
MIN_QUARTER_N = 300
MIN_FAMILY_N = 100

CATEGORY_WORDS = {
    "causal": ("因果解释", ["因为", "所以", "因此", "由于", "导致", "从而",
                           "于是", "之所以"]),
    "contrast": ("转折让步", ["但是", "不过", "然而", "虽然", "尽管", "其实",
                             "反而"]),
    "addition": ("递进并列", ["而且", "并且", "不仅", "甚至", "同时", "以及",
                             "此外"]),
    "sequence": ("序列结构", ["首先", "其次", "然后", "最后", "另外", "接下来",
                             "另一方面"]),
    "exemplify": ("举例说明", ["比如", "例如", "譬如"]),
    "summary": ("总结收束", ["总之", "总的来说", "综上所述", "简单来说",
                            "换句话说"]),
}
CATEGORY_LABELS = {k: v[0] for k, v in CATEGORY_WORDS.items()}
CATEGORY_COLORS = {"causal": "#C44E52", "contrast": "#4C72B0",
                   "addition": "#DD8452", "sequence": "#55A868",
                   "exemplify": "#8172B3", "summary": "#937860"}
CATEGORY_ORDER = ["causal", "contrast", "addition", "sequence",
                  "exemplify", "summary"]
SUBSET_ORDER = ["human", "aigc", "exploring"]
ALL_WORDS = [w for k in CATEGORY_ORDER for w in CATEGORY_WORDS[k][1]]
WORD_CATEGORY = {w: k for k in CATEGORY_ORDER for w in CATEGORY_WORDS[k][1]}

SPOKEN_WORDS = ["因为", "所以", "但是", "然后", "不过", "其实", "比如", "而且"]
WRITTEN_WORDS = [w for w in ALL_WORDS if w not in SPOKEN_WORDS]
WORD_REGISTER = {w: "spoken" for w in SPOKEN_WORDS}
WORD_REGISTER.update({w: "written" for w in WRITTEN_WORDS})
REGISTER_ORDER = ["spoken", "written"]
REGISTER_LABELS = {"spoken": "口语连接", "written": "书面连接"}
REGISTER_COLORS = {"spoken": "#DD8452", "written": "#4C72B0"}

FAMILY_NAME = {"anthropic": "Claude", "deepseek": "DeepSeek", "gemini": "Gemini",
               "glm": "GLM", "gpt": "GPT", "qwen": "Qwen"}
FAMILIES = ["anthropic", "deepseek", "gemini", "glm", "gpt", "qwen"]
TARGETS = (("likes", "点赞"), ("collections", "收藏"), ("comments", "评论"))


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def word_counts(text: str) -> dict:
    """对一段正文统计六个类别 + 每个连接词的命中次数。

    返回 {"causal": {"因为": 2, ...}, ..., "_words": {词: 次数}}；计数口径
    为逐词的 `str.count`（C 层非重叠计数）。词表内唯一需要扣除的嵌套组合是
    「之所以」中的「所以」，因此 `所以` 只计入不以「之所以」形式出现的位置
    （与 `(?<!之)所以` 等价）。
    """
    out: dict = {}
    words: Counter = Counter()
    for cat, (_, keyword_list) in CATEGORY_WORDS.items():
        found = {}
        for w in keyword_list:
            n = text.count(w)
            if w == "所以":
                n -= text.count("之所以")
            if n:
                found[w] = n
                words[w] += n
        out[cat] = found
    out["_words"] = dict(words)
    return out


def category_richness(counts: dict) -> int:
    """一篇帖子命中的不同连接类别数（0..6），衡量逻辑组织丰富度。"""
    return int(sum(1 for k in CATEGORY_ORDER if counts[k]))


def register_counts(counts: dict) -> dict:
    """把逐词计数汇总为口语/书面两个语域的次数。"""
    return {r: sum(counts["_words"].get(w, 0) for w in ALL_WORDS
                   if WORD_REGISTER[w] == r) for r in REGISTER_ORDER}


def density_per_100(count: int, body_len: int) -> float:
    """次数 / 正文字数 * 100；空正文返回 0。"""
    return count / body_len * 100 if body_len > 0 else 0.0


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
_WORD_INDEX = {w: i for i, w in enumerate(ALL_WORDS)}


def scan_corpus(path: str) -> dict:
    """单次流式扫描一个语料，返回逐记录数组。

    每条记录保存：正文长度、note_id、元数据完整度（用于去重取舍）、
    月份、领域、模型家族/模型（AIGC），以及六个类别的命中次数、逐词命中
    次数（int32 平铺矩阵）、口语/书面语域次数与连接丰富度。
    """
    body_len, note_id, meta, month, domain = [], [], [], [], []
    family, model = [], []
    word_index = _WORD_INDEX
    n_words = len(ALL_WORDS)
    cat_counts = {k: [] for k in CATEGORY_WORDS}
    reg_counts = {r: [] for r in REGISTER_ORDER}
    richness = []
    flat_words = array("i")
    likes = []
    n_total = 0
    for rec in iter_jsonl(path):
        n_total += 1
        content = content_of(rec)
        body = clean_body(content)
        L = len(body)
        counts = word_counts(body)
        body_len.append(L)
        note_id.append(rec.get("note_id"))
        dom = rec.get("domain")
        domain.append(dom if dom not in (None, "") else "未知")
        likes_v, colls, comms = engagement_of(rec)
        likes.append(likes_v)
        has_meta = bool(rec.get("user_id")) and dom not in (None, "", "未知")
        dt = parse_local_time(rec.get("local_time"))
        m = f"{dt.year:04d}{dt.month:02d}" if dt else None
        month.append(m)
        score = (int(has_meta) + int(bool(likes_v)) + int(bool(colls))
                 + int(bool(comms)) + int(dt is not None)
                 + int(bool(content.strip())))
        meta.append(score)
        family.append(rec.get("model_family") or "")
        model.append(rec.get("model") or "")
        for k in CATEGORY_WORDS:
            cat_counts[k].append(sum(counts[k].values()))
        regs = register_counts(counts)
        for r in REGISTER_ORDER:
            reg_counts[r].append(regs[r])
        richness.append(category_richness(counts))
        row = [0] * n_words
        for w, n in counts["_words"].items():
            row[word_index[w]] = n
        flat_words.extend(row)
    word_matrix = np.frombuffer(flat_words, dtype=np.int32).reshape(-1, n_words)
    return {
        "path": path, "n_total": n_total,
        "body_len": np.asarray(body_len, dtype=np.int32),
        "note_id": note_id, "meta": np.asarray(meta, dtype=np.int32),
        "month": month, "domain": domain, "likes": np.asarray(likes, dtype=float),
        "family": family, "model": model,
        "cat_counts": {k: np.asarray(v, dtype=np.int32)
                       for k, v in cat_counts.items()},
        "reg_counts": {r: np.asarray(v, dtype=np.int32)
                       for r, v in reg_counts.items()},
        "richness": np.asarray(richness, dtype=np.int32),
        "word_counts": {w: word_matrix[:, i] for i, w in enumerate(ALL_WORDS)},
    }


def dedup_keep(scan: dict) -> np.ndarray:
    """按 note_id 去重（保留元数据最完整一条），返回排序后的保留索引。"""
    keep = dedup_indices(scan["note_id"], scan["meta"])
    return np.asarray(sorted(keep), dtype=int)


def rates_block(scan: dict, keep: np.ndarray, min_len: int = MIN_BODY_LEN) -> dict:
    """一个子集的类别/语域/逐词使用率与连接丰富度（仅正文 >= min_len 的记录）。"""
    sel = keep[scan["body_len"][keep] >= min_len]
    n = int(len(sel))
    chars = int(scan["body_len"][sel].sum())
    cat = {}
    for k in CATEGORY_ORDER:
        c = scan["cat_counts"][k][sel].astype(float)
        cat[k] = {
            "n": n, "hits": int(c.sum()), "chars": chars,
            "density100": float(c.sum() / chars * 100) if chars else 0.0,
            "share": float((c > 0).mean()) if n else 0.0,
        }
    any_c = np.zeros(n, dtype=float)
    for k in CATEGORY_ORDER:
        any_c += scan["cat_counts"][k][sel]
    cat["any"] = {
        "n": n, "hits": int(any_c.sum()), "chars": chars,
        "density100": float(any_c.sum() / chars * 100) if chars else 0.0,
        "share": float((any_c > 0).mean()) if n else 0.0,
    }
    reg = {}
    for r in REGISTER_ORDER:
        c = scan["reg_counts"][r][sel].astype(float)
        reg[r] = {
            "n": n, "hits": int(c.sum()), "chars": chars,
            "density100": float(c.sum() / chars * 100) if chars else 0.0,
            "share": float((c > 0).mean()) if n else 0.0,
        }
    total_hits = reg["spoken"]["hits"] + reg["written"]["hits"]
    reg["written_share"] = (float(reg["written"]["hits"] / total_hits)
                            if total_hits else 0.0)
    words = {}
    for w, arr in scan["word_counts"].items():
        c = arr[sel].astype(float)
        words[w] = {
            "n": n,
            "per10k_chars": float(c.sum() / chars * 10000) if chars else 0.0,
            "share": float((c > 0).mean()) if n else 0.0,
        }
    return {"n": n, "chars": chars, "categories": cat, "registers": reg,
            "words": words,
            "richness_mean": float(scan["richness"][sel].mean()) if n else 0.0}


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_lexicon(subset_rates: dict, word_table: dict) -> None:
    """左：三个子集的类别密度；右：逐词 AIGC−人类使用差（探索集为圆点）。"""
    init_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 9.2),
                             gridspec_kw={"width_ratios": [1.0, 1.55]})

    # ---- (a) 类别 × 子集密度 ----
    ax = axes[0]
    y = np.arange(len(CATEGORY_ORDER))
    h = 0.24
    for j, subset in enumerate(SUBSET_ORDER):
        vals = [subset_rates[subset]["categories"][k]["density100"]
                for k in CATEGORY_ORDER]
        pos = y + (j - 1) * h
        ax.barh(pos, vals, height=h, color=PALETTE[subset],
                label={"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
                       "exploring": "探索集(后LLM)"}[subset])
        for yy, v in zip(pos, vals):
            ax.annotate(f"{v:.3f}", xy=(v, yy), xytext=(3, -3),
                        textcoords="offset points", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([CATEGORY_LABELS[k] for k in CATEGORY_ORDER])
    ax.set_xlabel("连接词密度（次/百字，正文 ≥ %d 字）" % MIN_BODY_LEN)
    ax.set_title("(a) 三类语料的篇章连接词使用密度")
    ax.legend(loc="lower right")

    # ---- (b) 逐词 AIGC−人类 使用差 ----
    ax = axes[1]
    rows = []
    for w, rec in word_table.items():
        rows.append((w, rec["category"], rec["register"],
                     rec["aigc"]["per10k_chars"] - rec["human"]["per10k_chars"],
                     rec["exploring"]["per10k_chars"]))
    rows.sort(key=lambda r: (CATEGORY_ORDER.index(r[1]), -r[3]))
    ypos = np.arange(len(rows))
    for cat_i, k in enumerate(CATEGORY_ORDER):
        idx = [i for i, r in enumerate(rows) if r[1] == k]
        if cat_i % 2 == 1 and idx:
            ax.axhspan(min(idx) - 0.5, max(idx) + 0.5, color="#EEEEEE", zorder=0)
    diffs = np.asarray([r[3] for r in rows])
    colors = [CATEGORY_COLORS[r[1]] for r in rows]
    ax.barh(ypos, diffs, color=colors, height=0.72, alpha=0.9, zorder=2)
    ax.scatter([r[4] for r in rows], ypos, s=22, color="black", zorder=4,
               edgecolor="white", linewidth=0.5, label="探索集使用率")
    ax.axvline(0, color="black", lw=0.8, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] + ("（口）" if r[2] == "spoken" else "")
                        for r in rows], fontsize=8.5)
    for i, r in enumerate(rows):
        v = r[3]
        if abs(v) < 0.02:
            continue
        ax.annotate(f"{v:+.2f}", xy=(v, ypos[i]),
                    xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points", fontsize=7.2,
                    ha="left" if v >= 0 else "right", va="center")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("AIGC − 人类 使用率差（次/万字符；越右=生成集越爱用）")
    ax.set_title("(b) 生成集对每个连接词的过量使用（黑点=探索集使用率）")
    handles = [plt.Line2D([], [], color=CATEGORY_COLORS[k], lw=6,
                          label=CATEGORY_LABELS[k]) for k in CATEGORY_ORDER]
    handles.append(plt.Line2D([], [], color="black", marker="o", ls="",
                              ms=4, label="探索集使用率"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    save_fig(fig, INSIGHTS + "discourse_lexicon.png")


def plot_gradient(grad_cat: dict, word_rho: dict, eng: dict,
                  family_table: dict, n_obs: int, n_cells: int) -> None:
    """四联图：P(AI) 十分位梯度 / 逐词单元内 ρ / 互动偏效应 / 家族指纹。"""
    init_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 13.2))

    # ---- (a) P(AI) 十分位密度曲线（6 类别 + 合计） ----
    ax = axes[0, 0]
    for k in CATEGORY_ORDER:
        dec = grad_cat[k]["deciles"]
        x = [d["decile"] for d in dec]
        yv = [d["mean"] for d in dec]
        lo = [d["ci_low"] for d in dec]
        hi = [d["ci_high"] for d in dec]
        ax.plot(x, yv, "o-", color=CATEGORY_COLORS[k], lw=1.8, ms=5,
                label=f"{CATEGORY_LABELS[k]}（ρ={grad_cat[k]['weighted_rho']:+.3f}）")
        ax.fill_between(x, lo, hi, color=CATEGORY_COLORS[k], alpha=0.14)
    dec_any = grad_cat["any"]["deciles"]
    ax.plot([d["decile"] for d in dec_any], [d["mean"] for d in dec_any],
            "s--", color="black", lw=1.6, ms=4.5,
            label=f"合计（ρ={grad_cat['any']['weighted_rho']:+.3f}）")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("检测器 P(AI) 十分位（1=最像人，10=最像 AI）")
    ax.set_ylabel("密度（次/百字）")
    ax.set_title("(a) 探索集：连接词密度随 P(AI) 的梯度（含 95% CI）")
    ax.legend(fontsize=8.5)

    # ---- (b) 逐词单元内 ρ ----
    ax = axes[0, 1]
    items = sorted(word_rho.items(), key=lambda kv: (CATEGORY_ORDER.index(
        kv[1]["category"]), kv[1]["weighted_rho"]))
    ypos = np.arange(len(items))
    vals = [kv[1]["weighted_rho"] for kv in items]
    colors = [CATEGORY_COLORS[kv[1]["category"]] for kv in items]
    ax.barh(ypos, vals, color=colors, alpha=0.9, height=0.72)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([kv[0] for kv in items], fontsize=8.5)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.3f}", xy=(v, ypos[i]),
                    xytext=(5 if v >= 0 else -5, -3),
                    textcoords="offset points", fontsize=7,
                    ha="left" if v >= 0 else "right")
    for k in CATEGORY_ORDER:
        rho = grad_cat[k]["weighted_rho"]
        ax.axvline(rho, color=CATEGORY_COLORS[k], lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("月份×领域单元内加权 Spearman ρ（词密度 vs P(AI)）")
    ax.set_title("(b) 逐词连接词的野外 P(AI) 梯度（虚线=类别整体 ρ）")
    handles = [plt.Line2D([], [], color=CATEGORY_COLORS[k], lw=6,
                          label=CATEGORY_LABELS[k]) for k in CATEGORY_ORDER]
    ax.legend(handles=handles, loc="lower left", fontsize=8)

    # ---- (c) 互动偏效应森林图（点赞/收藏/评论） ----
    ax = axes[1, 0]
    names = list(eng["fe"]["likes"]["coef"].keys())
    order = list(reversed(names))
    ypos = np.arange(len(order))
    for target, marker, offset, color in (
            ("likes", "o", -0.20, "#4C72B0"),
            ("collections", "^", 0.00, "#DD8452"),
            ("comments", "s", 0.20, "#55A868")):
        fe = eng["fe"][target]
        for i, name in enumerate(order):
            b = fe["coef"][name]
            lo, hi = fe["ci_low"][name], fe["ci_high"][name]
            sig = lo > 0 or hi < 0
            col = color if sig else "#9E9E9E"
            ax.plot([lo, hi], [ypos[i] + offset, ypos[i] + offset], color=col,
                    lw=1.6, solid_capstyle="butt", zorder=2)
            ax.scatter([b], [ypos[i] + offset], marker=marker, s=30,
                       facecolor=col if sig else "white", edgecolor=col,
                       zorder=3,
                       label={"likes": "点赞", "collections": "收藏",
                              "comments": "评论"}[target]
                       if i == 0 else None)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    labels = dict(CATEGORY_LABELS)
    labels.update({"p_ai": "P(AI)", "content_len": "正文长度",
                   "n_hashtags": "标签数", "any": "合计"})
    ax.set_yticklabels([labels.get(n, n) for n in order])
    ax.set_xlabel("单元内 FE β（每 1 SD → log10(1+互动)）")
    ax.set_title(f"(c) 连接特征的互动偏效应（n={n_obs:,}，{n_cells} 单元，"
                 f"整群 bootstrap；灰=CI 跨 0）")
    ax.legend(loc="lower right")

    # ---- (d) 家族指纹（按类别列内 z-score 着色的热力图） ----
    ax = axes[1, 1]
    fams = [f for f in FAMILIES if family_table.get(f, {}).get("n", 0) >= MIN_FAMILY_N]
    mat = np.asarray([[family_table[f]["density100"][k] for k in CATEGORY_ORDER]
                      for f in fams], dtype=float)
    z = (mat - mat.mean(axis=0)) / np.where(mat.std(axis=0) > 0,
                                            mat.std(axis=0), 1.0)
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
    ax.grid(False)
    for i in range(len(fams)):
        for j in range(len(CATEGORY_ORDER)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(z[i, j]) > 1.15 else "black")
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels([CATEGORY_LABELS[k] for k in CATEGORY_ORDER], fontsize=9)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f"{FAMILY_NAME[f]}（n={family_table[f]['n']}）"
                        for f in fams], fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("家族差异（列内 z-score）", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("(d) AIGC 训练集的家族连接词指纹（格内为密度，次/百字）")
    save_fig(fig, INSIGHTS + "discourse_gradient.png")


def plot_register(subset_rates: dict, grad: dict, quarterly: dict) -> None:
    """三联图：语域跨子集密度 / 语域 P(AI) 梯度 / 语域季度趋势。"""
    init_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.4))

    # ---- (a) 语域 × 子集密度 ----
    ax = axes[0]
    x = np.arange(len(SUBSET_ORDER))
    w = 0.36
    for j, r in enumerate(REGISTER_ORDER):
        vals = [subset_rates[s]["registers"][r]["density100"]
                for s in SUBSET_ORDER]
        ax.bar(x + (j - 0.5) * w, vals, width=w, color=REGISTER_COLORS[r],
               label=REGISTER_LABELS[r])
        for xx, v in zip(x + (j - 0.5) * w, vals):
            ax.annotate(f"{v:.2f}", xy=(xx, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["人类(前LLM)", "AIGC(17模型)", "探索集(后LLM)"])
    ax.set_ylabel("密度（次/百字，正文 ≥ %d 字）" % MIN_BODY_LEN)
    ax.set_title("(a) 口语 vs 书面连接词的跨子集密度")
    tops = []
    for i, s in enumerate(SUBSET_ORDER):
        reg = subset_rates[s]["registers"]
        top = max(reg["spoken"]["density100"], reg["written"]["density100"])
        tops.append(top)
        ax.annotate(f"书面占比 {reg['written_share']:.0%}",
                    xy=(x[i], top + 0.02), ha="center", fontsize=8.5,
                    color="#555555")
    ax.set_ylim(0, max(tops) * 1.18)
    ax.legend(loc="upper center", ncol=2)

    # ---- (b) 语域 P(AI) 梯度 ----
    ax = axes[1]
    for r in REGISTER_ORDER:
        dec = grad[r]["deciles"]
        x = [d["decile"] for d in dec]
        yv = [d["mean"] for d in dec]
        lo = [d["ci_low"] for d in dec]
        hi = [d["ci_high"] for d in dec]
        ax.plot(x, yv, "o-", color=REGISTER_COLORS[r], lw=1.8, ms=5,
                label=f"{REGISTER_LABELS[r]}（ρ={grad[r]['weighted_rho']:+.3f}）")
        ax.fill_between(x, lo, hi, color=REGISTER_COLORS[r], alpha=0.16)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("检测器 P(AI) 十分位（1=最像人，10=最像 AI）")
    ax.set_ylabel("密度（次/百字）")
    ax.set_title("(b) 探索集：语域密度随 P(AI) 的梯度（含 95% CI）")
    ax.legend(fontsize=9)

    # ---- (c) 季度趋势（固定领域权重） ----
    ax = axes[2]
    quarters = quarterly["quarters"]
    x = np.arange(len(quarters))
    for r in REGISTER_ORDER:
        vals = quarterly["domain_std"][r]
        ax.plot(x, vals, "o-", color=REGISTER_COLORS[r], lw=1.8, ms=5,
                label=REGISTER_LABELS[r])
    any_vals = quarterly["domain_std"]["any"]
    ax.plot(x, any_vals, "s--", color="black", lw=1.5, ms=4.5, label="合计")
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("密度（次/百字，领域标准化）")
    ax.set_title("(c) 探索集季度趋势（固定领域权重）")
    tr_sp = quarterly["trend"]["spoken"]
    tr_wr = quarterly["trend"]["written"]
    ax.legend(title=f"趋势 ρ：口语 {tr_sp['rho_domain_std']:+.2f} (p={tr_sp['p_domain_std']:.2f})"
                    f" / 书面 {tr_wr['rho_domain_std']:+.2f} (p={tr_wr['p_domain_std']:.2f})",
              fontsize=8.5, title_fontsize=8.5)
    save_fig(fig, INSIGHTS + "discourse_register.png")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def grad_stats(dens, p_ai, cells, strat, groups, groups_strat,
               p_ai_ranks, p_ai_ranks_strat, p_hi, p_lo) -> dict:
    """一个密度序列的完整梯度统计（十分位 / 单元内 ρ / 长度分层 / 首尾 δ）。"""
    dec = decile_table(dens, p_ai)
    wc = within_cell_spearman(dens, p_ai, cells, min_cell_n=MIN_CELL_N,
                              groups=groups, y_ranks=p_ai_ranks)
    st = within_cell_spearman(dens, p_ai, strat, min_cell_n=MIN_CELL_N,
                              groups=groups_strat, y_ranks=p_ai_ranks_strat)
    hi_p = p_ai >= p_hi
    lo_p = p_ai <= p_lo
    delta, dp = cliffs_delta(dens[hi_p], dens[lo_p])
    return {
        "deciles": dec,
        "weighted_rho": round(wc["weighted_rho"], 4),
        "neg_cell_share": round(wc["neg_cell_share"], 4),
        "n_cells": wc["n_cells"],
        "stratified_rho": round(st["weighted_rho"], 4),
        "stratified_cells": st["n_cells"],
        "cliff_delta_top_vs_bottom": round(delta, 4),
        "cliff_p": float(dp),
        "decile_p_range": [dec[0]["p_range"][0] if dec else float("nan"),
                           dec[-1]["p_range"][1] if dec else float("nan")],
    }


def main() -> None:
    print("扫描三个语料（全量，进程池并行）…")
    scans = dict(zip(DATASETS, run_parallel(
        [partial(scan_corpus, path) for path in DATASETS.values()])))
    for name, sc in scans.items():
        print(f"  {name}: {sc['n_total']:,} 条记录")

    # ---- 对齐探索集 p_ai ----
    scores = pd.read_csv(SCORES_PATH, usecols=["p_ai", "content_len"])
    n_score = len(scores)
    n_scanned = scans["exploring"]["n_total"]
    if n_score != n_scanned:
        raise SystemExit(f"exploring JSONL 行数 {n_scanned} 与 p_ai 行数 {n_score} 不一致")
    mismatch = int((scores["content_len"].to_numpy()
                    != scans["exploring"]["body_len"]).sum())
    if mismatch:
        raise SystemExit(f"探索集 p_ai 行序未对齐：content_len mismatch={mismatch}")

    # ---- 去重 + 子集使用率 ----
    keeps = {name: dedup_keep(sc) for name, sc in scans.items()}
    subset_rates = {name: rates_block(scans[name], keeps[name])
                    for name in SUBSET_ORDER}
    for name in SUBSET_ORDER:
        sc = scans[name]
        short = int((sc["body_len"][keeps[name]] < MIN_BODY_LEN).sum())
        print(f"  {name}: 去重后 {len(keeps[name]):,} 条，正文 <{MIN_BODY_LEN} 字 "
              f"排除 {short:,} 条")

    word_table = {}
    for w in ALL_WORDS:
        word_table[w] = {
            "category": WORD_CATEGORY[w],
            "register": WORD_REGISTER[w],
            "human": subset_rates["human"]["words"][w],
            "aigc": subset_rates["aigc"]["words"][w],
            "exploring": subset_rates["exploring"]["words"][w],
        }

    # ---- 探索集 P(AI) 梯度（去重 + 长度门槛）----
    expl = scans["exploring"]
    keep = keeps["exploring"]
    sel = keep[expl["body_len"][keep] >= MIN_BODY_LEN]
    p_ai = scores["p_ai"].to_numpy(dtype=float)[sel]
    body = expl["body_len"][sel].astype(float)
    months = np.asarray([expl["month"][i] or "未知" for i in sel])
    domains = np.asarray([expl["domain"][i] for i in sel])
    cells = pd.factorize(pd.Series([f"{m}|{d}" for m, d in zip(months, domains)]),
                         sort=True)[0]
    strat = pd.factorize(pd.Series(
        [f"{m}|{d}|{b}" for m, d, b in
         zip(months, domains, length_bin_codes(body))]), sort=True)[0]
    print(f"梯度样本：{len(sel):,} 条（去重 + 正文 ≥{MIN_BODY_LEN} 字），"
          f"单元 {len(np.unique(cells))} 个")

    groups = cell_groups(cells, MIN_CELL_N)
    # p_ai 是所有类别/逐词/语域的同一对侧向量：预算一次单元内平均秩供全部
    # 调用复用；上下十分位阈值同样只与 p_ai 有关，循环外只算一次。
    p_ai_ranks = cell_rank_cache(p_ai, groups)
    groups_strat = cell_groups(strat, MIN_CELL_N)
    p_ai_ranks_strat = cell_rank_cache(p_ai, groups_strat)
    p_hi, p_lo = float(np.quantile(p_ai, 0.9)), float(np.quantile(p_ai, 0.1))

    any_arr = np.zeros(len(sel), dtype=float)
    for k in CATEGORY_ORDER:
        any_arr += expl["cat_counts"][k][sel]
    grad_cat = {}
    for k in CATEGORY_ORDER + ["any"]:
        counts = any_arr if k == "any" else expl["cat_counts"][k][sel].astype(float)
        dens = counts / body * 100
        grad_cat[k] = grad_stats(dens, p_ai, cells, strat, groups, groups_strat,
                                 p_ai_ranks, p_ai_ranks_strat, p_hi, p_lo)
        print(f"  {k}: wcell ρ={grad_cat[k]['weighted_rho']:+.4f} "
              f"(长度分层 {grad_cat[k]['stratified_rho']:+.4f})，"
              f"δ={grad_cat[k]['cliff_delta_top_vs_bottom']:+.4f}")

    grad_reg = {}
    for r in REGISTER_ORDER:
        dens = expl["reg_counts"][r][sel].astype(float) / body * 100
        grad_reg[r] = grad_stats(dens, p_ai, cells, strat, groups, groups_strat,
                                 p_ai_ranks, p_ai_ranks_strat, p_hi, p_lo)
        print(f"  {r}: wcell ρ={grad_reg[r]['weighted_rho']:+.4f} "
              f"(长度分层 {grad_reg[r]['stratified_rho']:+.4f})，"
              f"δ={grad_reg[r]['cliff_delta_top_vs_bottom']:+.4f}")

    rich = expl["richness"][sel].astype(float)
    grad_rich = grad_stats(rich, p_ai, cells, strat, groups, groups_strat,
                           p_ai_ranks, p_ai_ranks_strat, p_hi, p_lo)
    print(f"  丰富度: wcell ρ={grad_rich['weighted_rho']:+.4f} "
          f"(长度分层 {grad_rich['stratified_rho']:+.4f})")

    word_rho = {}
    for w in ALL_WORDS:
        arr = expl["word_counts"][w]
        dens = arr[sel].astype(float) / body * 100
        if dens.std() == 0:
            continue
        wc = within_cell_spearman(dens, p_ai, cells, min_cell_n=MIN_CELL_N,
                                  groups=groups, y_ranks=p_ai_ranks)
        word_rho[w] = {"category": WORD_CATEGORY[w],
                       "register": WORD_REGISTER[w],
                       "weighted_rho": round(wc["weighted_rho"], 4),
                       "neg_cell_share": round(wc["neg_cell_share"], 4),
                       "n_cells": wc["n_cells"]}

    # ---- 互动关联（与 engagement_drivers 同协议）----
    model = pd.read_csv(FEATURES_PATH,
                        usecols=["month", "domain", "content_len", "n_hashtags",
                                 "likes", "collections", "comments"])
    if len(model) != n_scanned:
        raise SystemExit("features 与 scores 行数不一致")
    mm2 = int((model["content_len"].to_numpy()
               != scans["exploring"]["body_len"]).sum())
    if mm2:
        raise SystemExit(f"features content_len 与扫描不一致：{mm2}")
    likes_mismatch = int((model["likes"].to_numpy()
                          != scans["exploring"]["likes"]).sum())
    disc_dens = np.column_stack([
        scans["exploring"]["cat_counts"][k].astype(float)
        / np.maximum(scans["exploring"]["body_len"], 1) * 100
        for k in CATEGORY_ORDER])
    targets = {name: np.log10(1 + np.clip(model[name].to_numpy(dtype=float), 0, None))
               for name, _ in TARGETS}
    month_all = model["month"].astype(str).to_numpy()
    dom_all = model["domain"].astype(str).to_numpy()
    keep_eng = (month_all >= MONTH_FROM) & (scans["exploring"]["body_len"] >= MIN_BODY_LEN)
    eng_cells_raw = np.asarray([f"{m}|{d}" for m, d in zip(month_all, dom_all)])
    p_ai_all = scores["p_ai"].to_numpy(dtype=float)
    X = np.column_stack([disc_dens, p_ai_all,
                         scans["exploring"]["body_len"].astype(float),
                         model["n_hashtags"].to_numpy(dtype=float)])
    feature_names = list(CATEGORY_ORDER) + ["p_ai", "content_len", "n_hashtags"]
    eng = {"fe": {}, "univariate": {}}
    prep = prepare_fe(X[keep_eng], eng_cells_raw[keep_eng], MIN_CELL_N)
    for target, _ in TARGETS:
        yk = targets[target][keep_eng]
        fit = fe_ols(X[keep_eng], yk, eng_cells_raw[keep_eng], prepared=prep)
        boot = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                        fit["cell_codes"], n_boot=N_BOOT,
                                        seed=BOOT_SEED)
        ci = np.percentile(boot, [2.5, 97.5], axis=0)
        eng["fe"][target] = {
            "features": feature_names,
            "coef": {n: round(float(v), 4) for n, v in zip(feature_names, fit["coef"])},
            "ci_low": {n: round(float(v), 4) for n, v in zip(feature_names, ci[0])},
            "ci_high": {n: round(float(v), 4) for n, v in zip(feature_names, ci[1])},
            "n_obs": fit["n_obs"], "n_cells": fit["n_cells"],
        }
        print(f"  互动 FE（{target}）: n={fit['n_obs']:,}, "
              f"causal β={fit['coef'][0]:+.4f} [{ci[0][0]:+.4f}, {ci[1][0]:+.4f}]")

    # 单变量：6 类别 + 合计 + p_ai × 3 指标（预分组一次共享）
    groups_eng = cell_groups(eng_cells_raw[keep_eng], MIN_CELL_N)
    y_ranks = {t: cell_rank_cache(targets[t][keep_eng], groups_eng)
               for t, _ in TARGETS}
    uni_cols = {}
    for i, k in enumerate(CATEGORY_ORDER):
        uni_cols[k] = disc_dens[keep_eng][:, i]
    uni_cols["any"] = np.sum(disc_dens[keep_eng], axis=1)
    uni_cols["p_ai"] = p_ai_all[keep_eng]
    for k, xk in uni_cols.items():
        x_ranks = cell_rank_cache(xk, groups_eng)
        eng["univariate"][k] = {
            t: within_cell_spearman(xk, targets[t][keep_eng],
                                    eng_cells_raw[keep_eng],
                                    min_cell_n=MIN_CELL_N, groups=groups_eng,
                                    x_ranks=x_ranks, y_ranks=y_ranks[t])
            for t, _ in TARGETS
        }

    # ---- AIGC 家族 / 单模型指纹 ----
    aigc = scans["aigc"]
    ok = aigc["body_len"] >= MIN_BODY_LEN
    fam_agg: dict = defaultdict(lambda: {"n": 0, "chars": 0,
                                         "counts": defaultdict(int),
                                         "reg": defaultdict(int)})
    mod_agg: dict = defaultdict(lambda: {"n": 0, "chars": 0,
                                         "counts": defaultdict(int),
                                         "reg": defaultdict(int)})
    for i in np.flatnonzero(ok):
        for agg, key in ((fam_agg, aigc["family"][i]), (mod_agg, aigc["model"][i])):
            slot = agg[key]
            slot["n"] += 1
            slot["chars"] += int(aigc["body_len"][i])
            for k in CATEGORY_ORDER:
                slot["counts"][k] += int(aigc["cat_counts"][k][i])
            for r in REGISTER_ORDER:
                slot["reg"][r] += int(aigc["reg_counts"][r][i])

    def agg_to_table(agg):
        out = {}
        for key, slot in agg.items():
            if not key:
                continue
            chars = slot["chars"]
            out[key] = {
                "n": slot["n"],
                "density100": {
                    k: round(slot["counts"][k] / chars * 100, 4) if chars else 0.0
                    for k in CATEGORY_ORDER},
                "register_density100": {
                    r: round(slot["reg"][r] / chars * 100, 4) if chars else 0.0
                    for r in REGISTER_ORDER},
            }
        return out

    family_table = agg_to_table(fam_agg)
    model_table = agg_to_table(mod_agg)

    # ---- 季度趋势（去重；原始 + 固定领域权重）----
    q_sel = keeps["exploring"][expl["body_len"][keeps["exploring"]] >= MIN_BODY_LEN]
    q_df = pd.DataFrame({
        "quarter": [f"{m[:4]}Q{(int(m[4:6]) - 1) // 3 + 1}" if m else "未知"
                    for m in (expl["month"][i] for i in q_sel)],
        "domain": [expl["domain"][i] for i in q_sel],
        "body_len": expl["body_len"][q_sel].astype(float),
    })
    for k in CATEGORY_ORDER:
        q_df[k] = expl["cat_counts"][k][q_sel].astype(float) / q_df["body_len"] * 100
    for r in REGISTER_ORDER:
        q_df[r] = expl["reg_counts"][r][q_sel].astype(float) / q_df["body_len"] * 100
    q_df["any"] = q_df[CATEGORY_ORDER].sum(axis=1)
    q_df["richness"] = expl["richness"][q_sel].astype(float)
    q_df = q_df[q_df["quarter"] != "未知"]
    q_counts = q_df.groupby("quarter").size()
    valid_q = sorted(q_counts[q_counts >= MIN_QUARTER_N].index)
    q_df = q_df[q_df["quarter"].isin(valid_q)]
    quarterly = {"quarters": valid_q, "n": {q: int(q_counts[q]) for q in valid_q},
                 "raw": {}, "domain_std": {}, "trend": {}, "share_any": {}}
    for k in CATEGORY_ORDER + ["any"] + REGISTER_ORDER:
        raw = q_df.groupby("quarter")[k].mean().reindex(valid_q)
        std = domain_standardized_series(q_df, k, "quarter").reindex(valid_q)
        rho, p = spearmanr(range(len(valid_q)), raw.to_numpy())
        rho_s, p_s = spearmanr(range(len(valid_q)), std.to_numpy())
        quarterly["raw"][k] = [round(float(v), 4) for v in raw]
        quarterly["domain_std"][k] = [round(float(v), 4) for v in std]
        quarterly["trend"][k] = {"rho": round(float(rho), 4), "p": float(p),
                                 "rho_domain_std": round(float(rho_s), 4),
                                 "p_domain_std": float(p_s)}
        share = q_df.assign(hit=q_df[k] > 0).groupby("quarter")["hit"].mean().reindex(valid_q)
        quarterly["share_any"][k] = [round(float(v), 4) for v in share]

    # ---- 结果 ----
    out = {
        "meta": {
            "min_body_len": MIN_BODY_LEN, "min_cell_n": MIN_CELL_N,
            "month_from": MONTH_FROM, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "min_quarter_n": MIN_QUARTER_N,
            "exploring_content_len_mismatch": mismatch,
            "exploring_likes_mismatch": likes_mismatch,
            "note": "人类/探索集按 note_id 去重（保留元数据最完整一条）；"
                    "AIGC 无 note_id 逐记录；p_ai 按行序对齐 "
                    "paper_exploring_scores.csv.gz；连接词表为先验定义，"
                    "「所以」排除「之所以」；语域标签（口语/书面）为先验判断。",
        },
        "lexicon": {k: {"label": CATEGORY_LABELS[k], "words": CATEGORY_WORDS[k][1],
                        "register": {w: WORD_REGISTER[w] for w in CATEGORY_WORDS[k][1]}}
                    for k in CATEGORY_ORDER},
        "registers": {r: {"label": REGISTER_LABELS[r],
                          "words": [w for w in ALL_WORDS if WORD_REGISTER[w] == r]}
                      for r in REGISTER_ORDER},
        "subset_rates": {
            name: {
                "n": subset_rates[name]["n"],
                "chars": subset_rates[name]["chars"],
                "richness_mean": round(subset_rates[name]["richness_mean"], 4),
                "categories": subset_rates[name]["categories"],
                "registers": subset_rates[name]["registers"],
            } for name in SUBSET_ORDER},
        "word_rates": word_table,
        "exploring_gradient": {
            "n": int(len(sel)),
            "short_excluded": int((expl["body_len"][keep] < MIN_BODY_LEN).sum()),
            "n_cells": int(len(np.unique(cells))),
            "categories": grad_cat,
            "registers": grad_reg,
            "richness": grad_rich,
            "words": word_rho,
        },
        "engagement": {
            "month_from": MONTH_FROM, "min_body_len": MIN_BODY_LEN,
            "univariate": {
                k: {t: {"weighted_rho": round(v[t]["weighted_rho"], 4),
                        "neg_cell_share": round(v[t]["neg_cell_share"], 4),
                        "n_cells": v[t]["n_cells"]}
                    for t in ("likes", "collections", "comments")}
                for k, v in eng["univariate"].items()},
            "fe": eng["fe"],
        },
        "family": {"families": family_table, "models": model_table},
        "quarterly": quarterly,
    }
    save_json(out, BASE + "discourse_results.json")
    print("已写出 discourse_results.json")
    write_markdown(out)
    plot_lexicon(subset_rates, word_table)
    plot_gradient(grad_cat, word_rho, eng,
                  family_table,
                  eng["fe"]["likes"]["n_obs"], eng["fe"]["likes"]["n_cells"])
    plot_register(subset_rates, grad_reg, quarterly)
    print("已写出 discourse_results.md 与 insights/discourse_*.png")


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def write_markdown(out: dict) -> None:
    meta = out["meta"]
    rates = out["subset_rates"]
    grad = out["exploring_gradient"]
    eng = out["engagement"]
    q = out["quarterly"]
    labels = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
              "exploring": "探索集(后LLM)"}
    lines = [
        "# 篇章连接层分析结果（Discourse）",
        "",
        f"> 由 `discourse_analysis.py` 全量运行生成；正文清洗与 `paper_features.py` 一致，"
        f"人类/探索集按 note_id 去重；正文 ≥{meta['min_body_len']} 字；"
        f"探索集 p_ai 按行序对齐 `paper_exploring_scores.csv.gz`"
        f"（content_len 校验 mismatch={meta['exploring_content_len_mismatch']}）。",
        "",
        "## 1 跨子集连接词使用密度（次/百字）",
        "",
        "| 子集 | 有效帖数 | " + " | ".join(CATEGORY_LABELS[k] for k in CATEGORY_ORDER)
        + " | 合计 | 帖均类别数 |",
        "|---|---:|" + "---:|" * (len(CATEGORY_ORDER) + 2),
    ]
    for name in SUBSET_ORDER:
        cats = rates[name]["categories"]
        lines.append(
            f"| {labels[name]} | {rates[name]['n']:,} | "
            + " | ".join(f"{cats[k]['density100']:.3f}" for k in CATEGORY_ORDER)
            + f" | {cats['any']['density100']:.3f} "
            f"| {rates[name]['richness_mean']:.2f} |")
    lines += [
        "",
        "**口语 vs 书面语域（次/百字；书面占比=书面命中/总命中）：**",
        "",
        "| 子集 | 口语连接 | 书面连接 | 书面占比 |",
        "|---|---:|---:|---:|",
    ]
    for name in SUBSET_ORDER:
        reg = rates[name]["registers"]
        lines.append(
            f"| {labels[name]} | {reg['spoken']['density100']:.3f} "
            f"| {reg['written']['density100']:.3f} "
            f"| {reg['written_share']:.1%} |")
    lines += [
        "",
        "> 注：口语连接 = 因为/所以/但是/然后/不过/其实/比如/而且（其中因为、所以、"
        "但是、然后同时在论文特征 `hum_lex100` 的人类词表里），其余 29 词记为书面连接；"
        "语域标签为先验判断，逐词结果见 `discourse_results.json`。",
        "",
        "## 2 探索集内的 P(AI) 梯度（去重，正文 ≥%d 字）" % meta["min_body_len"],
        "",
        "| 类别 | 单元内加权 ρ | 负单元占比 | 长度分层 ρ | 首尾十分位 δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in CATEGORY_ORDER + ["any"]:
        g = grad["categories"][k]
        label = CATEGORY_LABELS.get(k, "合计")
        lines.append(
            f"| {label} | {g['weighted_rho']:+.4f} | {g['neg_cell_share']:.1%} "
            f"| {g['stratified_rho']:+.4f} | {g['cliff_delta_top_vs_bottom']:+.3f} |")
    for r in REGISTER_ORDER:
        g = grad["registers"][r]
        lines.append(
            f"| {REGISTER_LABELS[r]} | {g['weighted_rho']:+.4f} | "
            f"{g['neg_cell_share']:.1%} | {g['stratified_rho']:+.4f} | "
            f"{g['cliff_delta_top_vs_bottom']:+.3f} |")
    g = grad["richness"]
    lines.append(
        f"| 帖均类别数 | {g['weighted_rho']:+.4f} | {g['neg_cell_share']:.1%} | "
        f"{g['stratified_rho']:+.4f} | {g['cliff_delta_top_vs_bottom']:+.3f} |")
    lines += [
        "",
        f"（样本 {grad['n']:,} 条 / {grad['n_cells']} 个「月份×领域」单元；"
        f"长度分层 = 月份×领域×长度分箱单元。）",
        "",
        "**逐词单元内 ρ（Top/Bottom 8）：**",
        "",
        "| 词 | 类别 | 语域 | 单元内 ρ | 负单元占比 |",
        "|---|---|---|---:|---:|",
    ]
    items = sorted(grad["words"].items(), key=lambda kv: kv[1]["weighted_rho"])
    show = items[:8] + items[-8:]
    for w, r in show:
        lines.append(f"| {w} | {CATEGORY_LABELS[r['category']]} | "
                     f"{REGISTER_LABELS[r['register']]} | "
                     f"{r['weighted_rho']:+.4f} | {r['neg_cell_share']:.0%} |")
    lines += [
        "",
        "## 3 互动偏效应（月份×领域 FE + 整群 bootstrap %d 次）" % meta["n_boot"],
        "",
        "| 特征 | 点赞 β [95% CI] | 收藏 β [95% CI] | 评论 β [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for name in eng["fe"]["likes"]["features"]:
        cells_txt = []
        for t in ("likes", "collections", "comments"):
            fe = eng["fe"][t]
            cells_txt.append(f"{fe['coef'][name]:+.4f} "
                             f"[{fe['ci_low'][name]:+.4f}, {fe['ci_high'][name]:+.4f}]")
        label = CATEGORY_LABELS.get(name, {"p_ai": "P(AI)",
                                           "content_len": "正文长度",
                                           "n_hashtags": "标签数"}.get(name, name))
        lines.append(f"| {label} | " + " | ".join(cells_txt) + " |")
    lines += [
        "",
        f"（n={eng['fe']['likes']['n_obs']:,}，{eng['fe']['likes']['n_cells']} 个单元；"
        f"特征已 z-score，β 为每 1 SD 的偏效应；{meta['month_from']} 起、正文 ≥"
        f"{meta['min_body_len']} 字。）",
        "",
        "## 4 AIGC 家族连接词指纹（次/百字）",
        "",
        "| 家族 | n | " + " | ".join(CATEGORY_LABELS[k] for k in CATEGORY_ORDER)
        + " | 口语 | 书面 |",
        "|---|---:|" + "---:|" * (len(CATEGORY_ORDER) + 2),
    ]
    for fam in FAMILIES:
        slot = out["family"]["families"].get(fam)
        if not slot:
            continue
        reg = slot["register_density100"]
        lines.append(
            f"| {FAMILY_NAME.get(fam, fam)} | {slot['n']} | "
            + " | ".join(f"{slot['density100'][k]:.3f}" for k in CATEGORY_ORDER)
            + f" | {reg['spoken']:.3f} | {reg['written']:.3f} |")
    lines += [
        "",
        "## 5 季度趋势（探索集去重）",
        "",
        "| 季度 | n | " + " | ".join(CATEGORY_LABELS[k] for k in CATEGORY_ORDER)
        + " | 口语 | 书面 |",
        "|---|---:|" + "---:|" * (len(CATEGORY_ORDER) + 2),
    ]
    for i, qq in enumerate(q["quarters"]):
        vals = [q["raw"][k][i] for k in CATEGORY_ORDER]
        lines.append(
            f"| {qq} | {q['n'][qq]:,} | "
            + " | ".join(f"{v:.3f}" for v in vals)
            + f" | {q['raw']['spoken'][i]:.3f} | {q['raw']['written'][i]:.3f} |")
    lines.append("")
    for k in CATEGORY_ORDER + ["any"] + REGISTER_ORDER:
        t = q["trend"][k]
        label = CATEGORY_LABELS.get(k, REGISTER_LABELS.get(k, "合计"))
        lines.append(f"- {label}：季度趋势 ρ={t['rho']:+.3f}（p={t['p']:.3f}），"
                     f"固定领域权重后 ρ={t['rho_domain_std']:+.3f}（p={t['p_domain_std']:.3f}）。")
    lines += [
        "",
        "图表见 `insights/discourse_lexicon.png`、`insights/discourse_gradient.png`、"
        "`insights/discourse_register.png`。",
    ]
    with open(BASE + "discourse_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
