# -*- coding: utf-8 -*-
"""深潜分析之十七：互动引导语（CTA，call-to-action）。

既有分析把「AI 味惩罚」拆到了互动结构、逐特征载体与作者层，但还没有系统
刻画**帖子对读者的行动号召**：生成集是否更爱「喊话」读者（求评论、求赞藏、
求关注、催抢购）？野外的 CTA 使用与 P(AI) 梯度如何？喊话是否真的换回更多
评论/点赞？「AI 味惩罚」能否被 CTA 对冲？本脚本用四个先验定义的标记类别
（不做任何数据驱动筛选）回答这些问题：

1. 跨子集使用率：三个语料的 CTA 密度与帖子层普及率、逐标记过量使用；
2. 探索集内的 P(AI) 梯度：类别密度随检测器得分的十分位曲线、月份×领域
   单元内加权 ρ、长度分层稳健性、首尾十分位 Cliff's δ，以及逐标记 ρ；
3. 互动回报：控制 P(AI)、正文长度与标签数后，CTA 对点赞、收藏、**评论**
   的月份×领域固定效应偏效应（整群 bootstrap CI）。评论是 CTA 的直接目标，
   本脚本首次把评论端纳入同一 FE 协议；
4. 惩罚能否被对冲：在同一交互模型中加入 P(AI)×任一 CTA，检验喊话是否
   改变了 AI 味与互动的负向关系，并给出「有 CTA / 无 CTA」两处的 P(AI)
   边际斜率（同一单元集，避免分层拟合的样本选择差异）；
5. 生成端指纹：17 个模型/6 个家族的 CTA 密度；探索集季度趋势（原始 +
   固定领域权重）。

口径说明：
- 正文清洗复用 analysis_utils.clean_body（与 paper_features.py 一致）；
- 人类/探索集按 note_id 去重（保留元数据最完整一条，规则同 deep_dive_analysis），
  AIGC 无 note_id，逐记录处理；
- 探索集 p_ai 按行序对齐 paper_exploring_scores.csv.gz，脚本内用 content_len
  逐条校验（不一致时直接报错退出）；
- CTA 密度统一按「正文每百字」计；正文不足 MIN_BODY_LEN 字的帖子
  （纯标签/空正文等）不参与统计，数量单独报告；
- 互动回归与 engagement_drivers_analysis.py 完全同协议（特征先 z-score
  再单元内去均值、单元 ≥100 条、整群 bootstrap 300 次），被解释变量为
  log10(1+互动)，并额外加入四个 CTA 密度特征；
- 标记为**先验短语**（不做数据驱动的筛选），对高歧义词做显式排除
  （等价于否定前瞻）：「收藏」排除收藏夹/收藏品/收藏家这类名词，
  「聊聊」排除「聊聊天」，「留言」不重复计入「留言区」，「冲鸭」不
  重复计入「快冲鸭」；跨类别仅按所属类别计数（类别内不重复计数）。

输出：
  insights/cta_lexicon.png   跨子集类别密度 + 逐标记 AIGC−人类使用差
  insights/cta_gradient.png  P(AI) 梯度 / 逐标记 ρ / 三指标互动偏效应 /
                             家族指纹 / 季度趋势 / 分层 P(AI) 偏效应
  cta_results.json           全部数值结果
  cta_results.md             结果摘要
"""
from __future__ import annotations

from array import array
from collections import defaultdict
from functools import partial

import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from analysis_utils import (BASE, INSIGHTS, PALETTE, body_cache, init_plot_style,
                            save_fig, save_json, cliffs_delta, cell_rank_cache,
                            cell_groups, clean_body, content_of, decile_table,
                            dedup_indices, domain_standardized_series,
                            engagement_of, length_bin_codes,
                            parse_local_time, run_parallel, within_cell_spearman)
from derived_cache import iter_cached
from engagement_drivers_analysis import (cluster_bootstrap_fe_ols, fe_ols,
                                         prepare_fe)

DATASETS = {
    "human": BASE + "training_set_human_fixed.jsonl",
    "aigc": BASE + "training_set_aigc.jsonl",
    "exploring": BASE + "exploring_set_fixed.jsonl",
}
SCORES_PATH = BASE + "paper_exploring_scores.csv.gz"
FEATURES_PATH = BASE + "paper_features_exploring.csv.gz"

MIN_BODY_LEN = 50        # CTA 密度在更短正文（纯标签/空正文）上无意义
MIN_CELL_N = 100         # 单元内统计的最小单元样本量（与论文口径一致）
MONTH_FROM = "202301"    # 互动分析起始月份（与 paper_engagement 一致）
N_BOOT = 300             # 整群 bootstrap 次数
BOOT_SEED = 42
MIN_QUARTER_N = 300
MIN_FAMILY_N = 100

# 先验标记：类别 -> (中文标签, [标记短语, ...])
CATEGORY_DEFS = {
    "comment": ("评论征集", [
        "评论区", "留言区", "留言", "告诉我", "聊聊", "分享一下",
        "你们呢", "你呢", "交流",
    ]),
    "reaction": ("点赞收藏", ["点赞", "收藏", "码住"]),
    "follow": ("关注转发", ["关注我", "关注一下", "转发", "艾特", "分享给"]),
    "urge": ("催购催促", ["快冲", "冲鸭", "别错过"]),
}

# 高歧义短语的显式排除词（等价于原始正则的否定前瞻）：
# 「留言→留言区」「冲鸭→快冲鸭」是同类别内最长优先匹配的重复计数扣除，
# 用 `str.count` 之差计数；以标记本身开头的排除词（收藏夹/聊聊天）
# 则用「标记 + 否定前瞻」正则精确计数——与排除词邻接重叠的出现
#（如「聊聊聊天」）无法用简单扣除表达。
GUARDED = {
    "收藏": ("收藏夹", "收藏品", "收藏家"),
    "聊聊": ("聊聊天",),
    "留言": ("留言区",),
    "冲鸭": ("快冲鸭",),
}
GUARDED_RE = {
    w: re.compile(re.escape(w) + "(?!"
                  + "|".join(re.escape(g[len(w):]) for g in guards
                             if g.startswith(w)) + ")")
    for w, guards in GUARDED.items()
    if any(g.startswith(w) for g in guards)
}
# 逐标记展平（模块级预计算）：(标记, 类别, 前瞻正则或 None, 需扣除的后缀元组)。
# 热循环内不再逐词查 GUARDED/GUARDED_RE、不再做 startswith 过滤。
_FLAT_CTA_WORDS = [
    (w, cat, GUARDED_RE.get(w),
     tuple(g for g in GUARDED.get(w, ()) if not g.startswith(w)))
    for cat, (_, words) in CATEGORY_DEFS.items() for w in words
]
CATEGORY_LABELS = {k: v[0] for k, v in CATEGORY_DEFS.items()}
CATEGORY_COLORS = {"comment": "#4C72B0", "reaction": "#C44E52",
                   "follow": "#DD8452", "urge": "#8172B3"}
CATEGORY_ORDER = ["comment", "reaction", "follow", "urge"]
SUBSET_ORDER = ["human", "aigc", "exploring"]
ANY_LABEL = "任一 CTA"
ANY_COLOR = "#55A868"

# 图例名 -> 类别；保证类别内最长优先（同长保持定义顺序，与原正则 alternation 一致）
PATTERNS = [(display, cat)
            for cat in CATEGORY_ORDER
            for display in sorted(CATEGORY_DEFS[cat][1],
                                  key=lambda w: -len(w))]
PATTERN_DISPLAY = [p[0] for p in PATTERNS]
PATTERN_CATEGORY = {p[0]: p[1] for p in PATTERNS}

FAMILY_NAME = {"anthropic": "Claude", "deepseek": "DeepSeek", "gemini": "Gemini",
               "glm": "GLM", "gpt": "GPT", "qwen": "Qwen"}
FAMILIES = ["anthropic", "deepseek", "gemini", "glm", "gpt", "qwen"]
TARGETS = (("likes", "点赞"), ("collections", "收藏"), ("comments", "评论"))


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def cta_counts(text: str) -> dict:
    """统计一段正文的四个类别命中数与逐标记命中数。

    返回 {"comment": n, ..., "_patterns": {图例名: n}}；同一类别内
    最长优先匹配、不重复计数（如「留言区」计为一次「留言区」而不是
    「留言」+「区」），类别计数等于该类逐标记计数之和。

    实现：无排除词的标记用 `str.count`（C 层非重叠计数）；以标记开头
    的排除词（收藏夹/聊聊天）用预编译的「标记 + 否定前瞻」正则计数；
    其余排除词（快冲鸭）按最长优先从计数中扣除。与原「最长优先正则
    alternation」在本词表上逐位一致（含「聊聊聊天」这类邻接重叠）。
    """
    out: dict = {cat: 0 for cat in CATEGORY_ORDER}
    patterns: dict = {}
    for w, cat, guard_re, sub_guards in _FLAT_CTA_WORDS:
        if guard_re is not None:
            n = len(guard_re.findall(text))
        else:
            n = text.count(w)
        for g in sub_guards:
            n -= text.count(g)
        if n:
            out[cat] += n
            patterns[w] = patterns.get(w, 0) + n
    out["_patterns"] = patterns
    return out


def has_any(counts: dict) -> int:
    """任一类别命中即为 1（帖子层 CTA 普及率）。"""
    return int(any(counts[k] > 0 for k in CATEGORY_ORDER))


def density_per_100(count: int, body_len: int) -> float:
    """次数 / 正文字数 * 100；空正文返回 0。"""
    return count / body_len * 100 if body_len > 0 else 0.0


def any_density(counts: dict, body_len: int) -> float:
    """四个类别的合计密度（次/百字）。"""
    return density_per_100(sum(counts[k] for k in CATEGORY_ORDER), body_len)


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
_PATTERN_INDEX = {w: i for i, w in enumerate(PATTERN_DISPLAY)}


def scan_corpus(path: str) -> dict:
    """单次流式扫描一个语料，返回逐记录数组。

    每条记录保存：正文长度、note_id、元数据完整度（用于去重取舍）、
    月份、领域、模型家族/模型（AIGC），以及四个类别的命中数与逐标记命中数。

    逐标记计数写入按 `PATTERN_DISPLAY` 顺序拉平的 `array('i')`，最后一次性
    reshape/切片：避免每帖 20 次 `dict.get` + `list.append`（全量三语料
    约 300 万次 Python 调用），且 int32 平铺存储比 20 个 Python int 列表
    更省内存。`content.strip()` 只用于「正文是否有非空白内容」的元数据
    评分，与原先 `"".join(content.split())` 的布尔值完全一致。
    """
    body_len, note_id, meta, month, domain = [], [], [], [], []
    family, model = [], []
    word_index = _PATTERN_INDEX
    n_words = len(PATTERN_DISPLAY)
    cat_counts = {k: [] for k in CATEGORY_ORDER}
    flat_words = array("i")
    likes = []
    n_total = 0
    # 重复正文（探索集约 45%）直接复用上一次的命中计数，只读不改。
    cached = body_cache(cta_counts, max_entries=60_000)
    for rec in iter_cached(path):
        n_total += 1
        content = content_of(rec)
        body = clean_body(content)
        L = len(body)
        counts = cached(body)
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
        for k in CATEGORY_ORDER:
            cat_counts[k].append(counts[k])
        row = [0] * n_words
        for w, n in counts["_patterns"].items():
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
        "pattern_counts": {w: word_matrix[:, i]
                           for i, w in enumerate(PATTERN_DISPLAY)},
    }


def dedup_keep(scan: dict) -> np.ndarray:
    """按 note_id 去重（保留元数据最完整一条），返回排序后的保留索引。"""
    keep = dedup_indices(scan["note_id"], scan["meta"])
    return np.asarray(sorted(keep), dtype=int)


def rates_block(scan: dict, keep: np.ndarray, min_len: int = MIN_BODY_LEN) -> dict:
    """一个子集的类别/逐标记使用率（仅正文 >= min_len 的记录）。

    类别与「任一 CTA」都报告帖子层普及率（share）与合并密度；逐标记
    额外报告每万字符使用率。
    """
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
    words = {}
    for w, arr in scan["pattern_counts"].items():
        c = arr[sel].astype(float)
        words[w] = {
            "n": n,
            "per10k_chars": float(c.sum() / chars * 10000) if chars else 0.0,
            "share": float((c > 0).mean()) if n else 0.0,
        }
    return {"n": n, "chars": chars, "categories": cat, "words": words}


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_lexicon(subset_rates: dict, word_table: dict) -> None:
    """左：三个子集的类别密度；右：逐标记 AIGC−人类使用差（探索集为圆点）。"""
    init_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.6),
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
    ax.set_xlabel("CTA 密度（次/百字，正文 ≥ %d 字）" % MIN_BODY_LEN)
    ax.set_title("(a) 三类语料的互动引导语（CTA）密度")
    ax.legend(loc="lower right")

    # ---- (b) 逐标记 AIGC−人类 使用差 ----
    ax = axes[1]
    rows = []
    for w, rec in word_table.items():
        rows.append((w, rec["category"],
                     rec["aigc"]["per10k_chars"] - rec["human"]["per10k_chars"],
                     rec["exploring"]["per10k_chars"]))
    rows.sort(key=lambda r: (CATEGORY_ORDER.index(r[1]), -r[2]))
    ypos = np.arange(len(rows))
    diffs = np.asarray([r[2] for r in rows])
    colors = [CATEGORY_COLORS[r[1]] for r in rows]
    for cat_i, k in enumerate(CATEGORY_ORDER):
        idx = [i for i, r in enumerate(rows) if r[1] == k]
        if cat_i % 2 == 1 and idx:
            ax.axhspan(min(idx) - 0.5, max(idx) + 0.5, color="#EEEEEE", zorder=0)
    ax.barh(ypos, diffs, color=colors, height=0.72, alpha=0.9, zorder=2)
    ax.scatter([r[3] for r in rows], ypos, s=22, color="black", zorder=4,
               edgecolor="white", linewidth=0.5, label="探索集使用率")
    ax.axvline(0, color="black", lw=0.8, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    for i, r in enumerate(rows):
        v = r[2]
        ax.annotate(r[0], xy=(0, ypos[i]), xytext=(-7 if v >= 0 else 7, 0),
                    textcoords="offset points", ha="right" if v >= 0 else "left",
                    va="center", fontsize=8.5)
        ax.annotate(f"{v:+.2f}", xy=(v, ypos[i]),
                    xytext=(6 if v >= 0 else -6, 4),
                    textcoords="offset points", fontsize=7.2,
                    ha="left" if v >= 0 else "right")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("AIGC − 人类 使用率差（次/万字符；越右=生成集越爱用）")
    ax.set_title("(b) 生成集对每个 CTA 标记的过量使用（黑点=探索集使用率）")
    handles = [plt.Line2D([], [], color=CATEGORY_COLORS[k], lw=6,
                          label=CATEGORY_LABELS[k]) for k in CATEGORY_ORDER]
    handles.append(plt.Line2D([], [], color="black", marker="o", ls="",
                              ms=4, label="探索集使用率"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    save_fig(fig, INSIGHTS + "cta_lexicon.png")


def plot_gradient(grad: dict, word_rho: dict, eng: dict, family_table: dict,
                  quarterly: dict, moderation: dict) -> None:
    """六联图：P(AI) 梯度 / 逐标记 ρ / 三指标偏效应 / 家族指纹 /
    季度趋势 / 交互模型下的 P(AI) 边际斜率。"""
    init_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 12.4))

    # ---- (a) P(AI) 十分位密度曲线 ----
    ax = axes[0, 0]
    for k in CATEGORY_ORDER:
        dec = grad["categories"][k]["deciles"]
        x = [d["decile"] for d in dec]
        yv = [d["mean"] for d in dec]
        lo = [d["ci_low"] for d in dec]
        hi = [d["ci_high"] for d in dec]
        ax.plot(x, yv, "o-", color=CATEGORY_COLORS[k], lw=1.8, ms=5,
                label=f"{CATEGORY_LABELS[k]}（ρ={grad['categories'][k]['weighted_rho']:+.3f}）")
        ax.fill_between(x, lo, hi, color=CATEGORY_COLORS[k], alpha=0.16)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("检测器 P(AI) 十分位（1=最像人，10=最像 AI）")
    ax.set_ylabel("密度（次/百字）")
    ax.set_title("(a) 探索集：CTA 密度随 P(AI) 的梯度（含 95% CI）")
    ax.legend(fontsize=8.5)

    # ---- (b) 逐标记单元内 ρ ----
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
        rho = grad["categories"][k]["weighted_rho"]
        ax.axvline(rho, color=CATEGORY_COLORS[k], lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("月份×领域单元内加权 Spearman ρ（密度 vs P(AI)）")
    ax.set_title("(b) 逐标记的野外 P(AI) 梯度（虚线=类别整体 ρ）")
    handles = [plt.Line2D([], [], color=CATEGORY_COLORS[k], lw=6,
                          label=CATEGORY_LABELS[k]) for k in CATEGORY_ORDER]
    ax.legend(handles=handles, loc="lower left", fontsize=8)

    # ---- (c) 三指标互动偏效应森林图 ----
    ax = axes[0, 2]
    fe_likes = eng["fe"]["likes"]
    names = list(fe_likes["coef"].keys())
    order = list(reversed(names))
    ypos = np.arange(len(order))
    markers = (("likes", "o", -0.24, "#4C72B0", "点赞"),
               ("collections", "^", 0.0, "#DD8452", "收藏"),
               ("comments", "s", 0.24, "#55A868", "评论"))
    for target, marker, offset, color, tlabel in markers:
        fe = eng["fe"][target]
        first = True
        for i, name in enumerate(order):
            b = fe["coef"][name]
            lo, hi = fe["ci_low"][name], fe["ci_high"][name]
            sig = lo > 0 or hi < 0
            col = color if sig else "#9E9E9E"
            ax.plot([lo, hi], [ypos[i] + offset, ypos[i] + offset], color=col,
                    lw=1.6, solid_capstyle="butt", zorder=2)
            ax.scatter([b], [ypos[i] + offset], marker=marker, s=30,
                       facecolor=col if sig else "white", edgecolor=col,
                       zorder=3, label=tlabel if first else None)
            first = False
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    labels = {"comment": "评论征集", "reaction": "点赞收藏", "follow": "关注转发",
              "urge": "催购催促", "any": ANY_LABEL, "p_ai": "P(AI)",
              "content_len": "正文长度", "n_hashtags": "标签数"}
    ax.set_yticklabels([labels.get(n, n) for n in order])
    ax.set_xlabel("单元内 FE β（每 1 SD → log10(1+互动)）")
    ax.set_title(f"(c) CTA 的互动偏效应（n={fe_likes['n_obs']:,}，"
                 f"{fe_likes['n_cells']} 单元，整群 bootstrap）")
    ax.legend(loc="lower right")

    # ---- (d) 家族指纹（按类别列内 z-score 着色的热力图） ----
    ax = axes[1, 0]
    fams = [f for f in FAMILIES if family_table.get(f, {}).get("n", 0) >= MIN_FAMILY_N]
    mat = np.asarray([[family_table[f]["density100"][k] for k in CATEGORY_ORDER]
                      for f in fams], dtype=float)
    z = (mat - mat.mean(axis=0)) / np.where(mat.std(axis=0) > 0,
                                            mat.std(axis=0), 1.0)
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
    ax.grid(False)
    for i in range(len(fams)):
        for j in range(len(CATEGORY_ORDER)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=9.5,
                    color="white" if abs(z[i, j]) > 1.15 else "black")
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels([CATEGORY_LABELS[k] for k in CATEGORY_ORDER], fontsize=9)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f"{FAMILY_NAME[f]}（n={family_table[f]['n']}）"
                        for f in fams], fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("家族差异（列内 z-score）", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("(d) AIGC 训练集的家族 CTA 指纹（格内为密度，次/百字）")

    # ---- (e) 季度趋势（原始 + 固定领域权重同图对照） ----
    ax = axes[1, 1]
    qs = quarterly["quarters"]
    x = np.arange(len(qs))
    for k in CATEGORY_ORDER:
        raw = quarterly["raw"][k]
        std = quarterly["domain_std"][k]
        t = quarterly["trend"][k]
        line, = ax.plot(x, raw, "o-", color=CATEGORY_COLORS[k], lw=1.7, ms=4,
                        label=f"{CATEGORY_LABELS[k]}（ρ={t['rho']:+.2f}，"
                              f"p={t['p']:.2f}）")
        ax.plot(x, std, "--", color=line.get_color(), lw=1.2, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("密度（次/百字）")
    ax.set_title("(e) 探索集季度趋势（实线=原始均值，虚线=固定领域权重）")
    ax.legend(fontsize=8)

    # ---- (f) 交互模型：P(AI) 边际斜率（无 CTA vs 有 CTA） ----
    ax = axes[1, 2]
    groups = (("no_cta", "无 CTA", "#4C72B0"), ("cta", "有 CTA", "#C44E52"))
    for j, (target, tlabel) in enumerate((("likes", "点赞"), ("comments", "评论"))):
        for i, (key, glabel, color) in enumerate(groups):
            slot = moderation[target]["marginal_slope"][key]
            yv = j + (i - 0.5) * 0.26
            sig = slot["ci_low"] > 0 or slot["ci_high"] < 0
            ax.plot([slot["ci_low"], slot["ci_high"]], [yv, yv], color=color,
                    lw=2.0, solid_capstyle="butt")
            ax.scatter([slot["coef"]], [yv], s=46, color=color if sig else "white",
                       edgecolor=color, zorder=3)
            ax.annotate(f"{slot['coef']:+.3f}", xy=(slot["coef"], yv),
                        xytext=(7, -3), textcoords="offset points", fontsize=8)
    handles = [plt.Line2D([], [], color=c, lw=6, label=g) for _, g, c in groups]
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(2))
    ax.set_yticklabels(["点赞", "评论"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("P(AI) 边际斜率（每 1 SD → log10(1+互动)，含 95% CI）")
    ax.set_title("(f) 交互模型：AI 味惩罚在有无 CTA 帖子间是否不同")
    ax.legend(handles=handles, fontsize=8.5, loc="lower right")
    save_fig(fig, INSIGHTS + "cta_gradient.png")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
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
    for w in PATTERN_DISPLAY:
        word_table[w] = {
            "category": PATTERN_CATEGORY[w],
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

    any_sel = np.zeros(len(sel), dtype=float)
    for k in CATEGORY_ORDER:
        any_sel += expl["cat_counts"][k][sel]
    any_sel = (any_sel > 0).astype(float)

    groups = cell_groups(cells, MIN_CELL_N)
    # p_ai 是所有类别/逐标记的同一对侧向量：预算一次单元内平均秩供全部调用
    # 复用（原先每个类别与逐标记都会重算一次，共 25 次重复排序）；上下十分位
    # 阈值同样只与 p_ai 有关，循环外只算一次；「任一 CTA」的十分位份额只对
    # k=="any" 输出，不再对每个类别都算一遍却丢弃。
    p_ai_ranks = cell_rank_cache(p_ai, groups)
    groups_strat = cell_groups(strat, MIN_CELL_N)
    p_ai_ranks_strat = cell_rank_cache(p_ai, groups_strat)
    p_hi, p_lo = float(np.quantile(p_ai, 0.9)), float(np.quantile(p_ai, 0.1))
    grad_cat = {}
    for k in CATEGORY_ORDER + ["any"]:
        if k == "any":
            tot = np.zeros(len(sel), dtype=float)
            for kk in CATEGORY_ORDER:
                tot += expl["cat_counts"][kk][sel].astype(float)
        else:
            tot = expl["cat_counts"][k][sel].astype(float)
        dens = tot / body * 100
        dec = decile_table(dens, p_ai)
        wc = within_cell_spearman(dens, p_ai, cells, min_cell_n=MIN_CELL_N,
                                  groups=groups, y_ranks=p_ai_ranks)
        st = within_cell_spearman(dens, p_ai, strat, min_cell_n=MIN_CELL_N,
                                  groups=groups_strat, y_ranks=p_ai_ranks_strat)
        lo_p = p_ai <= p_lo
        hi_p = p_ai >= p_hi
        delta, dp = cliffs_delta(dens[hi_p], dens[lo_p])
        grad_cat[k] = {
            "deciles": dec,
            "any_share_deciles": (decile_table(any_sel, p_ai)
                                  if k == "any" else None),
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
        print(f"  {k}: wcell ρ={wc['weighted_rho']:+.4f} "
              f"(长度分层 {st['weighted_rho']:+.4f})，δ={delta:+.4f}")

    word_rho = {}
    for w in PATTERN_DISPLAY:
        arr = expl["pattern_counts"][w]
        dens = arr[sel].astype(float) / body * 100
        if dens.std() == 0:
            continue
        wc = within_cell_spearman(dens, p_ai, cells, min_cell_n=MIN_CELL_N,
                                  groups=groups, y_ranks=p_ai_ranks)
        word_rho[w] = {"category": PATTERN_CATEGORY[w],
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
    cta_dens = np.column_stack([
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
    X = np.column_stack([cta_dens, p_ai_all,
                         scans["exploring"]["body_len"].astype(float),
                         model["n_hashtags"].to_numpy(dtype=float)])
    feature_names = ["comment", "reaction", "follow", "urge", "p_ai",
                     "content_len", "n_hashtags"]
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
              f"comment β={fit['coef'][0]:+.4f} "
              f"[{ci[0][0]:+.4f}, {ci[1][0]:+.4f}]")

    # 单变量：4 类别 + any × 3 指标（预分组一次共享）
    groups_eng = cell_groups(eng_cells_raw[keep_eng], MIN_CELL_N)
    y_ranks = {t: cell_rank_cache(targets[t][keep_eng], groups_eng)
               for t, _ in TARGETS}
    uni_cols = {}
    for i, k in enumerate(CATEGORY_ORDER):
        uni_cols[k] = cta_dens[keep_eng][:, i]
    any_eng = (np.sum(cta_dens[keep_eng], axis=1) > 0).astype(float)
    uni_cols["any"] = any_eng
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

    # ---- 交互模型：CTA 是否改变 P(AI) 的互动偏效应 ----
    # 单一模型（同单元集）比分层拟合更干净：y ~ z(p_ai) + z(any) + z(p_ai×any)
    # + 正文长度 + 标签数；交互项显著说明「AI 味惩罚」在有无 CTA 的帖子间不同。
    # 边际斜率 = ∂y/∂z(p_ai) = β_p + β_int×(被重新标准化的交互项对 z(p_ai) 的
    # 斜率修正)，这里直接在 bootstrap 系数上按 za 取值线性换算。
    p_ai_ke = p_ai_all[keep_eng]
    len_ke = scans["exploring"]["body_len"][keep_eng].astype(float)
    tags_ke = model["n_hashtags"].to_numpy(dtype=float)[keep_eng]
    mean_a, sd_a = float(any_eng.mean()), float(any_eng.std())
    zp = (p_ai_ke - p_ai_ke.mean()) / p_ai_ke.std()
    za = (any_eng - mean_a) / (sd_a if sd_a > 0 else 1.0)
    Xm = np.column_stack([zp, za, zp * za, len_ke, tags_ke])
    mnames = ["p_ai", "any", "p_ai_x_any", "content_len", "n_hashtags"]
    prep_m = prepare_fe(Xm, eng_cells_raw[keep_eng], MIN_CELL_N)
    za0 = (0.0 - mean_a) / (sd_a if sd_a > 0 else 1.0)
    za1 = (1.0 - mean_a) / (sd_a if sd_a > 0 else 1.0)

    def marginal_slopes(draws: np.ndarray, scale_zp: float, scale_prod: float):
        """由系数抽样换算 any=0/1 处的 P(AI) 边际斜率（每 1 SD P(AI)）。"""
        out = {}
        for key, za_v in (("no_cta", za0), ("cta", za1)):
            s = draws[:, 0] / scale_zp + draws[:, 2] * za_v / scale_prod
            out[key] = s
        return out

    moderation = {}
    for target, _ in (("likes", "点赞"), ("comments", "评论")):
        yk = targets[target][keep_eng]
        fit = fe_ols(Xm, yk, eng_cells_raw[keep_eng], prepared=prep_m)
        boot = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                        fit["cell_codes"], n_boot=N_BOOT,
                                        seed=BOOT_SEED)
        ci = np.percentile(boot, [2.5, 97.5], axis=0)
        slopes = marginal_slopes(boot, float(prep_m["scale"][0]),
                                 float(prep_m["scale"][2]))
        slope_point = {
            key: float(fit["coef"][0] / prep_m["scale"][0]
                       + fit["coef"][2] * za_v / prep_m["scale"][2])
            for key, za_v in (("no_cta", za0), ("cta", za1))}
        moderation[target] = {
            "features": mnames,
            "coef": {n: round(float(v), 4) for n, v in zip(mnames, fit["coef"])},
            "ci_low": {n: round(float(v), 4) for n, v in zip(mnames, ci[0])},
            "ci_high": {n: round(float(v), 4) for n, v in zip(mnames, ci[1])},
            "marginal_slope": {
                key: {
                    "coef": round(slope_point[key], 4),
                    "ci_low": round(float(np.percentile(slopes[key], 2.5)), 4),
                    "ci_high": round(float(np.percentile(slopes[key], 97.5)), 4),
                } for key in ("no_cta", "cta")},
            "n_obs": fit["n_obs"], "n_cells": fit["n_cells"],
        }
        b_int = fit["coef"][2]
        p_int = 2 * min(float((boot[:, 2] <= 0).mean()),
                        float((boot[:, 2] >= 0).mean()))
        moderation[target]["interaction_p_boot"] = round(p_int, 4)
        print(f"  交互模型（{target}）：p_ai×any β={b_int:+.4f} "
              f"[{ci[0][2]:+.4f}, {ci[1][2]:+.4f}]（bootstrap p={p_int:.3f}）；"
              f"边际斜率 无CTA {slope_point['no_cta']:+.4f} / 有CTA {slope_point['cta']:+.4f}")

    # ---- AIGC 家族 / 单模型指纹 ----
    aigc = scans["aigc"]
    ok = aigc["body_len"] >= MIN_BODY_LEN
    fam_agg: dict = defaultdict(lambda: {"n": 0, "chars": 0, "counts": defaultdict(int)})
    mod_agg: dict = defaultdict(lambda: {"n": 0, "chars": 0, "counts": defaultdict(int)})
    for i in np.flatnonzero(ok):
        for agg, key in ((fam_agg, aigc["family"][i]), (mod_agg, aigc["model"][i])):
            slot = agg[key]
            slot["n"] += 1
            slot["chars"] += int(aigc["body_len"][i])
            for k in CATEGORY_ORDER:
                slot["counts"][k] += int(aigc["cat_counts"][k][i])

    def agg_to_table(agg):
        out = {}
        for key, slot in agg.items():
            if not key:
                continue
            out[key] = {
                "n": slot["n"],
                "density100": {
                    k: round(slot["counts"][k] / slot["chars"] * 100, 4)
                    if slot["chars"] else 0.0
                    for k in CATEGORY_ORDER},
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
    q_df["any"] = q_df[CATEGORY_ORDER].sum(axis=1)
    q_df = q_df[q_df["quarter"] != "未知"]
    q_counts = q_df.groupby("quarter").size()
    valid_q = sorted(q_counts[q_counts >= MIN_QUARTER_N].index)
    q_df = q_df[q_df["quarter"].isin(valid_q)]
    quarterly = {"quarters": valid_q, "n": {q: int(q_counts[q]) for q in valid_q},
                 "raw": {}, "domain_std": {}, "trend": {}, "share_any": {}}
    for k in CATEGORY_ORDER:
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
    any_share_q = q_df.assign(hit=q_df["any"] > 0).groupby("quarter")["hit"].mean().reindex(valid_q)
    quarterly["share_any"]["any"] = [round(float(v), 4) for v in any_share_q]

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
                    "paper_exploring_scores.csv.gz；标记为先验短语，"
                    "收藏排除收藏夹/收藏品/收藏家，聊聊排除聊聊天",
        },
        "lexicon": {k: {"label": CATEGORY_LABELS[k],
                        "patterns": [d for d, c in PATTERNS if c == k]}
                    for k in CATEGORY_ORDER},
        "subset_rates": {
            name: {
                "n": subset_rates[name]["n"],
                "chars": subset_rates[name]["chars"],
                "categories": subset_rates[name]["categories"],
            } for name in SUBSET_ORDER},
        "word_rates": word_table,
        "exploring_gradient": {
            "n": int(len(sel)),
            "short_excluded": int((expl["body_len"][keep] < MIN_BODY_LEN).sum()),
            "n_cells": int(len(np.unique(cells))),
            "categories": grad_cat,
            "words": word_rho,
        },
        "engagement": {
            "month_from": MONTH_FROM, "min_body_len": MIN_BODY_LEN,
            "univariate": {
                k: {t: {"weighted_rho": round(v[t]["weighted_rho"], 4),
                        "neg_cell_share": round(v[t]["neg_cell_share"], 4),
                        "n_cells": v[t]["n_cells"]}
                    for t, _ in TARGETS}
                for k, v in eng["univariate"].items()},
            "fe": eng["fe"],
            "moderation": moderation,
        },
        "family": {"families": family_table, "models": model_table},
        "quarterly": quarterly,
    }
    save_json(out, BASE + "cta_results.json")
    print("已写出 cta_results.json")
    write_markdown(out)
    plot_lexicon(subset_rates, word_table)
    plot_gradient({"categories": grad_cat}, word_rho, eng,
                  family_table, quarterly, moderation)
    print("已写出 cta_results.md 与 insights/cta_*.png")


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
        "# 互动引导语（CTA）分析结果",
        "",
        f"> 由 `cta_analysis.py` 全量运行生成；正文清洗与 `paper_features.py` 一致，"
        f"人类/探索集按 note_id 去重；正文 ≥{meta['min_body_len']} 字；"
        f"探索集 p_ai 按行序对齐 `paper_exploring_scores.csv.gz`"
        f"（content_len 校验 mismatch={meta['exploring_content_len_mismatch']}）。"
        f"CTA 标记为先验短语（不做数据驱动筛选）。",
        "",
        "## 1 跨子集 CTA 使用密度与普及率",
        "",
        "| 子集 | 有效帖数 | " + " | ".join(
            [f"{CATEGORY_LABELS[k]} 密度" for k in CATEGORY_ORDER]
            + [f"{ANY_LABEL} 普及率"]) + " |",
        "|---|---:|" + "---:|" * (len(CATEGORY_ORDER) + 1),
    ]
    for name in SUBSET_ORDER:
        cats = rates[name]["categories"]
        lines.append(
            f"| {labels[name]} | {rates[name]['n']:,} | "
            + " | ".join(f"{cats[k]['density100']:.4f}" for k in CATEGORY_ORDER)
            + f" | {cats['any']['share']:.1%} |")
    lines += [
        "",
        "> 注：密度为「总命中次数/总字符数×100」；任一 CTA 普及率为帖子层占"
        "比（≥1 次命中）。三个子集正文长度分布不同，跨子集比较以帖子层普及率"
        "更稳健，密度供梯度与互动分析使用。",
        "",
        "**逐标记 AIGC−人类过量使用（次/万字符）：**",
        "",
        "| 标记 | 类别 | 人类 | AIGC | 探索集 | AIGC−人类 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    rows = sorted(out["word_rates"].items(),
                  key=lambda kv: (CATEGORY_ORDER.index(kv[1]["category"]),
                                  -(kv[1]["aigc"]["per10k_chars"]
                                    - kv[1]["human"]["per10k_chars"])))
    for w, rec in rows:
        d = rec["aigc"]["per10k_chars"] - rec["human"]["per10k_chars"]
        lines.append(f"| {w} | {CATEGORY_LABELS[rec['category']]} | "
                     f"{rec['human']['per10k_chars']:.3f} | "
                     f"{rec['aigc']['per10k_chars']:.3f} | "
                     f"{rec['exploring']['per10k_chars']:.3f} | {d:+.3f} |")
    lines += [
        "",
        "## 2 探索集内的 P(AI) 梯度（去重，正文 ≥%d 字）" % meta["min_body_len"],
        "",
        "| 类别 | 单元内加权 ρ | 负单元占比 | 长度分层 ρ | 首尾十分位 δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in CATEGORY_ORDER + ["any"]:
        g = grad["categories"][k]
        name = CATEGORY_LABELS.get(k, ANY_LABEL)
        lines.append(
            f"| {name} | {g['weighted_rho']:+.4f} | "
            f"{g['neg_cell_share']:.1%} | {g['stratified_rho']:+.4f} | "
            f"{g['cliff_delta_top_vs_bottom']:+.3f} |")
    lines += [
        "",
        f"（样本 {grad['n']:,} 条 / {grad['n_cells']} 个「月份×领域」单元；"
        f"长度分层 = 月份×领域×长度分箱单元。）",
        "",
        "**逐标记单元内 ρ（Top/Bottom 8）：**",
        "",
    ]
    items = sorted(grad["words"].items(), key=lambda kv: kv[1]["weighted_rho"])
    show = items[:8] + items[-8:]
    lines.append("| 标记 | 类别 | 单元内 ρ | 负单元占比 |")
    lines.append("|---|---|---:|---:|")
    for w, r in show:
        lines.append(f"| {w} | {CATEGORY_LABELS[r['category']]} | "
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
        for t, _ in TARGETS:
            fe = eng["fe"][t]
            cells_txt.append(
                f"{fe['coef'][name]:+.4f} "
                f"[{fe['ci_low'][name]:+.4f}, {fe['ci_high'][name]:+.4f}]")
        lines.append(f"| {name} | " + " | ".join(cells_txt) + " |")
    lines += [
        "",
        f"（n={eng['fe']['likes']['n_obs']:,}，{eng['fe']['likes']['n_cells']} "
        f"个单元；特征已 z-score，β 为每 1 SD 的偏效应；{meta['month_from']} "
        f"起、正文 ≥{meta['min_body_len']} 字；评论端为本次新增口径。）",
        "",
        "**交互模型：CTA 是否改变 P(AI) 的互动偏效应**",
        "",
        "| 被解释变量 | p_ai×any 交互 β [95% CI] | 无 CTA 边际斜率 [95% CI] | 有 CTA 边际斜率 [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for target, tlabel in TARGETS:
        if target not in eng["moderation"]:
            continue
        mo = eng["moderation"][target]
        s0, s1 = mo["marginal_slope"]["no_cta"], mo["marginal_slope"]["cta"]
        lines.append(
            f"| {tlabel} | {mo['coef']['p_ai_x_any']:+.4f} "
            f"[{mo['ci_low']['p_ai_x_any']:+.4f}, {mo['ci_high']['p_ai_x_any']:+.4f}] | "
            f"{s0['coef']:+.4f} [{s0['ci_low']:+.4f}, {s0['ci_high']:+.4f}] | "
            f"{s1['coef']:+.4f} [{s1['ci_low']:+.4f}, {s1['ci_high']:+.4f}] |")
    mo_l = eng["moderation"]["likes"]
    lines += [
        "",
        f"（交互模型 n={mo_l['n_obs']:,}，{mo_l['n_cells']} 个单元；"
        f"特征：z(P(AI))、z(任一 CTA)、z(P(AI)×任一 CTA)、正文长度、标签数；"
        f"交互项 bootstrap p={mo_l['interaction_p_boot']:.3f}（点赞）。"
        f"边际斜率为每 1 SD P(AI) 对 log10(1+互动) 的斜率。）",
        "",
        "## 4 AIGC 家族 CTA 指纹（次/百字）",
        "",
        "| 家族 | n | " + " | ".join(CATEGORY_LABELS[k] for k in CATEGORY_ORDER) + " |",
        "|---|---:|" + "---:|" * len(CATEGORY_ORDER),
    ]
    for fam in FAMILIES:
        slot = out["family"]["families"].get(fam)
        if not slot:
            continue
        lines.append(
            f"| {FAMILY_NAME.get(fam, fam)} | {slot['n']} | "
            + " | ".join(f"{slot['density100'][k]:.4f}" for k in CATEGORY_ORDER)
            + " |")
    lines += [
        "",
        "## 5 季度趋势（探索集去重）",
        "",
        "| 季度 | n | " + " | ".join(CATEGORY_LABELS[k] for k in CATEGORY_ORDER) + " |",
        "|---|---:|" + "---:|" * len(CATEGORY_ORDER),
    ]
    for i, qq in enumerate(q["quarters"]):
        lines.append(
            f"| {qq} | {q['n'][qq]:,} | "
            + " | ".join(f"{q['raw'][k][i]:.4f}" for k in CATEGORY_ORDER)
            + " |")
    lines.append("")
    for k in CATEGORY_ORDER:
        t = q["trend"][k]
        lines.append(f"- {CATEGORY_LABELS[k]}：季度趋势 ρ={t['rho']:+.3f}（p={t['p']:.3f}），"
                     f"固定领域权重后 ρ={t['rho_domain_std']:+.3f}（p={t['p_domain_std']:.3f}）。")
    lines += [
        "",
        "图表见 `insights/cta_lexicon.png`、`insights/cta_gradient.png`。",
    ]
    with open(BASE + "cta_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
