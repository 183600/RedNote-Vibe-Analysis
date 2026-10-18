# -*- coding: utf-8 -*-
"""深潜分析之十八：标点与排版层（typography）。

既有分析已经发现「高密度标点 + 规整句式」是 AI 味互动惩罚的主要载体
（`engagement_drivers_analysis.py` 中逗号密度的多元偏效应最强），也刻画了
开篇/收尾模板与标签/表情生态，但还没有把**标点符号本身**与**帖子排版**当作
一个独立维度系统分析。本脚本用先验定义的 12 个标点族与 7 个排版指标
（不做数据驱动筛选）回答：

1. 跨子集差异：三个语料的标点族密度/帖子层普及率、逐标记 AIGC−人类过量使用、
   排版画像（行数、平均行长、碎片化行、列表行、冒号行、单行帖）与
   「标点丰富度」（一篇帖子用到的标点族数量）；
2. 野外梯度：探索集内各标点族/排版指标随检测器 P(AI) 的十分位曲线、
   月份×领域单元内加权 ρ（含月份×领域×长度分箱的稳健性）、首尾十分位
   Cliff's δ；
3. 互动回报：与 `engagement_drivers_analysis.py` 完全同协议的月份×领域固定
   效应 OLS（特征先 z-score 再单元内去均值、单元 ≥100 条、整群 bootstrap
   300 次），在只控制其余标点族与 P(AI)/正文长度/标签数的模型里给出每个
   标点族的偏效应（点赞、收藏、评论三端）；
4. 生成端指纹：17 个模型 / 6 个家族的标点族密度；
5. 时间演化：探索集标点族密度的季度趋势（原始 + 固定领域权重）。

口径说明：
- 正文清洗复用 `analysis_utils.clean_body`（与 `paper_features.py` 一致）；
- 人类/探索集按 note_id 去重（保留元数据最完整一条，规则同
  `deep_dive_analysis.py`），AIGC 无 note_id，逐记录处理；
- 探索集 p_ai 按行序对齐 `paper_exploring_scores.csv.gz`，脚本内用
  content_len 逐条校验（不一致直接报错退出）；
- 标点密度统一按「正文每百字」计；正文不足 MIN_BODY_LEN 字的帖子
  （纯标签/空正文等）不参与统计，数量单独报告；
- 标点族聚合了全角/半角变体（如感叹号同时统计 `！` 与 `!`）；逗号、句号、
  顿号、感叹号、问号、省略号、波浪号与 `paper_features.py` 的
  `comma100/period100/pause100/excl100/ques100/ellip100/tilde100` 同源，
  只是本脚本额外计入半角变体。省略号内的 `。` 同时计入句号（与
  `paper_features.py` 的 ellip100/period100 口径一致）；
- 排版指标在清洗后的正文上计算：行 = 非空行；列表行沿用
  `paper_features.LIST_LINE`；冒号行沿用 `paper_features` 的「前 14 字内
  出现冒号」定义；
- 互动回归的样本窗口自 2023.01 起（同 `paper_engagement.py`），被解释变量
  为 log10(1+互动)。

输出：
  insights/typography_lexicon.png   跨子集标点密度 + 逐标记过量使用 +
                                    标点丰富度分布
  insights/typography_gradient.png  P(AI) 梯度 / 逐标记 ρ / 互动偏效应 /
                                    家族指纹 / 季度趋势
  insights/typography_layout.png    排版指标梯度与子集画像
  typography_results.json           全部数值结果
  typography_results.md             结果摘要
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
                            clean_body, content_of, decile_table, dedup_indices,
                            domain_standardized_series, engagement_of, iter_jsonl,
                            length_bin_codes, parse_local_time, run_parallel,
                            within_cell_spearman)
from engagement_drivers_analysis import (cluster_bootstrap_fe_ols, fe_ols,
                                         prepare_fe)
from paper_features import LIST_LINE

DATASETS = {
    "human": BASE + "training_set_human_fixed.jsonl",
    "aigc": BASE + "training_set_aigc.jsonl",
    "exploring": BASE + "exploring_set_fixed.jsonl",
}
SCORES_PATH = BASE + "paper_exploring_scores.csv.gz"
FEATURES_PATH = BASE + "paper_features_exploring.csv.gz"

MIN_BODY_LEN = 50        # 标点密度在更短正文（纯标签/空正文）上无意义
MIN_CELL_N = 100         # 单元内统计的最小单元样本量（与论文口径一致）
MONTH_FROM = "202301"    # 互动分析起始月份（与 paper_engagement 一致）
N_BOOT = 300             # 整群 bootstrap 次数
BOOT_SEED = 42
MIN_QUARTER_N = 300
MIN_FAMILY_N = 100
SHORT_LINE = 15          # 碎片化行阈值（字符）

# 先验标点族：类别 -> (中文标签, [token, ...])。
# token 为字符串时按子串计数；token 为字符元组时按各字符分别计数再求和
# （用于成对的引号/括号）。
CATEGORY_DEFS = {
    "excl": ("感叹号", ("！", "!")),
    "ques": ("问号", ("？", "?")),
    "ellip": ("省略号", ("…", "。。。", "...")),
    "tilde": ("波浪号", ("～", "~")),
    "comma": ("逗号", ("，", ",")),
    "pause": ("顿号", ("、",)),
    "period": ("句号", ("。",)),
    "colon": ("冒号", ("：", ":")),
    "semcolon": ("分号", ("；", ";")),
    "quote": ("引号", (("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"))),
    "bracket": ("括号", (("（", "）"), ("(", ")"), ("【", "】"))),
    "dash": ("破折号", ("—", "―", "－")),
}
CATEGORY_ORDER = ["excl", "ques", "ellip", "tilde", "comma", "pause", "period",
                  "colon", "semcolon", "quote", "bracket", "dash"]
CATEGORY_LABELS = {k: v[0] for k, v in CATEGORY_DEFS.items()}
CATEGORY_COLORS = {
    "excl": "#d62728", "ques": "#ff7f0e", "ellip": "#9467bd",
    "tilde": "#e377c2", "comma": "#1f77b4", "pause": "#17becf",
    "period": "#2ca02c", "colon": "#8c564b", "semcolon": "#7f7f7f",
    "quote": "#bcbd22", "bracket": "#aec7e8", "dash": "#ffbb78",
}
EXPRESSIVE = ["excl", "ques", "ellip", "tilde"]
REGULAR = ["comma", "pause", "period", "colon"]
STRUCTURED = ["semcolon", "quote", "bracket", "dash"]
SUBSET_ORDER = ["human", "aigc", "exploring"]
SUBSET_LABELS = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
                 "exploring": "探索集(后LLM)"}
TOTAL_LABEL = "标点总量"
TOTAL_COLOR = "#55A868"

# 排版指标：key -> 中文标签
LAYOUT_LABELS = {
    "n_lines": "行数",
    "avg_line_len": "平均行长",
    "short_line_ratio": "碎片化行占比（<%d字）" % SHORT_LINE,
    "list_line_ratio": "列表行占比",
    "colon_line_ratio": "冒号行占比",
    "single_line": "单行帖",
    "punct_types": "标点族数量（0–12）",
}
LAYOUT_KEYS = ["single_line", "short_line_ratio", "list_line_ratio",
               "colon_line_ratio", "avg_line_len", "n_lines", "punct_types"]
LAYOUT_METRICS = ["n_lines", "avg_line_len", "short_line_ratio",
                  "list_line_ratio", "colon_line_ratio", "single_line"]

FAMILY_NAME = {"anthropic": "Claude", "deepseek": "DeepSeek", "gemini": "Gemini",
               "glm": "GLM", "gpt": "GPT", "qwen": "Qwen"}
FAMILIES = ["anthropic", "deepseek", "gemini", "glm", "gpt", "qwen"]
TARGETS = (("likes", "点赞"), ("collections", "收藏"), ("comments", "评论"))


# --------------------------------------------------------------------------
# 标点词表展开（模块级常量）
# --------------------------------------------------------------------------
def _category_tokens(cat: str):
    """把一个类别的 token 定义展开为「逐标记」列表（成对字符拆开）。"""
    out = []
    for tok in CATEGORY_DEFS[cat][1]:
        if isinstance(tok, str):
            out.append(tok)
        else:
            out.extend(tok)
    return out


MARKERS: list[tuple[str, str]] = [
    (tok, cat) for cat in CATEGORY_ORDER for tok in _category_tokens(cat)]
MARKER_DISPLAY = [m for m, _ in MARKERS]
MARKER_CATEGORY = {m: cat for m, cat in MARKERS}
MARKER_INDEX = {m: i for i, m in enumerate(MARKER_DISPLAY)}


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def _count_token(text: str, tok) -> int:
    """统计一个 token 的出现次数；元组 token 表示逐字符计数后求和。"""
    if isinstance(tok, str):
        return text.count(tok)
    return sum(text.count(ch) for ch in tok)


def typography_counts(text: str) -> dict:
    """统计一段正文的 12 个标点族命中数与逐标记命中数。

    返回 {"excl": n, ..., "_markers": {标记: n}}；类别计数等于该类别
    逐标记计数之和。同一类别内的 token 互不重叠（半角/全角、成对引号/括号
    均为不同字符），因此不会重复计数；省略号内部的 `。` 同时计入句号，
    与 `paper_features.py` 的 ellip100/period100 口径一致。
    """
    out: dict = {}
    marker_counts: Counter = Counter()
    for cat in CATEGORY_ORDER:
        total = 0
        for tok in CATEGORY_DEFS[cat][1]:
            if isinstance(tok, str):
                n = text.count(tok)
                if n:
                    marker_counts[tok] += n
                total += n
            else:
                for ch in tok:
                    n = text.count(ch)
                    if n:
                        marker_counts[ch] += n
                    total += n
        out[cat] = int(total)
    out["_markers"] = dict(marker_counts)
    return out


def punct_type_count(counts: dict) -> int:
    """一篇帖子用到的标点族数量（0–12）。"""
    return int(sum(1 for k in CATEGORY_ORDER if counts[k] > 0))


def layout_metrics(body: str) -> dict:
    """清洗正文的排版指标（纯函数）。

    行 = 非空行；`avg_line_len` 与 `paper_features.py` 同为「全文字符数/非空
    行数」（保留浮点精度，汇总时再取整）；列表行正则直接复用
    `paper_features.LIST_LINE`；冒号行沿用「前 14 字内出现冒号」的定义。
    """
    lines = [ln for ln in body.split("\n") if ln.strip()]
    n_lines = len(lines)
    n_safe = max(n_lines, 1)
    short = sum(1 for ln in lines if len(ln.strip()) < SHORT_LINE)
    list_lines = sum(1 for ln in lines if LIST_LINE.match(ln))
    colon_lines = sum(1 for ln in lines if ("：" in ln[:14] or ":" in ln[:14]))
    return {
        "n_lines": int(n_lines),
        "avg_line_len": float(len(body) / n_safe),
        "short_line_ratio": float(short / n_safe),
        "list_line_ratio": float(list_lines / n_safe),
        "colon_line_ratio": float(colon_lines / n_safe),
        "single_line": int(n_lines <= 1),
    }


def density_per_100(count: float, body_len: float) -> float:
    """次数 / 正文字数 * 100；空正文返回 0。"""
    return float(count) / float(body_len) * 100 if body_len > 0 else 0.0


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
def scan_corpus(path: str) -> dict:
    """单次流式扫描一个语料，返回逐记录数组。

    每条记录保存：正文长度、note_id、元数据完整度（去重取舍）、月份、领域、
    模型家族/模型（AIGC）、12 个标点族计数、逐标记计数（按 MARKER_DISPLAY
    顺序平铺为 int32 数组）与 6 个排版指标 + 标点族数量。
    """
    body_len, note_id, meta, month, domain = [], [], [], [], []
    family, model = [], []
    cat_counts = {k: [] for k in CATEGORY_ORDER}
    layout = {k: [] for k in LAYOUT_METRICS}
    punct_types, likes = [], []
    flat_marker = array("i")
    n_markers = len(MARKER_DISPLAY)
    n_total = 0
    for rec in iter_jsonl(path):
        n_total += 1
        content = content_of(rec)
        body = clean_body(content)
        L = len(body)
        counts = typography_counts(body)
        lay = layout_metrics(body)
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
        for k in LAYOUT_METRICS:
            layout[k].append(lay[k])
        punct_types.append(punct_type_count(counts))
        row = [0] * n_markers
        for w, n in counts["_markers"].items():
            row[MARKER_INDEX[w]] = n
        flat_marker.extend(row)
    marker_matrix = np.frombuffer(flat_marker, dtype=np.int32).reshape(
        -1, n_markers)
    return {
        "path": path, "n_total": n_total,
        "body_len": np.asarray(body_len, dtype=np.int32),
        "note_id": note_id, "meta": np.asarray(meta, dtype=np.int32),
        "month": month, "domain": domain, "likes": np.asarray(likes, dtype=float),
        "family": family, "model": model,
        "cat_counts": {k: np.asarray(v, dtype=np.int32)
                       for k, v in cat_counts.items()},
        "layout": {k: np.asarray(v, dtype=float) for k, v in layout.items()},
        "punct_types": np.asarray(punct_types, dtype=np.int32),
        "marker_counts": {w: marker_matrix[:, i]
                          for i, w in enumerate(MARKER_DISPLAY)},
    }


def dedup_keep(scan: dict) -> np.ndarray:
    """按 note_id 去重（保留元数据最完整一条），返回排序后的保留索引。"""
    keep = dedup_indices(scan["note_id"], scan["meta"])
    return np.asarray(sorted(keep), dtype=int)


def rates_block(scan: dict, keep: np.ndarray,
                min_len: int = MIN_BODY_LEN) -> dict:
    """一个子集的标点族/逐标记/排版指标描述统计（仅正文 >= min_len 的记录）。"""
    sel = keep[scan["body_len"][keep] >= min_len]
    n = int(len(sel))
    chars = int(scan["body_len"][sel].sum())
    cat = {}
    for k in CATEGORY_ORDER:
        c = scan["cat_counts"][k][sel].astype(float)
        cat[k] = {
            "n": n, "hits": int(c.sum()), "chars": chars,
            "density100": density_per_100(c.sum(), chars),
            "share": float((c > 0).mean()) if n else 0.0,
        }
    total = np.zeros(n, dtype=float)
    for k in CATEGORY_ORDER:
        total += scan["cat_counts"][k][sel]
    cat["total"] = {
        "n": n, "hits": int(total.sum()), "chars": chars,
        "density100": density_per_100(total.sum(), chars),
        "share": float((total > 0).mean()) if n else 0.0,
    }
    markers = {}
    for w, arr in scan["marker_counts"].items():
        c = arr[sel].astype(float)
        markers[w] = {
            "n": n,
            "per10k_chars": float(c.sum() / chars * 10000) if chars else 0.0,
            "share": float((c > 0).mean()) if n else 0.0,
        }
    layout = {}
    for k in LAYOUT_METRICS + ["punct_types"]:
        v = (scan["layout"][k] if k in scan["layout"]
             else scan["punct_types"].astype(float))[sel]
        layout[k] = {
            "n": n,
            "mean": float(v.mean()) if n else 0.0,
            "median": float(np.median(v)) if n else 0.0,
        }
    layout["single_line"]["share"] = float(scan["layout"]["single_line"][sel].mean()) \
        if n else 0.0
    return {"n": n, "chars": chars, "categories": cat,
            "markers": markers, "layout": layout}


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_lexicon(subset_rates: dict, word_table: dict,
                 type_dist: dict) -> None:
    """跨子集标点密度、逐标记 AIGC−人类过量使用、标点丰富度分布。"""
    init_plot_style()
    fig = plt.figure(figsize=(19.5, 11.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5], height_ratios=[1.45, 1.0])

    # ---- (a) 标点族 × 子集密度 ----
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(CATEGORY_ORDER))
    h = 0.26
    for j, subset in enumerate(SUBSET_ORDER):
        vals = [subset_rates[subset]["categories"][k]["density100"]
                for k in CATEGORY_ORDER]
        pos = y + (j - 1) * h
        ax.barh(pos, vals, height=h, color=PALETTE[subset],
                label=SUBSET_LABELS[subset])
        for yy, v in zip(pos, vals):
            ax.annotate(f"{v:.2f}", xy=(v, yy), xytext=(3, -3),
                        textcoords="offset points", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([CATEGORY_LABELS[k] for k in CATEGORY_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("密度（次/百字，正文 ≥ %d 字）" % MIN_BODY_LEN)
    ax.set_title("(a) 三类语料的标点族密度")
    ax.legend(loc="lower right")

    # ---- (b) 逐标记 AIGC−人类 使用差 ----
    ax = fig.add_subplot(gs[0, 1])
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
    ax.scatter([r[3] for r in rows], ypos, s=20, color="black", zorder=4,
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
    ax.invert_yaxis()
    # 逗号（约 390/万字符）与感叹号（+102）量级远超其余标记，普通线性轴下
    # 大多数标记不可读；symlog 轴保留 0 附近的线性区，同时压缩长尾。
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xticks([-100, -10, 0, 10, 100, 400])
    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("AIGC − 人类 使用率差（次/万字符，symlog 轴；越右=生成集越爱用）")
    ax.set_title("(b) 生成集对每个标点标记的过量使用（黑点=探索集使用率）")
    handles = [plt.Line2D([], [], color=CATEGORY_COLORS[k], lw=6,
                          label=CATEGORY_LABELS[k]) for k in CATEGORY_ORDER]
    handles.append(plt.Line2D([], [], color="black", marker="o", ls="",
                              ms=4, label="探索集使用率"))
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, ncol=2)

    # ---- (c) 标点丰富度（帖子用到的标点族数量）分布 ----
    ax = fig.add_subplot(gs[1, 1])
    xs = np.arange(0, len(CATEGORY_ORDER) + 1)
    for subset in SUBSET_ORDER:
        share = [type_dist[subset][k] for k in xs]
        ax.plot(xs, share, "o-", color=PALETTE[subset], lw=1.8, ms=4,
                label=f"{SUBSET_LABELS[subset]}"
                      f"（均值 {subset_rates[subset]['layout']['punct_types']['mean']:.2f} 类）")
    ax.set_xticks(xs)
    ax.set_xlabel("一篇帖子使用的标点族数量（0–%d）" % len(CATEGORY_ORDER))
    ax.set_ylabel("帖子占比")
    ax.set_title("(c) 标点丰富度分布：生成集用到的标点种类更多还是更少？")
    ax.legend()
    save_fig(fig, INSIGHTS + "typography_lexicon.png")


def plot_gradient(grad: dict, word_rho: dict, eng: dict, family_table: dict,
                  quarterly: dict) -> None:
    """六联图：P(AI) 梯度 / 逐标记 ρ / 互动偏效应 / 家族指纹 / 季度趋势。"""
    init_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 12.4))

    # ---- (a) 表现性标点的 P(AI) 十分位曲线 ----
    ax = axes[0, 0]
    for k in EXPRESSIVE:
        dec = grad["categories"][k]["deciles"]
        x = [d["decile"] for d in dec]
        yv = [d["mean"] for d in dec]
        lo = [d["ci_low"] for d in dec]
        hi = [d["ci_high"] for d in dec]
        ax.plot(x, yv, "o-", color=CATEGORY_COLORS[k], lw=1.8, ms=5,
                label=f"{CATEGORY_LABELS[k]}（ρ={grad['categories'][k]['weighted_rho']:+.3f}）")
        ax.fill_between(x, lo, hi, color=CATEGORY_COLORS[k], alpha=0.15)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("检测器 P(AI) 十分位（1=最像人，10=最像 AI）")
    ax.set_ylabel("密度（次/百字）")
    ax.set_title("(a) 表现性标点随 P(AI) 的梯度（含 95% CI）")
    ax.legend(fontsize=8.5)

    # ---- (b) 常规标点的 P(AI) 十分位曲线 ----
    ax = axes[0, 1]
    for k in REGULAR:
        dec = grad["categories"][k]["deciles"]
        x = [d["decile"] for d in dec]
        ax.plot(x, [d["mean"] for d in dec], "o-", color=CATEGORY_COLORS[k],
                lw=1.8, ms=4.5,
                label=f"{CATEGORY_LABELS[k]}（ρ={grad['categories'][k]['weighted_rho']:+.3f}）")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("检测器 P(AI) 十分位（1=最像人，10=最像 AI）")
    ax.set_ylabel("密度（次/百字）")
    ax.set_title("(b) 常规标点随 P(AI) 的梯度")
    ax.legend(fontsize=8.5)

    # ---- (c) 逐标记单元内 ρ ----
    ax = axes[0, 2]
    items = sorted(word_rho.items(), key=lambda kv: (CATEGORY_ORDER.index(
        kv[1]["category"]), kv[1]["weighted_rho"]))
    ypos = np.arange(len(items))
    vals = [kv[1]["weighted_rho"] for kv in items]
    colors = [CATEGORY_COLORS[kv[1]["category"]] for kv in items]
    ax.barh(ypos, vals, color=colors, alpha=0.9, height=0.72)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([kv[0] for kv in items], fontsize=7.8)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.3f}", xy=(v, ypos[i]),
                    xytext=(5 if v >= 0 else -5, -3),
                    textcoords="offset points", fontsize=6.8,
                    ha="left" if v >= 0 else "right")
    ax.invert_yaxis()
    ax.set_xlabel("月份×领域单元内加权 Spearman ρ（密度 vs P(AI)）")
    ax.set_title("(c) 逐标记的野外 P(AI) 梯度")
    handles = [plt.Line2D([], [], color=CATEGORY_COLORS[k], lw=6,
                          label=CATEGORY_LABELS[k]) for k in CATEGORY_ORDER]
    ax.legend(handles=handles, loc="lower left", fontsize=6.8, ncol=2)

    # ---- (d) 三指标互动偏效应森林图 ----
    ax = axes[1, 0]
    fe_likes = eng["fe"]["likes"]
    order = list(reversed(fe_likes["features"]))
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
            ax.scatter([b], [ypos[i] + offset], marker=marker, s=28,
                       facecolor=col if sig else "white", edgecolor=col,
                       zorder=3, label=tlabel if first else None)
            first = False
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    labels = {k: CATEGORY_LABELS[k] for k in CATEGORY_ORDER}
    labels.update({"p_ai": "P(AI)", "content_len": "正文长度",
                   "n_hashtags": "标签数"})
    ax.set_yticklabels([labels.get(n, n) for n in order])
    ax.set_xlabel("单元内 FE β（每 1 SD → log10(1+互动)）")
    ax.set_title(f"(d) 标点族的互动偏效应（n={fe_likes['n_obs']:,}，"
                 f"{fe_likes['n_cells']} 单元，整群 bootstrap）")
    ax.legend(loc="lower right")

    # ---- (e) 家族指纹（列内 z-score 着色）----
    ax = axes[1, 1]
    fams = [f for f in FAMILIES
            if family_table.get(f, {}).get("n", 0) >= MIN_FAMILY_N]
    mat = np.asarray([[family_table[f]["density100"][k] for k in CATEGORY_ORDER]
                      for f in fams], dtype=float)
    z = (mat - mat.mean(axis=0)) / np.where(mat.std(axis=0) > 0,
                                            mat.std(axis=0), 1.0)
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
    ax.grid(False)
    for i in range(len(fams)):
        for j in range(len(CATEGORY_ORDER)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(z[i, j]) > 1.15 else "black")
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels([CATEGORY_LABELS[k] for k in CATEGORY_ORDER],
                       fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f"{FAMILY_NAME[f]}（n={family_table[f]['n']}）"
                        for f in fams], fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("家族差异（列内 z-score）", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("(e) AIGC 训练集的家族标点指纹（格内为密度，次/百字）")

    # ---- (f) 季度趋势（取 |ρ| 最大的 6 类）----
    ax = axes[1, 2]
    qs = quarterly["quarters"]
    x = np.arange(len(qs))
    ranked = sorted(CATEGORY_ORDER,
                    key=lambda k: -abs(quarterly["trend"][k]["rho"]))
    for k in ranked[:6]:
        raw = quarterly["raw"][k]
        t = quarterly["trend"][k]
        ax.plot(x, raw, "o-", color=CATEGORY_COLORS[k], lw=1.6, ms=3.5,
                label=f"{CATEGORY_LABELS[k]}（ρ={t['rho']:+.2f}，p={t['p']:.2f}）")
    ax.set_xticks(x)
    ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("密度（次/百字）")
    ax.set_title("(f) 探索集季度趋势（|ρ| 最大的 6 个标点族）")
    ax.legend(fontsize=7.5)
    save_fig(fig, INSIGHTS + "typography_gradient.png")


def plot_layout(grad: dict, subset_rates: dict) -> None:
    """排版层：单元内 ρ、子集画像、标点丰富度与单行帖的 P(AI) 梯度。

    `grad` 为 `exploring_gradient.layout`（6 个排版指标 + punct_types）。
    """
    init_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.4))

    # ---- (a) 排版指标与 P(AI) 的单元内 ρ ----
    ax = axes[0]
    keys = LAYOUT_KEYS
    ypos = np.arange(len(keys))
    vals = [grad[k]["weighted_rho"] for k in keys]
    strat = [grad[k]["stratified_rho"] for k in keys]
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in vals]
    ax.barh(ypos, vals, color=colors, alpha=0.85, height=0.66,
            label="月份×领域单元内 ρ")
    ax.scatter(strat, ypos, marker="D", s=30, color="black", zorder=4,
               label="长度分层后 ρ")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([LAYOUT_LABELS[k] for k in keys], fontsize=8.5)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.3f}", xy=(v, ypos[i]),
                    xytext=(6 if v >= 0 else -6, 5),
                    textcoords="offset points", fontsize=7.5,
                    ha="left" if v >= 0 else "right")
    ax.invert_yaxis()
    ax.set_xlabel("与 P(AI) 的单元内加权 Spearman ρ")
    ax.set_title("(a) 排版指标与 P(AI) 的关联")
    ax.legend(loc="lower right", fontsize=8)

    # ---- (b) 子集排版画像 ----
    ax = axes[1]
    metrics = ["single_line", "short_line_ratio", "list_line_ratio",
               "colon_line_ratio"]
    labels = {m: LAYOUT_LABELS[m] for m in metrics}
    x = np.arange(len(metrics))
    width = 0.26
    for j, subset in enumerate(SUBSET_ORDER):
        vals = [subset_rates[subset]["layout"][m]["mean"] if m != "single_line"
                else subset_rates[subset]["layout"]["single_line"]["share"]
                for m in metrics]
        ax.bar(x + (j - 1) * width, vals, width=width, color=PALETTE[subset],
               label=SUBSET_LABELS[subset])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in metrics], fontsize=8.5)
    ax.set_ylabel("帖子/行占比（均值）")
    ax.set_title("(b) 三类语料的排版画像")
    ax.legend(fontsize=8)

    # ---- (c) 标点丰富度与单行帖的 P(AI) 梯度 ----
    ax = axes[2]
    dec_types = grad["punct_types"]["deciles"]
    dec_single = grad["single_line"]["deciles"]
    x = [d["decile"] for d in dec_types]
    ax.plot(x, [d["mean"] for d in dec_types], "o-", color="#9467bd", lw=1.8,
            ms=5, label=f"标点族数量（ρ={grad['punct_types']['weighted_rho']:+.3f}）")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("检测器 P(AI) 十分位（1=最像人，10=最像 AI）")
    ax.set_ylabel("平均标点族数量", color="#9467bd")
    ax.tick_params(axis="y", labelcolor="#9467bd")
    ax2 = ax.twinx()
    ax2.plot(x, [d["mean"] for d in dec_single], "s--", color="#C44E52", lw=1.6,
             ms=4.5, label=f"单行帖占比（ρ={grad['single_line']['weighted_rho']:+.3f}）")
    ax2.set_ylabel("单行帖占比", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax.set_title("(c) 标点丰富度与单行帖随 P(AI) 的变化")
    save_fig(fig, INSIGHTS + "typography_layout.png")


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
    for w in MARKER_DISPLAY:
        word_table[w] = {
            "category": MARKER_CATEGORY[w],
            "human": subset_rates["human"]["markers"][w],
            "aigc": subset_rates["aigc"]["markers"][w],
            "exploring": subset_rates["exploring"]["markers"][w],
        }

    # 标点丰富度分布（0..12 类）
    type_dist = {}
    for name in SUBSET_ORDER:
        sc = scans[name]
        sel = keeps[name][sc["body_len"][keeps[name]] >= MIN_BODY_LEN]
        pt = sc["punct_types"][sel]
        counts = np.bincount(pt, minlength=len(CATEGORY_ORDER) + 1)
        type_dist[name] = {k: float(counts[k] / len(pt)) if len(pt) else 0.0
                           for k in range(len(CATEGORY_ORDER) + 1)}

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
    p_ai_ranks = cell_rank_cache(p_ai, groups)
    groups_strat = cell_groups(strat, MIN_CELL_N)
    p_ai_ranks_strat = cell_rank_cache(p_ai, groups_strat)
    p_hi, p_lo = float(np.quantile(p_ai, 0.9)), float(np.quantile(p_ai, 0.1))

    def gradient_entry(values):
        wc = within_cell_spearman(values, p_ai, cells, min_cell_n=MIN_CELL_N,
                                  groups=groups, y_ranks=p_ai_ranks)
        st = within_cell_spearman(values, p_ai, strat, min_cell_n=MIN_CELL_N,
                                  groups=groups_strat, y_ranks=p_ai_ranks_strat)
        delta, dp = cliffs_delta(values[p_ai >= p_hi], values[p_ai <= p_lo])
        return {
            "weighted_rho": round(wc["weighted_rho"], 4),
            "neg_cell_share": round(wc["neg_cell_share"], 4),
            "n_cells": wc["n_cells"],
            "stratified_rho": round(st["weighted_rho"], 4),
            "stratified_cells": st["n_cells"],
            "cliff_delta_top_vs_bottom": round(delta, 4),
            "cliff_p": float(dp),
        }

    grad_cat = {}
    total_sel = np.zeros(len(sel), dtype=float)
    for k in CATEGORY_ORDER:
        total_sel += expl["cat_counts"][k][sel]
    for k in CATEGORY_ORDER + ["total"]:
        tot = (total_sel if k == "total"
               else expl["cat_counts"][k][sel].astype(float))
        dens = tot / body * 100
        entry = gradient_entry(dens)
        entry["deciles"] = decile_table(dens, p_ai)
        grad_cat[k] = entry
        print(f"  {k}: wcell ρ={entry['weighted_rho']:+.4f} "
              f"(长度分层 {entry['stratified_rho']:+.4f})，"
              f"δ={entry['cliff_delta_top_vs_bottom']:+.4f}")

    word_rho = {}
    for w in MARKER_DISPLAY:
        dens = expl["marker_counts"][w][sel].astype(float) / body * 100
        if dens.std() == 0:
            continue
        wc = within_cell_spearman(dens, p_ai, cells, min_cell_n=MIN_CELL_N,
                                  groups=groups, y_ranks=p_ai_ranks)
        word_rho[w] = {"category": MARKER_CATEGORY[w],
                       "weighted_rho": round(wc["weighted_rho"], 4),
                       "neg_cell_share": round(wc["neg_cell_share"], 4),
                       "n_cells": wc["n_cells"]}

    grad_layout = {}
    for k in LAYOUT_METRICS:
        vals = expl["layout"][k][sel]
        entry = gradient_entry(vals)
        entry["deciles"] = decile_table(vals, p_ai)
        grad_layout[k] = entry
    pt_sel = expl["punct_types"][sel].astype(float)
    grad_layout["punct_types"] = gradient_entry(pt_sel)
    grad_layout["punct_types"]["deciles"] = decile_table(pt_sel, p_ai)
    grad_layout["single_line"]["deciles"] = decile_table(
        expl["layout"]["single_line"][sel], p_ai)
    print(f"  排版：punct_types ρ={grad_layout['punct_types']['weighted_rho']:+.4f}，"
          f"single_line ρ={grad_layout['single_line']['weighted_rho']:+.4f}")

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
    cat_dens = np.column_stack([
        scans["exploring"]["cat_counts"][k].astype(float)
        / np.maximum(scans["exploring"]["body_len"], 1) * 100
        for k in CATEGORY_ORDER])
    targets = {name: np.log10(1 + np.clip(model[name].to_numpy(dtype=float), 0, None))
               for name, _ in TARGETS}
    month_all = model["month"].astype(str).to_numpy()
    dom_all = model["domain"].astype(str).to_numpy()
    keep_eng = ((month_all >= MONTH_FROM)
                & (scans["exploring"]["body_len"] >= MIN_BODY_LEN))
    eng_cells_raw = np.asarray([f"{m}|{d}" for m, d in zip(month_all, dom_all)])
    p_ai_all = scores["p_ai"].to_numpy(dtype=float)
    X = np.column_stack([cat_dens, p_ai_all,
                         scans["exploring"]["body_len"].astype(float),
                         model["n_hashtags"].to_numpy(dtype=float)])
    feature_names = CATEGORY_ORDER + ["p_ai", "content_len", "n_hashtags"]
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
              f"comma β={fit['coef'][4]:+.4f} "
              f"[{ci[0][4]:+.4f}, {ci[1][4]:+.4f}]，"
              f"p_ai β={fit['coef'][-3]:+.4f} "
              f"[{ci[0][-3]:+.4f}, {ci[1][-3]:+.4f}]")

    # 单变量：12 标点族 + 总量 + p_ai × 3 指标（预分组一次共享）
    groups_eng = cell_groups(eng_cells_raw[keep_eng], MIN_CELL_N)
    y_ranks = {t: cell_rank_cache(targets[t][keep_eng], groups_eng)
               for t, _ in TARGETS}
    uni_cols = {k: cat_dens[keep_eng][:, i]
                for i, k in enumerate(CATEGORY_ORDER)}
    uni_cols["total"] = cat_dens[keep_eng].sum(axis=1)
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
    q_df["total"] = q_df[CATEGORY_ORDER].sum(axis=1)
    q_df = q_df[q_df["quarter"] != "未知"]
    q_counts = q_df.groupby("quarter").size()
    valid_q = sorted(q_counts[q_counts >= MIN_QUARTER_N].index)
    q_df = q_df[q_df["quarter"].isin(valid_q)]
    quarterly = {"quarters": valid_q, "n": {q: int(q_counts[q]) for q in valid_q},
                 "raw": {}, "domain_std": {}, "trend": {}, "share_any": {}}
    for k in CATEGORY_ORDER + ["total"]:
        raw = q_df.groupby("quarter")[k].mean().reindex(valid_q)
        std = domain_standardized_series(q_df, k, "quarter").reindex(valid_q)
        rho, p = spearmanr(range(len(valid_q)), raw.to_numpy())
        rho_s, p_s = spearmanr(range(len(valid_q)), std.to_numpy())
        quarterly["raw"][k] = [round(float(v), 4) for v in raw]
        quarterly["domain_std"][k] = [round(float(v), 4) for v in std]
        quarterly["trend"][k] = {"rho": round(float(rho), 4), "p": float(p),
                                 "rho_domain_std": round(float(rho_s), 4),
                                 "p_domain_std": float(p_s)}
        share = (q_df.assign(hit=q_df[k] > 0).groupby("quarter")["hit"].mean()
                 .reindex(valid_q))
        quarterly["share_any"][k] = [round(float(v), 4) for v in share]

    # ---- 结果 ----
    out = {
        "meta": {
            "min_body_len": MIN_BODY_LEN, "min_cell_n": MIN_CELL_N,
            "month_from": MONTH_FROM, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "min_quarter_n": MIN_QUARTER_N, "short_line_len": SHORT_LINE,
            "exploring_content_len_mismatch": mismatch,
            "exploring_likes_mismatch": likes_mismatch,
            "note": "人类/探索集按 note_id 去重（保留元数据最完整一条）；"
                    "AIGC 无 note_id 逐记录；p_ai 按行序对齐 "
                    "paper_exploring_scores.csv.gz；标点族与逐标记均为先验"
                    "定义（含全角/半角变体），省略号内的句号同时计入句号。",
        },
        "lexicon": {k: {"label": CATEGORY_LABELS[k],
                        "tokens": [t if isinstance(t, str) else "".join(t)
                                   for t in CATEGORY_DEFS[k][1]]}
                    for k in CATEGORY_ORDER},
        "markers": {w: MARKER_CATEGORY[w] for w in MARKER_DISPLAY},
        "subset_rates": {
            name: {
                "n": subset_rates[name]["n"],
                "chars": subset_rates[name]["chars"],
                "categories": subset_rates[name]["categories"],
                "layout": subset_rates[name]["layout"],
                "punct_type_dist": type_dist[name],
            } for name in SUBSET_ORDER},
        "word_rates": word_table,
        "exploring_gradient": {
            "n": int(len(sel)),
            "short_excluded": int((expl["body_len"][keep] < MIN_BODY_LEN).sum()),
            "n_cells": int(len(np.unique(cells))),
            "categories": grad_cat,
            "markers": word_rho,
            "layout": grad_layout,
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
        },
        "family": {"families": family_table, "models": model_table},
        "quarterly": quarterly,
    }
    save_json(out, BASE + "typography_results.json")
    print("已写出 typography_results.json")
    write_markdown(out)
    plot_lexicon(subset_rates, word_table, type_dist)
    plot_gradient({"categories": grad_cat}, word_rho, eng,
                  family_table, quarterly)
    plot_layout(grad_layout, subset_rates)
    print("已写出 typography_results.md 与 insights/typography_*.png")


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def write_markdown(out: dict) -> None:
    meta = out["meta"]
    rates = out["subset_rates"]
    grad = out["exploring_gradient"]
    eng = out["engagement"]
    q = out["quarterly"]
    labels = SUBSET_LABELS
    lines = [
        "# 标点与排版（typography）分析结果",
        "",
        f"> 由 `typography_analysis.py` 全量运行生成；正文清洗与 `paper_features.py`"
        f" 一致，人类/探索集按 note_id 去重；正文 ≥{meta['min_body_len']} 字；"
        f"探索集 p_ai 按行序对齐 `paper_exploring_scores.csv.gz`"
        f"（content_len 校验 mismatch={meta['exploring_content_len_mismatch']}）。"
        f"12 个标点族为**先验定义**（含全角/半角变体），省略号内的句号同时"
        f"计入句号。",
        "",
        "## 1 跨子集标点密度与普及率",
        "",
        "| 子集 | 有效帖数 | 标点总量密度 | 含任一标点占比 | "
        + " | ".join(f"{CATEGORY_LABELS[k]} 密度" for k in EXPRESSIVE)
        + " |",
        "|---|---:|---:|---:|" + "---:|" * len(EXPRESSIVE),
    ]
    for name in SUBSET_ORDER:
        cats = rates[name]["categories"]
        lines.append(
            f"| {labels[name]} | {rates[name]['n']:,} | "
            f"{cats['total']['density100']:.3f} | {cats['total']['share']:.1%} | "
            + " | ".join(f"{cats[k]['density100']:.3f}" for k in EXPRESSIVE)
            + " |")
    lines += [
        "",
        "> 注：密度为「总命中次数/总字符数×100」；标点总量为 12 个族之和。",
        "",
        "**常规/结构标点密度（次/百字）：**",
        "",
        "| 子集 | " + " | ".join(CATEGORY_LABELS[k] for k in REGULAR + STRUCTURED)
        + " |",
        "|---|" + "---:|" * (len(REGULAR) + len(STRUCTURED)),
    ]
    for name in SUBSET_ORDER:
        cats = rates[name]["categories"]
        lines.append(f"| {labels[name]} | " + " | ".join(
            f"{cats[k]['density100']:.3f}" for k in REGULAR + STRUCTURED) + " |")
    lines += [
        "",
        "**逐标记 AIGC−人类过量使用（次/万字符，按类别内差值排序）：**",
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
        "**排版画像（均值）：**",
        "",
        "| 子集 | 行数 | 平均行长 | 碎片化行占比 | 列表行占比 | 冒号行占比 | 单行帖 | 标点族数量 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SUBSET_ORDER:
        lay = rates[name]["layout"]
        lines.append(
            f"| {labels[name]} | {lay['n_lines']['mean']:.2f} | "
            f"{lay['avg_line_len']['mean']:.2f} | "
            f"{lay['short_line_ratio']['mean']:.1%} | "
            f"{lay['list_line_ratio']['mean']:.1%} | "
            f"{lay['colon_line_ratio']['mean']:.1%} | "
            f"{lay['single_line']['share']:.1%} | "
            f"{lay['punct_types']['mean']:.2f} |")
    lines += [
        "",
        "## 2 探索集内的 P(AI) 梯度（去重，正文 ≥%d 字）" % meta["min_body_len"],
        "",
        "| 标点族 | 单元内加权 ρ | 负单元占比 | 长度分层 ρ | 首尾十分位 δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in CATEGORY_ORDER + ["total"]:
        g = grad["categories"][k]
        name = CATEGORY_LABELS.get(k, TOTAL_LABEL)
        lines.append(
            f"| {name} | {g['weighted_rho']:+.4f} | {g['neg_cell_share']:.1%} | "
            f"{g['stratified_rho']:+.4f} | {g['cliff_delta_top_vs_bottom']:+.3f} |")
    lines += [
        "",
        f"（样本 {grad['n']:,} 条 / {grad['n_cells']} 个「月份×领域」单元；"
        f"长度分层 = 月份×领域×长度分箱单元。）",
        "",
        "**逐标记单元内 ρ（最低 8 / 最高 8）：**",
        "",
        "| 标记 | 类别 | 单元内 ρ | 负单元占比 |",
        "|---|---|---:|---:|",
    ]
    items = sorted(grad["markers"].items(), key=lambda kv: kv[1]["weighted_rho"])
    for w, r in items[:8] + items[-8:]:
        lines.append(f"| {w} | {CATEGORY_LABELS[r['category']]} | "
                     f"{r['weighted_rho']:+.4f} | {r['neg_cell_share']:.0%} |")
    lines += [
        "",
        "**排版指标与 P(AI) 的关联：**",
        "",
        "| 指标 | 单元内加权 ρ | 负单元占比 | 长度分层 ρ | 首尾十分位 δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in LAYOUT_KEYS:
        g = grad["layout"][k]
        lines.append(
            f"| {LAYOUT_LABELS[k]} | {g['weighted_rho']:+.4f} | "
            f"{g['neg_cell_share']:.1%} | {g['stratified_rho']:+.4f} | "
            f"{g['cliff_delta_top_vs_bottom']:+.3f} |")
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
        f"起、正文 ≥{meta['min_body_len']} 字。模型只控制其余标点族与 "
        f"P(AI)/正文长度/标签数，不含其他风格特征。）",
        "",
        "## 4 AIGC 家族标点指纹（次/百字）",
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
            + " | ".join(f"{slot['density100'][k]:.3f}" for k in CATEGORY_ORDER)
            + " |")
    lines += [
        "",
        "## 5 季度趋势（探索集去重）",
        "",
        "| 季度 | n | " + " | ".join(CATEGORY_LABELS[k] for k in CATEGORY_ORDER)
        + " | 总量 |",
        "|---|---:|" + "---:|" * (len(CATEGORY_ORDER) + 1),
    ]
    for i, qq in enumerate(q["quarters"]):
        lines.append(
            f"| {qq} | {q['n'][qq]:,} | "
            + " | ".join(f"{q['raw'][k][i]:.3f}" for k in CATEGORY_ORDER)
            + f" | {q['raw']['total'][i]:.3f} |")
    lines.append("")
    for k in CATEGORY_ORDER + ["total"]:
        t = q["trend"][k]
        name = CATEGORY_LABELS.get(k, TOTAL_LABEL)
        lines.append(f"- {name}：季度趋势 ρ={t['rho']:+.3f}（p={t['p']:.3f}），"
                     f"固定领域权重后 ρ={t['rho_domain_std']:+.3f}"
                     f"（p={t['p_domain_std']:.3f}）。")
    lines += [
        "",
        "图表见 `insights/typography_lexicon.png`、"
        "`insights/typography_gradient.png`、`insights/typography_layout.png`。",
    ]
    with open(BASE + "typography_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
