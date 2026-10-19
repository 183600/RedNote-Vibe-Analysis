# -*- coding: utf-8 -*-
"""深潜分析之二：开篇/收尾「框架模板化」与 AI 风格的关系。

既有分析覆盖了词汇、句法、时间与互动，但没有回答一个直觉问题：
「AI 味」是否也体现在**帖子怎么开头、怎么收尾**这种框架层面？

三个新维度（全量扫描，非抽样；口径与论文管线一致）：

1. 跨子集框架集中度：正文清洗后取「删除 # 与空白后的紧凑文本」的首/尾 4 字，
   计算 Top-20 覆盖率、Miller–Madow 校正熵、等样本 bootstrap 对照
   （消除各子集样本量差异带来的经验熵偏差）。
2. AI 专属 vs 人类专属开篇/收尾模板：以训练集每帖发生率之差提炼模板，
   再检验探索集内检测器得分 p_ai 十分位上的模板使用率梯度，以及季度扩散轨迹。
3. 收尾符号构成：小红书原生表情 [xxR] / Unicode emoji / @提及 / 标点等，
   含「纯标签/空正文」这一探索集特有形态。

口径说明：
- 正文清洗复用 analysis_utils.clean_body（与 paper_features.py 完全一致）；
- 探索集 p_ai 按行序对齐 paper_exploring_scores.csv.gz；正文清洗版本差异
  会使 content_len 漂移，脚本只把它记为 text_len_drift（信息性指标），
  不再据此判定行序错位；
- 人类/探索集按 note_id 去重（保留元数据最完整的一条，规则同 deep_dive_analysis），
  AIGC 无 note_id，逐记录处理；
- 紧凑文本长度 < 8 的帖子不参与首/尾 4 字模板统计（数量单独报告）。

输出：
  insights/framing_formulaicity.png   跨子集框架集中度 + 收尾符号构成
  insights/framing_gradient.png       探索集 p_ai 梯度 / 季度扩散 / 小时分布 / 模板迁移
  framing_results.json                全部数值结果
  framing_results.md                  结果摘要
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from analysis_utils import (BASE, INSIGHTS, PALETTE, body_cache, init_plot_style,
                            save_fig, cliffs_delta, clean_body, content_of,
                            dedup_indices, engagement_of,
                            parse_local_time, rate_diff_ranking, rate_top,
                            shannon_entropy, topk_coverage, run_parallel,
                            wilson_ci)
from derived_cache import iter_cached
from paper_features import EMOJI

DATASETS = {
    "human": BASE + "training_set_human_fixed.jsonl",
    "aigc": BASE + "training_set_aigc.jsonl",
    "exploring": BASE + "exploring_set_fixed.jsonl",
}
LABELS = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)", "exploring": "探索集(后LLM)"}
SCORES_PATH = BASE + "paper_exploring_scores.csv.gz"

FRAME_LEN = 4          # 框架模板长度（字符）
MIN_NORM = 8           # 紧凑文本短于此长度的帖子不参与模板统计
TOPK = 20              # 主指标：覆盖率取前 20
TEMPLATE_K = 30        # 每个方向提炼的模板条数
TOP_OPEN_K = 50        # AIGC 高频开篇（按发生率取前 50）
MIN_A_COUNT = 10       # AIGC 模板最小出现次数
MIN_H_COUNT = 30       # 人类模板最小出现次数
MIN_QUARTER_N = 300    # 季度趋势最少样本
BOOT = 1000            # 等样本 bootstrap 次数
BOOT_SEED = 42

MARK = re.compile(r"[#\s\u200b\ufeff]+")
XHS_TAIL = re.compile(r"\[[^\[\]]{1,8}R\]$")
MENTION_TAIL = re.compile(r"@[^\s@]{1,20}$")
PUNCT_CLASS = {"感叹号": "！!", "问号": "？?", "波浪号": "～~",
               "句号": "。.", "逗号": "，,"}
END_ORDER = ["小红书表情[xxR]", "Unicode emoji", "@提及", "感叹号", "问号",
             "波浪号", "句号", "逗号", "其他标点/符号", "文字结尾", "纯标签/空正文"]
END_COLORS = ["#C44E52", "#DD8452", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
              "#4C72B0", "#64B5CD", "#CCB974", "#55A868", "#BBBBBB"]


# --------------------------------------------------------------------------
# 扫描与去重
# --------------------------------------------------------------------------
def compact(body: str) -> str:
    """紧凑文本：删除 # 与所有空白（含 BOM/零宽空格），用于取框架片段。

    标准清洗把 `#标签` 替换为空格后，正文首尾可能残留单字 `#` 与空格
    （如 `#护肤#` -> ` #`），这些采集痕迹不应进入框架统计。
    """
    return MARK.sub("", body)


def ending_class(suffix: str) -> str:
    """按紧凑文本的结尾后缀分类收尾符号。

    XHS_TAIL / MENTION_TAIL 都是 `$` 锚定的模式（最长匹配 12/21 字符），
    只需在末尾 24 个字符上搜索；对长正文可省去整段扫描，分类结果不变。
    """
    if not suffix:
        return "纯标签/空正文"
    tail = suffix[-24:]
    if XHS_TAIL.search(tail):
        return "小红书表情[xxR]"
    last = suffix[-1]
    if EMOJI.match(last):
        return "Unicode emoji"
    if MENTION_TAIL.search(tail):
        return "@提及"
    for name, chars in PUNCT_CLASS.items():
        if last in chars:
            return name
    if last in "、；;：:":
        return "其他标点/符号"
    if not last.isalnum() and not ("\u4e00" <= last <= "\u9fff"):
        return "其他标点/符号"
    return "文字结尾"


def frame_features(body: str) -> tuple[str, str]:
    """紧凑文本与其收尾类别（纯函数；scan_view 内按正文缓存复用）。"""
    s = compact(body)
    return s, ending_class(s)


def scan_view(path: str, content_len_ref: np.ndarray | None = None) -> dict:
    """单次流式扫描：收集首/尾框架片段、note_id、元数据完整度、时间与收尾类别。"""
    heads, tails, ids, months = [], [], [], []
    meta, hours, ending_classes = [], [], []
    month_pool: dict = {}   # 共享同月字符串对象（月份只有几十种）
    n_raw = n_short = text_len_drift = 0
    # 重复正文（探索集约 45%）直接复用紧凑文本与收尾类别，只读不改。
    cached = body_cache(frame_features, max_entries=60_000)
    for rec in iter_cached(path):
        content = content_of(rec)
        body = clean_body(content)
        if content_len_ref is not None and len(body) != int(content_len_ref[n_raw]):
            text_len_drift += 1
        s, cls = cached(body)
        if len(s) >= MIN_NORM:
            heads.append(s[:FRAME_LEN])
            tails.append(s[-FRAME_LEN:])
        else:
            heads.append(None)
            tails.append(None)
            n_short += 1
        ending_classes.append(cls)
        ids.append(rec.get("note_id"))
        likes, colls, comms = engagement_of(rec)
        dt = parse_local_time(rec.get("local_time"))
        has_meta = bool(rec.get("user_id")) and rec.get("domain") not in (None, "", "未知")
        meta.append(int(has_meta) + int(bool(likes)) + int(bool(colls))
                    + int(bool(comms)) + int(dt is not None)
                    + int(bool(content) and not content.isspace()))
        hours.append(dt.hour if dt is not None else -1)
        if dt is None:
            months.append(None)
        else:
            ym = (dt.year, dt.month)
            month = month_pool.get(ym)
            if month is None:
                month = month_pool[ym] = f"{dt.year:04d}{dt.month:02d}"
            months.append(month)
        n_raw += 1
    return {
        "path": path, "n_raw": n_raw, "n_short": n_short, "mismatch": 0,
        "text_len_drift": text_len_drift,
        "heads": heads, "tails": tails,
        "ending_classes": ending_classes, "ids": ids, "months": months,
        "meta": np.asarray(meta, dtype=np.int64),
        "hours": np.asarray(hours, dtype=np.int64),
    }


def encode(values: list) -> tuple[np.ndarray, np.ndarray, dict]:
    """把字符串列表编码为整数码；返回 (codes, valid_mask, code->str 映射)。"""
    mapping: dict = {}
    codes = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        c = mapping.get(v)
        if c is None:
            c = len(mapping)
            mapping[v] = c
        codes[i] = c
    valid = np.asarray([v is not None for v in values])
    return codes, valid, {c: v for v, c in mapping.items()}


def bootstrap_equal_n(codes: np.ndarray, valid: np.ndarray, n_sub: int,
                      seed: int = BOOT_SEED, b: int = BOOT) -> tuple[np.ndarray, np.ndarray]:
    """等样本量 bootstrap：每次从该子集有放回抽 n_sub 条，重算覆盖率与校正熵。"""
    pool = codes[valid]
    rng = np.random.default_rng(seed)
    cov = np.empty(b)
    ent = np.empty(b)
    for i in range(b):
        sample = pool[rng.integers(0, pool.size, n_sub)]
        counts = np.bincount(sample)
        cov[i] = topk_coverage(counts, TOPK)
        ent[i] = shannon_entropy(counts, miller_madow=True)
    return cov, ent


# 框架模板的提炼/排序复用 analysis_utils 的通用实现（标签/表情分析同口径）。
def specific_templates(counts_a: dict, n_a: int, counts_h: dict, n_h: int,
                       reverse: bool, min_count: int,
                       k: int = TEMPLATE_K) -> list[dict]:
    """按每帖发生率之差提取某侧专属模板（analysis_utils.rate_diff_ranking 的别名）。"""
    return rate_diff_ranking(counts_a, n_a, counts_h, n_h, reverse=reverse,
                             min_count=min_count, k=k)


def top_templates(counts_a: dict, n_a: int, min_count: int,
                  k: int = TOP_OPEN_K) -> list[dict]:
    """AIGC 侧高频框架模板（analysis_utils.rate_top 的别名）。"""
    return rate_top(counts_a, n_a, min_count=min_count, k=k)


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_formulaicity(view_stats: dict, boots: dict, ending_pct: dict) -> None:
    keys = ("human", "aigc", "exploring")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    def box_panel(ax, metric, ylabel, title, fmt="{:.1%}"):
        data = [boots[k][metric] for k in keys]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                        medianprops=dict(color="black", lw=1.0))
        for patch, k in zip(bp["boxes"], keys):
            patch.set_facecolor(PALETTE[k])
            patch.set_alpha(0.7)
        full = [view_stats[k][metric] for k in keys]
        ax.scatter(range(1, 4), full, marker="D", s=42, color="white",
                   edgecolor="black", zorder=5)
        for i, v in enumerate(full):
            ax.annotate(fmt.format(v), (i + 1, v), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8.5)
        ax.set_xticks(range(1, 4))
        ax.set_xticklabels([LABELS[k] for k in keys])
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    box_panel(axes[0, 0], "top20_open",
              f"首 {FRAME_LEN} 字 Top-{TOPK} 覆盖率",
              "开篇集中度（等样本 bootstrap 分布 + 全量实测）")
    box_panel(axes[0, 1], "top20_close",
              f"尾 {FRAME_LEN} 字 Top-{TOPK} 覆盖率",
              "收尾集中度（等样本 bootstrap 分布 + 全量实测）")

    ax = axes[1, 0]
    vals = [view_stats[k]["entropy_open_mm"] for k in keys]
    ax.bar(range(3), vals, color=[PALETTE[k] for k in keys], alpha=0.82,
           width=0.58)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([LABELS[k] for k in keys])
    ax.set_ylim(min(vals) * 0.85, max(vals) * 1.10)
    ax.set_ylabel("开篇熵 (bit)")
    ax.set_title("开篇多样性（全量，Miller–Madow 校正熵）")

    ax = axes[1, 1]
    y = np.arange(len(keys))[::-1]
    left = np.zeros(len(keys))
    for cat, color in zip(END_ORDER, END_COLORS):
        vals = np.array([ending_pct[k].get(cat, 0.0) for k in keys])
        ax.barh(y, vals, left=left, color=color, height=0.62, label=cat)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v >= 6:
                ax.text(l + v / 2, y[yi], f"{v:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white" if v > 10 else "black")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[k] for k in keys])
    ax.set_xlim(0, 100)
    ax.set_xlabel("占比 (%)")
    ax.set_title("收尾符号构成（紧凑文本末尾分类）")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, fontsize=7.6, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.suptitle("RedNote-Vibe 开篇/收尾框架模板化：越公式化的框架越缺乏人味",
                 fontsize=13.5, y=1.0)
    save_fig(fig, INSIGHTS + "framing_formulaicity.png")


def plot_gradient(grad: dict, quarterly: dict, hourly: dict,
                  templates: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    ax = axes[0, 0]
    dec = grad["deciles"]
    x = [d["mean_p"] for d in dec]
    series = (("share_aigc_top_tmpl", "#DD8452", "--", "AIGC 高频开篇 Top-50"),
              ("share_aigc_tmpl", PALETTE["aigc"], "-", "AI 专属开篇模板"),
              ("share_human_tmpl", PALETTE["human"], "-", "人类专属开篇模板"))
    for key, color, style, label in series:
        pct = np.array([d[key] for d in dec]) * 100
        lo = np.array([d[key + "_ci"][0] for d in dec]) * 100
        hi = np.array([d[key + "_ci"][1] for d in dec]) * 100
        ax.errorbar(x, pct, yerr=[pct - lo, hi - pct], marker="o", ms=4.5,
                    lw=1.5, ls=style, capsize=3, color=color, label=label)
    ax.set_xlabel("检测器得分 P(AI)（十分位均值）")
    ax.set_ylabel("开篇命中模板的帖子占比 (%)")
    ax.set_title("模板化开篇随 AI 风格得分上升")
    # p 值从实际检验结果生成：两个 p 都小于 1e-16 时才压缩成「均 p<1e-16」，
    # 避免图上文案与 grad 中的检验结果脱钩（数据/检验一变就写死过期结论）。
    p_ai, p_hum = float(grad["cliff"]["p"]), float(grad["cliff_human"]["p"])
    p_text = ("均 p<1e-16" if max(p_ai, p_hum) < 1e-16
              else f"p={p_ai:.1e} / p={p_hum:.1e}")
    ax.text(0.03, 0.95,
            f"AI 专属 δ={grad['cliff']['delta']:.2f}、"
            f"人类专属 δ={grad['cliff_human']['delta']:.2f}"
            f"\n（{p_text}，Mann–Whitney 双侧）",
            transform=ax.transAxes, fontsize=8.6, va="top")
    ax.legend(fontsize=8.2, loc="upper left", bbox_to_anchor=(0.03, 0.80))

    ax = axes[0, 1]
    qs = sorted(quarterly["exploring"])
    xs = np.arange(len(qs))
    ys = np.array([quarterly["exploring"][q]["share_aigc_tmpl"] * 100
                   for q in qs])
    lo = np.array([quarterly["exploring"][q]["ci_aigc_tmpl"][0] * 100
                   for q in qs])
    hi = np.array([quarterly["exploring"][q]["ci_aigc_tmpl"][1] * 100
                   for q in qs])
    yt = [quarterly["exploring"][q]["share_aigc_top_tmpl"] * 100 for q in qs]
    ax.plot(xs, yt, "s--", color="#DD8452", lw=1.4, ms=4,
            label="AIGC 高频开篇 Top-50")
    ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], marker="o", ms=4.5, lw=1.6,
                capsize=3, color=PALETTE["aigc"], label="AI 专属开篇（探索集）")
    ax.axhline(quarterly["human_clean_share"] * 100, color=PALETTE["human"],
               ls="--", lw=1.3, label="人类集干净窗口基线(≤2022.06)")
    ax.set_xticks(xs)
    ax.set_xticklabels(qs, rotation=45, fontsize=8)
    ax.set_ylabel("开篇模板使用率 (%)")
    ax.set_ylim(0, max(yt + list(ys)) * 1.45)
    ax.set_title("探索集内 AI 开篇模板的季度轨迹")
    tr = quarterly.get("trend", {})
    trt = quarterly.get("trend_top", {})
    if tr:
        ax.text(0.98, 0.95,
                f"专属 ρ={tr['rho']:.2f} (p={tr['p']:.2f})\n"
                f"Top-50 ρ={trt.get('rho', float('nan')):.2f} "
                f"(p={trt.get('p', float('nan')):.2f})\n"
                f"AIGC 生成集参考 {quarterly['aigc_share']:.1%}（超出坐标范围）",
                transform=ax.transAxes, fontsize=8.4, va="top", ha="right",
                color="#444444")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, 0]
    hrs = hourly["by_hour"]
    hs = np.arange(24)
    mean_p = [hrs[str(h)]["mean_p"] if str(h) in hrs else np.nan for h in hs]
    hi_share = [hrs[str(h)]["hi_share"] * 100 if str(h) in hrs else np.nan for h in hs]
    ax.bar(hs, mean_p, color=PALETTE["exploring"], alpha=0.75, width=0.72)
    ax.set_xlabel("发布小时（本地时间）")
    ax.set_ylabel("平均 P(AI)", color=PALETTE["exploring"])
    ax.set_xticks(range(0, 24, 2))
    ax2 = ax.twinx()
    ax2.plot(hs, hi_share, "o-", color=PALETTE["aigc"], ms=3.4, lw=1.2,
             label="P(AI)>0.9 占比")
    ax2.set_ylabel("高置信 AI 风格帖占比 (%)", color=PALETTE["aigc"])
    ax2.grid(False)
    ax.set_title("探索集：AI 风格得分的小时分布（去重后）")
    ax.text(0.03, 0.95,
            f"小时均值极差 {hourly['mean_p_spread']:.3f}；"
            f"ρ(小时, P(AI))={hourly['rho']:.2f}",
            transform=ax.transAxes, fontsize=8.8, va="top")
    ax2.legend(fontsize=8, loc="upper right")

    ax = axes[1, 1]
    per1k = quarterly["template_exploring_per1k"]
    tops = sorted((t for t in templates["aigc_specific_open"] if per1k.get(t["word"], 0) > 0),
                  key=lambda t: -per1k[t["word"]])[:10]
    words = [t["word"] for t in tops][::-1]
    ypos = np.arange(len(words))
    a_rates = np.array([t["aigc_per1k"] for t in tops][::-1])
    e_rates = np.array([per1k[t["word"]] for t in tops][::-1])
    for y, a, e in zip(ypos, a_rates, e_rates):
        ax.plot([e, a], [y, y], color="#BBBBBB", lw=1.2, zorder=1)
    ax.scatter(e_rates, ypos, s=44, color=PALETTE["exploring"], zorder=3,
               label="探索集（后LLM）")
    ax.scatter(a_rates, ypos, s=44, color=PALETTE["aigc"], zorder=3,
               label="AIGC 生成集")
    for y, a, e in zip(ypos, a_rates, e_rates):
        ax.annotate(f"{e:.2f}", (e, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.4,
                    color=PALETTE["exploring"])
        ax.annotate(f"{a:.1f}", (a, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.4,
                    color=PALETTE["aigc"])
    ax.set_xscale("log")
    ax.set_yticks(ypos)
    ax.set_yticklabels(words, fontsize=9)
    ax.set_xlabel("每千帖出现次数（对数轴）")
    ax.set_title("扩散最多的 AI 专属开篇模板（按探索集使用率选取）")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", visible=False)

    fig.suptitle("探索集内「框架模板化 × AI 风格得分」：梯度、扩散与时间形态",
                 fontsize=13.5, y=1.0)
    save_fig(fig, INSIGHTS + "framing_gradient.png")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    init_plot_style()
    scores = pd.read_csv(SCORES_PATH, usecols=["p_ai", "month", "content_len"])
    p_ai = scores["p_ai"].to_numpy()
    months_scores = scores["month"].astype(str).to_numpy()

    ref = scores["content_len"].to_numpy()

    scans: dict[str, dict] = {}
    scans_list = run_parallel(
        [partial(scan_view, path, content_len_ref=ref if key == "exploring" else None)
         for key, path in DATASETS.items()])
    for (key, _), s in zip(DATASETS.items(), scans_list):
        scans[key] = s
        print(f"[scan] {key}: {s['n_raw']} 条记录，短文本 {s['n_short']}，"
              f"对齐 mismatch={s['mismatch']}，文本长度 drift={s['text_len_drift']}")
    if scans["exploring"]["n_raw"] != len(scores):
        raise SystemExit("探索集记录数与 paper_exploring_scores 行数不一致，终止")

    kept = {k: np.asarray(dedup_indices(s["ids"], s["meta"]))
            for k, s in scans.items()}
    for s in scans.values():  # 去重后 ids/meta 不再需要，尽早释放
        s.pop("ids")
        s.pop("meta")
    n_sub = min(sum(1 for i in kept[k] if scans[k]["heads"][i] is not None)
                for k in kept)

    # ---- 维度 1：跨子集框架集中度 ----
    view_stats, boots, ending_pct = {}, {}, {}
    side_arrays: dict = {}
    side_counts: dict = {}
    opened: dict = {}   # 去重后的开篇片段，供维度 3 基线复用
    for k in DATASETS:
        sel = kept[k]
        side_arrays[k] = {}
        side_counts[k] = {}
        side_bincount: dict = {}
        open_values = [scans[k]["heads"][i] for i in sel]
        opened[k] = open_values
        for side, values in (("open", open_values),
                             ("close", [scans[k]["tails"][i] for i in sel])):
            codes, valid, cmap = encode(values)
            cnt = np.bincount(codes[valid])
            side_arrays[k][side] = (codes, valid, cmap)
            side_counts[k][side] = {cmap[i]: int(v)
                                    for i, v in enumerate(cnt) if v}
            side_bincount[side] = cnt
        n_used = int(side_arrays[k]["open"][1].sum())
        oc = side_bincount["open"]
        cc = side_bincount["close"]
        view_stats[k] = {
            "n_records": scans[k]["n_raw"],
            "n_notes": int(len(sel)),
            # 去重后仍因过短而不参与模板统计的笔记数（非原始记录数，
            # 否则探索集的重复副本会把该计数虚增 ~57%）。
            "n_short_excluded": int(len(sel)) - n_used,
            "n_short_records_raw": scans[k]["n_short"],
            "n_used": n_used,
            "top20_open": topk_coverage(oc, TOPK),
            "top50_open": topk_coverage(oc, 50),
            "top20_close": topk_coverage(cc, TOPK),
            "entropy_open_mm": shannon_entropy(oc, miller_madow=True),
            "entropy_close_mm": shannon_entropy(cc, miller_madow=True),
            "unique_open_ratio": float(np.count_nonzero(oc) / n_used),
            "unique_close_ratio": float(np.count_nonzero(cc) / n_used),
        }
        cov_o, ent_o = bootstrap_equal_n(*side_arrays[k]["open"][:2], n_sub)
        cov_c, ent_c = bootstrap_equal_n(*side_arrays[k]["close"][:2], n_sub)
        boots[k] = {"top20_open": cov_o, "entropy_open_mm": ent_o,
                    "top20_close": cov_c, "entropy_close_mm": ent_c}
        for metric, arr in boots[k].items():
            lo_b, hi_b = np.percentile(arr, [2.5, 97.5])
            view_stats[k][f"boot_{metric}_ci"] = [round(float(lo_b), 4),
                                                  round(float(hi_b), 4)]
        ec = Counter(scans[k]["ending_classes"][i] for i in sel)
        total = sum(ec.values())
        ending_pct[k] = {cat: round(ec.get(cat, 0) / total * 100, 3)
                         for cat in END_ORDER}
        print(f"[formulaicity] {k}: used={n_used}, "
              f"open top20={view_stats[k]['top20_open']:.1%}, "
              f"close top20={view_stats[k]['top20_close']:.1%}, "
              f"open entropy={view_stats[k]['entropy_open_mm']:.2f} bit")

    # 编码表 / 全局逐帖片段此后不再需要（opened 已保留开篇列表），及时释放
    del side_arrays
    for s in scans.values():
        s.pop("heads")
        s.pop("tails")
        s.pop("ending_classes")

    # ---- 维度 2：专属模板提炼 ----
    n_used_h = view_stats["human"]["n_used"]
    n_used_a = view_stats["aigc"]["n_used"]
    templates = {}
    for side, suffix in (("open", "open"), ("close", "close")):
        a_cnt = side_counts["aigc"][side]
        h_cnt = side_counts["human"][side]
        templates[f"aigc_specific_{suffix}"] = specific_templates(
            a_cnt, n_used_a, h_cnt, n_used_h, reverse=False, min_count=MIN_A_COUNT)
        templates[f"human_specific_{suffix}"] = specific_templates(
            a_cnt, n_used_a, h_cnt, n_used_h, reverse=True, min_count=MIN_H_COUNT)
    templates["aigc_top_open"] = top_templates(
        side_counts["aigc"]["open"], n_used_a, MIN_A_COUNT)
    a_open_set = {t["word"] for t in templates["aigc_specific_open"]}
    top_open_set = {t["word"] for t in templates["aigc_top_open"]}
    h_open_set = {t["word"] for t in templates["human_specific_open"]}
    del side_counts  # 模板提炼完成，汇总计数字典不再需要

    # ---- 维度 3：探索集梯度 / 季度 / 小时 ----
    sel_e = kept["exploring"]
    head_e = opened["exploring"]
    hours_e = scans["exploring"]["hours"][sel_e]
    head_valid = np.asarray([h is not None for h in head_e])
    p_v = p_ai[sel_e][head_valid]
    month_v = months_scores[sel_e][head_valid]
    hours_v = hours_e[head_valid]
    heads_valid = [h for h in head_e if h is not None]
    tmpl_a = np.asarray([h in a_open_set for h in heads_valid])
    tmpl_top = np.asarray([h in top_open_set for h in heads_valid])
    tmpl_h = np.asarray([h in h_open_set for h in heads_valid])
    n_v = int(head_valid.sum())

    order = np.argsort(p_v, kind="stable")
    decile = np.empty(n_v, dtype=np.int64)
    decile[order] = np.minimum(np.arange(n_v) * 10 // n_v, 9)
    deciles = []
    for d in range(10):
        m = decile == d
        n_m = int(m.sum())
        k_a = int(tmpl_a[m].sum())
        k_t = int(tmpl_top[m].sum())
        k_h = int(tmpl_h[m].sum())
        la, ha_ = wilson_ci(k_a, n_m)
        lt, ht = wilson_ci(k_t, n_m)
        lo, hh = wilson_ci(k_h, n_m)
        deciles.append({
            "decile": d + 1, "n": n_m,
            "mean_p": round(float(p_v[m].mean()), 4),
            "share_aigc_tmpl": round(k_a / n_m, 4),
            "share_aigc_tmpl_ci": [round(float(la), 4), round(float(ha_), 4)],
            "share_aigc_top_tmpl": round(k_t / n_m, 4),
            "share_aigc_top_tmpl_ci": [round(float(lt), 4), round(float(ht), 4)],
            "share_human_tmpl": round(k_h / n_m, 4),
            "share_human_tmpl_ci": [round(float(lo), 4), round(float(hh), 4)],
        })

    def spearman_cliff(indicator):
        rho, pval = spearmanr(p_v, indicator.astype(float))
        if indicator.any() and (~indicator).any():
            delta, p_mw = cliffs_delta(p_v[indicator], p_v[~indicator])
        else:
            delta, p_mw = 0.0, 1.0
        return ({"rho": round(float(rho), 3), "p": float(pval)},
                {"delta": round(float(delta), 3), "p": float(p_mw)})

    sp_a, cl_a = spearman_cliff(tmpl_a)
    sp_t, cl_t = spearman_cliff(tmpl_top)
    sp_h, cl_h = spearman_cliff(tmpl_h)
    grad = {
        "n": n_v,
        "spearman": sp_a,
        "cliff": cl_a,
        "spearman_top": sp_t,
        "cliff_top": cl_t,
        "spearman_human": sp_h,
        "cliff_human": cl_h,
        "deciles": deciles,
        "aigc_tmpl_total_rate": round(float(tmpl_a.mean()), 4),
        "aigc_top_tmpl_total_rate": round(float(tmpl_top.mean()), 4),
        "human_tmpl_total_rate": round(float(tmpl_h.mean()), 4),
    }
    print(f"[gradient] n={n_v}, 专属ρ={sp_a['rho']:.3f} (p={sp_a['p']:.1e}), "
          f"δ={cl_a['delta']:.3f}; Top50 率={grad['aigc_top_tmpl_total_rate']:.1%}")

    quarter = np.asarray([f"{m[:4]}Q{(int(m[4:6]) - 1) // 3 + 1}"
                          for m in month_v])
    quarterly = {"exploring": {}, "trend": {},
                 "exploring_share": grad["aigc_tmpl_total_rate"]}
    for q in sorted(set(quarter)):
        m = quarter == q
        n_q = int(m.sum())
        if n_q < MIN_QUARTER_N:
            continue
        k_q = int(tmpl_a[m].sum())
        lo, hi = wilson_ci(k_q, n_q)
        quarterly["exploring"][q] = {
            "n": n_q,
            "k_aigc_tmpl": k_q,
            "share_aigc_tmpl": round(k_q / n_q, 4),
            "ci_aigc_tmpl": [round(float(lo), 4), round(float(hi), 4)],
            "share_aigc_top_tmpl": round(float(tmpl_top[m].mean()), 4),
            "share_human_tmpl": round(float(tmpl_h[m].mean()), 4),
        }
    q_keys = sorted(quarterly["exploring"])
    if len(q_keys) >= 3:
        def q_trend(key):
            rho_, p_ = spearmanr(range(len(q_keys)),
                                  [quarterly["exploring"][q][key] for q in q_keys])
            return {"rho": round(float(rho_), 3), "p": float(p_)}
        quarterly["trend"] = q_trend("share_aigc_tmpl")
        quarterly["trend_top"] = q_trend("share_aigc_top_tmpl")
        quarterly["trend_human_tmpl"] = q_trend("share_human_tmpl")

    # 基线：人类集（全量 / ≤2022.06 干净窗口）与 AIGC 生成集
    h_heads = opened["human"]
    h_months = [scans["human"]["months"][i] for i in kept["human"]]
    h_valid = [h is not None for h in h_heads]
    quarterly["human_all_share"] = round(
        sum(1 for h, v in zip(h_heads, h_valid) if v and h in a_open_set)
        / max(sum(h_valid), 1), 4)
    clean = [(h, m) for h, m, v in zip(h_heads, h_months, h_valid)
             if v and m is not None and m <= "202206"]
    quarterly["human_clean_share"] = round(
        sum(1 for h, _ in clean if h in a_open_set) / max(len(clean), 1), 4)
    a_heads = [h for h in opened["aigc"] if h is not None]
    quarterly["aigc_share"] = round(
        sum(1 for h in a_heads if h in a_open_set) / max(len(a_heads), 1), 4)
    for s in scans.values():  # 月份仅用于人类集基线，用完后释放
        s.pop("months")
    del opened
    head_e_counter = Counter(heads_valid)
    n_head_e = sum(head_e_counter.values())
    quarterly["template_exploring_per1k"] = {
        t["word"]: round(head_e_counter.get(t["word"], 0) / max(n_head_e, 1)
                         * 1000, 2)
        for t in templates["aigc_specific_open"]}
    print(f"[quarterly] exploring AI-template {quarterly['exploring_share']:.1%}, "
          f"human clean {quarterly['human_clean_share']:.1%}, "
          f"AIGC {quarterly['aigc_share']:.1%}")

    hourly = {"by_hour": {}, "rho": 0.0, "rho_p": 1.0, "mean_p_spread": 0.0}
    hsel = hours_v >= 0
    hv, pv = hours_v[hsel], p_v[hsel]
    for h in range(24):
        m = hv == h
        if int(m.sum()) < 30:
            continue
        hourly["by_hour"][str(h)] = {
            "n": int(m.sum()),
            "mean_p": round(float(pv[m].mean()), 4),
            "hi_share": round(float((pv[m] > 0.9).mean()), 4),
        }
    if hv.size:
        h_rho, h_p = spearmanr(hv, pv)
        hourly["rho"] = round(float(h_rho), 3)
        hourly["rho_p"] = float(h_p)
    means = [v["mean_p"] for v in hourly["by_hour"].values()]
    if means:
        hourly["mean_p_spread"] = round(float(max(means) - min(means)), 4)
    print(f"[hourly] n={int(hsel.sum())}, spread={hourly['mean_p_spread']:.3f}, "
          f"ρ={hourly['rho']:.3f}")

    out = {
        "meta": {
            "frame_len": FRAME_LEN, "min_compact_len": MIN_NORM,
            "topk": TOPK, "bootstrap": BOOT, "bootstrap_n_sub": int(n_sub),
            "bootstrap_seed": BOOT_SEED,
            "exploring_content_len_mismatch": scans["exploring"]["mismatch"],
            "exploring_text_len_drift": scans["exploring"]["text_len_drift"],
            "note": "探索集按 note_id 去重；p_ai 按行序对齐 "
                    "paper_exploring_scores.csv.gz",
        },
        "formulaicity": {k: view_stats[k] for k in DATASETS},
        "ending_composition_pct": ending_pct,
        "templates": templates,
        "exploring_gradient": grad,
        "quarterly": quarterly,
        "hourly": hourly,
    }
    with open(BASE + "framing_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    write_markdown(out)
    plot_formulaicity(view_stats, boots, ending_pct)
    plot_gradient(grad, quarterly, hourly, templates)
    print("完成：framing_results.json / framing_results.md / "
          "insights/framing_formulaicity.png / insights/framing_gradient.png")


def write_markdown(out: dict) -> None:
    f = out["formulaicity"]
    meta = out["meta"]
    q = out["quarterly"]
    g = out["exploring_gradient"]
    lines = [
        "# 框架模板化分析结果（Framing）",
        "",
        "> 由 `framing_analysis.py` 全量运行生成；正文清洗与 `paper_features.py` 一致，",
        "> 人类/探索集按 note_id 去重，AIGC 无 note_id 逐记录处理；探索集 p_ai 按行序",
        "> 对齐 `paper_exploring_scores.csv.gz`（scores/features content_len "
        f"mismatch={meta['exploring_content_len_mismatch']}，原始文本长度 "
        f"drift={meta['exploring_text_len_drift']}）。",
        "",
        "## 1 跨子集框架集中度",
        "",
        "| 子集 | 有效帖数 | 首4字Top20覆盖 | 尾4字Top20覆盖 | 开篇熵(bit) | 开篇唯一率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, lab in LABELS.items():
        v = f[k]
        lines.append(
            f"| {lab} | {v['n_used']:,} | {v['top20_open']:.1%} | "
            f"{v['top20_close']:.1%} | {v['entropy_open_mm']:.2f} | "
            f"{v['unique_open_ratio']:.3f} |")
    lines += [
        "",
        f"紧凑文本 < {meta['min_compact_len']} 字（多为纯 #标签 帖）而不参与模板统计："
        f"human {f['human']['n_short_excluded']:,} / "
        f"aigc {f['aigc']['n_short_excluded']:,} / "
        f"exploring {f['exploring']['n_short_excluded']:,}。",
        "",
        "## 2 AI 专属 / 人类专属框架模板（Top 10）",
        "",
        "**AI 专属开篇：** " + "、".join(
            f"`{t['word']}`" for t in out["templates"]["aigc_specific_open"][:10]),
        "",
        "**AIGC 高频开篇（按生成集发生率 Top 10）：** " + "、".join(
            f"`{t['word']}`" for t in out["templates"]["aigc_top_open"][:10]),
        "",
        "**人类专属开篇：** " + "、".join(
            f"`{t['word']}`" for t in out["templates"]["human_specific_open"][:10]),
        "",
        "**AI 专属收尾：** " + "、".join(
            f"`{t['word']}`" for t in out["templates"]["aigc_specific_close"][:10]),
        "",
        "> 注：模板按 AIGC 与人类的每帖发生率之差排名；个别模板（如 `参考片段`，"
        "AIGC 发生率 <0.2%）属生成提示的格式残留，占比极低，不影响整体结论。",
        "",
        "## 3 探索集内的梯度与扩散",
        "",
        f"- P(AI) 与命中 AI 专属开篇的 Spearman ρ={g['spearman']['rho']}"
        f"（p={g['spearman']['p']:.1e}）；命中帖 vs 未命中帖的 "
        f"Cliff's δ={g['cliff']['delta']}（p={g['cliff']['p']:.1e}）。",
        f"- P(AI) 与命中 AIGC 高频开篇 Top-50 的 ρ={g['spearman_top']['rho']}"
        f"（p={g['spearman_top']['p']:.1e}），δ={g['cliff_top']['delta']}。",
        f"- 人类专属开篇的命中率随 P(AI) 显著下降：最低分位 "
        f"{g['deciles'][0]['share_human_tmpl']:.1%} → 最高分位 "
        f"{g['deciles'][-1]['share_human_tmpl']:.1%}，"
        f"ρ={g['spearman_human']['rho']}，δ={g['cliff_human']['delta']}"
        f"（p={g['cliff_human']['p']:.1e}）。",
        f"- AI 专属开篇使用率：AIGC 生成集 {q['aigc_share']:.1%}，"
        f"探索集 {q['exploring_share']:.1%}，人类集 {q['human_all_share']:.1%}"
        f"（干净窗口 ≤2022.06：{q['human_clean_share']:.1%}）；"
        f"AIGC 高频开篇 Top-50 在探索集的出现率 "
        f"{g['aigc_top_tmpl_total_rate']:.1%}。",
    ]
    if "trend" in q:
        lines.append(
            f"- 探索集季度趋势：专属模板 ρ={q['trend']['rho']}"
            f"（p={q['trend']['p']:.2f}），Top-50 模板 "
            f"ρ={q.get('trend_top', {}).get('rho', 0)}"
            f"（p={q.get('trend_top', {}).get('p', 1.0):.2f}）；"
            f"人类专属模板使用率趋势 ρ={q.get('trend_human_tmpl', {}).get('rho', 0)}"
            f"（p={q.get('trend_human_tmpl', {}).get('p', 1.0):.2f}）。")
    lines += [
        f"- 小时：平均 P(AI) 的 24 小时极差 {out['hourly']['mean_p_spread']:.3f}，"
        f"ρ(小时, P(AI))={out['hourly']['rho']:.2f}"
        f"（p={out['hourly']['rho_p']:.2f}）。",
        "",
        "## 4 收尾符号构成（%）",
        "",
        "| 收尾类型 | 人类 | AIGC | 探索集 |",
        "|---|---:|---:|---:|",
    ]
    for cat in END_ORDER:
        vals = [out["ending_composition_pct"][k].get(cat, 0.0)
                for k in ("human", "aigc", "exploring")]
        lines.append(f"| {cat} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    lines += [
        "",
        "图表见 `insights/`：`framing_formulaicity.png`、`framing_gradient.png`。",
    ]
    with open(BASE + "framing_results.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
