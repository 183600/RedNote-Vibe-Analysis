# -*- coding: utf-8 -*-
"""互动结构与「AI 味惩罚」分解（探索集，2023.01–2025.06）。

既有分析（paper_engagement.py）刻画了 AI 风格得分与互动总量的关系
（点赞 ρ=-0.139、收藏 -0.132、评论 -0.052，中心化单元内 -0.091），
但只看总量会掩盖三件事：

1. **互动构成**：AI 味更重的帖子是否只是"点赞少"，还是连收藏/评论的
   *相对*意愿也一起变化（收藏/点赞比、评论/点赞比）？
2. **零互动密度**：低总互动是来自整体下移，还是来自更多"完全无人互动"
   的帖子（零点赞/零收藏/零评论占比）？
3. **长尾集中度**：同一分位内部的互动分布形状是否不同（基尼系数、
   前 10% 帖子的互动占比）——高 AI 味帖子是否更"赢家通吃"？

另外把"月份×领域"单元内的秩相关同时对点赞/收藏/评论三个指标计算，
检验三种互动上的惩罚方向是否一致（paper_engagement 只报告了总量层面
的收藏/评论系数）。

全部计算复用 `paper_exploring_scores.csv.gz`（逐帖得分，行序与
`paper_features_exploring.csv.gz` 对齐），不重新训练检测器；分位标签
与 paper_engagement.py 完全同口径（同一 p_ai 十分位切分）。

输出：
  insights/engagement_structure.png   零互动率 / 基尼系数 / 互动构成比
  engagement_structure_results.json   全部数值结果
  engagement_structure_results.md     结果摘要
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from analysis_utils import (BASE, INSIGHTS, init_plot_style, save_fig,
                            save_json, gini_coefficient, wilson_ci)

MIN_MONTH_N = 100      # 月度单元最少帖子数（同 paper_engagement）
MIN_CELL_N = 100       # 月份×领域单元最少帖子数（同 paper_engagement）
N_DECILES = 10
TOP_SHARE = 0.10       # 长尾集中度：前 10% 帖子的互动占比

METRICS = (("likes", "点赞"), ("collections", "收藏"), ("comments", "评论"))


def load_scores() -> pd.DataFrame:
    # 只读用到列（p_ai_full / content_len 等从未参与分析），避免整表载入
    df = pd.read_csv(BASE + "paper_exploring_scores.csv.gz",
                     usecols=["month", "domain", "likes", "collections",
                              "comments", "p_ai"])
    df["month"] = df["month"].astype(str)
    df = df[df["month"] >= "202301"].copy()
    df["decile"] = pd.qcut(df["p_ai"], N_DECILES, labels=False,
                           duplicates="drop")
    return df


def decile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("decile"):
        row = {
            "decile": int(d),
            "n": int(len(g)),
            "p_lo": float(g["p_ai"].min()),
            "p_hi": float(g["p_ai"].max()),
        }
        for m, _ in METRICS:
            v = g[m].to_numpy(dtype=float)
            zero = int((v == 0).sum())
            lo, hi = wilson_ci(zero, len(v))
            row[f"{m}_median"] = float(np.median(v))
            row[f"{m}_zero_share"] = zero / len(v)
            row[f"{m}_zero_ci"] = [float(lo), float(hi)]
            row[f"{m}_gini"] = gini_coefficient(v)
            k = max(1, int(round(len(v) * TOP_SHARE)))
            tops = np.sort(v)[::-1][:k].sum()
            total = v.sum()
            row[f"{m}_top{int(TOP_SHARE*100)}_share"] = (
                float(tops / total) if total > 0 else 0.0)
        row["coll_per_like"] = float(np.median(
            g["collections"] / (g["likes"] + 1)))
        row["comm_per_like"] = float(np.median(
            g["comments"] / (g["likes"] + 1)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("decile")


def within_cell_rho(df: pd.DataFrame) -> dict:
    """月份×领域单元内 Spearman(p_ai, 指标)，按样本量加权汇总三个指标。

    三个指标共用同一次 groupby 分组（分组与指标无关），避免对同一份数据
    重复分组三遍。
    """
    out = {}
    # 只复制分组与相关计算需要的列（原实现复制整张 9 列大表）
    work = df[["month", "domain", "p_ai"]].copy()
    log_cols = {}
    for m, _ in METRICS:
        log_cols[m] = f"_log_{m}"
        work[log_cols[m]] = np.log10(1 + df[m])
    rhos = {m: [] for m, _ in METRICS}
    ns = {m: [] for m, _ in METRICS}
    for _, g in work.groupby(["month", "domain"], sort=True):
        n = len(g)
        if n < MIN_CELL_N:
            continue
        for m, _ in METRICS:
            r, _ = spearmanr(g["p_ai"], g[log_cols[m]])
            if not np.isnan(r):
                rhos[m].append(r)
                ns[m].append(n)
    for m, _ in METRICS:
        rhos_arr = np.asarray(rhos[m])
        if rhos_arr.size == 0:
            out[m] = {"weighted_rho": float("nan"),
                      "neg_cell_share": float("nan"), "n_cells": 0}
            continue
        out[m] = {
            "weighted_rho": float(np.average(rhos_arr, weights=ns[m])),
            "neg_cell_share": float((rhos_arr < 0).mean()),
            "n_cells": int(len(rhos_arr)),
        }
    return out


def trend_across_deciles(table: pd.DataFrame, col: str) -> dict:
    rho, p = spearmanr(table.index, table[col])
    return {"rho": float(rho), "p": float(p)}


def plot_structure(table: pd.DataFrame, out_path: str) -> None:
    init_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = table.index.to_numpy()
    labels = ["最像人"] + [""] * (N_DECILES - 2) + ["最像AI"]

    ax = axes[0]
    for (m, lab), color in zip(METRICS, ("#C44E52", "#55A868", "#4C72B0")):
        share = table[f"{m}_zero_share"] * 100
        lo = np.array([c[0] for c in table[f"{m}_zero_ci"]]) * 100
        hi = np.array([c[1] for c in table[f"{m}_zero_ci"]]) * 100
        ax.plot(x, share, "o-", color=color, ms=4, lw=1.4, label=f"零{lab}")
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    ax.set_xlabel("AI 风格得分十分位（0=最像人 … 9=最像AI）")
    ax.set_ylabel("零互动帖子占比（%，95% Wilson CI）")
    ax.set_title("零互动密度随 AI 味上升")
    ax.legend(fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)

    ax = axes[1]
    for (m, lab), color in zip(METRICS, ("#C44E52", "#55A868", "#4C72B0")):
        ax.plot(x, table[f"{m}_gini"], "o-", color=color, ms=4, lw=1.4,
                label=lab)
    ax.set_xlabel("AI 风格得分十分位")
    ax.set_ylabel("帖内互动基尼系数")
    ax.set_title("互动长尾集中度")
    ax.legend(fontsize=9)
    ax.set_xticks(list(x))

    ax = axes[2]
    ax.plot(x, table["coll_per_like"], "s-", color="#55A868", ms=4, lw=1.4,
            label="收藏/点赞")
    ax.plot(x, table["comm_per_like"], "^-", color="#4C72B0", ms=4, lw=1.4,
            label="评论/点赞")
    ax.set_xlabel("AI 风格得分十分位")
    ax.set_ylabel("逐帖比率的中位数")
    ax.set_title("互动构成比（×/(点赞+1)）")
    ax.legend(fontsize=9)
    ax.set_xticks(list(x))

    fig.suptitle("「AI 味惩罚」的互动结构分解（探索集 2023.01–2025.06）",
                 fontsize=13)
    fig.tight_layout()
    save_fig(fig, out_path)


def write_markdown(out: dict, t: pd.DataFrame) -> str:
    cells = out["within_cell"]
    lines = [
        "# 互动结构与「AI 味惩罚」分解（Engagement Structure）",
        "",
        "> 由 `engagement_structure_analysis.py` 全量运行生成；数据源 "
        "`paper_exploring_scores.csv.gz`（逐帖 P(AI)），口径与 "
        "`paper_engagement.py` 一致（2023.01 起、同一十分位切分、"
        "月份×领域单元内秩相关）。",
        "",
        "## 1 零互动密度（零点赞 / 零收藏 / 零评论占比）",
        "",
        "| 十分位 | n | P(AI) 区间 | 零点赞 | 零收藏 | 零评论 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for d, r in t.iterrows():
        lines.append(
            f"| {d} | {int(r['n']):,} | {r['p_lo']:.3f}–{r['p_hi']:.3f} | "
            f"{r['likes_zero_share']:.1%} | {r['collections_zero_share']:.1%} | "
            f"{r['comments_zero_share']:.1%} |")
    lines += [
        "",
        "## 2 长尾集中度（基尼系数 / 前 10% 帖子互动占比）",
        "",
        "| 十分位 | 点赞基尼 | 点赞前10%占比 | 收藏基尼 | 评论基尼 |",
        "|---|---:|---:|---:|---:|",
    ]
    for d, r in t.iterrows():
        lines.append(
            f"| {d} | {r['likes_gini']:.3f} | "
            f"{r['likes_top10_share']:.1%} | {r['collections_gini']:.3f} | "
            f"{r['comments_gini']:.3f} |")
    lines += [
        "",
        "## 3 互动构成比（逐帖中位数）",
        "",
        "| 十分位 | 收藏/点赞 | 评论/点赞 |",
        "|---|---:|---:|",
    ]
    for d, r in t.iterrows():
        lines.append(f"| {d} | {r['coll_per_like']:.3f} | "
                     f"{r['comm_per_like']:.3f} |")
    lines += [
        "",
        "## 4 月份×领域单元内的秩相关（三个互动指标）",
        "",
        "| 指标 | 加权 ρ | 负相关单元占比 | 单元数 |",
        "|---|---:|---:|---:|",
    ]
    for m, lab in (("likes", "点赞"), ("collections", "收藏"),
                   ("comments", "评论")):
        c = cells[m]
        lines.append(f"| {lab} | {c['weighted_rho']:+.4f} | "
                     f"{c['neg_cell_share']:.1%} | {c['n_cells']} |")
    tr = out["decile_trends"]
    lines += [
        "",
        "## 5 十分位趋势（Spearman）",
        "",
        f"- 零点赞占比：ρ={tr['likes_zero_share']['rho']:+.3f} "
        f"(p={tr['likes_zero_share']['p']:.2g})",
        f"- 点赞基尼：ρ={tr['likes_gini']['rho']:+.3f} "
        f"(p={tr['likes_gini']['p']:.2g})",
        f"- 收藏/点赞比：ρ={tr['coll_per_like']['rho']:+.3f} "
        f"(p={tr['coll_per_like']['p']:.2g})（非单调，见构成比表）",
        "",
        f"> 一致性校验：本表十分位点赞中位数与 `paper_engagement_results.json` "
        f"的最大绝对差 = {out['consistency']['likes_median_max_abs_diff']}。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load_scores()
    print(f"探索集（2023.01 起）: {len(df):,} 条（去重前逐记录口径）")

    table = decile_table(df)
    cells = within_cell_rho(df)
    print("\n零互动占比（decile 0 → 9）:")
    for m, lab in METRICS:
        print(f"  {lab}: {table[f'{m}_zero_share'].iloc[0]:.1%} → "
              f"{table[f'{m}_zero_share'].iloc[-1]:.1%}")
    print("点赞基尼: "
          f"{table['likes_gini'].iloc[0]:.3f} → {table['likes_gini'].iloc[-1]:.3f}")
    print("收藏/点赞比: "
          f"{table['coll_per_like'].iloc[0]:.3f} → 峰值 "
          f"{table['coll_per_like'].max():.3f} → {table['coll_per_like'].iloc[-1]:.3f}")
    for m, lab in METRICS:
        c = cells[m]
        if c["n_cells"]:
            print(f"单元内 ρ({lab}): {c['weighted_rho']:+.4f} "
                  f"(负单元 {c['neg_cell_share']:.0%})")
        else:
            print(f"单元内 ρ({lab}): 无满足样本量要求的单元")

    with open(BASE + "paper_engagement_results.json", encoding="utf-8") as f:
        paper = json.load(f)
    paper_med = {int(d["decile"]): d["likes_median"]
                 for d in paper["decile_table"]}
    max_diff = max(abs(table.loc[d, "likes_median"] - paper_med[d])
                   for d in table.index if d in paper_med)

    out = {
        "meta": {
            "source": "paper_exploring_scores.csv.gz",
            "month_from": "202301",
            "n_deciles": N_DECILES,
            "min_cell_n": MIN_CELL_N,
            "top_share": TOP_SHARE,
            "note": "收藏/点赞比、评论/点赞比按逐帖 (x+1) 平滑后取中位数",
        },
        "decile_table": {
            str(d): {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in r.items()}
            for d, r in table.iterrows()
        },
        "within_cell": {m: {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in c.items()}
                        for m, c in cells.items()},
        "decile_trends": {
            "likes_zero_share": trend_across_deciles(table, "likes_zero_share"),
            "likes_gini": trend_across_deciles(table, "likes_gini"),
            "coll_per_like": trend_across_deciles(table, "coll_per_like"),
        },
        "consistency": {
            "likes_median_max_abs_diff": round(float(max_diff), 4),
            "against": "paper_engagement_results.json decile_table",
        },
    }
    save_json(out, BASE + "engagement_structure_results.json")
    with open(BASE + "engagement_structure_results.md", "w",
              encoding="utf-8") as f:
        f.write(write_markdown(out, table))
    plot_structure(table, INSIGHTS + "engagement_structure.png")
    print("已保存 engagement_structure_results.json/.md 与 "
          "insights/engagement_structure.png")


if __name__ == "__main__":
    main()
