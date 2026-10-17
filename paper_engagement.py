# -*- coding: utf-8 -*-
"""第六步：AI 风格得分与用户互动的关系（探索集，2023.01–2025.06）。

方法要点：
- 点赞数重尾，使用 log10(1+likes)
- 爬取分批次（月份）差异大，主分析在「月份×领域」单元内做秩相关，再汇总
- 同时报告分位组对比与按月相关系数的分布
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from analysis_utils import BASE, setup_chinese_fonts

FIGS = BASE + "paper_figs/"
os.makedirs(FIGS, exist_ok=True)
setup_chinese_fonts()


def main():
    df = pd.read_csv(BASE + "paper_exploring_scores.csv.gz",
                     usecols=["month", "domain", "likes", "collections",
                              "comments", "p_ai"])
    df["month"] = df["month"].astype(str)
    df = df[df["month"] >= "202301"].copy()
    df["log_likes"] = np.log10(1 + df["likes"])
    df["log_colls"] = np.log10(1 + df["collections"])
    df["log_comms"] = np.log10(1 + df["comments"])

    # ---- 1. 总体与单元内秩相关 ----
    rho_all, p_all = spearmanr(df["p_ai"], df["log_likes"])
    print(f"总体 Spearman(p_ai, likes): rho={rho_all:.3f} p={p_all:.2e} n={len(df)}")

    cell_rhos, cell_ns = [], []
    p_ai_np = df["p_ai"].to_numpy()
    log_likes_np = df["log_likes"].to_numpy()
    for (m, d), idx in df.groupby(["month", "domain"]).indices.items():
        if len(idx) >= 100:
            r, _ = spearmanr(p_ai_np[idx], log_likes_np[idx])
            if not np.isnan(r):
                cell_rhos.append(r)
                cell_ns.append(len(idx))
    cell_rhos = np.array(cell_rhos)
    cell_ns = np.array(cell_ns)
    w_mean = float(np.average(cell_rhos, weights=cell_ns))
    print(f"月份×领域单元内 rho: 加权均值={w_mean:.3f}, 单元数={len(cell_rhos)}, "
          f"负相关单元占比={(cell_rhos<0).mean()*100:.0f}%")

    # 其他互动指标
    rho_c, _ = spearmanr(df["p_ai"], df["log_colls"])
    rho_m, _ = spearmanr(df["p_ai"], df["log_comms"])
    print(f"收藏 rho={rho_c:.3f}  评论 rho={rho_m:.3f}")

    # ---- 2. 得分十分位 vs 互动 ----
    df["decile"] = pd.qcut(df["p_ai"], 10, labels=False, duplicates="drop")
    dec = df.groupby("decile").agg(
        n=("log_likes", "size"),
        likes_med=("likes", "median"),
        log_likes_mean=("log_likes", "mean"),
        log_likes_se=("log_likes", lambda s: s.std() / np.sqrt(len(s))),
        comments_med=("comments", "median"),
        p_lo=("p_ai", "min"), p_hi=("p_ai", "max"),
    )
    print("\n得分十分位 vs 互动:")
    print(dec[["n", "likes_med", "log_likes_mean"]].to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.errorbar(range(len(dec)), dec["log_likes_mean"],
                yerr=1.96 * dec["log_likes_se"], fmt="o-", color="#C44E52",
                lw=1.4, capsize=3)
    ax.set_xlabel("AI 风格得分十分位（0=最像人 … 9=最像AI）")
    ax.set_ylabel("log10(1+点赞) 均值（95%CI）")
    ax.set_title(f"AI 风格与点赞量（Spearman ρ={rho_all:.3f}）")
    ax.set_xticks(range(len(dec)))

    ax = axes[1]
    groups = [("P(AI)<0.1", df[df["p_ai"] < 0.1]), ("0.1–0.5", df[(df["p_ai"] >= 0.1) & (df["p_ai"] < 0.5)]),
              ("0.5–0.9", df[(df["p_ai"] >= 0.5) & (df["p_ai"] < 0.9)]), ("P(AI)≥0.9", df[df["p_ai"] >= 0.9])]
    meds_l = [g["likes"].median() for _, g in groups]
    meds_c = [g["comments"].median() for _, g in groups]
    x = np.arange(len(groups))
    ax.bar(x - 0.2, meds_l, 0.4, label="点赞中位数", color="#C44E52")
    ax.bar(x + 0.2, [m * 10 for m in meds_c], 0.4, label="评论中位数×10", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels([k for k, _ in groups])
    ax.set_ylabel("中位数")
    ax.set_title("按 AI 风格分组的互动中位数")
    ax.legend()
    for i, (v, g) in enumerate(zip(meds_l, groups)):
        ax.text(i - 0.2, v + 30, f"{int(v)}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS + "engagement_vs_score.png", dpi=150)
    plt.close(fig)

    # ---- 3. 按月 rho 时间线（互动惩罚是否随时间变化）----
    mr = []
    for m, idx in df.groupby("month").indices.items():
        if len(idx) >= 300:
            r, _ = spearmanr(p_ai_np[idx], log_likes_np[idx])
            mr.append((m, r, len(idx)))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([m for m, _, _ in mr], [r for _, r, _ in mr], color="#4C72B0")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(0, len(mr), 3))
    ax.set_xticklabels([mr[i][0] for i in range(0, len(mr), 3)], rotation=45, fontsize=8)
    ax.set_ylabel("月内 Spearman ρ")
    ax.set_title("AI 风格得分与点赞的月内相关（按月）")
    fig.tight_layout()
    fig.savefig(FIGS + "engagement_rho_monthly.png", dpi=150)
    plt.close(fig)
    trend_rho, trend_p = spearmanr(range(len(mr)), [r for _, r, _ in mr])
    print(f"\n月内ρ的时间趋势: rho={trend_rho:.3f} p={trend_p:.3f}")

    out = {
        "overall_spearman_likes": {"rho": round(float(rho_all), 4), "p": float(p_all), "n": int(len(df))},
        "within_cell_weighted_rho": round(w_mean, 4),
        "neg_cell_share": round(float((cell_rhos < 0).mean()), 3),
        "n_cells": int(len(cell_rhos)),
        "spearman_collections": round(float(rho_c), 4),
        "spearman_comments": round(float(rho_m), 4),
        "decile_table": [
            {"decile": int(i), "n": int(r["n"]), "likes_median": float(r["likes_med"]),
             "log_likes_mean": round(float(r["log_likes_mean"]), 4),
             "p_range": [round(float(r["p_lo"]), 3), round(float(r["p_hi"]), 3)]}
            for i, r in dec.iterrows()
        ],
        "group_medians": {k: {"likes": float(g["likes"].median()),
                              "comments": float(g["comments"].median()),
                              "collections": float(g["collections"].median()),
                              "n": int(len(g))} for k, g in groups},
        "monthly_rho": [{"month": m, "rho": round(float(r), 4), "n": n} for m, r, n in mr],
        "rho_time_trend": {"rho": round(float(trend_rho), 3), "p": float(trend_p)},
    }
    with open(BASE + "paper_engagement_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("已保存 paper_engagement_results.json")


if __name__ == "__main__":
    main()
