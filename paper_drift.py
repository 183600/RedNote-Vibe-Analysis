# -*- coding: utf-8 -*-
"""第五步：后 LLM 时代的风格漂移分析。

把训练好的风格检测器逐月应用到探索集（2023.01–2025.06），并：
1. 阴性对照：对人类集（2020.12–2022.11，全部为前 LLM 时代）逐月打分，应当平稳
2. 领域调整：固定领域权重的月度得分，消除领域构成漂移的影响
3. 单特征时间曲线：6 个对爬取格式稳健的特征
4. Spearman 趋势检验

用法：`python3 paper_drift.py`（默认同时刷新派生得分文件
`paper_exploring_scores.csv.gz`，内容不变时保持原文件字节不动）；
`python3 paper_drift.py --figs-only` 只重算图表与结果 JSON，不触碰该数据文件。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from analysis_utils import (BASE, setup_chinese_fonts, detector_oof_scores,
                            detector_score, write_csv_gz_if_changed)

FIGS = BASE + "paper_figs/"
os.makedirs(FIGS, exist_ok=True)
setup_chinese_fonts()

MIN_MONTH_N = 100

score = detector_score


def monthly_series(df, col="p_ai"):
    g = df.groupby("month").agg(n=(col, "size"), mean=(col, "mean"),
                                hi=(col, lambda s: (s > 0.9).mean()))
    g = g[g["n"] >= MIN_MONTH_N]
    return g.sort_index()


def domain_adjusted(df, domains, weights):
    """固定领域权重的月度均值。

    先用一次 groupby 建好 (月, 领域) 的行索引，后续按索引切片取均值，
    避免每个月对 10 个领域做重复的布尔扫描；每个单元格的均值仍走
    `Series.mean()` 的同一计算路径，数值逐位一致。
    """
    rows = {}
    size_m = df.groupby("month").size()
    cell_idx = df.groupby(["month", "domain"], sort=True).indices
    p = df["p_ai"]
    for m in size_m.index:
        if size_m[m] < MIN_MONTH_N:
            continue
        vals, ws = [], []
        for d in domains:
            idx = cell_idx.get((m, d))
            if idx is not None and len(idx) >= 20:
                vals.append(p.iloc[idx].mean())
                ws.append(weights[d])
        if sum(ws) > 0.5:  # 至少覆盖一半权重
            rows[m] = float(np.average(vals, weights=ws))
    return pd.Series(rows).sort_index()


def main(with_scores: bool = True):
    with open(BASE + "paper_detector_model.json", encoding="utf-8") as f:
        model = json.load(f)

    # 只加载各帧实际用到的列：模型特征列表来自已保存的 detector 参数，
    # 人类/AIGC 帧仅稳健特征参与打分，探索集需要全特征与互动/时间列。
    model_feats = list(model["features"])
    robust_feats = list(model["robust_features"])
    human = pd.read_csv(BASE + "paper_features_human.csv.gz",
                        usecols=["month", "domain"] + robust_feats)
    expl = pd.read_csv(BASE + "paper_features_exploring.csv.gz",
                       usecols=["month", "domain", "likes", "collections",
                                "comments"] + model_feats)
    aigc_feats = pd.read_csv(BASE + "paper_features_aigc.csv.gz",
                             usecols=robust_feats)
    human["month"] = human["month"].astype(str)
    expl["month"] = expl["month"].astype(str)

    human["p_ai"] = detector_oof_scores(human, aigc_feats, model, robust=True)
    expl["p_ai"] = score(expl, model, robust=True)
    expl["p_ai_full"] = score(expl, model, robust=False)
    # 人类集基线与月度均值只算一次，图/JSON/摘要复用。
    # 基线口径为论文定义的「干净回溯窗口」（≤2022.06）：2022.07 起是实时
    # 爬取窗口，存在采样断层（高置信占比 1.93% vs 干净窗口 0.67%），混入
    # 会系统性抬高基线、削弱与探索集台阶的对比。
    human_clean = human[human["month"] <= "202206"]
    human_mean_p = float(human_clean["p_ai"].mean())
    human_hi_share = float((human_clean["p_ai"] > 0.9).mean())

    ms_h = monthly_series(human)
    ms_e = monthly_series(expl)

    # 领域调整（用探索集整体领域占比作为固定权重，只用两集共有领域）
    common = sorted(set(human["domain"].unique()) & set(expl["domain"].unique()))
    w = expl[expl["domain"].isin(common)]["domain"].value_counts(normalize=True).to_dict()
    adj = domain_adjusted(expl, common, w)

    # 趋势检验（探索集）
    idx = np.arange(len(ms_e))
    rho, p = spearmanr(idx, ms_e["mean"].values)
    rho_hi, p_hi = spearmanr(idx, ms_e["hi"].values)
    idx_h = np.arange(len(ms_h))
    rho_h, p_h = spearmanr(idx_h, ms_h["mean"].values)
    rho_adj, p_adj = spearmanr(np.arange(len(adj)), adj.values)
    print(f"探索集月度均值趋势: rho={rho:.3f} p={p:.2e}")
    print(f"探索集高置信(>0.9)占比趋势: rho={rho_hi:.3f} p={p_hi:.2e}")
    print(f"领域调整后趋势: rho={rho_adj:.3f} p={p_adj:.2e}")
    print(f"人类集阴性对照趋势: rho={rho_h:.3f} p={p_h:.2e}")

    # ---- 图1：AI风格得分时间线 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    human_monthly_base = human_mean_p
    x_h = range(len(ms_h))
    ax.plot(x_h, ms_h["mean"], "o-", color="#4C72B0", ms=3, lw=1.2)
    ax.axhline(human_monthly_base, color="gray", ls="--", lw=0.8)
    ax.set_xticks(list(x_h)[::4])
    ax.set_xticklabels(ms_h.index[::4], rotation=45, fontsize=8)
    ax.set_ylabel("平均 P(AI)")
    ax.set_title(f"人类集（前LLM时代）阴性对照\nρ={rho_h:.2f}, p={p_h:.2g}")
    ax.set_ylim(0, 0.65)

    ax = axes[1]
    x_e = range(len(ms_e))
    ax.plot(x_e, ms_e["mean"], "o-", color="#C44E52", ms=3, lw=1.2, label="原始月度均值")
    month_pos = {m: i for i, m in enumerate(ms_e.index)}
    adj_x = [month_pos[m] for m in adj.index if m in month_pos]
    ax.plot(adj_x, [adj[m] for m in adj.index if m in month_pos],
            "s--", color="#55A868", ms=3, lw=1.2, label="领域调整后")
    ax.axhline(human_monthly_base, color="gray", ls="--", lw=0.8, label="人类集基线")
    ax.set_xticks(list(x_e)[::3])
    ax.set_xticklabels(ms_e.index[::3], rotation=45, fontsize=8)
    ax.set_title(f"探索集（后LLM时代）AI 风格得分\n原始 ρ={rho:.2f} (p={p:.2g})，调整后 ρ={rho_adj:.2f} (p={p_adj:.2g})")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=9)
    fig.suptitle("社交媒体文本的 AI 风格漂移（稳健特征检测器，P(AI) 月度均值）", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS + "drift_score_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2：高置信占比 ----
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x_e, ms_e["hi"] * 100, color="#C44E52", alpha=0.85)
    hi_base = human_hi_share * 100
    ax.axhline(hi_base, color="gray", ls="--", lw=1,
               label=f"人类集基线 {hi_base:.1f}%")
    ax.set_xticks(list(x_e)[::3])
    ax.set_xticklabels(ms_e.index[::3], rotation=45, fontsize=8)
    ax.set_ylabel("P(AI) > 0.9 的帖子占比 (%)")
    ax.set_title(f"探索集高置信 AI 风格帖子占比（ρ={rho_hi:.2f}, p={p_hi:.2g}）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS + "drift_high_confidence.png", dpi=150)
    plt.close(fig)

    # ---- 图3：单特征时间曲线（季度）----
    expl["quarter"] = expl["month"].str[:4] + "Q" + (
        (expl["month"].str[4:6].astype(int) - 1) // 3 + 1).astype(str)
    expl_q = expl[expl["month"] >= "202301"]
    feat_show = [("excl100", "感叹号/百字"), ("emoji100", "emoji/百字"),
                 ("ai_lex100", "AI味词/百字"), ("hum_lex100", "逻辑连接词/百字"),
                 ("uniq_char_ratio", "独特字比率"), ("content_len", "正文长度")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    quarters = sorted(expl_q["quarter"].unique())
    # 季度均值与两套基线一次性按列算好（与逐特征单独算完全等价）。
    feat_cols = [f for f, _ in feat_show]
    qm_all = expl_q.groupby("quarter")[feat_cols].mean().reindex(quarters)
    # 与上方数值基线保持同一干净窗口，避免实时爬取窗口的采样断层
    # 把单特征曲线的人类基线整体抬高。
    h_bases = human_clean[feat_cols].mean()
    a_bases = aigc_feats[feat_cols].mean()
    for ax, (f, lab) in zip(axes.flat, feat_show):
        qm = qm_all[f]
        ax.plot(range(len(quarters)), qm.values, "o-", color="#C44E52", lw=1.4)
        h_base = float(h_bases[f])
        a_base = float(a_bases[f])
        ax.axhline(h_base, color="#4C72B0", ls="--", lw=1.1, label="人类基线")
        ax.axhline(a_base, color="#937860", ls=":", lw=1.3, label="AIGC基线")
        ax.set_xticks(range(len(quarters)))
        ax.set_xticklabels(quarters, rotation=45, fontsize=8)
        ax.set_title(lab, fontsize=11)
        rho_f, p_f = spearmanr(range(len(quarters)), qm.values)
        ax.text(0.03, 0.92, f"ρ={rho_f:.2f}", transform=ax.transAxes, fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("探索集（2023Q1–2025Q2）核心风格特征的季度演化", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS + "drift_features_quarterly.png", dpi=150)
    plt.close(fig)

    # ---- 数值输出 ----
    out = {
        "human_baseline_mean_p": round(human_mean_p, 4),
        "human_baseline_hi_share": round(human_hi_share, 4),
        "trend": {
            "exploring_mean": {"rho": round(rho, 3), "p": float(p)},
            "exploring_hi": {"rho": round(rho_hi, 3), "p": float(p_hi)},
            "exploring_domain_adj": {"rho": round(rho_adj, 3), "p": float(p_adj)},
            "human_control": {"rho": round(rho_h, 3), "p": float(p_h)},
        },
        "exploring_monthly": {
            m: {"n": int(r["n"]), "mean_p": round(float(r["mean"]), 4),
                "hi_share": round(float(r["hi"]), 4)}
            for m, r in ms_e.iterrows()
        },
        "human_monthly": {
            m: {"n": int(r["n"]), "mean_p": round(float(r["mean"]), 4)}
            for m, r in ms_h.iterrows()
        },
        "domain_adjusted_monthly": {m: round(float(v), 4) for m, v in adj.items()},
    }
    with open(BASE + "paper_drift_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # 摘要打印
    e23 = expl[(expl["month"] >= "202301") & (expl["month"] <= "202306")]
    e25 = expl[(expl["month"] >= "202501") & (expl["month"] <= "202506")]
    print(f"\n人类集基线 mean P(AI) = {human_mean_p:.4f}, >0.9 占比 {human_hi_share*100:.1f}%")
    print(f"探索集 2023上半年: mean={e23['p_ai'].mean():.4f}, >0.9 {(e23['p_ai']>0.9).mean()*100:.1f}% (n={len(e23)})")
    print(f"探索集 2025上半年: mean={e25['p_ai'].mean():.4f}, >0.9 {(e25['p_ai']>0.9).mean()*100:.1f}% (n={len(e25)})")

    # 保存逐帖得分供互动分析（内容未变化时不写文件，保持研究数据字节不动）
    if with_scores:
        scores = expl[["month", "domain", "likes", "collections", "comments",
                       "p_ai", "p_ai_full", "content_len"]]
        if write_csv_gz_if_changed(scores, BASE + "paper_exploring_scores.csv.gz"):
            print("已更新 paper_exploring_scores.csv.gz")
        else:
            print("paper_exploring_scores.csv.gz 内容未变化，保持原文件不变")
    else:
        print("[figs-only] 跳过 paper_exploring_scores.csv.gz 写入")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figs-only", action="store_true",
                        help="只重算图表与结果 JSON，不写 paper_exploring_scores.csv.gz")
    args = parser.parse_args()
    main(with_scores=not args.figs_only)
