# -*- coding: utf-8 -*-
"""第五步：后 LLM 时代的风格漂移分析。

把训练好的风格检测器逐月应用到探索集（2023.01–2025.06），并：
1. 阴性对照：对人类集（2020.12–2022.11，全部为前 LLM 时代）逐月打分，应当平稳
2. 领域调整：固定领域权重的月度得分，消除领域构成漂移的影响
3. 单特征时间曲线：6 个对爬取格式稳健的特征
4. Spearman 趋势检验
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr

BASE = "/home/qwe12345678/RedNote-Vibe-Analysis/"
FIGS = BASE + "paper_figs/"
os.makedirs(FIGS, exist_ok=True)

font_dir = "/usr/share/fonts/noto-cjk"
for f in sorted(os.listdir(font_dir)):
    if "NotoSansCJK" in f and f.endswith(".ttc"):
        font_manager.fontManager.addfont(os.path.join(font_dir, f))
        prop = font_manager.FontProperties(fname=os.path.join(font_dir, f))
        plt.rcParams["font.family"] = prop.get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

MIN_MONTH_N = 100


def score(df, model, robust=True):
    """用保存的逻辑回归参数打分。robust=True 用稳健特征模型。"""
    if robust:
        feats = model["robust_features"]
        mean = np.array(model["robust_scaler_mean"])
        scale = np.array(model["robust_scaler_scale"])
        coef = np.array(model["robust_coef"])
        b = model["robust_intercept"]
    else:
        feats = model["features"]
        mean = np.array(model["scaler_mean"])
        scale = np.array(model["scaler_scale"])
        coef = np.array(model["coef"])
        b = model["intercept"]
    X = df[feats].fillna(0).values
    z = (X - mean) / scale
    logit = z @ coef + b
    return 1.0 / (1.0 + np.exp(-logit))


def monthly_series(df, col="p_ai"):
    g = df.groupby("month").agg(n=(col, "size"), mean=(col, "mean"),
                                hi=(col, lambda s: (s > 0.9).mean()))
    g = g[g["n"] >= MIN_MONTH_N]
    return g.sort_index()


def domain_adjusted(df, domains, weights):
    """固定领域权重的月度均值。"""
    rows = {}
    for m, gm in df.groupby("month"):
        if len(gm) < MIN_MONTH_N:
            continue
        vals, ws = [], []
        for d in domains:
            gd = gm[gm["domain"] == d]
            if len(gd) >= 20:
                vals.append(gd["p_ai"].mean())
                ws.append(weights[d])
        if sum(ws) > 0.5:  # 至少覆盖一半权重
            rows[m] = float(np.average(vals, weights=ws))
    return pd.Series(rows).sort_index()


def main():
    with open(BASE + "paper_detector_model.json", encoding="utf-8") as f:
        model = json.load(f)

    human = pd.read_csv(BASE + "paper_features_human.csv.gz")
    expl = pd.read_csv(BASE + "paper_features_exploring.csv.gz")
    human["month"] = human["month"].astype(str)
    expl["month"] = expl["month"].astype(str)

    human["p_ai"] = score(human, model, robust=True)
    expl["p_ai"] = score(expl, model, robust=True)
    expl["p_ai_full"] = score(expl, model, robust=False)

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
    x_h = range(len(ms_h))
    ax.plot(x_h, ms_h["mean"], "o-", color="#4C72B0", ms=3, lw=1.2)
    ax.axhline(ms_h["mean"].mean(), color="gray", ls="--", lw=0.8)
    ax.set_xticks(list(x_h)[::4])
    ax.set_xticklabels(ms_h.index[::4], rotation=45, fontsize=8)
    ax.set_ylabel("平均 P(AI)")
    ax.set_title(f"人类集（前LLM时代）阴性对照\nρ={rho_h:.2f}, p={p_h:.2g}")
    ax.set_ylim(0, 0.65)

    ax = axes[1]
    x_e = range(len(ms_e))
    ax.plot(x_e, ms_e["mean"], "o-", color="#C44E52", ms=3, lw=1.2, label="原始月度均值")
    adj_x = [list(ms_e.index).index(m) for m in adj.index if m in set(ms_e.index)]
    ax.plot(adj_x, [adj[m] for m in adj.index if m in set(ms_e.index)],
            "s--", color="#55A868", ms=3, lw=1.2, label="领域调整后")
    ax.axhline(ms_h["mean"].mean(), color="gray", ls="--", lw=0.8, label="人类集基线")
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
    hi_base = (human["p_ai"] > 0.9).mean() * 100
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
    for ax, (f, lab) in zip(axes.flat, feat_show):
        qm = expl_q.groupby("quarter")[f].mean().reindex(quarters)
        ax.plot(range(len(quarters)), qm.values, "o-", color="#C44E52", lw=1.4)
        h_base = human[f].mean()
        a_base = pd.read_csv(BASE + "paper_features_aigc.csv.gz")[f].mean()
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
        "human_baseline_mean_p": round(float(human["p_ai"].mean()), 4),
        "human_baseline_hi_share": round(float((human["p_ai"] > 0.9).mean()), 4),
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
    print(f"\n人类集基线 mean P(AI) = {human['p_ai'].mean():.4f}, >0.9 占比 {(human['p_ai']>0.9).mean()*100:.1f}%")
    print(f"探索集 2023上半年: mean={e23['p_ai'].mean():.4f}, >0.9 {(e23['p_ai']>0.9).mean()*100:.1f}% (n={len(e23)})")
    print(f"探索集 2025上半年: mean={e25['p_ai'].mean():.4f}, >0.9 {(e25['p_ai']>0.9).mean()*100:.1f}% (n={len(e25)})")

    # 保存逐帖得分供互动分析
    expl[["month", "domain", "likes", "collections", "comments",
          "p_ai", "p_ai_full", "content_len"]].to_csv(
        BASE + "paper_exploring_scores.csv.gz", index=False, compression="gzip")
    print("已保存 paper_exploring_scores.csv.gz")


if __name__ == "__main__":
    main()
