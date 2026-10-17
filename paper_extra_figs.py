# -*- coding: utf-8 -*-
"""第七步：补充论文图。

1. 重绘漂移时间线：标注人类集 2022.07 的采样断层，阴性对照只在干净窗口内检验
2. 差中差对比图（同采样层的前后 LLM 对比）
3. 词表 log-odds 图
并把修正后的趋势数字合并进 paper_drift_results.json。
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from analysis_utils import BASE, setup_chinese_fonts, detector_score

FIGS = BASE + "paper_figs/"
setup_chinese_fonts()

score = detector_score


with open(BASE + "paper_detector_model.json", encoding="utf-8") as f:
    model = json.load(f)
# 人类集只用到月份/点赞与稳健特征，按需读列压低内存与解析耗时
human = pd.read_csv(BASE + "paper_features_human.csv.gz",
                    usecols=["month", "likes"] + list(model["robust_features"]))
human["month"] = human["month"].astype(str)
human["p_ai"] = score(human, model)
# 探索集逐帖稳健得分已由 paper_drift.py 落盘（p_ai 列即稳健模型分），
# 直接复用，避免重算全特征并保持与 drift 结果逐位一致
expl = pd.read_csv(BASE + "paper_exploring_scores.csv.gz",
                   usecols=["month", "likes", "p_ai"])
expl["month"] = expl["month"].astype(str)

ms_h = human.groupby("month")["p_ai"].agg(["size", "mean"])
ms_h = ms_h[ms_h["size"] >= 100].sort_index()
ms_e = expl.groupby("month")["p_ai"].agg(["size", "mean"])
ms_e = ms_e[ms_e["size"] >= 100].sort_index()

clean_months = [m for m in ms_h.index if m <= "202206"]
rho_clean, p_clean = spearmanr(range(len(clean_months)),
                               ms_h.loc[clean_months, "mean"].values)
rho_e, p_e = spearmanr(range(len(ms_e)), ms_e["mean"].values)

with open(BASE + "paper_drift_results.json", encoding="utf-8") as f:
    drift = json.load(f)
adj = drift["domain_adjusted_monthly"]

# ---- 图1：修正版时间线 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8),
                         gridspec_kw={"width_ratios": [1, 1.45]})
ax = axes[0]
x_h = np.arange(len(ms_h))
split = ms_h.index.get_loc("202207")
ax.plot(x_h[:split], ms_h["mean"].values[:split], "o-", color="#4C72B0",
        ms=3.5, lw=1.3, label="回溯爬取窗口")
ax.plot(x_h[split:], ms_h["mean"].values[split:], "o-", color="#DD8452",
        ms=3.5, lw=1.3, label="实时爬取窗口")
ax.axvspan(split - 0.5, len(ms_h) - 0.5, color="#DD8452", alpha=0.08)
base_clean = ms_h.loc[clean_months, "mean"].mean()
ax.axhline(base_clean, color="gray", ls="--", lw=0.9)
ax.annotate("采样断层\n(爬取批次切换)", xy=(split, 0.26), fontsize=9,
            ha="center", color="#7f3b08")
ax.set_xticks(x_h[::4])
ax.set_xticklabels(ms_h.index[::4], rotation=45, fontsize=8)
ax.set_ylabel("平均 P(AI)")
ax.set_title(f"人类集（前LLM时代）按月得分\n干净窗口趋势 ρ={rho_clean:.2f} (p={p_clean:.2f})")
ax.set_ylim(0, 0.65)
ax.legend(fontsize=8, loc="upper left")

ax = axes[1]
x_e = np.arange(len(ms_e))
ax.plot(x_e, ms_e["mean"].values, "o-", color="#C44E52", ms=3.5, lw=1.3,
        label="原始月度均值")
month_pos = {m: i for i, m in enumerate(ms_e.index)}
adj_pairs = [(month_pos[m], adj[m]) for m in adj if m in month_pos]
ax.plot([i for i, _ in adj_pairs], [v for _, v in adj_pairs],
        "s--", color="#55A868", ms=3.5, lw=1.2, label="领域调整后")
ax.axhline(base_clean, color="gray", ls="--", lw=0.9, label=f"前LLM干净基线 {base_clean:.2f}")
ax.set_xticks(x_e[::3])
ax.set_xticklabels(ms_e.index[::3], rotation=45, fontsize=8)
ax.set_title(f"探索集（后LLM时代，热帖）按月得分\n趋势 ρ={rho_e:.2f} (p={p_e:.2f})")
ax.set_ylim(0, 0.65)
ax.legend(fontsize=8, loc="lower right")
fig.suptitle("AI 风格得分的时间演化（稳健特征检测器）", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIGS + "drift_score_timeline.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- 图2：差中差 ----
clean = human[human["month"] <= "202206"]
late = human[human["month"] >= "202207"]
e = expl[expl["month"] >= "202301"]
fresh_post = e[e["likes"] < 100]["p_ai"]
hot_pre = clean[clean["likes"] > 1000]["p_ai"]
hot_post = e[e["likes"] >= 1000]["p_ai"]
# 均值/高置信占比各算一次，图与 JSON 复用（避免重复全列扫描）
stats = {
    name: (float(s.mean()), float((s > 0.9).mean()))
    for name, s in (("clean", clean["p_ai"]), ("late", late["p_ai"]),
                    ("fresh_post", fresh_post), ("hot_pre", hot_pre),
                    ("hot_post", hot_post))
}
groups = {
    "新帖/低互动层": (late["p_ai"], fresh_post),
    "热帖层": (hot_pre, hot_post),
}
# 柱状图直接复用上面已算好的 stats（mean / >0.9 占比各只算一次）
pre_stats = [stats["late"], stats["hot_pre"]]
post_stats = [stats["fresh_post"], stats["hot_post"]]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for i, metric in enumerate(["mean", "hi"]):
    ax = axes[i]
    labels = list(groups)
    scale = 100.0 if metric == "hi" else 1.0
    pre = [s[i] * scale for s in pre_stats]
    post = [s[i] * scale for s in post_stats]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, pre, 0.36, label="前LLM (≤2022.11)", color="#4C72B0")
    ax.bar(x + 0.18, post, 0.36, label="后LLM (2023.01–2025.06)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for xi, (a, b) in enumerate(zip(pre, post)):
        fmt = "{:.3f}" if metric == "mean" else "{:.1f}%"
        ax.text(xi - 0.18, a, fmt.format(a), ha="center", va="bottom", fontsize=9)
        ax.text(xi + 0.18, b, fmt.format(b), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("平均 P(AI)" if metric == "mean" else "P(AI)>0.9 占比 (%)")
    ax.set_title("平均 AI 风格得分" if metric == "mean" else "高置信 AI 风格帖占比")
    ax.legend(fontsize=9)
fig.suptitle("同类采样层的前后 LLM 时代对比（差中差）", fontsize=13)
fig.tight_layout()
fig.savefig(FIGS + "did_comparison.png", dpi=150)
plt.close(fig)

# ---- 图3：词表 log-odds ----
with open(BASE + "paper_lexicon_results.json", encoding="utf-8") as f:
    lex = json.load(f)
STOP = {"能", "们", "但", "小", "得", "啊", "搭", "穿", "夏日", "搭配", "像", "吃",
        "去", "做", "买", "孩子", "他", "是", "的", "有", "不", "很", "也", "等",
        "什么", "这个", "一个", "过", "会", "以", "并"}
aw = [d for d in lex["aigc_words"] if d["word"] not in STOP][:15]
hw = [d for d in lex["human_words"] if d["word"] not in STOP][:15]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
ax = axes[0]
ax.barh([d["word"] for d in aw][::-1], [d["z"] for d in aw][::-1], color="#C44E52")
ax.set_title("AIGC 显著偏好词（log-odds z 分数）")
ax.set_xlabel("z 分数")
ax = axes[1]
ax.barh([d["word"] for d in hw][::-1], [-d["z"] for d in hw][::-1], color="#4C72B0")
ax.set_title("人类显著偏好词（log-odds |z| 分数）")
ax.set_xlabel("|z| 分数")
fig.suptitle("区分 AIGC 与人类的风格词（informative Dirichlet log-odds）", fontsize=12)
fig.tight_layout()
fig.savefig(FIGS + "lexicon_logodds.png", dpi=150)
plt.close(fig)

# ---- 合并修正数字 ----
drift["trend"]["human_control_clean_window"] = {
    "rho": round(float(rho_clean), 3), "p": float(p_clean),
    "months": f"202012-202206 ({len(clean_months)}个月)"}
drift["sampling_discontinuity"] = {
    "note": "人类集2022.07起点赞>0比例从~70%骤降至~20%，且各领域得分同步上升，判定为爬取批次切换（回溯爬取->实时爬取）",
    "clean_window_meanP": round(stats["clean"][0], 4),
    "clean_window_hi": round(stats["clean"][1], 4),
    "late_window_meanP": round(stats["late"][0], 4),
    "late_window_hi": round(stats["late"][1], 4),
}
drift["did"] = {
    "fresh_stratum": {"pre": round(stats["late"][0], 3),
                      "post": round(stats["fresh_post"][0], 3),
                      "pre_hi": round(stats["late"][1], 4),
                      "post_hi": round(stats["fresh_post"][1], 4)},
    "hot_stratum": {"pre": round(stats["hot_pre"][0], 3),
                    "post": round(stats["hot_post"][0], 3),
                    "pre_hi": round(stats["hot_pre"][1], 4),
                    "post_hi": round(stats["hot_post"][1], 4)},
}
with open(BASE + "paper_drift_results.json", "w", encoding="utf-8") as f:
    json.dump(drift, f, ensure_ascii=False, indent=1)
print("完成：drift_score_timeline.png(修正), did_comparison.png, lexicon_logodds.png")
print(f"干净窗口阴性对照: rho={rho_clean:.3f} p={p_clean:.3f}")
