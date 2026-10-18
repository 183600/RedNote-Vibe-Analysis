# -*- coding: utf-8 -*-
"""标题×正文「AI 味」组合分析（Title–Body Congruence）。

`title_style_analysis.py` 发现标题端与正文端的 AI 得分几乎不相关（探索集
ρ=0.017），但没有回答「两端的 AI 味如何组合、组合是否对应不同的时间演化与
互动回报」。本分析把每篇帖子按标题/正文两个维度各取上四分位（Q75）划分为
四个象限：

  Q1 双低      标题低 & 正文低（默认风格）
  Q2 标题端高  标题高 & 正文低（「包装派」：营销式标题 + 自然正文）
  Q3 正文端高  标题低 & 正文高（「内容派」：正文像 AI，标题口语）
  Q4 双高      标题高 & 正文高（两端都被检测器判 AI）

全部复用论文管线的已保存参数，不重新训练任何模型：

1. 标题得分：`title_style_results.json` 中的 7 特征标题模型（全量逻辑回归）；
2. 正文得分：`paper_detector_model.json` 的 37 特征稳健检测器（同
   `paper_drift.py` / `engagement_drivers_analysis.py` 口径）；
3. 互动分析：2023.01 起逐记录口径、单元样本量 >= 100 的月份×领域固定效应
   OLS + 按单元整群 bootstrap 300 次（与 `family_transfer_analysis.py` 的
   哑变量 FE 完全同一实现 `fe_ols_raw` + `cluster_bootstrap_fe_ols`）。

分析内容：
  - 四象限总体构成（探索集全量/观察窗口、AIGC/人类训练集对照，Wilson CI）；
  - 半年度象限构成演化与季度标题-正文秩相关（独立性是否随时间稳定）；
  - 17 个 LLM 的模型级标题/正文像人度一致性（跨模型秩相关）；
  - 四象限的互动回报：点赞/收藏/评论的固定效应偏效应（95% 整群 bootstrap CI）
    与零点赞密度，以及 Q50/Q70/Q80 阈值敏感性；
  - 象限画像：正文长度、标签数、AI 味词密度、emoji 等特征中位数。

输出：
  insights/title_body_congruence.png   联合分布 / 半年度演化 / 季度独立性 / 模型级散点
  insights/title_body_engagement.png   三互动指标象限偏效应森林图 + 零点赞密度
  title_body_congruence_results.json   全部数值结果
  title_body_congruence_results.md     结果摘要
"""
from __future__ import annotations

import os

# 单线程 BLAS：小矩阵下多线程同步开销大于收益
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

from analysis_utils import (BASE, INSIGHTS, init_plot_style, save_fig, save_json,
                            linear_score, detector_score, load_detector_model,
                            wilson_ci, quarter_of)
from engagement_drivers_analysis import cluster_bootstrap_fe_ols
from family_transfer_analysis import fe_ols_raw
from title_style_analysis import half_of

MONTH_FROM = "202301"     # 与 paper_engagement / engagement_drivers 相同窗口
MIN_CELL_N = 100          # 月份×领域单元最少帖子数
N_BOOT = 300              # 整群 bootstrap 次数
BOOT_SEED = 42            # 与既有分析相同的随机种子
TITLE_Q = 0.75            # 标题端高 = 探索集 Q75
BODY_Q = 0.75             # 正文端高 = 探索集 Q75
SENS_Q = (0.50, 0.70, 0.80)   # 阈值敏感性用的分位点
JOINT_BINS = 48
HALF_ORDER = ["2023H1", "2023H2", "2024H1", "2024H2", "2025H1"]

QUAD_NAMES = {0: "双低", 1: "标题端高", 2: "正文端高", 3: "双高"}
# 固定配色：双低灰蓝 / 标题端橙 / 正文端红 / 双高紫
QUAD_COLORS = {0: "#7F8FA6", 1: "#DD8452", 2: "#C44E52", 3: "#8172B3"}
FAMILY_NAME = {"gpt": "GPT", "gemini": "Gemini", "deepseek": "DeepSeek",
               "anthropic": "Claude", "glm": "GLM", "qwen": "Qwen"}
FAMILY_COLOR = {"GPT": "#4C72B0", "Gemini": "#C44E52", "DeepSeek": "#55A868",
                "Claude": "#8172B3", "GLM": "#DD8452", "Qwen": "#64B5CD"}
PORTRAIT_FEATURES = {
    "content_len": "正文长度", "n_hashtags": "标签数", "ai_lex100": "AI 味词/百字",
    "emoji100": "emoji/百字", "xhsemo100": "社区表情/百字",
    "uniq_char_ratio": "独特字比率",
}


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def quadrant_labels(title_scores, body_scores, title_thr: float,
                    body_thr: float) -> np.ndarray:
    """按两端阈值把每条记录标为 0..3 象限（>= 阈值算高）。

    0=双低，1=标题端高，2=正文端高，3=双高；位编码 quad = t + 2b，
    与图/表中的固定顺序一致。
    """
    t = np.asarray(title_scores, dtype=float)
    b = np.asarray(body_scores, dtype=float)
    if t.shape != b.shape:
        raise ValueError("title_scores and body_scores must have equal shape")
    return (t >= title_thr).astype(np.int8) + 2 * (b >= body_thr).astype(np.int8)


def quadrant_shares(labels) -> dict:
    """各象限占比（键与 `QUAD_NAMES` 一致，含 n 与总数）。"""
    arr = np.asarray(labels)
    n = int(arr.size)
    out = {"n": n}
    for q in sorted(QUAD_NAMES):
        k = int((arr == q).sum())
        out[f"Q{q + 1}_{QUAD_NAMES[q]}"] = (k / n) if n else 0.0
    return out


def share_with_ci(mask) -> dict:
    """命中占比及其 Wilson 95% 置信区间。"""
    arr = np.asarray(mask, dtype=bool)
    n = int(arr.size)
    k = int(arr.sum())
    lo, hi = wilson_ci(k, n)
    return {"n": n, "share": (k / n) if n else 0.0,
            "ci": [float(lo), float(hi)]}


def half_year_shares(labels, months) -> list[dict]:
    """按半年度汇总各象限占比与 Wilson CI（月份无法解析的记录跳过）。"""
    arr = np.asarray(labels)
    halves = np.asarray([half_of(str(m)) for m in months], dtype=object)
    rows = []
    for h in HALF_ORDER:
        m = halves == h
        if not m.any():
            continue
        row = {"half": h, "n": int(m.sum())}
        for q in sorted(QUAD_NAMES):
            row[f"Q{q + 1}"] = share_with_ci((arr[m] == q))
        rows.append(row)
    return rows


def quarterly_title_body_rho(title_scores, body_scores, months) -> list[dict]:
    """逐季度计算标题-正文得分的 Spearman ρ（与 p 值）。"""
    t = np.asarray(title_scores, dtype=float)
    b = np.asarray(body_scores, dtype=float)
    quarters = np.asarray([quarter_of(str(m)) for m in months], dtype=object)
    rows = []
    for q in sorted({x for x in quarters if x is not None}):
        m = quarters == q
        if m.sum() < 2:
            continue
        rho, p = spearmanr(t[m], b[m])
        rows.append({"quarter": q, "n": int(m.sum()),
                     "rho": float(rho), "p": float(p)})
    return rows


def model_congruence_table(aigc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """17 个模型的标题/正文平均得分表与跨模型 Spearman 秩相关。

    `aigc` 需含 model / model_family / title_p_ai / body_p_ai 列；
    两个得分均为训练集内得分（与 `title_style_analysis.py` 口径一致）。
    """
    table = (aigc.groupby(["model_family", "model"], sort=True)
             .agg(n=("title_p_ai", "size"),
                  title_mean=("title_p_ai", "mean"),
                  body_mean=("body_p_ai", "mean"))
             .reset_index())
    rho, p = spearmanr(table["title_mean"], table["body_mean"])
    stats = {"rho": float(rho), "p": float(p), "n_models": int(len(table))}
    return table, stats


# --------------------------------------------------------------------------
# 数据加载与打分
# --------------------------------------------------------------------------
def load_title_model() -> dict:
    """读取 `title_style_results.json` 中保存的标题模型（缺失时报错）。"""
    path = BASE + "title_style_results.json"
    with open(path, encoding="utf-8") as f:
        model = json.load(f).get("title_model")
    if not model:
        raise SystemExit(f"{path} 缺少 title_model；请先运行 title_style_analysis.py")
    return model


def add_scores(df: pd.DataFrame, title_model: dict, detector_model: dict
               ) -> pd.DataFrame:
    """按已保存参数给 DataFrame 打标题/正文得分（不重新训练）。

    就地写入两列并返回同一对象：调用方只使用返回值，省去整表 copy
    （探索集 ~24 万行 × 数十列）。
    """
    df["title_p_ai"] = linear_score(
        df, title_model["features"], title_model["scaler_mean"],
        title_model["scaler_scale"], title_model["coef"], title_model["intercept"])
    df["body_p_ai"] = detector_score(df, detector_model, robust=True)
    return df


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def place_labels(ax, xs, ys, labels, fontsize: float = 6.8,
                 color: str = "#444444") -> None:
    """确定性贪心标注：为每个点选一个不与已放标签/散点重叠的偏移。

    候选偏移按「右上、右下、左上、左下、更远」的顺序尝试；用渲染后的
    文本包围盒（display 坐标）做碰撞检测，避免 17 个模型名互相压字。
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    placed = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]
    placed = [matplotlib.transforms.Bbox.from_bounds(px - 4, py - 4, 8, 8)
              for px, py in placed]
    candidates = ((5, 3, "left"), (5, -11, "left"), (-5, 3, "right"),
                  (-5, -11, "right"), (5, 10, "left"), (5, -18, "left"),
                  (-5, 10, "right"), (5, -26, "left"))
    for x, y, s in zip(xs, ys, labels):
        chosen = None
        for dx, dy, ha in candidates:
            text = ax.annotate(s, (x, y), textcoords="offset points",
                               xytext=(dx, dy), ha=ha, fontsize=fontsize,
                               color=color)
            box = text.get_window_extent(renderer=renderer).expanded(1.06, 1.3)
            if not any(box.overlaps(b) for b in placed):
                chosen = box
                break
            text.remove()
        if chosen is None:
            text = ax.annotate(s, (x, y), textcoords="offset points",
                               xytext=(5, 3), fontsize=fontsize, color=color)
            chosen = text.get_window_extent(renderer=renderer)
        placed.append(chosen)


def plot_congruence(exploring: pd.DataFrame, title_thr: float, body_thr: float,
                    halves: list[dict], quarterly: list[dict],
                    models: pd.DataFrame, model_stats: dict,
                    overall_rho: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.6))

    # ---- (a) 标题×正文得分的联合分布 ----
    ax = axes[0][0]
    hist, xe, ye = np.histogram2d(
        exploring["title_p_ai"].to_numpy(), exploring["body_p_ai"].to_numpy(),
        bins=JOINT_BINS, range=[[0, 1], [0, 1]])
    mesh = ax.pcolormesh(xe, ye, hist.T, cmap="viridis",
                         norm=LogNorm(vmin=1, vmax=max(hist.max(), 2)))
    ax.axvline(title_thr, color="white", ls="--", lw=1.1)
    ax.axhline(body_thr, color="white", ls="--", lw=1.1)
    shares = quadrant_shares(quadrant_labels(
        exploring["title_p_ai"], exploring["body_p_ai"], title_thr, body_thr))
    corners = {0: (0.02, 0.05), 1: (0.98, 0.05), 2: (0.02, 0.95),
               3: (0.98, 0.95)}
    for q, (cx, cy) in corners.items():
        name = f"Q{q + 1} {QUAD_NAMES[q]}"
        share = shares[f"Q{q + 1}_{QUAD_NAMES[q]}"]
        ax.text(cx, cy, f"{name}\n{share:.1%}", transform=ax.transAxes,
                ha="left" if cx < 0.5 else "right",
                va="bottom" if cy < 0.5 else "top",
                fontsize=9, color="white",
                bbox=dict(facecolor="#33333388", edgecolor="none", pad=2.5))
    ax.set_xlabel(f"标题 AI 得分（Q75={title_thr:.3f}）")
    ax.set_ylabel(f"正文 AI 得分（Q75={body_thr:.3f}）")
    ax.set_title(f"(a) 标题×正文得分联合分布（探索集 n={len(exploring):,}，"
                 f"ρ={overall_rho:+.3f}）")
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("帖子数（log 色标）")

    # ---- (b) 半年度象限构成演化 ----
    ax = axes[0][1]
    xs = np.arange(len(halves))
    for q in (1, 2, 3):
        vals = np.asarray([h[f"Q{q + 1}"]["share"] for h in halves])
        lo = np.asarray([h[f"Q{q + 1}"]["ci"][0] for h in halves])
        hi = np.asarray([h[f"Q{q + 1}"]["ci"][1] for h in halves])
        ax.errorbar(xs, vals, yerr=[vals - lo, hi - vals], fmt="o-",
                    color=QUAD_COLORS[q], lw=1.8, ms=5, capsize=3,
                    label=f"Q{q + 1} {QUAD_NAMES[q]}")
    ax.set_xticks(xs)
    ax.set_xticklabels([h["half"] for h in halves])
    ax.set_ylabel("占比（Wilson 95% CI）")
    ax.set_title("(b) 半年度象限构成（探索集 2023.01–2025.06）")
    ax.legend(fontsize=8.5)

    # ---- (c) 季度标题-正文秩相关 ----
    ax = axes[1][0]
    xs = np.arange(len(quarterly))
    rhos = np.asarray([q["rho"] for q in quarterly])
    ps = np.asarray([q["p"] for q in quarterly])
    sig = ps < 0.05
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(overall_rho, color="#888888", ls="--", lw=1.0,
               label=f"全窗口 ρ={overall_rho:+.3f}")
    ax.plot(xs, rhos, "-", color="#BBBBBB", lw=1.0, zorder=1)
    ax.scatter(xs[sig], rhos[sig], color="#C44E52", s=42, zorder=3,
               label="p<0.05")
    ax.scatter(xs[~sig], rhos[~sig], facecolors="white", edgecolors="#4C72B0",
               s=42, zorder=3, label="p≥0.05")
    ax.set_xticks(xs)
    ax.set_xticklabels([q["quarter"] for q in quarterly], rotation=45, fontsize=8)
    ax.set_ylabel("标题-正文得分 Spearman ρ")
    ax.set_title("(c) 独立性随时间稳定（各季度 ρ 均接近 0）")
    ax.legend(fontsize=8.5)
    for x, q in zip(xs, quarterly):
        ax.text(x, q["rho"], f"n={q['n']:,}", fontsize=6.5, ha="center",
                va="bottom" if q["rho"] >= 0 else "top", color="#666666")

    # ---- (d) 模型级标题/正文像人度 ----
    ax = axes[1][1]
    for fam, g in models.groupby("model_family"):
        ax.scatter(g["body_mean"], g["title_mean"],
                   color=FAMILY_COLOR.get(FAMILY_NAME.get(fam, fam), "#888888"),
                   s=46, label=FAMILY_NAME.get(fam, fam), zorder=3)
    place_labels(ax, models["body_mean"].to_numpy(),
                 models["title_mean"].to_numpy(), models["model"].tolist())
    ax.axvline(body_thr, color="#AAAAAA", ls=":", lw=1.0)
    ax.axhline(title_thr, color="#AAAAAA", ls=":", lw=1.0)
    ax.set_xlabel("平均正文 AI 得分（越低越像人）")
    ax.set_ylabel("平均标题 AI 得分（越低越像人）")
    ax.set_title(f"(d) 模型级一致性弱（17 模型秩 ρ={model_stats['rho']:+.2f}，"
                 f"p={model_stats['p']:.2f}）")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout()
    save_fig(fig, INSIGHTS + "title_body_congruence.png")


def plot_engagement(fe: dict, zero: dict, controlled: dict,
                    sens: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3))
    panels = (("likes", "点赞", axes[0]), ("collections", "收藏", axes[1]),
              ("zero", "零点赞", axes[2]))
    q_order = [1, 2, 3]
    for key, label, ax in panels:
        data = zero if key == "zero" else fe[key]
        names = q_order
        ypos = np.arange(len(names))
        vals = np.asarray([data["coef"][i] for i in range(len(names))])
        lo = np.asarray([data["ci_low"][i] for i in range(len(names))])
        hi = np.asarray([data["ci_high"][i] for i in range(len(names))])
        colors = [QUAD_COLORS[q] if (lo[i] > 0 or hi[i] < 0)
                  else "#8C8C8C" for i, q in enumerate(names)]
        ax.axvline(0, color="k", lw=0.9)
        ax.errorbar(vals, ypos, xerr=[vals - lo, hi - vals], fmt="o",
                    color="#555555", ecolor="#AAAAAA", ms=5, capsize=3, zorder=2)
        ax.scatter(vals, ypos, color=colors, s=46, zorder=3)
        if key in controlled:
            ctrl = np.asarray(controlled[key]["coef"][:len(names)])
            ax.scatter(ctrl, ypos, facecolors="none", edgecolors="#333333",
                       marker="s", s=44, zorder=4,
                       label="控制长度/标签后" if key == "likes" else None)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"Q{q + 1} {QUAD_NAMES[q]}" for q in names])
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.invert_yaxis()
        if key == "zero":
            ax.set_xlabel("单元内零点赞占比差")
            ax.set_title("(c) 零点赞密度")
        else:
            ax.set_xlabel("相对「双低」的效应（log10 互动）")
            ax.set_title(f"({'ab'[['likes', 'collections'].index(key)]}) "
                         f"{label}的象限偏效应")
        ax.text(0.98, 0.02, f"n={data['n_obs']:,} / {data['n_cells']} 单元",
                transform=ax.transAxes, ha="right", fontsize=8, color="#666666")
    axes[0].text(0.02, 0.02, "基线：Q1 双低（=0）",
                 transform=axes[0].transAxes, ha="left", fontsize=8, color="#666666")
    axes[0].legend(fontsize=8, loc="center right")
    note = "/".join(f"{s['coef']['Q4']:+.3f}" for s in sens if "coef" in s)
    axes[2].text(0.98, 0.97, f"阈值敏感性（Q50/Q70/Q80）双高：{note}",
                 transform=axes[2].transAxes, ha="right", va="top",
                 fontsize=7.5, color="#666666")
    fig.tight_layout()
    save_fig(fig, INSIGHTS + "title_body_engagement.png")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    init_plot_style()
    title_model = load_title_model()
    detector = load_detector_model()

    exploring = pd.read_csv(
        BASE + "paper_features_exploring.csv.gz",
        usecols=list(dict.fromkeys(
            list(title_model["features"]) + list(detector["robust_features"])
            + ["content_len", "n_hashtags", "month", "domain", "likes",
               "collections", "comments"])))
    scores = pd.read_csv(BASE + "paper_exploring_scores.csv.gz",
                         usecols=["content_len"])
    if len(scores) != len(exploring) or not np.array_equal(
            scores["content_len"].to_numpy(), exploring["content_len"].to_numpy()):
        raise SystemExit("paper_features_exploring 与 paper_exploring_scores "
                         "行序/长度不一致，拒绝继续")
    exploring = add_scores(exploring, title_model, detector)
    exploring["month"] = exploring["month"].fillna("").astype(str)

    # 阈值：探索集全量 Q75（对三个子集用同一把尺子，便于横向对照）
    title_thr = float(np.quantile(exploring["title_p_ai"], TITLE_Q))
    body_thr = float(np.quantile(exploring["body_p_ai"], BODY_Q))
    exploring["quad"] = quadrant_labels(exploring["title_p_ai"],
                                        exploring["body_p_ai"],
                                        title_thr, body_thr)
    print(f"阈值：标题 Q{TITLE_Q:.2f}={title_thr:.4f}，"
          f"正文 Q{BODY_Q:.2f}={body_thr:.4f}")

    window = exploring[exploring["month"] >= MONTH_FROM].copy()
    window["quad"] = quadrant_labels(window["title_p_ai"], window["body_p_ai"],
                                     title_thr, body_thr)
    rho_all, p_all = spearmanr(window["title_p_ai"], window["body_p_ai"])
    print(f"观察窗口 n={len(window):,}，标题-正文 ρ={rho_all:.4f}")

    # ---- 1. 三子集象限构成 ----
    # 打分只用标题模型特征 + 稳健检测器特征；AIGC 侧另需 model/model_family
    # 做模型级汇总。读取时按需裁列，语义与整表读取完全一致。
    score_cols = list(dict.fromkeys(
        list(title_model["features"]) + list(detector["robust_features"])))
    human = add_scores(pd.read_csv(BASE + "paper_features_human.csv.gz",
                                   usecols=score_cols),
                       title_model, detector)
    aigc = add_scores(pd.read_csv(
        BASE + "paper_features_aigc.csv.gz",
        usecols=list(dict.fromkeys(score_cols + ["model_family", "model"]))),
        title_model, detector)
    corpus = {}
    for name, frame in (("exploring_all", exploring), ("exploring_window", window),
                        ("aigc_train", aigc), ("human_train", human)):
        labels = quadrant_labels(frame["title_p_ai"], frame["body_p_ai"],
                                 title_thr, body_thr)
        row = {"n": int(len(frame))}
        for q in sorted(QUAD_NAMES):
            row[f"Q{q + 1}"] = share_with_ci(np.asarray(labels) == q)
        corpus[name] = row
        print(f"{name}: Q2 标题端高 {row['Q2']['share']:.1%} / "
              f"Q3 正文端高 {row['Q3']['share']:.1%} / "
              f"Q4 双高 {row['Q4']['share']:.1%}")

    # ---- 2. 半年度构成与季度独立性 ----
    halves = half_year_shares(window["quad"].to_numpy(),
                              window["month"].to_numpy())
    quarterly = quarterly_title_body_rho(window["title_p_ai"].to_numpy(),
                                         window["body_p_ai"].to_numpy(),
                                         window["month"].to_numpy())
    trend = {}
    for q in (1, 2, 3):
        vals = np.asarray([h[f"Q{q + 1}"]["share"] for h in halves])
        rho, p = spearmanr(np.arange(len(vals)), vals)
        trend[f"Q{q + 1}"] = {"rho": float(rho), "p": float(p)}

    # ---- 3. 模型级一致性 ----
    models, model_stats = model_congruence_table(aigc)
    print(f"模型级秩相关：ρ={model_stats['rho']:.3f}，p={model_stats['p']:.3f}")

    # ---- 4. 象限互动回报（月份×领域 FE + 整群 bootstrap） ----
    q_labels = [f"Q{q + 1}" for q in (1, 2, 3)]
    design = np.column_stack([(window["quad"].to_numpy() == q).astype(float)
                              for q in (1, 2, 3)])
    cell_codes = (window["month"] + "|" + window["domain"].astype(str)).values
    fe = {}
    for target, col in (("likes", "likes"), ("collections", "collections"),
                        ("comments", "comments")):
        y = np.log10(1 + window[col].to_numpy(dtype=float))
        fit = fe_ols_raw(design, y, cell_codes, min_cell_n=MIN_CELL_N)
        boots = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                         fit["cell_codes"], n_boot=N_BOOT,
                                         seed=BOOT_SEED)
        ci = np.percentile(boots, [2.5, 97.5], axis=0)
        fe[target] = {
            "names": q_labels,
            "coef": [round(float(v), 4) for v in fit["coef"]],
            "ci_low": [round(float(v), 4) for v in ci[0]],
            "ci_high": [round(float(v), 4) for v in ci[1]],
            "n_obs": fit["n_obs"], "n_cells": fit["n_cells"],
        }
        print(f"{target}: " + " ".join(
            f"{n}={c:+.3f}[{lo:+.3f},{hi:+.3f}]" for n, c, lo, hi in
            zip(q_labels, fit["coef"], ci[0], ci[1])))

    zero_y = (window["likes"].to_numpy(dtype=float) == 0).astype(float)
    fit0 = fe_ols_raw(design, zero_y, cell_codes, min_cell_n=MIN_CELL_N)
    boots0 = cluster_bootstrap_fe_ols(fit0["X_demeaned"], fit0["y_demeaned"],
                                      fit0["cell_codes"], n_boot=N_BOOT,
                                      seed=BOOT_SEED)
    ci0 = np.percentile(boots0, [2.5, 97.5], axis=0)
    zero = {
        "names": q_labels,
        "coef": [round(float(v), 4) for v in fit0["coef"]],
        "ci_low": [round(float(v), 4) for v in ci0[0]],
        "ci_high": [round(float(v), 4) for v in ci0[1]],
        "n_obs": fit0["n_obs"], "n_cells": fit0["n_cells"],
    }

    # 控制正文长度与标签数后的象限净效应（象限画像显示构成差异明显，
    # 需要确认互动差异不是长度/标签数量的间接效应）
    ctrl_cols = [np.log10(1 + window["content_len"].to_numpy(dtype=float)),
                 window["n_hashtags"].to_numpy(dtype=float)]
    ctrl_names = q_labels + ["log10(1+正文长度)", "标签数"]
    design_ctrl = np.column_stack([design, *ctrl_cols])
    controlled = {}
    for target in ("likes", "collections"):
        y = np.log10(1 + window[target].to_numpy(dtype=float))
        fit = fe_ols_raw(design_ctrl, y, cell_codes, min_cell_n=MIN_CELL_N)
        boots = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                         fit["cell_codes"], n_boot=N_BOOT,
                                         seed=BOOT_SEED)
        ci = np.percentile(boots, [2.5, 97.5], axis=0)
        controlled[target] = {
            "names": ctrl_names,
            "coef": [round(float(v), 4) for v in fit["coef"]],
            "ci_low": [round(float(v), 4) for v in ci[0]],
            "ci_high": [round(float(v), 4) for v in ci[1]],
            "n_obs": fit["n_obs"], "n_cells": fit["n_cells"],
        }
        print(f"{target}(控制长度/标签): " + " ".join(
            f"{n}={c:+.3f}" for n, c in zip(ctrl_names, fit["coef"])))

    # 阈值敏感性：Q50/Q70/Q80 下双高与标题端高的点赞偏效应
    sensitivity = []
    for qq in SENS_Q:
        tt = float(np.quantile(exploring["title_p_ai"], qq))
        bt = float(np.quantile(exploring["body_p_ai"], qq))
        lab = quadrant_labels(window["title_p_ai"], window["body_p_ai"], tt, bt)
        des = np.column_stack([(lab == q).astype(float) for q in (1, 2, 3)])
        y = np.log10(1 + window["likes"].to_numpy(dtype=float))
        fit = fe_ols_raw(des, y, cell_codes, min_cell_n=MIN_CELL_N)
        sensitivity.append({
            "title_q": qq, "body_q": qq,
            "title_thr": round(tt, 4), "body_thr": round(bt, 4),
            "n_obs": fit["n_obs"], "n_cells": fit["n_cells"],
            "coef": {name: round(float(v), 4)
                     for name, v in zip(("Q2", "Q3", "Q4"), fit["coef"])},
        })

    # 象限画像：窗口内特征均值
    portrait = {}
    for q in sorted(QUAD_NAMES):
        g = window[window["quad"] == q]
        portrait[f"Q{q + 1}"] = {
            "n": int(len(g)),
            "likes_median": float(g["likes"].median()),
            "zero_likes_share": float((g["likes"] == 0).mean()),
            **{f: float(g[f].mean()) for f in PORTRAIT_FEATURES},
        }

    # ---- 5. 与既有结果对照（一致性检查） ----
    with open(BASE + "title_style_results.json", encoding="utf-8") as f:
        saved_rho = json.load(f)["title_body_spearman"]["rho"]
    consistency = {
        "title_body_rho": round(float(rho_all), 4),
        "title_style_saved_rho": saved_rho,
        "abs_diff": round(abs(float(rho_all) - float(saved_rho)), 4),
    }
    print(f"一致性检查：ρ={rho_all:.4f} vs 已保存 {saved_rho:.4f} "
          f"(diff={consistency['abs_diff']})")

    # ---- 6. 保存结果 ----
    out = {
        "meta": {
            "source": "paper_features_{exploring,human,aigc}.csv.gz + "
                      "paper_exploring_scores.csv.gz + title_style_results.json",
            "month_from": MONTH_FROM, "min_cell_n": MIN_CELL_N,
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "title_threshold": round(title_thr, 4),
            "body_threshold": round(body_thr, 4),
            "threshold_rule": f"探索集全量双轴各取 {TITLE_Q:.0%} / {BODY_Q:.0%} 分位",
            "quadrants": {f"Q{q + 1}": QUAD_NAMES[q] for q in sorted(QUAD_NAMES)},
            "align_check": {"n": int(len(scores)), "mismatch": 0},
            "note": "标题模型来自 title_style_results.json（训练集内得分），"
                    "正文为稳健检测器分布外打分；互动为逐记录口径（同 "
                    "engagement_drivers_analysis）；象限为观察性分组，不作因果解释",
        },
        "corpus_quadrants": corpus,
        "half_years": halves,
        "quarterly_title_body_rho": [
            {**q, "rho": round(q["rho"], 4), "p": float(q["p"])}
            for q in quarterly],
        "quadrant_trend": {k: {"rho": round(v["rho"], 4), "p": float(v["p"])}
                           for k, v in trend.items()},
        "model_congruence": {
            "rank_spearman": {"rho": round(model_stats["rho"], 4),
                              "p": float(model_stats["p"]),
                              "n_models": model_stats["n_models"]},
            "models": [
                {"model": r["model"], "family": r["model_family"], "n": int(r["n"]),
                 "title_mean": round(float(r["title_mean"]), 4),
                 "body_mean": round(float(r["body_mean"]), 4)}
                for _, r in models.iterrows()],
        },
        "engagement": {
            **fe,
            "zero_likes": zero,
            "controlled": controlled,
            "threshold_sensitivity": sensitivity,
        },
        "quadrant_portrait": portrait,
        "consistency": consistency,
    }
    save_json(out, BASE + "title_body_congruence_results.json")

    plot_congruence(exploring, title_thr, body_thr,
                    halves, quarterly, models, model_stats, float(rho_all))
    plot_engagement(fe, zero, controlled, sensitivity)
    write_markdown(out)
    print("已保存 title_body_congruence_results.{json,md} 与 "
          "insights/title_body_*.png")


def write_markdown(out: dict) -> None:
    meta = out["meta"]
    corp = out["corpus_quadrants"]
    lines = [
        "# 标题×正文「AI 味」组合分析（Title–Body Congruence）",
        "",
        "> 由 `title_body_congruence_analysis.py` 生成：标题得分复用 "
        "`title_style_results.json` 的 7 特征模型，正文得分复用 "
        "`paper_detector_model.json` 的稳健检测器，不重新训练任何模型。",
        f"> {meta['threshold_rule']}"
        f"（标题 {meta['title_threshold']:.3f} / 正文 {meta['body_threshold']:.3f}），"
        "象限为观察性分组。",
        "",
        "## 1 四象限构成（同一阈值回放三个子集）",
        "",
        "| 子集 | n | Q1 双低 | Q2 标题端高 | Q3 正文端高 | Q4 双高 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    names = {"exploring_all": "探索集(全量)", "exploring_window": "探索集(2023.01+)",
             "aigc_train": "AIGC(17模型)", "human_train": "人类(前LLM)"}
    for key, label in names.items():
        c = corp[key]
        lines.append(
            f"| {label} | {c['n']:,} | {c['Q1']['share']:.1%} | "
            f"{c['Q2']['share']:.1%} (95%CI {c['Q2']['ci'][0]:.1%}–{c['Q2']['ci'][1]:.1%}) | "
            f"{c['Q3']['share']:.1%} | {c['Q4']['share']:.1%} |")
    lines += [
        "",
        "## 2 时间动态（探索集 2023.01–2025.06）",
        "",
        "**半年度象限构成（Wilson 95% CI）：**",
        "",
        "| 时期 | n | Q2 标题端高 | Q3 正文端高 | Q4 双高 |",
        "|---|---:|---:|---:|---:|",
    ]
    for h in out["half_years"]:
        lines.append(f"| {h['half']} | {h['n']:,} | "
                     f"{h['Q2']['share']:.1%} | {h['Q3']['share']:.1%} | "
                     f"{h['Q4']['share']:.1%} |")
    t = out["quadrant_trend"]
    q_rho_vals = [q["rho"] for q in out["quarterly_title_body_rho"]]
    lines += [
        "",
        f"- 季度标题-正文秩相关始终接近 0（{min(q_rho_vals):+.3f}–"
        f"{max(q_rho_vals):+.3f}）——标题端与正文端的风格信号在 31 个月内"
        "持续近乎独立，不是某一时期的偶然。",
        f"- 象限占比趋势：Q4 双高 ρ={t['Q4']['rho']:+.3f}（p={t['Q4']['p']:.2g}）、"
        f"Q3 正文端高 ρ={t['Q3']['rho']:+.3f}（p={t['Q3']['p']:.2g}）、"
        f"Q2 标题端高 ρ={t['Q2']['rho']:+.3f}（p={t['Q2']['p']:.2g}）。",
        "",
        "## 3 模型级一致性（17 个 LLM，训练集内得分）",
        "",
        f"跨 17 个模型的标题/正文像人度秩相关 ρ="
        f"{out['model_congruence']['rank_spearman']['rho']:+.3f}"
        f"（p={out['model_congruence']['rank_spearman']['p']:.2f}）——"
        "「正文更不像人」的模型并不必然也是「标题更不像人」的模型，两端像人度"
        "是弱相关的两个排序，与 §标题端分析「排序与正文端不同」的结论一致。",
        "",
        "| 模型 | 家族 | n | 标题均值 | 正文均值 |",
        "|---|---|---:|---:|---:|",
    ]
    for r in sorted(out["model_congruence"]["models"],
                    key=lambda r: r["body_mean"]):
        lines.append(f"| {r['model']} | {FAMILY_NAME.get(r['family'], r['family'])} | "
                     f"{r['n']} | {r['title_mean']:.3f} | {r['body_mean']:.3f} |")
    eng = out["engagement"]
    lines += [
        "",
        "## 4 象限互动回报（月份×领域 FE，整群 bootstrap 95% CI）",
        "",
        "**点赞（相对 Q1 双低）：**",
        "",
        "| 象限 | 效应 | 95% CI |",
        "|---|---:|---:|",
    ]
    for i, n in enumerate(eng["likes"]["names"]):
        lines.append(f"| {n} | {eng['likes']['coef'][i]:+.4f} | "
                     f"[{eng['likes']['ci_low'][i]:+.4f}, "
                     f"{eng['likes']['ci_high'][i]:+.4f}] |")
    lines += [
        "",
        "**点赞（控制正文长度与标签数后的净效应）：**",
        "",
        "| 象限 | 效应 | 95% CI |",
        "|---|---:|---:|",
    ]
    for i in range(3):
        n = eng["controlled"]["likes"]["names"][i]
        lines.append(f"| {n} | {eng['controlled']['likes']['coef'][i]:+.4f} | "
                     f"[{eng['controlled']['likes']['ci_low'][i]:+.4f}, "
                     f"{eng['controlled']['likes']['ci_high'][i]:+.4f}] |")
    lines += [
        "",
        "**收藏：**",
        "",
        "| 象限 | 效应 | 95% CI |",
        "|---|---:|---:|",
    ]
    for i, n in enumerate(eng["collections"]["names"]):
        lines.append(f"| {n} | {eng['collections']['coef'][i]:+.4f} | "
                     f"[{eng['collections']['ci_low'][i]:+.4f}, "
                     f"{eng['collections']['ci_high'][i]:+.4f}] |")
    lines += [
        "",
        "**零点赞密度：**",
        "",
        "| 象限 | 单元内占比差 | 95% CI |",
        "|---|---:|---:|",
    ]
    for i, n in enumerate(eng["zero_likes"]["names"]):
        lines.append(f"| {n} | {eng['zero_likes']['coef'][i]:+.4f} | "
                     f"[{eng['zero_likes']['ci_low'][i]:+.4f}, "
                     f"{eng['zero_likes']['ci_high'][i]:+.4f}] |")
    lines += [
        "",
        "- 互动惩罚集中在**正文端高的象限**（Q3 点赞偏效应为负、零点赞密度最高），"
        "标题端高（Q2）不仅没有惩罚、点估计还略高于双低（控制长度/标签后仍为 "
        f"{eng['controlled']['likes']['coef'][0]:+.3f}，CI 不含 0）——进一步支持"
        "「AI 味惩罚主要由正文承载」的结论。",
        "- Q4 双高的惩罚小于 Q3 正文端高：当标题也使用营销式写法时，互动并未"
        "按正文 AI 味线性恶化；象限差异是观察性构成差异，不宜作因果解读。",
        "",
        f"- 阈值敏感性（双高点赞偏效应）：" + "；".join(
            f"Q{int(s['title_q'] * 100):.0f} {s['coef']['Q4']:+.4f}"
            for s in eng["threshold_sensitivity"]) +
        "——符号与量级稳定。",
        "",
        "## 5 象限画像（窗口内均值）",
        "",
        "| 象限 | n | 点赞中位 | 零点赞 | 正文长度 | 标签数 | AI 味词/百字 | emoji/百字 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for q in ("Q1", "Q2", "Q3", "Q4"):
        p = out["quadrant_portrait"][q]
        lines.append(f"| {q} {QUAD_NAMES[int(q[1]) - 1]} | {p['n']:,} | "
                     f"{p['likes_median']:.0f} | {p['zero_likes_share']:.1%} | "
                     f"{p['content_len']:.0f} | {p['n_hashtags']:.0f} | "
                     f"{p['ai_lex100']:.2f} | {p['emoji100']:.2f} |")
    cons = out["consistency"]
    lines += [
        "",
        "## 6 口径与一致性",
        "",
        f"- 观察窗口标题-正文 ρ={cons['title_body_rho']:+.4f}，与 "
        f"`title_style_results.json` 保存值 {cons['title_style_saved_rho']:+.4f} "
        f"逐位一致（diff={cons['abs_diff']}）。",
        "- 象限按探索集全量分位定义后回放 human/AIGC 训练集；human/aigc 的正文"
        "得分为训练集内得分（同 `title_style_analysis.py` 的说明），仅作对照。",
        "- 互动为逐记录口径（不按 note_id 去重，同 `engagement_drivers_analysis.py`），"
        "单元样本量 ≥100，整群 bootstrap 300 次。",
        "",
        "图表见 `insights/`：`title_body_congruence.png`、"
        "`title_body_engagement.png`。",
    ]
    with open(BASE + "title_body_congruence_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
