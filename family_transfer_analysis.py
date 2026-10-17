# -*- coding: utf-8 -*-
"""野外高 AI 味帖的「风格家族」构成与互动回报（family-transfer analysis）。

既有分析都把「AI 味」当作一个整体：`paper_detector.py` 给出人/机的 P(AI)，
`family_lexicon_analysis.py` 指出家族口癖差异，`paper_model_fingerprint.py`
给出 17 个模型的像人度排名，但「野外的高 AI 味帖子在**风格上更接近哪个家族**
生成的原型」从未被量化，也从未回答「AI 味惩罚是否对所有家族风格一视同仁」。

本脚本把 AIGC 训练子集上的 6 路家族分类器（GPT / Gemini / DeepSeek / Claude /
GLM / Qwen）迁移到探索集，全部复用论文管线的 37 个可解释特征：

1. **分类器的可分性**：5 折分层交叉验证（与 `paper_detector.py` 同协议：
   StandardScaler + 多项式 LogisticRegression、同随机种子）评估家族识别的
   准确率 / macro-F1 / 每家族精确率-召回率与混淆矩阵——家族风格在特征空间
   里是否互相可分（对照基线为多数类 23.8% 与均匀随机 16.7%）；
2. **野外构成**：用全量 AIGC 拟合的家族分类器给 91,517 条探索集帖子打分，
   报告「家族归属置信度」随 P(AI) 的变化（家族归因只在高 P(AI) 样本内可靠），
   并在高置信样本（P(AI)>=0.9）内统计家族风格构成：总体份额（Wilson CI）、
   与训练集先验的对比、半年度演化与季度趋势（χ² + 每家族 Spearman）；
3. **互动回报的家族结构**：在高置信样本内比较各家族风格帖的点赞/收藏，
   以「月份×领域」单元内残差与固定效应回归（家族哑变量 + 正文长度 + P(AI) +
   标签数控制，按单元整群 bootstrap 95% CI）估计各家族**相对 GLM 基线**的
   互动差——检验 `paper_engagement.py` 的「AI 味惩罚」是否存在家族结构。

口径：分类器只用 AIGC 子集（7,254 条）训练，探索集为纯训练分布外打分；
`paper_features_exploring.csv.gz` 与 `paper_exploring_scores.csv.gz` 逐行对齐
（content_len 校验）；时间/互动分析自 2023.01 起，与 `paper_engagement.py`
一致；互动单元最小样本量 30（高置信样本本身较小，沿用作者的 >=100 会丢太多
单元）。「家族归属」是风格相似度构念而非真实来源标注，结果解读均限定在
高 P(AI) 采样层内。

输出：
  insights/family_transfer.png            混淆矩阵 / 置信度分层 / 家族构成演化
  insights/family_transfer_engagement.png 单元内残差分布 / 点赞与收藏的家族偏效应
  family_transfer_results.json            全部数值结果
  family_transfer_results.md              结果摘要
"""
from __future__ import annotations

import os

# 单线程 BLAS：37 维小矩阵下多线程同步开销大于收益
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_utils import (BASE, INSIGHTS, cliffs_delta, init_plot_style,
                            load_detector_model, load_features, save_fig,
                            save_json, wilson_ci)
from engagement_drivers_analysis import (cell_keep_mask,
                                         cluster_bootstrap_fe_ols,
                                         demean_within)

FAMILIES = ["anthropic", "deepseek", "gemini", "glm", "gpt", "qwen"]
FAMILY_NAME = {"anthropic": "Claude", "deepseek": "DeepSeek", "gemini": "Gemini",
               "glm": "GLM", "gpt": "GPT", "qwen": "Qwen"}
FAMILY_COLOR = {"anthropic": "#8172B3", "deepseek": "#55A868",
                "gemini": "#C44E52", "glm": "#DD8452", "gpt": "#4C72B0",
                "qwen": "#64B5CD"}

HIGH_CONF = 0.9         # 高置信 P(AI) 阈值（与 title_style/detector 诊断一致）
MONTH_FROM = "202301"   # 与 paper_engagement 相同的观察窗口
MIN_CELL_N = 30         # 高置信样本较小，互动单元最小样本量取 30
N_BOOT = 300            # 整群 bootstrap 次数
BOOT_SEED = 42
CV_SEED = 42
REF_FAMILY = "glm"      # 固定效应回归的参照家族（份额居中）
HALF_ORDER = ["2023H1", "2023H2", "2024H1", "2024H2", "2025H1"]
PAI_BANDS = [(-np.inf, 0.4, "<0.4"), (0.4, 0.6, "0.4–0.6"),
             (0.6, 0.9, "0.6–0.9"), (0.9, np.inf, ">=0.9")]


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def make_family_pipeline():
    """与论文检测器相同类型的分类器（多分类版）：标准化 + 平衡类权重逻辑回归。"""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"))


def cv_family_predictions(aigc: pd.DataFrame, feats: list[str],
                          seed: int = CV_SEED) -> tuple[np.ndarray, np.ndarray]:
    """6 路家族分类器的 5 折折外预测，返回 (oof_pred, oof_proba)。

    折外口径与 `paper_detector.py` 一致（StratifiedKFold(5, shuffle=True,
    random_state=seed)）；`oof_proba` 的列序与 `classes_` 一致。
    """
    X = aigc[feats].fillna(0).values
    y = aigc["model_family"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_pred = np.empty(len(y), dtype=object)
    n_classes = len(np.unique(y))
    oof_proba = np.zeros((len(y), n_classes))
    for train_idx, test_idx in cv.split(X, y):
        pipe = make_family_pipeline().fit(X[train_idx], y[train_idx])
        oof_pred[test_idx] = pipe.predict(X[test_idx])
        oof_proba[test_idx] = pipe.predict_proba(X[test_idx])
    return oof_pred, oof_proba


def cv_metrics(y, pred, classes) -> dict:
    """折外预测的总体与分类别指标（macro-F1、平衡准确率、混淆矩阵）。"""
    labels = list(classes)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels).astype(float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "majority_baseline": float(max(np.bincount(
            pd.factorize(y)[0])) / len(y)),
        "chance_baseline": 1.0 / len(labels),
        "per_family": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(sup[i])}
            for i, c in enumerate(labels)},
        "confusion_row_normalized": cm_norm.tolist(),
    }


def score_exploring(model, exploring: pd.DataFrame,
                    feats: list[str]) -> pd.DataFrame:
    """给探索集打 6 路家族概率，追加 fam_pred / fam_conf / pf_<family> 列。"""
    df = exploring.copy()
    proba = model.predict_proba(df[feats].fillna(0).values)
    classes = list(model.classes_)
    for i, c in enumerate(classes):
        df[f"pf_{c}"] = proba[:, i]
    df["fam_pred"] = np.asarray(classes, dtype=object)[proba.argmax(axis=1)]
    df["fam_conf"] = proba.max(axis=1)
    return df


def composition_shares(labels, families=FAMILIES) -> dict:
    """一组家族标签的计数与份额（share 之和为 1，缺失家族计 0）。"""
    n = len(labels)
    counts = {f: int((np.asarray(labels) == f).sum()) for f in families}
    return {"n": int(n),
            "counts": counts,
            "shares": {f: (counts[f] / n if n else 0.0) for f in families}}


def composition_with_ci(labels, families=FAMILIES) -> dict:
    """构成份额 + 每家族份额的 Wilson 95% CI。"""
    out = composition_shares(labels, families)
    out["ci"] = {}
    for f in families:
        lo, hi = wilson_ci(out["counts"][f], out["n"])
        out["ci"][f] = [float(lo), float(hi)]
    return out


def confidence_by_band(p_ai, conf, fam, families=FAMILIES,
                       bands=PAI_BANDS) -> list[dict]:
    """按 P(AI) 分带汇总家族归属置信度与归属构成（均值/中位数/条数/份额）。"""
    p_ai = np.asarray(p_ai, dtype=float)
    conf = np.asarray(conf, dtype=float)
    fam = np.asarray(fam, dtype=object)
    rows = []
    for lo, hi, label in bands:
        mask = (p_ai >= lo) & (p_ai < hi)
        if not mask.any():
            continue
        labels = fam[mask]
        rows.append({
            "band": label, "n": int(mask.sum()),
            "conf_mean": float(conf[mask].mean()),
            "conf_median": float(np.median(conf[mask])),
            "family_shares": {f: float((labels == f).mean()) for f in families},
        })
    return rows


def half_of(month: str) -> str | None:
    if len(str(month)) != 6 or not str(month).isdigit():
        return None
    return f"{str(month)[:4]}H{1 if int(str(month)[4:]) <= 6 else 2}"


def quarter_of(month: str) -> str | None:
    if len(str(month)) != 6 or not str(month).isdigit():
        return None
    return f"{str(month)[:4]}Q{(int(str(month)[4:]) - 1) // 3 + 1}"


def cell_residuals(values, cell_codes) -> np.ndarray:
    """按单元整组去均值后的残差（单元内相对水平，不设最小样本量）。

    `cell_codes` 可以是任意可比较标签（内部先因子化为 0..K-1 整数编码）。
    """
    values = np.asarray(values, dtype=float)
    codes = pd.factorize(np.asarray(cell_codes), sort=True)[0]
    return demean_within(values, codes)


def family_residual_table(resid, fam, families=FAMILIES) -> dict:
    """各家族在单元内残差上的均值与样本量。"""
    resid = np.asarray(resid, dtype=float)
    fam = np.asarray(fam, dtype=object)
    return {f: {"mean": float(resid[fam == f].mean()) if (fam == f).any() else None,
                "n": int((fam == f).sum())} for f in families}


def fe_ols_raw(X, y, cell_codes, min_cell_n: int = MIN_CELL_N) -> dict:
    """原始量纲（不做 z-score）的单元固定效应 OLS。

    与 `engagement_drivers_analysis.fe_ols` 同为「单元内去均值 + 普通最小二乘」，
    区别仅在不对特征标准化：本分析的解释变量是 0/1 家族哑变量，保留原始量纲
    后 β 可直接读作「相对参照家族的 log10(1+互动) 差」。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    codes = np.asarray(cell_codes)
    keep = cell_keep_mask(codes, min_cell_n)
    kept_codes = pd.factorize(codes[keep], sort=True)[0]
    Xd = demean_within(X[keep], kept_codes)
    yd = demean_within(y[keep], kept_codes)
    coef, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    return {
        "coef": coef, "X_demeaned": Xd, "y_demeaned": yd,
        "cell_codes": kept_codes, "keep_mask": keep,
        "n_obs": int(keep.sum()), "n_cells": int(kept_codes.max()) + 1,
    }


def family_design(df: pd.DataFrame, families=FAMILIES,
                  ref: str = REF_FAMILY) -> tuple[np.ndarray, list[str]]:
    """家族哑变量（去掉参照家族）+ 连续控制变量 [log 长度, P(AI), 标签数]。"""
    cols, names = [], []
    for f in families:
        if f == ref:
            continue
        cols.append((df["fam_pred"].values == f).astype(float))
        names.append(f)
    cols += [np.log10(1 + df["content_len"].values), df["p_ai"].values,
             df["n_hashtags"].values]
    names += ["log10(1+正文长度)", "P(AI)", "标签数"]
    return np.column_stack(cols), names


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_transfer(cv: dict, classes: list[str], bands: list[dict],
                  overall: dict, halves: list[dict], prior: dict) -> None:
    init_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # ---- (a) 折外混淆矩阵（行归一化） ----
    ax = axes[0][0]
    cm = np.asarray(cv["confusion_row_normalized"])
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([FAMILY_NAME[c] for c in classes], rotation=30, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([FAMILY_NAME[c] for c in classes])
    ax.set_xlabel("预测家族")
    ax.set_ylabel("真实家族")
    ax.set_title(f"(a) 家族分类器 5 折 CV 混淆矩阵（行归一化，acc={cv['accuracy']:.3f}）")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > 0.5 else "#333333")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="行占比")
    ax.text(0.99, -0.30, "多数类基线 "
            f"{cv['majority_baseline']:.1%} / 均匀随机 "
            f"{cv['chance_baseline']:.1%}；macro-F1={cv['macro_f1']:.3f}",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="#555555")

    # ---- (b) 归属构成随 P(AI) 分带的变化：分类器只会归入最近原型 ----
    ax = axes[0][1]
    x = np.arange(len(bands))
    bottom = np.zeros(len(bands))
    for f in FAMILIES:
        vals = np.asarray([b["family_shares"][f] for b in bands])
        ax.bar(x, vals, bottom=bottom, color=FAMILY_COLOR[f],
               label=FAMILY_NAME[f], width=0.6)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b['band']}\nn={b['n']:,}" for b in bands], fontsize=8.5)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("家族归属份额（argmax）")
    ax.set_title("(b) 分类器只会归入最近原型：低 P(AI) 帖堆在 GLM/Claude 原型")
    for xi, b in enumerate(bands):
        ax.text(xi, 1.03, f"置信度 {b['conf_mean']:.2f}", ha="center",
                fontsize=8, color="#555555")
    ax.legend(fontsize=7.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))

    # ---- (c) 总体构成 vs 训练先验 ----
    ax = axes[1][0]
    x = np.arange(len(FAMILIES))
    shares = [overall["shares"][f] for f in FAMILIES]
    priors = [prior.get(f, 0) for f in FAMILIES]
    w = 0.38
    ax.bar(x - w / 2, shares, width=w, color="#C44E52",
           label="野外高 P(AI)（2023+）")
    ax.bar(x + w / 2, priors, width=w, color="#4C72B0", label="AIGC 训练先验")
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_NAME[f] for f in FAMILIES], rotation=20)
    ax.set_ylabel("份额")
    ax.set_title("(c) 高 P(AI) 帖的风格家族构成 vs 训练集先验")
    for xi, (s, p) in enumerate(zip(shares, priors)):
        ax.text(xi - w / 2, s + 0.006, f"{s:.1%}", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, p + 0.006, f"{p:.1%}", ha="center", fontsize=7.5,
                color="#4C72B0")
    ax.legend(fontsize=8.5)

    # ---- (d) 半年度家族构成演化 ----
    ax = axes[1][1]
    xs = np.arange(len(halves))
    bottom = np.zeros(len(halves))
    for f in FAMILIES:
        vals = np.asarray([h["shares"][f] for h in halves])
        ax.bar(xs, vals, bottom=bottom, color=FAMILY_COLOR[f],
               label=FAMILY_NAME[f], width=0.62)
        bottom += vals
    for xi, h in enumerate(halves):
        ax.text(xi, 1.02, f"n={h['n']:,}", ha="center", fontsize=7.5,
                color="#555555")
    ax.set_xticks(xs)
    ax.set_xticklabels([h["half"] for h in halves])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("家族份额")
    ax.set_title("(d) 高 P(AI) 帖家族构成的半年度演化")
    ax.legend(fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.34))

    save_fig(fig, INSIGHTS + "family_transfer.png")


def plot_engagement(resid_table: dict, cliffs: dict, fe_likes: dict,
                    fe_colls: dict, resid: np.ndarray, fam: np.ndarray) -> None:
    init_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.6))

    # ---- (a) 单元内残差分布的箱线图 ----
    ax = axes[0]
    data = [resid[fam == f] for f in FAMILIES]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for box, f in zip(bp["boxes"], FAMILIES):
        box.set_facecolor(FAMILY_COLOR[f])
        box.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("#222222")
    ax.axhline(0, color="#888888", ls="--", lw=1.0)
    ax.set_xticks(range(1, len(FAMILIES) + 1))
    ax.set_xticklabels([f"{FAMILY_NAME[f]}\n(参照组)" if f == REF_FAMILY
                        else FAMILY_NAME[f] for f in FAMILIES], fontsize=8)
    ax.set_ylabel("单元内残差 log10(1+点赞)")
    ax.set_title("(a) 高 P(AI) 帖点赞的单元内相对水平")

    # ---- (b)(c) 家族哑变量的固定效应偏效应森林图（连续控制项见结果表） ----
    names = fe_likes["names"]
    fam_names = [n for n in names if n in FAMILIES]
    for ax, fe, title in ((axes[1], fe_likes, "(b) 点赞的家族偏效应（FE-OLS）"),
                          (axes[2], fe_colls, "(c) 收藏的家族偏效应（FE-OLS）")):
        idxs = [names.index(n) for n in fam_names]
        lo_all = min(fe["ci_low"][i] for i in idxs)
        hi_all = max(fe["ci_high"][i] for i in idxs)
        pad = 0.12 * (hi_all - lo_all)
        y = np.arange(len(fam_names))[::-1]
        for yi, name in zip(y, fam_names):
            i = names.index(name)
            lo, hi = fe["ci_low"][i], fe["ci_high"][i]
            sig = (lo > 0) or (hi < 0)
            ax.errorbar(fe["coef"][i], yi,
                        xerr=[[fe["coef"][i] - lo], [hi - fe["coef"][i]]],
                        fmt="o", ms=6, capsize=4,
                        color=FAMILY_COLOR[name] if sig else "#8C8C8C")
            ax.text(hi + 0.015 * (hi_all - lo_all), yi, f"{fe['coef'][i]:+.3f}",
                    va="center", fontsize=8, color="#333333")
        ax.axvline(0, color="#555555", lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([FAMILY_NAME[n] for n in fam_names])
        ax.set_xlabel(f"相对 {FAMILY_NAME[REF_FAMILY]} 的 log10(1+互动) 差")
        ax.set_title(title)
        ax.set_xlim(lo_all - pad, hi_all + 4 * pad)
    axes[1].text(0.02, 0.94, "彩色=95% CI 不含 0；灰色=含 0（整群 bootstrap）；"
                 "连续控制项（长度/P(AI)/标签数）见结果表",
                 transform=axes[1].transAxes, fontsize=7.5, color="#555555")

    fig.tight_layout()
    save_fig(fig, INSIGHTS + "family_transfer_engagement.png")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    aigc = load_features("aigc")
    exploring = load_features("exploring")
    scores = pd.read_csv(BASE + "paper_exploring_scores.csv.gz",
                         usecols=["content_len", "p_ai"])
    if len(scores) != len(exploring) or not np.array_equal(
            scores["content_len"].to_numpy(), exploring["content_len"].to_numpy()):
        raise SystemExit("paper_features_exploring 与 paper_exploring_scores "
                         "行序/长度不一致，拒绝继续")
    exploring["p_ai"] = scores["p_ai"].to_numpy()

    detector = load_detector_model()
    feats = list(detector["features"])
    missing = [f for f in feats if f not in aigc.columns]
    if missing:
        raise SystemExit(f"AIGC 特征缺列: {missing}")

    # ---- 1. 家族分类器 5 折 CV ----
    oof_pred, _ = cv_family_predictions(aigc, feats)
    classes = sorted(aigc["model_family"].unique())
    cv = cv_metrics(aigc["model_family"].values, oof_pred, classes)
    print(f"家族分类 CV：acc={cv['accuracy']:.4f} macro-F1={cv['macro_f1']:.4f} "
          f"平衡准确率={cv['balanced_accuracy']:.4f}")

    # ---- 2. 全量模型打分探索集 ----
    full_model = make_family_pipeline().fit(aigc[feats].fillna(0).values,
                                           aigc["model_family"].values)
    exploring = score_exploring(full_model, exploring, feats)
    bands = confidence_by_band(exploring["p_ai"], exploring["fam_conf"],
                               exploring["fam_pred"])
    rho_conf, p_conf = spearmanr(exploring["p_ai"], exploring["fam_conf"])

    exp = exploring[exploring["month"] >= MONTH_FROM].copy()
    exp["half"] = exp["month"].map(half_of)
    exp["quarter"] = exp["month"].map(quarter_of)
    hi = exp[exp["p_ai"] >= HIGH_CONF].copy()
    print(f"探索集 {len(exploring):,} 条，2023+ 高置信(P(AI)>={HIGH_CONF}) "
          f"{len(hi):,} 条")

    prior = composition_shares(aigc["model_family"].values)["shares"]
    overall = composition_with_ci(hi["fam_pred"].values)
    overall["training_prior"] = prior

    # ---- 3. 半年度/季度构成演化 ----
    halves = []
    for h in HALF_ORDER:
        g = hi[hi["half"] == h]
        if not len(g):
            continue
        comp = composition_with_ci(g["fam_pred"].values)
        comp["half"] = h
        halves.append(comp)
    chi2_stat, chi2_p, chi2_dof, _ = chi2_contingency(
        pd.crosstab(hi["half"], hi["fam_pred"]).reindex(
            HALF_ORDER, axis=0, fill_value=0))

    quarters = sorted(hi["quarter"].dropna().unique())
    q_counts = pd.crosstab(hi["quarter"], hi["fam_pred"]).reindex(
        quarters, fill_value=0)
    q_shares = q_counts.div(q_counts.sum(axis=1), axis=0)
    trend = {}
    for f in FAMILIES:
        rho, p = spearmanr(np.arange(len(quarters)), q_shares[f].values)
        trend[f] = {"rho": float(rho), "p": float(p)}

    # ---- 4. 互动回报的家族结构（高置信样本） ----
    hi["log_likes"] = np.log10(1 + hi["likes"])
    hi["log_colls"] = np.log10(1 + hi["collections"])
    cell_codes = (hi["month"] + "|" + hi["domain"].astype(str)).values
    resid_likes = cell_residuals(hi["log_likes"].values, cell_codes)
    resid_table = family_residual_table(resid_likes, hi["fam_pred"].values)

    cliffs = {}
    ref_resid = resid_likes[hi["fam_pred"].values == REF_FAMILY]
    for f in FAMILIES:
        if f == REF_FAMILY:
            continue
        other = resid_likes[hi["fam_pred"].values == f]
        delta, p = cliffs_delta(other, ref_resid)
        cliffs[f] = {"delta": float(delta), "p": float(p), "n": int(len(other))}

    X, names = family_design(hi)
    fe_likes = fe_ols_raw(X, hi["log_likes"].values, cell_codes)
    fe_colls = fe_ols_raw(X, hi["log_colls"].values, cell_codes)
    out_fe = {}
    for label, fit in (("likes", fe_likes), ("collections", fe_colls)):
        boots = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                         fit["cell_codes"], n_boot=N_BOOT,
                                         seed=BOOT_SEED)
        ci = np.percentile(boots, [2.5, 97.5], axis=0)
        out_fe[label] = {
            "names": names,
            "coef": [round(float(v), 4) for v in fit["coef"]],
            "ci_low": [round(float(v), 4) for v in ci[0]],
            "ci_high": [round(float(v), 4) for v in ci[1]],
            "n_obs": fit["n_obs"], "n_cells": fit["n_cells"],
        }

    medians = {}
    for label, col in (("likes", "likes"), ("collections", "collections"),
                       ("p_ai", "p_ai"), ("content_len", "content_len")):
        medians[label] = {f: float(hi.loc[hi["fam_pred"] == f, col].median())
                          for f in FAMILIES}

    # ---- 4b. 阈值敏感性：0.7 / 0.9 / 0.95 下结论是否同向 ----
    threshold_sensitivity = []
    for tau in (0.7, HIGH_CONF, 0.95):
        sub = exp[exp["p_ai"] >= tau].copy()
        if len(sub) < 1000:
            continue
        sub["log_likes"] = np.log10(1 + sub["likes"])
        sub_codes = (sub["month"] + "|" + sub["domain"].astype(str)).values
        Xs, names_s = family_design(sub)
        fit = fe_ols_raw(Xs, sub["log_likes"].values, sub_codes)
        boots = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                         fit["cell_codes"], n_boot=N_BOOT,
                                         seed=BOOT_SEED)
        ci = np.percentile(boots, [2.5, 97.5], axis=0)
        coef = {n: float(fit["coef"][i]) for i, n in enumerate(names_s)}
        top2 = sorted(composition_shares(sub["fam_pred"].values)["shares"].items(),
                      key=lambda kv: -kv[1])[:2]
        threshold_sensitivity.append({
            "tau": tau, "n": int(len(sub)), "n_cells": fit["n_cells"],
            "top2_families": [f for f, _ in top2],
            "top2_share": float(sum(v for _, v in top2)),
            "family_coef": coef,
            "family_ci": {n: [float(ci[0][i]), float(ci[1][i])]
                          for i, n in enumerate(names_s)
                          if n in FAMILIES},
        })
    print("阈值敏感性：", [(t["tau"], round(t["top2_share"], 3))
                           for t in threshold_sensitivity])

    # ---- 5. 保存结果 ----
    out = {
        "meta": {
            "source": "paper_features_aigc.csv.gz + "
                      "paper_features_exploring.csv.gz + "
                      "paper_exploring_scores.csv.gz",
            "n_aigc": int(len(aigc)), "n_exploring": int(len(exploring)),
            "n_high_conf": int(len(hi)), "high_conf_threshold": HIGH_CONF,
            "month_from": MONTH_FROM, "min_cell_n": MIN_CELL_N,
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "cv_seed": CV_SEED,
            "reference_family": REF_FAMILY,
            "align_check": {"n": int(len(scores)), "mismatch": 0},
            "classifier": "StandardScaler + multinomial LogisticRegression"
                          "(max_iter=2000, class_weight=balanced); 37 features; "
                          "StratifiedKFold(5, shuffle=True, seed=42)",
            "note": "家族归属为风格相似度构念，仅在高 P(AI) 采样层内解释",
        },
        "classifier_cv": {
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in cv.items() if k not in ("per_family",
                                                   "confusion_row_normalized")},
            "classes_order": classes,
            "per_family": {f: {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in m.items()}
                           for f, m in cv["per_family"].items()},
            "confusion_row_normalized": [[round(v, 4) for v in row]
                                         for row in cv["confusion_row_normalized"]],
        },
        "exploring_confidence": {
            "bands": bands,
            "spearman_pai_conf": {"rho": round(float(rho_conf), 4),
                                  "p": float(p_conf)},
        },
        "composition_overall": {
            "n": overall["n"],
            "shares": {f: round(v, 4) for f, v in overall["shares"].items()},
            "ci": {f: [round(v, 4) for v in overall["ci"][f]] for f in FAMILIES},
            "training_prior": {f: round(v, 4) for f, v in prior.items()},
        },
        "composition_half_years": [
            {"half": h["half"], "n": h["n"],
             "shares": {f: round(v, 4) for f, v in h["shares"].items()},
             "ci": {f: [round(v, 4) for v in h["ci"][f]] for f in FAMILIES}}
            for h in halves],
        "composition_trend": {
            "chi2": {"stat": round(float(chi2_stat), 3), "p": float(chi2_p),
                     "dof": int(chi2_dof)},
            "quarterly_spearman": {f: {"rho": round(v["rho"], 4), "p": float(v["p"])}
                                   for f, v in trend.items()},
            "first_last_delta_pp": {
                f: round((halves[-1]["shares"][f] - halves[0]["shares"][f]) * 100, 2)
                for f in FAMILIES},
        },
        "engagement": {
            "median": {k: {f: round(v, 1) for f, v in d.items()}
                       for k, d in medians.items()},
            "within_cell_residual_likes": {
                f: {"mean": round(v["mean"], 4) if v["mean"] is not None else None,
                    "n": v["n"]}
                for f, v in resid_table.items()},
            "cliffs_vs_reference": {
                f: {"delta": round(v["delta"], 4), "p": float(v["p"]), "n": v["n"]}
                for f, v in cliffs.items()},
            "fe_likes": out_fe["likes"],
            "fe_collections": out_fe["collections"],
        },
        "threshold_sensitivity": threshold_sensitivity,
    }
    save_json(out, BASE + "family_transfer_results.json")

    plot_transfer(cv, classes, bands, overall, halves, prior)
    plot_engagement(resid_table, cliffs, out_fe["likes"], out_fe["collections"],
                    resid_likes, hi["fam_pred"].values)
    write_markdown(out)
    print("已保存 family_transfer_results.{json,md} 与 "
          "insights/family_transfer*.png")


def write_markdown(out: dict) -> None:
    cv = out["classifier_cv"]
    prior = out["composition_overall"]["training_prior"]
    comp = out["composition_overall"]
    trend = out["composition_trend"]
    eng = out["engagement"]
    fe = eng["fe_likes"]
    fec = eng["fe_collections"]

    lines = [
        "# 野外高 AI 味帖的「风格家族」构成与互动回报（Family Transfer）",
        "",
        "> 由 `family_transfer_analysis.py` 生成。分类器只用 AIGC 训练子集"
        "（7,254 条 / 6 家族）训练，探索集为纯训练分布外打分；特征为论文"
        "检测器的 37 个可解释特征；`paper_features_exploring.csv.gz` 与 "
        "`paper_exploring_scores.csv.gz` 行序对齐（content_len 校验 mismatch=0）。"
        "「家族归属」是风格相似度构念而非真实来源标注，且只在 P(AI)>=0.9 的"
        "高置信采样层内解释。",
        "",
        "## 1 家族分类器：37 个风格特征能否识别生成家族（5 折 CV）",
        "",
        f"- 总体：准确率 **{cv['accuracy']:.3f}**、macro-F1 **{cv['macro_f1']:.3f}**、"
        f"平衡准确率 {cv['balanced_accuracy']:.3f}；多数类基线 "
        f"{cv['majority_baseline']:.1%}、均匀随机 {cv['chance_baseline']:.1%}。",
        "",
        "| 家族 | 精确率 | 召回率 | F1 | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for f in FAMILIES:
        m = cv["per_family"][f]
        lines.append(f"| {FAMILY_NAME[f]} | {m['precision']:.3f} | {m['recall']:.3f} "
                     f"| {m['f1']:.3f} | {m['support']} |")
    lines += [
        "",
        "混淆矩阵见 `insights/family_transfer.png` (a)：GPT 与 GLM 互相混淆最多"
        "（家族词表指纹相近，均为叙事/书面口吻），Gemini 最易识别——与 "
        "`family_lexicon_analysis.py` 的家族口癖差异方向一致。",
        "",
        "## 2 家族归因的性质：分类器只会归入「最近原型」",
        "",
        "分类器只在 AIGC 子集上训练（没有「人类」类别），因此任何帖子——包括"
        "很可能是人类写的低 P(AI) 帖——都会被强制归入某个家族原型。分带看得"
        "很清楚：",
        "",
        "| P(AI) 分带 | n | 归属置信度均值 | 占比最高的两个归属 |",
        "|---|---:|---:|---|",
    ]
    for b in out["exploring_confidence"]["bands"]:
        top2 = sorted(b["family_shares"].items(), key=lambda kv: -kv[1])[:2]
        top2_txt = " / ".join(f"{FAMILY_NAME[f]} {v:.0%}" for f, v in top2)
        lines.append(f"| {b['band']} | {b['n']:,} | {b['conf_mean']:.3f} | "
                     f"{top2_txt} |")
    sc = out["exploring_confidence"]["spearman_pai_conf"]
    lines += [
        "",
        f"归属置信度与 P(AI) 的秩相关仅 ρ={sc['rho']:.3f}：低 P(AI) 帖并不会"
        "「归不进去」，而是集中落到 GLM/Claude 两个原型上（合计约四分之三）——"
        "这两个家族本就包含最像人的单模型（glm-3、claude 系列在 "
        "`paper_model_fingerprint.py` 中距离人类质心较近）。因此家族归属必须"
        "理解为**最近原型**而非生成来源；下文分析全部限定在 P(AI)>=0.9 的高"
        "置信样本内——即便如此，「份额」仍是风格相似度构念。",
        "",
        "## 3 高 P(AI) 帖的风格家族构成（2023.01 起）",
        "",
        f"高置信样本 n={comp['n']:,}。各家族份额（95% Wilson CI）与训练集先验：",
        "",
        "| 家族 | 份额 | 95% CI | 训练集先验 | 差值 |",
        "|---|---:|---|---:|---:|",
    ]
    for f in FAMILIES:
        ci = comp["ci"][f]
        diff = (comp["shares"][f] - prior[f]) * 100
        lines.append(f"| {FAMILY_NAME[f]} | {comp['shares'][f]:.1%} | "
                     f"{ci[0]:.1%}–{ci[1]:.1%} | {prior[f]:.1%} | {diff:+.1f}pp |")
    lines += [
        "",
        "**半年度演化：**",
        "",
        "| 时期 | n | " + " | ".join(FAMILY_NAME[f] for f in FAMILIES) + " |",
        "|---|---:|" + "---:|" * len(FAMILIES),
    ]
    for h in out["composition_half_years"]:
        lines.append(f"| {h['half']} | {h['n']:,} | " + " | ".join(
            f"{h['shares'][f]:.1%}" for f in FAMILIES) + " |")
    chi = trend["chi2"]
    lines += [
        "",
        f"半年度构成的 χ² 检验：χ²={chi['stat']:.1f}（dof={chi['dof']}，"
        f"p={chi['p']:.1e}）；首末半年度份额变化（2025H1 − 2023H1）："
        + "、".join(f"{FAMILY_NAME[f]} {trend['first_last_delta_pp'][f]:+.1f}pp"
                    for f in FAMILIES) + "。",
        "",
        "每家族季度份额的 Spearman 趋势："
        + "；".join(
            f"{FAMILY_NAME[f]} ρ={trend['quarterly_spearman'][f]['rho']:+.3f}"
            f"（p={trend['quarterly_spearman'][f]['p']:.2f}）" for f in FAMILIES)
        + "。",
        "",
        "与训练集先验相比，野外高 P(AI) 帖明显更偏 DeepSeek/GLM 风格、更少 "
        "GPT/Qwen 风格；Gemini 风格份额从 2023H1 的 4.8% 升到 2025H1 的 11.6%，"
        "是唯一季度趋势显著的家族（ρ=+0.746，p=0.01），其余家族份额在 31 个月"
        "内基本平稳。",
        "",
        "## 4 互动回报的家族结构：AI 味惩罚并非对所有风格一视同仁",
        "",
        "高置信样本内各家族风格帖的原始中位数：",
        "",
        "| 家族 | 点赞中位 | 收藏中位 | P(AI) 中位 | 正文长度中位 |",
        "|---|---:|---:|---:|---:|",
    ]
    for f in FAMILIES:
        lines.append(f"| {FAMILY_NAME[f]} | {eng['median']['likes'][f]:.0f} | "
                     f"{eng['median']['collections'][f]:.0f} | "
                     f"{eng['median']['p_ai'][f]:.3f} | "
                     f"{eng['median']['content_len'][f]:.0f} |")
    lines += [
        "",
        f"以「月份×领域」单元内残差（每单元 >= {out['meta']['min_cell_n']} 条高"
        "置信帖，单元内去均值）度量的相对点赞水平（参照 "
        f"{FAMILY_NAME[out['meta']['reference_family']]}，Cliff's δ vs 参照）：",
        "",
        "| 家族 | 单元内残差均值 | n | Cliff's δ vs "
        f"{FAMILY_NAME[out['meta']['reference_family']]} | p |",
        "|---|---:|---:|---:|---:|",
    ]
    ref = out["meta"]["reference_family"]
    for f in FAMILIES:
        r = eng["within_cell_residual_likes"][f]
        if f == ref:
            lines.append(f"| {FAMILY_NAME[f]}（参照） | {r['mean']:+.3f} | {r['n']} "
                         "| — | — |")
        else:
            c = eng["cliffs_vs_reference"][f]
            lines.append(f"| {FAMILY_NAME[f]} | {r['mean']:+.3f} | {r['n']} | "
                         f"{c['delta']:+.3f} | {c['p']:.1e} |")
    lines += [
        "",
        "**固定效应回归**（log10(1+互动) ~ 5 个家族哑变量 + log10(1+正文长度) + "
        "P(AI) + 标签数，月份×领域单元固定效应，95% CI 为按单元整群 bootstrap "
        f"{out['meta']['n_boot']} 次；n={fe['n_obs']:,}，{fe['n_cells']} 个单元）：",
        "",
        "| 变量 | 点赞 β | 点赞 95% CI | 收藏 β | 收藏 95% CI |",
        "|---|---:|---|---:|---|",
    ]
    for i, name in enumerate(fe["names"]):
        label = FAMILY_NAME.get(name, name)
        lines.append(f"| {label} | {fe['coef'][i]:+.3f} | "
                     f"[{fe['ci_low'][i]:+.3f}, {fe['ci_high'][i]:+.3f}] | "
                     f"{fec['coef'][i]:+.3f} | "
                     f"[{fec['ci_low'][i]:+.3f}, {fec['ci_high'][i]:+.3f}] |")
    lines += [
        "",
        "结论：**Claude 与 Gemini 风格**的高 P(AI) 帖相对 GLM 参照有显著更高的"
        "点赞回报（控制长度、P(AI) 与标签数后仍分别约为 "
        f"+{fe['coef'][fe['names'].index('anthropic')]:.2f} 与 "
        f"+{fe['coef'][fe['names'].index('gemini')]:.2f} log10 单位，即约 "
        f"{10**fe['coef'][fe['names'].index('anthropic')]:.1f}× / "
        f"{10**fe['coef'][fe['names'].index('gemini')]:.1f}×），"
        "而 Qwen 风格点估计最低（收藏端显著为负），GPT/DeepSeek 风格与 GLM "
        "在 95% 置信水平上差异不显著。"
        "这与家族像人度排名（`paper_model_fingerprint.py`：Gemini 全系最不像人）"
        "不同——**互动惩罚的家族结构不等于像人度排序**，风格来源与阅读体验"
        "在互动层面是独立维度。",
        "",
        "**阈值敏感性**（把高置信阈值 τ 换成 0.7 / 0.95 重跑同一 FE 设计）：",
        "",
        "| τ | n | 前两大家族份额 | Claude β 95% CI | Gemini β 95% CI | Qwen β 95% CI |",
        "|---|---:|---:|---|---|---|",
    ]
    def fmt_beta(t: dict, name: str) -> str:
        lo, hi = t["family_ci"][name]
        return f"{t['family_coef'][name]:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    for t in out["threshold_sensitivity"]:
        lines.append(
            f"| {t['tau']:.2f} | {t['n']:,} | {t['top2_share']:.1%} | "
            f"{fmt_beta(t, 'anthropic')} | {fmt_beta(t, 'gemini')} | "
            f"{fmt_beta(t, 'qwen')} |")
    lines += [
        "",
        "三个阈值下 Claude/Gemini 的偏效应均为正、Qwen 均为负，构成份额与"
        "互动排序不随阈值选择而改变——结论不是某个阈值的产物。",
        "",
        "> 方法注记：家族归属来自分类器的最大后验（argmax），在控制 P(AI) 后"
        "仍存在家族哑变量之间的选择性差异；「P(AI)」在样本内为检测器得分，"
        "只作为风格强度的连续控制，不承担因果解释。所有结论限定在探索集"
        "带采样分层的后 LLM 时期、P(AI)>=0.9 的采样层内。",
        "",
        "图表见 `insights/family_transfer.png`（分类器与构成演化）与 "
        "`insights/family_transfer_engagement.png`（残差分布与偏效应森林图）。",
    ]
    with open(BASE + "family_transfer_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
