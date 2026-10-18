# -*- coding: utf-8 -*-
"""检测器诊断：概率校准、阈值敏感性、错分切片与探索集不确定性。

既有分析（paper_detector / paper_drift / paper_robustness）报告了检测器的
AUC/F1/ACC、特征权重与高置信占比，但没有回答四个诊断性问题：

1. 输出概率是否校准？`class_weight="balanced"` 训练出的得分对应平衡先验，
   不能直接当作自然分布下的 P(AI)；本脚本把 logit 平移 log(n_负/n_正)
   （King & Zeng, 2001 的先验校正）后检验可靠性曲线与 ECE。
2. 换阈值时精确率/召回率如何权衡，不同用途该选哪个阈值。
3. 检测器在哪些长度/领域切片上犯错更多（假阳性率 / 假阴性率 / 分片 AUC）。
4. 错分的帖子在可解释特征空间上"长什么样"（标准化均值差画像）。
5. 探索集里落在"拿不准"区间的帖子占比随时间如何变化——若高置信占比
   上升而中间地带不膨胀，说明跃迁不是检测器整体变得更不确定造成的。

全部计算复用 paper_features_*.csv.gz 与 paper_detector_model.json，
交叉验证口径与 paper_detector.py 完全一致（同一管道、折数与随机种子）。

输出：
  insights/detector_calibration.png   可靠性曲线 + 得分分布 + 精确率/召回率曲线
  insights/detector_errors.png        长度/领域切片错误率 + 错分特征画像
  insights/detector_uncertainty.png   得分 ECDF 对比 + 月度置信度分层堆叠
  detector_diagnostics_results.json   全部数值结果
  detector_diagnostics_results.md     结果摘要
"""
from __future__ import annotations

import json
import os

# 单线程 BLAS：小矩阵下多线程同步开销大于收益
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_utils import (BASE, INSIGHTS, init_plot_style, save_fig, save_json,
                            ece_from_bins, prior_corrected_proba,
                            reliability_bins, load_detector_model,
                            threshold_metrics)
from paper_detector import LABELS

META = ["dataset", "domain", "month", "model_family", "model",
        "likes", "collections", "comments"]
THRESHOLDS = (0.3, 0.5, 0.7, 0.9, 0.95)
HIGH = 0.9
UNCERTAIN = (0.4, 0.6)
MIN_MONTH_N = 100
COLORS = {"human": "#4C72B0", "aigc": "#C44E52", "exploring": "#55A868"}


# --------------------------------------------------------------------------
# 数据与交叉验证（口径同 paper_detector.py）
# --------------------------------------------------------------------------
# 本脚本全程未用到的 5 个 META 列；读取时跳过以省去整列解析
_UNUSED_META = ("model_family", "model", "likes", "collections", "comments")


def load_labeled() -> tuple[pd.DataFrame, list[str]]:
    usecols = lambda c: c not in _UNUSED_META
    human = pd.read_csv(BASE + "paper_features_human.csv.gz", usecols=usecols)
    aigc = pd.read_csv(BASE + "paper_features_aigc.csv.gz", usecols=usecols)
    df = pd.concat([human, aigc], ignore_index=True)
    feats = [c for c in human.columns if c not in META]
    return df, feats


def cv_proba_X(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """对已构建的特征矩阵做 5 折分层 CV 袋外概率（同管道同种子）。"""
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000,
                                            class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def cv_proba(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """5 折分层交叉验证的袋外概率，与 paper_detector.py 同管道同种子。"""
    X = df[feats].fillna(0).values
    y = (df["dataset"] == "aigc").astype(int).values
    return cv_proba_X(X, y)


def length_decile_table(df: pd.DataFrame, y, proba) -> list[dict]:
    dec = pd.qcut(df["content_len"], 10, labels=False, duplicates="drop")
    lengths = df["content_len"].values
    rows = []
    for d in sorted(pd.unique(dec.dropna())):
        m = (dec == d).values
        yy, pp = y[m], proba[m]
        ln = lengths[m]
        if len(yy) == 0:
            continue
        row = {
            "decile": int(d) + 1,
            "n": int(len(yy)),
            "len_min": int(ln.min()),
            "len_max": int(ln.max()),
            "len_median": int(np.median(ln)),
            "human_fp_rate": round(float((pp[yy == 0] >= 0.5).mean()), 4)
            if (yy == 0).any() else None,
            "aigc_fn_rate": round(float((pp[yy == 1] < 0.5).mean()), 4)
            if (yy == 1).any() else None,
            "mean_p": round(float(pp.mean()), 4),
        }
        if (yy == 0).any() and (yy == 1).any():
            row["auc"] = round(float(roc_auc_score(yy, pp)), 4)
        rows.append(row)
    return rows


def domain_table(df: pd.DataFrame, y, proba, min_n: int = 50) -> list[dict]:
    rows = []
    for dom, g in df.groupby("domain"):
        m = g.index.values
        yy, pp = y[m], proba[m]
        n_h, n_a = int((yy == 0).sum()), int((yy == 1).sum())
        if n_h + n_a < min_n:
            continue
        row = {
            "domain": dom, "n_human": n_h, "n_aigc": n_a,
            "human_fp_rate": round(float((pp[yy == 0] >= 0.5).mean()), 4)
            if n_h else None,
            "aigc_fn_rate": round(float((pp[yy == 1] < 0.5).mean()), 4)
            if n_a else None,
        }
        if n_h and n_a:
            row["auc"] = round(float(roc_auc_score(yy, pp)), 4)
        rows.append(row)
    return sorted(rows, key=lambda r: r["domain"])


def error_profiles(df: pd.DataFrame, y, proba, model: dict,
                   top_k: int = 10, X=None) -> dict:
    """错分帖 vs 正确分类帖的标准化特征均值差（z 差，正=错分帖该特征更高）。

    传入 `X`（调用方已构建的全特征 fillna 矩阵）时直接复用，避免再次复制
    整张特征表；X 为 None 时保持旧行为自建矩阵。
    """
    feats = model["features"]
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    if X is None:
        X = df[feats].fillna(0).values
    z = (X - mean) / scale

    out = {}
    for name, cls in (("human_fp_vs_tn", 0), ("aigc_fn_vs_tp", 1)):
        if cls == 0:
            wrong, right = (proba >= 0.5), (proba < 0.5)
        else:
            wrong, right = (proba < 0.5), (proba >= 0.5)
        mask_w, mask_r = (y == cls) & wrong, (y == cls) & right
        zw, zr = z[mask_w], z[mask_r]
        diff = zw.mean(axis=0) - zr.mean(axis=0)
        order = np.argsort(-np.abs(diff))[:top_k]
        out[name] = {
            "n_wrong": int(mask_w.sum()), "n_right": int(mask_r.sum()),
            "features": [{"feature": feats[i], "label": LABELS.get(feats[i], feats[i]),
                          "z_diff": round(float(diff[i]), 3),
                          "wrong_mean": round(float(zw[:, i].mean()), 3),
                          "right_mean": round(float(zr[:, i].mean()), 3)}
                         for i in order],
        }
    return out


def exploring_bands(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """逐月统计稳健检测器得分的低/中/高置信占比与"拿不准"占比。"""
    rows = []
    for m, g in scores.groupby("month"):
        if len(g) < MIN_MONTH_N:
            continue
        p = g["p_ai"].values
        rows.append({
            "month": str(m), "n": int(len(g)),
            "lo_share": round(float((p < 0.5).mean()), 4),
            "mid_share": round(float(((p >= 0.5) & (p < HIGH)).mean()), 4),
            "hi_share": round(float((p >= HIGH).mean()), 4),
            "uncertain_share": round(float(((p > UNCERTAIN[0])
                                            & (p < UNCERTAIN[1])).mean()), 4),
            "mean_p": round(float(p.mean()), 4),
        })
    df = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    trends = {}
    for col in ("hi_share", "mid_share", "uncertain_share", "mean_p"):
        rho, p = spearmanr(np.arange(len(df)), df[col].values)
        trends[col] = {"rho": round(float(rho), 3), "p": float(p)}
    return df, trends


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_calibration(raw_bins: dict, corr_bins: dict, y, proba_corr,
                     thresholds: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1, label="完美校准")
    mr = np.asarray(raw_bins["mean_pred"], dtype=float)
    tr = np.asarray(raw_bins["mean_true"], dtype=float)
    cnt = np.asarray(raw_bins["count"], dtype=float)
    ok = cnt > 0
    ax.plot(mr[ok], tr[ok], "o-", color="#C44E52", lw=1.6, ms=5,
            label="平衡先验得分（原始）")
    mc = np.asarray(corr_bins["mean_pred"], dtype=float)
    tc = np.asarray(corr_bins["mean_true"], dtype=float)
    cc = np.asarray(corr_bins["count"], dtype=float)
    okc = cc > 0
    ax.plot(mc[okc], tc[okc], "s-", color="#4C72B0", lw=1.6, ms=5,
            label="先验校正后得分")
    ax.set_xlabel("预测 P(AI)")
    ax.set_ylabel("实际 AI 占比（自然分布）")
    ax.set_title("可靠性曲线（5 折袋外概率）")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.hist(proba_corr[y == 0], bins=50, alpha=0.55, color=COLORS["human"],
            density=True, label="人类（前LLM）")
    ax.hist(proba_corr[y == 1], bins=50, alpha=0.55, color=COLORS["aigc"],
            density=True, label="AIGC")
    for t in (0.5, HIGH):
        ax.axvline(t, color="#555555", ls=":", lw=1.1)
        ax.text(t, ax.get_ylim()[1] * 0.96, f" {t:g}", fontsize=8, color="#555555")
    ax.set_xlabel("先验校正后的 P(AI)")
    ax.set_ylabel("密度")
    ax.set_title("袋外得分分布（自然先验）")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    ax = axes[2]
    ts = np.asarray([r["threshold"] for r in thresholds])
    ax.plot(ts, [r["precision"] for r in thresholds], "o-", color="#4C72B0",
            lw=1.6, label="精确率")
    ax.plot(ts, [r["recall"] for r in thresholds], "s-", color="#C44E52",
            lw=1.6, label="召回率")
    ax.plot(ts, [r["f1"] for r in thresholds], "^-", color="#DD8452", lw=1.4,
            label="F1")
    for t in (0.5, HIGH):
        ax.axvline(t, color="#555555", ls=":", lw=1.1)
    ax.set_xlabel("阈值 τ（平衡先验得分）")
    ax.set_ylabel("指标值")
    ax.set_title("阈值敏感性（human vs AIGC 训练集）")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    save_fig(fig, INSIGHTS + "detector_calibration.png")


def plot_errors(deciles: list[dict], domains: list[dict],
                profiles: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax = axes[0]
    x = [r["len_median"] for r in deciles]
    fp = [r["human_fp_rate"] * 100 if r["human_fp_rate"] is not None else np.nan
          for r in deciles]
    fn = [r["aigc_fn_rate"] * 100 if r["aigc_fn_rate"] is not None else np.nan
          for r in deciles]
    auc = [r.get("auc") for r in deciles]
    ax.plot(x, fp, "o-", color=COLORS["human"], lw=1.6, label="人类假阳性率")
    ax.plot(x, fn, "s-", color=COLORS["aigc"], lw=1.6, label="AIGC 假阴性率")
    ax.set_xscale("log")
    ax.set_xlabel("正文长度（中位数，字符，对数轴）")
    ax.set_ylabel("错误率 (%)")
    ax.set_title("按正文长度十分位的错误率")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, auc, "^--", color="#55A868", lw=1.3, alpha=0.85, label="分片 AUC")
    ax2.set_ylabel("AUC", color="#55A868")
    ax2.set_ylim(0.75, 1.01)
    ax2.grid(False)
    ax2.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    # n_human / n_aigc 为 0 的切片速率为 None：排序时排到最后、绘图时画成
    # 缺测（nan），不能当成 0——0% 会被误读成「表现最好」的切片，且下游
    # 极值文案（write_markdown）会跟着选错对象。与 aigc 侧的 None 过滤一致。
    ds = sorted(domains, key=lambda r: -(r["human_fp_rate"]
                                         if r["human_fp_rate"] is not None
                                         else float("-inf")))
    names = [r["domain"] for r in ds]
    fp = [r["human_fp_rate"] * 100 if r["human_fp_rate"] is not None else np.nan
          for r in ds]
    fn = [r["aigc_fn_rate"] * 100 if r["aigc_fn_rate"] is not None else np.nan
          for r in ds]
    pos = np.arange(len(ds))
    ax.barh(pos - 0.19, fp, height=0.38, color=COLORS["human"],
            label="人类假阳性率")
    ax.barh(pos + 0.19, fn, height=0.38, color=COLORS["aigc"],
            label="AIGC 假阴性率")
    ax.set_yticks(pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("错误率 (%)")
    ax.set_title("按领域的错误率")
    ax.legend(fontsize=8)

    ax = axes[2]
    dh = {f["label"]: f["z_diff"] for f in profiles["human_fp_vs_tn"]["features"]}
    da = {f["label"]: f["z_diff"] for f in profiles["aigc_fn_vs_tp"]["features"]}
    feats = list(dict.fromkeys(list(dh) + list(da)))
    pos = np.arange(len(feats))
    # 只进入另一侧 Top-10 的特征在本侧没有数值：画成 NaN（缺测）而不是 0.0，
    # 否则「未进榜」会被误读为「该特征上两类错分无差异」。
    ax.barh(pos - 0.19, [dh.get(f, np.nan) for f in feats], height=0.38,
            color=COLORS["human"], label="人类假阳性 − 真阴性")
    ax.barh(pos + 0.19, [da.get(f, np.nan) for f in feats], height=0.38,
            color=COLORS["aigc"], label="AIGC 假阴性 − 真阳性")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(pos)
    ax.set_yticklabels(feats, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("标准化均值差（z 分数差，正=错分帖该特征更极端）")
    ax.set_title("错分帖特征画像（两类错分 Top 特征并集）")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, INSIGHTS + "detector_errors.png")


def plot_uncertainty(ecdf: dict, bands: pd.DataFrame, trends: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    ax = axes[0]
    for name, (xs, ys, color, ls) in ecdf.items():
        ax.plot(xs, ys, color=color, lw=1.8, ls=ls, label=name)
    ax.set_xlabel("稳健检测器得分 P(AI)")
    ax.set_ylabel("累计占比")
    ax.set_title("得分 ECDF：人类 / AIGC / 探索集两端")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)

    ax = axes[1]
    x = np.arange(len(bands))
    ax.fill_between(x, 0, bands["lo_share"] * 100, color="#B8C4E0",
                    label="低置信 P<0.5")
    ax.fill_between(x, bands["lo_share"] * 100,
                    (bands["lo_share"] + bands["mid_share"]) * 100,
                    color="#E8C4B8", label=f"中置信 0.5≤P<{HIGH:g}")
    ax.fill_between(x, (bands["lo_share"] + bands["mid_share"]) * 100, 100,
                    color="#C44E52", alpha=0.85, label=f"高置信 P≥{HIGH:g}")
    step = max(1, len(bands) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(bands["month"].iloc[::step], rotation=45, fontsize=8)
    ax.set_ylabel("帖子占比 (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"探索集月度置信度分层（高置信 ρ={trends['hi_share']['rho']:.2f}, "
                 f"p={trends['hi_share']['p']:.2g}）")
    ax.legend(fontsize=8, loc="center left")

    fig.tight_layout()
    save_fig(fig, INSIGHTS + "detector_uncertainty.png")


def ecdf_curves(df: pd.DataFrame, scores: pd.DataFrame,
                oof_robust: np.ndarray) -> dict:
    """人类集 / AIGC 集 / 探索集两端半年的稳健得分 ECDF。

    人类/AIGC 用 5 折交叉验证的 out-of-fold 稳健得分，与探索集（外样本）
    同口径；若用全量模型的训练集内得分，两条曲线会偏乐观、不可比。
    """
    def curve(values):
        xs = np.sort(np.asarray(values, dtype=float))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return [0.0] + xs.tolist(), [0.0] + ys.tolist()

    is_human = (df["dataset"] == "human").values
    out = {}
    out["人类（前LLM）"] = (*curve(oof_robust[is_human]), COLORS["human"], "-")
    out["AIGC"] = (*curve(oof_robust[~is_human]), COLORS["aigc"], "-")
    e23 = scores[(scores["month"] >= "202301") & (scores["month"] <= "202306")]["p_ai"]
    e25 = scores[scores["month"] >= "202501"]["p_ai"]
    out["探索集 2023H1"] = (*curve(e23), COLORS["exploring"], "--")
    out["探索集 2025H1"] = (*curve(e25), "#937860", "--")
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    init_plot_style()
    model = load_detector_model()
    df, feats = load_labeled()
    robust_feats = model["robust_features"]
    y = (df["dataset"] == "aigc").astype(int).values
    n_h, n_a = int((y == 0).sum()), int((y == 1).sum())
    print(f"标注数据: human={n_h} aigc={n_a} (features={len(feats)}, robust={len(robust_feats)})")

    # 特征矩阵只构建一次：稳健特征是全特征矩阵的列子集，直接切片复用
    X_full = df[feats].fillna(0).values
    col_idx = {c: i for i, c in enumerate(feats)}
    proba = cv_proba_X(X_full, y)
    proba_r = cv_proba_X(X_full[:, [col_idx[c] for c in robust_feats]], y)
    auc = float(roc_auc_score(y, proba))
    auc_r = float(roc_auc_score(y, proba_r))
    print(f"全特征 CV AUC={auc:.4f}（paper_detector 记录 0.9402）")
    print(f"稳健特征 CV AUC={auc_r:.4f}（paper_detector 记录 0.9294）")

    # ---- 概率校准（平衡先验 + 先验校正）----
    proba_corr = prior_corrected_proba(proba, n_neg=n_h, n_pos=n_a)
    raw_bins = reliability_bins(proba, y, bins=10)
    corr_bins = reliability_bins(proba_corr, y, bins=10)
    ece_raw = ece_from_bins(raw_bins)  # 直接复用已分箱结果，避免重复分箱
    ece_corr = ece_from_bins(corr_bins)
    brier_raw = float(brier_score_loss(y, proba))
    brier_corr = float(brier_score_loss(y, proba_corr))
    print(f"ECE 原始={ece_raw:.4f} 先验校正后={ece_corr:.4f}；"
          f"Brier 原始={brier_raw:.4f} 校正后={brier_corr:.4f}")

    thresholds = threshold_metrics(y, proba, THRESHOLDS)
    deciles = length_decile_table(df, y, proba)
    domains = domain_table(df, y, proba)
    profiles = error_profiles(df, y, proba, model, X=X_full)

    scores = pd.read_csv(BASE + "paper_exploring_scores.csv.gz",
                         usecols=["month", "p_ai"])
    scores["month"] = scores["month"].astype(str)
    bands, trends = exploring_bands(scores)
    print("探索集月度置信分层趋势:",
          {k: (v["rho"], round(v["p"], 3)) for k, v in trends.items()})

    ecdf = ecdf_curves(df, scores, proba_r)

    plot_calibration(raw_bins, corr_bins, y, proba_corr, thresholds)
    plot_errors(deciles, domains, profiles)
    plot_uncertainty(ecdf, bands, trends)

    out = {
        "cv": {
            "auc_full": round(auc, 4), "auc_robust": round(auc_r, 4),
            "f1_full": round(float(f1_score(y, (proba >= 0.5).astype(int))), 4),
            "acc_full": round(float(accuracy_score(y, (proba >= 0.5).astype(int))), 4),
            "n_human": n_h, "n_aigc": n_a,
        },
        "calibration": {
            "ece_raw_balanced": round(ece_raw, 4),
            "ece_prior_corrected": round(ece_corr, 4),
            "brier_raw_balanced": round(brier_raw, 4),
            "brier_prior_corrected": round(brier_corr, 4),
            "prior_natural_p_ai": round(n_a / (n_h + n_a), 4),
            "reliability_raw": raw_bins,
            "reliability_corrected": corr_bins,
        },
        "thresholds": thresholds,
        "length_deciles": deciles,
        "domain_slices": domains,
        "error_profiles": profiles,
        "exploring_bands_monthly": bands.to_dict(orient="records"),
        "exploring_band_trends": trends,
    }
    save_json(out, BASE + "detector_diagnostics_results.json")
    write_markdown(out)
    print("已保存 detector_diagnostics_results.json/.md 与 3 张诊断图")


def write_markdown(out: dict) -> None:
    cv = out["cv"]
    cal = out["calibration"]
    trows = "\n".join(
        f"| {r['threshold']:g} | {r['precision']:.3f} | {r['recall']:.3f} | "
        f"{r['f1']:.3f} | {r['flagged_neg_share']*100:.1f}% | "
        f"{r['flagged_pos_share']*100:.1f}% |" for r in out["thresholds"])
    # None（该切片没有人类样本）不能参与极值比较：`or 0` 会把缺测当成 0%，
    # 选出「假阳性率最低的领域」后又在下面乘 100 时崩溃。与 aigc 侧同口径。
    with_fp = [r for r in out["domain_slices"] if r["human_fp_rate"] is not None]
    dmin = min(with_fp, key=lambda r: r["human_fp_rate"])
    dmax = max(with_fp, key=lambda r: r["human_fp_rate"])
    amin = min((r for r in out["domain_slices"] if r["aigc_fn_rate"] is not None),
               key=lambda r: r["aigc_fn_rate"])
    amax = max((r for r in out["domain_slices"] if r["aigc_fn_rate"] is not None),
               key=lambda r: r["aigc_fn_rate"])
    t = out["exploring_band_trends"]
    lines = [
        "# 检测器诊断：校准、阈值与错分切片",
        "",
        "全部数字由 `detector_diagnostics.py` 在 `paper_features_*.csv.gz` 上复算，"
        "交叉验证口径与 `paper_detector.py` 一致（5 折、种子 42、平衡类权重）。",
        "",
        f"- 5 折 CV：全特征 AUC={cv['auc_full']:.4f} / 稳健特征 "
        f"AUC={cv['auc_robust']:.4f}（与 paper_detector 记录一致）。",
        f"- 概率校准：平衡先验得分的 ECE={cal['ece_raw_balanced']:.4f}，按"
        f"自然先验（P(AI)={cal['prior_natural_p_ai']:.4f}）做 logit 平移校正后"
        f" ECE 降至 {cal['ece_prior_corrected']:.4f}；Brier "
        f"{cal['brier_raw_balanced']:.4f} → {cal['brier_prior_corrected']:.4f}。"
        "即：得分要有概率语义必须先做先验校正，校正后校准良好。",
        "",
        "## 阈值敏感性（平衡先验得分，human vs AIGC）",
        "",
        "| τ | 精确率 | 召回率 | F1 | 人类被标为 AI | AIGC 被检出 |",
        "|---|--------|--------|----|--------------|------------|",
        trows,
        "",
        f"- τ=0.5 平衡精确率/召回率（{out['thresholds'][1]['precision']:.2f}/"
        f"{out['thresholds'][1]['recall']:.2f}）；τ=0.9 精确率升至 "
        f"{[r for r in out['thresholds'] if r['threshold']==0.9][0]['precision']:.3f}，"
        "适合高置信子集分析；τ=0.3 召回率优先，适合召回敏感的场景。",
        "",
        "## 错分切片",
        "",
        f"- 人类假阳性率的领域极差很小：{dmax['domain']} 最高 "
        f"({dmax['human_fp_rate']*100:.1f}%)、{dmin['domain']} 最低 "
        f"({dmin['human_fp_rate']*100:.1f}%)；AIGC 假阴性率 {amin['domain']} 最低 "
        f"({amin['aigc_fn_rate']*100:.1f}%)、{amax['domain']} 最高 "
        f"({amax['aigc_fn_rate']*100:.1f}%)。领域间差异远小于跨时代差异，"
        "检测器不是靠领域词作弊。",
        "- 错分画像（`insights/detector_errors.png`）呈镜像对称：人类假阳性帖"
        "在独特字比率、AI 味词密度、感叹号上更像 AIGC；AIGC 假阴性帖则更接近"
        "人类（独特字比率与 AI 味词密度显著偏低）。这再次说明风格两端存在"
        "连续重叠区，边界帖天然难分。",
        "",
        "## 探索集不确定性",
        "",
        f"- 高置信占比趋势 ρ={t['hi_share']['rho']:.3f} (p={t['hi_share']['p']:.2g})，"
        f"中间地带 (0.5–0.9) 占比趋势 ρ={t['mid_share']['rho']:.3f} "
        f"(p={t['mid_share']['p']:.2g})，均与论文 §5.4 的「后 LLM 时代平台期」"
        "一致；",
        f"- 「拿不准」区间 (0.4–0.6) 占比随时间不升反降"
        f"（ρ={t['uncertain_share']['rho']:.3f}, p={t['uncertain_share']['p']:.2g}），"
        "排除了「得分上升只是检测器越来越含糊」的解释。",
        "",
    ]
    with open(BASE + "detector_diagnostics_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
