# -*- coding: utf-8 -*-
"""第八步：稳健性检验，回答关于既有结论的四个审查问题。

1. 长度匹配：human 帖子比 AIGC 长，风格差异是否只是长度造成的假象？
   -> 在共同长度分箱内逐特征比较（均值 + Cliff's delta）
2. 领域一致性：单特征的人/机差异方向是否在所有领域内一致？
   -> 10 领域 x 12 特征的 Cliff's delta 热力图
3. 跨领域迁移：检测器是否只是记住了"领域教科书词汇"？
   -> 留一领域（LOSO）：训练时去掉目标领域，测其在未见领域上的 AUC
4. 检测器与扩散结论的稳定性：
   -> 5 折 CV 得分做 1000 次 bootstrap 的 AUC 置信区间、PR 曲线与工作点，
      以及 0.5/0.7/0.9/0.95 四个阈值下各时期的"高置信"占比敏感性

只使用既有派生特征（paper_features_*.csv.gz）与已保存的检测器参数，
除新增图表/JSON 外不改动任何产物。
"""
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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve)

from analysis_utils import (BASE, FIGS, FEATURE_META, setup_chinese_fonts,
                            load_features, load_detector_model, detector_score,
                            cliffs_delta, wilson_ci, bootstrap_auc, save_json)

setup_chinese_fonts()

C_HUMAN = "#4C72B0"
C_AIGC = "#C44E52"
C_GREEN = "#55A868"
C_ORANGE = "#DD8452"

LENGTH_EDGES = [150, 250, 350, 500, 700, 1000, np.inf]
LENGTH_LABELS = ["150-250", "250-350", "350-500", "500-700", "700-1000", "1000+"]

FEAT_LABEL = {
    "excl100": "感叹号/百字", "ques100": "问号/百字", "tilde100": "波浪号/百字",
    "emoji100": "emoji/百字", "xhsemo100": "小红书表情/百字",
    "ai_lex100": "AI味词/百字", "hum_lex100": "逻辑连接词/百字",
    "list_line_ratio": "列表行占比", "colon_line_ratio": "冒号行占比",
    "p_的": "「的」/百字", "p_是": "「是」/百字", "uniq_char_ratio": "独特字比率",
}

CONSISTENCY_FEATS = list(FEAT_LABEL)
LENGTH_FEATS = ["ai_lex100", "hum_lex100", "excl100", "emoji100",
                "xhsemo100", "uniq_char_ratio"]

MIN_BIN_N = 50
MIN_DOMAIN_N = 30


def part_length_matched(human: pd.DataFrame, aigc: pd.DataFrame) -> dict:
    rows = {}
    human_bins = pd.cut(human["content_len"], LENGTH_EDGES, right=False,
                        labels=LENGTH_LABELS)
    aigc_bins = pd.cut(aigc["content_len"], LENGTH_EDGES, right=False,
                       labels=LENGTH_LABELS)
    for lab in LENGTH_LABELS:
        gh = human[human_bins == lab]
        ga = aigc[aigc_bins == lab]
        if len(gh) < MIN_BIN_N or len(ga) < MIN_BIN_N:
            continue
        rows[lab] = {"n_human": int(len(gh)), "n_aigc": int(len(ga)), "features": {}}
        for f in CONSISTENCY_FEATS:
            delta, p = cliffs_delta(ga[f].values, gh[f].values)
            rows[lab]["features"][f] = {
                "human_mean": round(float(gh[f].mean()), 4),
                "aigc_mean": round(float(ga[f].mean()), 4),
                "cliffs_delta": round(float(delta), 3),
                "p_value": float(p),
            }

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    x = np.arange(len(rows))
    for ax, f in zip(axes.flat, LENGTH_FEATS):
        hm = [rows[b]["features"][f]["human_mean"] for b in rows]
        am = [rows[b]["features"][f]["aigc_mean"] for b in rows]
        ax.plot(x, hm, "o-", color=C_HUMAN, lw=1.6, ms=5, label="人类")
        ax.plot(x, am, "s-", color=C_AIGC, lw=1.6, ms=5, label="AIGC")
        deltas = [rows[b]["features"][f]["cliffs_delta"] for b in rows]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\nδ={d:+.2f}" for b, d in zip(rows, deltas)],
                           fontsize=8)
        ax.set_title(FEAT_LABEL[f], fontsize=11)
        ax.grid(alpha=0.25, lw=0.6)
    for ax in (axes[0][0], axes[1][0]):
        ax.set_ylabel("均值")
    axes[0][0].legend(fontsize=9)
    fig.suptitle("长度匹配比较：同一正文长度区间内，人/机风格差异依然存在\n"
                 "（δ 为区间内 AIGC vs 人类的 Cliff's delta）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS + "robustness_length_matched.png", dpi=150)
    plt.close(fig)
    return rows


def part_domain_consistency(human: pd.DataFrame, aigc: pd.DataFrame) -> dict:
    domains = sorted(set(human["domain"]) & set(aigc["domain"]))
    matrix = np.full((len(domains), len(CONSISTENCY_FEATS)), np.nan)
    pvals = np.full_like(matrix, np.nan)
    for i, d in enumerate(domains):
        gh = human[human["domain"] == d]
        ga = aigc[aigc["domain"] == d]
        if len(gh) < MIN_DOMAIN_N or len(ga) < MIN_DOMAIN_N:
            continue
        for j, f in enumerate(CONSISTENCY_FEATS):
            delta, p = cliffs_delta(ga[f].values, gh[f].values)
            matrix[i, j] = delta
            pvals[i, j] = p

    sign_consistency = {}
    for j, f in enumerate(CONSISTENCY_FEATS):
        col = matrix[:, j]
        col = col[~np.isnan(col)]
        sign_consistency[f] = {
            "positive_domains": int((col > 0).sum()),
            "negative_domains": int((col < 0).sum()),
            "n_domains": int(len(col)),
            "median_delta": round(float(np.median(col)), 3),
        }

    vmax = float(np.nanmax(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CONSISTENCY_FEATS)))
    ax.set_xticklabels([FEAT_LABEL[f] for f in CONSISTENCY_FEATS],
                       rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=10)
    for i in range(len(domains)):
        for j in range(len(CONSISTENCY_FEATS)):
            if np.isnan(matrix[i, j]):
                continue
            star = "*" if pvals[i, j] < 0.001 else ""
            ax.text(j, i, f"{matrix[i, j]:+.2f}{star}", ha="center", va="center",
                    fontsize=7.5, color="black")
    ax.set_title("各领域内人/机风格差异的方向一致性（Cliff's delta，红=AIGC更高，"
                 "蓝=人类更高；* p<0.001）", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Cliff's delta")
    fig.tight_layout()
    fig.savefig(FIGS + "robustness_effect_consistency.png", dpi=150)
    plt.close(fig)
    return {"delta_matrix": {d: {f: (None if np.isnan(matrix[i, j])
                                     else round(float(matrix[i, j]), 3))
                                 for j, f in enumerate(CONSISTENCY_FEATS)}
                             for i, d in enumerate(domains)},
            "sign_consistency": sign_consistency}


def part_cross_domain(df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                      ref_auc: float) -> dict:
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    domains = sorted(set(df.loc[df["dataset"] == "human", "domain"])
                     & set(df.loc[df["dataset"] == "aigc", "domain"]))
    is_human = (df["dataset"] == "human").values
    is_aigc = (df["dataset"] == "aigc").values
    rows = {}
    for d in domains:
        tr = (df["domain"] != d).values
        te = (df["domain"] == d).values
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[te])[:, 1]
        rows[d] = {"auc": round(float(roc_auc_score(y[te], proba)), 4),
                   "n_human": int((te & is_human).sum()),
                   "n_aigc": int((te & is_aigc).sum())}

    aucs = np.array([r["auc"] for r in rows.values()])
    weights = np.array([r["n_human"] + r["n_aigc"] for r in rows.values()], dtype=float)
    summary = {"min_auc": round(float(aucs.min()), 4),
               "max_auc": round(float(aucs.max()), 4),
               "mean_auc": round(float(aucs.mean()), 4),
               "weighted_mean_auc": round(float(np.average(aucs, weights=weights)), 4),
               "reference_cv_auc": round(float(ref_auc), 4)}

    fig, ax = plt.subplots(figsize=(9, 4.6))
    order = sorted(rows, key=lambda d: rows[d]["auc"])
    vals = [rows[d]["auc"] for d in order]
    ax.bar(range(len(order)), vals, color=C_GREEN, alpha=0.9)
    ax.axhline(ref_auc, color=C_AIGC, ls="--", lw=1.2,
               label=f"全数据 5 折 CV AUC={ref_auc:.3f}")
    ax.set_ylim(0.8, 1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel("留一领域测试 AUC")
    ax.set_title("跨领域泛化：训练集剔除该领域后，检测器在未见领域上的人/机区分能力")
    for i, d in enumerate(order):
        ax.text(i, vals[i] + 0.004, f"{rows[d]['auc']:.3f}", ha="center", fontsize=9)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIGS + "robustness_domain_transfer.png", dpi=150)
    plt.close(fig)
    return {"per_domain": rows, "summary": summary}


def part_detector_ci(X: np.ndarray, y: np.ndarray) -> dict:
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    boot = bootstrap_auc(y, proba, n_boot=1000)
    ap = float(average_precision_score(y, proba))
    precision, recall, thresholds = precision_recall_curve(y, proba)
    i90 = int(np.argmin(np.abs(thresholds - 0.9)))
    at09 = {"precision": round(float(precision[i90]), 4),
            "recall": round(float(recall[i90]), 4),
            "f1": round(float(2 * precision[i90] * recall[i90] /
                                (precision[i90] + recall[i90])), 4)}
    baseline = float(y.mean())

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(recall, precision, color=C_AIGC, lw=1.8,
            label=f"PR 曲线（AP={ap:.3f}）")
    ax.axhline(baseline, color="gray", ls="--", lw=1,
               label=f"正类比例基线 {baseline*100:.1f}%")
    ax.plot(at09["recall"], at09["precision"], "o", color=C_HUMAN, ms=8,
            label=f"阈值 P≥0.9：P={at09['precision']:.2f}, R={at09['recall']:.2f}")
    ax.set_xlabel("召回率（AIGC）")
    ax.set_ylabel("精确率")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"检测器的精确率-召回率曲线\n5 折 CV AUC={boot['auc']:.3f} "
                 f"(95%CI {boot['ci_low']:.3f}–{boot['ci_high']:.3f})", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIGS + "robustness_precision_recall.png", dpi=150)
    plt.close(fig)
    return {"auc_bootstrap": boot, "average_precision": round(ap, 4),
            "pr_at_threshold_0.9": at09, "positive_rate": round(baseline, 4)}


def part_threshold_sensitivity(human: pd.DataFrame,
                               expl: pd.DataFrame) -> dict:
    model = load_detector_model()
    human["p_ai"] = detector_score(human, model, robust=True)
    expl["p_ai"] = detector_score(expl, model, robust=True)

    periods = {
        "human_clean(2020.12-2022.06)": human[human["month"] <= "202206"],
        "human_late(2022.07-11)": human[human["month"] >= "202207"],
        "exploring_2023H1": expl[(expl["month"] >= "202301") & (expl["month"] <= "202306")],
        "exploring_2023H2": expl[(expl["month"] >= "202307") & (expl["month"] <= "202312")],
        "exploring_2024H1": expl[(expl["month"] >= "202401") & (expl["month"] <= "202406")],
        "exploring_2024H2": expl[(expl["month"] >= "202407") & (expl["month"] <= "202412")],
        "exploring_2025H1": expl[(expl["month"] >= "202501") & (expl["month"] <= "202506")],
    }
    thresholds = [0.5, 0.7, 0.9, 0.95]
    table = {}
    for name, g in periods.items():
        entry = {"n": int(len(g)), "thresholds": {}}
        for t in thresholds:
            k = int((g["p_ai"] > t).sum())
            lo, hi = wilson_ci(k, len(g))
            entry["thresholds"][f"{t}"] = {
                "share": round(k / len(g), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
            }
        table[name] = entry

    ratios = {}
    base = table["human_clean(2020.12-2022.06)"]["thresholds"]
    for name in periods:
        if name.startswith("human"):
            continue
        ratios[name] = {
            f"{t}": round(table[name]["thresholds"][f"{t}"]["share"] /
                          base[f"{t}"]["share"], 1)
            for t in thresholds
        }

    fig, ax = plt.subplots(figsize=(11, 5))
    names = list(periods)
    x = np.arange(len(names))
    for t, color in zip(thresholds, [C_HUMAN, C_GREEN, C_AIGC, C_ORANGE]):
        vals = [table[n]["thresholds"][f"{t}"]["share"] * 100 for n in names]
        ax.plot(x, vals, "o-", color=color, lw=1.6, ms=5, label=f"P(AI)>{t}")
    ax.axvspan(-0.5, 1.5, color=C_HUMAN, alpha=0.06)
    ax.text(0.5, ax.get_ylim()[1] * 0.95, "前 LLM 时代", ha="center",
            fontsize=10, color=C_HUMAN)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("高于阈值的帖子占比 (%)")
    ax.set_title("阈值敏感性：无论取 0.5/0.7/0.9/0.95，干净前 LLM 窗口占比均最低，\n"
                 "后 LLM 时代即达高位并在 2023–2025 保持平稳", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIGS + "robustness_threshold_sensitivity.png", dpi=150)
    plt.close(fig)
    return {"periods": table, "post_pre_ratio": ratios}


def load_reference_auc() -> float:
    """读取 paper_detector.py 保存的全数据 5 折 CV AUC，作为稳健性比较基准。

    直接从 paper_detector_results.json 读取，避免硬编码值随检测器重训而失配；
    文件缺失或结构异常时回退到当前基准 0.9402。
    """
    path = BASE + "paper_detector_results.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f)["cv_metrics"]["auc"])
    except Exception:
        return 0.9402


def main() -> None:
    human = load_features("human")
    aigc = load_features("aigc")
    expl = load_features("exploring")

    feats = [c for c in human.columns if c not in FEATURE_META]
    df = pd.concat([human, aigc], ignore_index=True)
    X = df[feats].fillna(0).values
    y = (df["dataset"] == "aigc").astype(int).values

    print("=== 1. 长度匹配 ===")
    length_matched = part_length_matched(human, aigc)
    for f in LENGTH_FEATS:
        ds = [v["features"][f]["cliffs_delta"] for v in length_matched.values()]
        print(f"{FEAT_LABEL[f]:12s} 各长度箱 delta: {ds}")

    print("=== 2. 领域一致性 ===")
    consistency = part_domain_consistency(human, aigc)
    for f, s in consistency["sign_consistency"].items():
        print(f"{FEAT_LABEL[f]:12s} 同向领域 {s['positive_domains']}↑/"
              f"{s['negative_domains']}↓ (共{s['n_domains']}) 中位δ={s['median_delta']:+.2f}")

    ref_auc = load_reference_auc()
    print("=== 3. 留一领域 ===")
    transfer = part_cross_domain(df, X, y, ref_auc)
    for d, r in sorted(transfer["per_domain"].items(), key=lambda x: x[1]["auc"]):
        print(f"{d}: AUC={r['auc']:.4f} (H={r['n_human']}, A={r['n_aigc']})")
    print("汇总:", transfer["summary"])

    print("=== 4. 检测器置信区间与 PR ===")
    det = part_detector_ci(X, y)
    print("AUC bootstrap:", det["auc_bootstrap"])
    print("AP:", det["average_precision"], "P≥0.9工作点:", det["pr_at_threshold_0.9"])

    print("=== 5. 阈值敏感性 ===")
    sens = part_threshold_sensitivity(human, expl)
    for name, r in sens["post_pre_ratio"].items():
        print(f"{name}: 相对干净基线的放大倍数 {r}")

    out = {
        "length_matched": length_matched,
        "domain_effect_consistency": consistency,
        "leave_one_domain_out": transfer,
        "detector_pr": det,
        "threshold_sensitivity": sens,
    }
    save_json(out, BASE + "paper_robustness_results.json")
    print("已保存 paper_robustness_results.json 及 5 张稳健性图")


if __name__ == "__main__":
    main()
