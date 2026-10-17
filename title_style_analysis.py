# -*- coding: utf-8 -*-
"""标题端「AI 味」分析：标题是否也能单独辨认人机，以及标题风格如何随时间与互动变化。

既有分析（paper_detector / paper_drift / paper_engagement）把所有特征混在一起，
标题只在稳健性消融里作为整体被剔除，标题自身的信息量从未被单独量化。本脚本
补上这一维度，全部复用论文管线的数据与口径：

1. 检测器消融：同一套 5 折分层交叉验证下比较「正文特征（29）/ 标题特征（7）/
   全部特征（37）」的 AUC，量化标题单独可分性与增量贡献；
2. 标题 AI 得分：用标题特征训练的全量逻辑回归给三个子集打分，给出 17 个 LLM 的
   标题像人度排名与探索集半年度高置信占比；
3. 时间与互动：探索集内标题得分与正文得分的月度轨迹、标题得分与点赞的
   总体/单元内秩相关（口径同 paper_engagement.py）；
4. 标题模板化：全量扫描标题，统计首 4 字 Top-20 覆盖率、前缀熵与 AI 专属/人类
   专属标题前缀（口径同 framing_analysis.py，人类/探索集按 note_id 去重）。

输出：
  insights/title_detector.png          检测器消融 + 标题模型系数
  insights/title_dynamics.png          月度标题/正文得分轨迹 + 17 模型标题像人度
  insights/title_formulaicity.png      标题前缀集中度 + 标题得分十分位互动
  title_style_results.json             全部数值结果
  title_style_results.md               结果摘要
"""
from __future__ import annotations

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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_utils import (BASE, INSIGHTS, init_plot_style, save_fig,
                            save_json, engagement_of, iter_jsonl,
                            linear_score, load_detector_model, parse_local_time,
                            title_of, detector_score, content_of, compact_text,
                            prefix_ngram_counts, shannon_entropy, topk_coverage,
                            bootstrap_median_ci, cliffs_delta, wilson_ci)
from framing_analysis import dedup_indices, specific_templates

SCORES_INPUTS = {
    "human": BASE + "training_set_human_fixed.jsonl",
    "aigc": BASE + "training_set_aigc.jsonl",
    "exploring": BASE + "exploring_set_fixed.jsonl",
}
FAMILY_NAME = {"gpt": "GPT", "gemini": "Gemini", "deepseek": "DeepSeek",
               "anthropic": "Claude", "glm": "GLM", "qwen": "Qwen"}
FAMILY_COLOR = {"GPT": "#4C72B0", "Gemini": "#C44E52", "DeepSeek": "#55A868",
                "Claude": "#8172B3", "GLM": "#DD8452", "Qwen": "#64B5CD"}

TITLE_LABELS = {
    "title_len": "标题长度", "title_excl": "标题含感叹号", "title_ques": "标题含问号",
    "title_emoji": "标题含emoji", "title_pipe": "标题含分隔符｜",
    "title_num": "标题含数字", "title_bracket": "标题含【】",
}
TITLE_PREFIX_LEN = 4
TOPK = 20
HIGH_CONF = 0.9
MIN_CELL_N = 100
MIN_AIGC_TPL = 10
MIN_HUMAN_TPL = 30
BOOT = 1000
BOOT_SEED = 42
HALF_ORDER = ["2023H1", "2023H2", "2024H1", "2024H2", "2025H1"]


# --------------------------------------------------------------------------
# 数据加载与打分
# --------------------------------------------------------------------------
def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    human = pd.read_csv(BASE + "paper_features_human.csv.gz")
    aigc = pd.read_csv(BASE + "paper_features_aigc.csv.gz")
    return human, aigc


def cv_auc(df: pd.DataFrame, feats: list[str], cv: StratifiedKFold,
           X=None) -> dict:
    """与 paper_detector.py 完全相同的逻辑回归交叉验证口径。

    传入 `X`（调用方按 feats 顺序切好的特征矩阵）时直接复用，避免为
    每个特征子集重复构造整张矩阵；X 为 None 时保持旧行为。
    """
    if X is None:
        X = df[feats].fillna(0).values
    y = (df["dataset"] == "aigc").astype(int).values
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return {
        "auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, (proba >= 0.5).astype(int))),
        "acc": float(accuracy_score(y, (proba >= 0.5).astype(int))),
        "n_features": len(feats),
    }


def fit_title_model(df: pd.DataFrame, title_feats: list[str],
                    X=None) -> dict:
    if X is None:
        X = df[title_feats].fillna(0).values
    y = (df["dataset"] == "aigc").astype(int).values
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(scaler.transform(X), y)
    return {
        "features": title_feats,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
    }


def scan_titles(path: str) -> tuple[list[str], list, np.ndarray]:
    """流式扫描标题；meta 分数用于按 note_id 去重时保留最完整的一条。

    主流程只用标题的紧凑前缀（TITLE_PREFIX_LEN 字符）做 n-gram 计数，
    因此扫描时就地截断前缀，避免长期驻留全量标题字符串；过短标题存 None
    （下游 compact_text 会把 None 视为空串，自动跳过）。
    """
    prefixes, ids, metas = [], [], []
    for rec in iter_jsonl(path):
        compact = compact_text(title_of(rec))
        prefixes.append(compact[:TITLE_PREFIX_LEN]
                        if len(compact) >= TITLE_PREFIX_LEN else None)
        likes, colls, comms = engagement_of(rec)
        dt = parse_local_time(rec.get("local_time"))
        has_meta = bool(rec.get("user_id")) and rec.get("domain") not in (None, "", "未知")
        content = content_of(rec)
        if not isinstance(content, str):
            content = str(content)
        metas.append(int(has_meta) + int(bool(likes)) + int(bool(colls))
                     + int(bool(comms)) + int(dt is not None)
                     + int(bool(content) and not content.isspace()))
        ids.append(rec.get("note_id"))
    return prefixes, ids, np.asarray(metas, dtype=np.int64)


def half_of(month: str) -> str | None:
    if len(month) != 6 or not month.isdigit():
        return None
    year, mm = month[:4], int(month[4:])
    return f"{year}H{1 if mm <= 6 else 2}"


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_detector(ablation: dict, coefs: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    order = ["body_only", "title_only", "full"]
    names = {"body_only": "正文特征 (29)", "title_only": "标题特征 (7)",
             "full": "全部特征 (37)"}
    colors = ["#55A868", "#DD8452", "#4C72B0"]
    vals = [ablation[k]["auc"] for k in order]
    ax = axes[0]
    bars = ax.bar([names[k] for k in order], vals, color=colors, width=0.55)
    ax.set_ylim(min(vals) - 0.06, max(vals) + 0.03)
    ax.set_ylabel("5 折 CV AUC")
    ax.set_title("标题单独的可分性（human vs AIGC）")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", fontsize=10.5)
    ax.annotate(f"标题单独 ≈ 正文模型的 "
                f"{ablation['title_only']['auc']/ablation['body_only']['auc']:.0%}",
                xy=(0.5, 0.03), xycoords="axes fraction",
                ha="center", fontsize=9, color="#555555")

    ax = axes[1]
    labels = [TITLE_LABELS.get(f, f) for f in coefs.index]
    colors = ["#4C72B0" if v < 0 else "#C44E52" for v in coefs.values]
    ax.barh(labels, coefs.values, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("标准化逻辑回归系数（正=AI 信号，负=人类信号）")
    ax.set_title("标题模型特征权重")
    fig.tight_layout()
    save_fig(fig, INSIGHTS + "title_detector.png")


def plot_dynamics(monthly: pd.DataFrame, model_rank: pd.DataFrame,
                  dyad_rho: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    x = np.arange(len(monthly))
    ax.plot(x, monthly["title_mean"], "o-", color="#DD8452", lw=1.8, ms=4,
            label="标题得分")
    ax.plot(x, monthly["body_mean"], "s--", color="#4C72B0", lw=1.5, ms=3.5,
            label="正文得分（稳健检测器）")
    step = max(1, len(monthly) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(monthly["month"].iloc[::step], rotation=45, fontsize=8)
    ax.set_ylabel("月均 P(AI)")
    ax.set_title(f"探索集月度标题/正文风格轨迹（标题×正文 ρ={dyad_rho:.2f}）")
    ax.legend()

    ax = axes[1]
    y = np.arange(len(model_rank))
    colors = [FAMILY_COLOR.get(FAMILY_NAME.get(f, f), "#888888")
              for f in model_rank["family"]]
    ax.barh(y, model_rank["title_mean"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(model_rank["model"], fontsize=8)
    ax.axvline(float(model_rank["aigc_overall"].iloc[0]), color="#555555",
               ls="--", lw=1.0, label="AIGC 总体均值")
    ax.set_xlabel("平均标题 AI 得分（越低越像人）")
    ax.set_title("17 个 LLM 的标题像人度")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save_fig(fig, INSIGHTS + "title_dynamics.png")


def plot_formulaicity(tpl_stats: dict, decile: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    keys = ("human", "aigc", "exploring")
    names = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)", "exploring": "探索集(后LLM)"}
    colors = {"human": "#4C72B0", "aigc": "#C44E52", "exploring": "#55A868"}
    ax = axes[0]
    cov = [tpl_stats[k]["coverage_topk"] for k in keys]
    bars = ax.bar([names[k] for k in keys], cov, color=[colors[k] for k in keys],
                  width=0.55)
    ax.set_ylabel(f"标题首 {TITLE_PREFIX_LEN} 字 Top-{TOPK} 覆盖率")
    ax.set_title("标题模板集中度")
    for b, k, v in zip(bars, keys, cov):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003,
                f"{v:.1%}\nH={tpl_stats[k]['entropy']:.1f}bit",
                ha="center", fontsize=9)

    ax = axes[1]
    x = np.arange(len(decile))
    ax.bar(x, decile["likes_median"], color="#DD8452", width=0.65)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, len(decile) + 1))
    ax.set_xlabel("标题 AI 得分十分位（1=最像人 … 10=最像AI）")
    ax.set_ylabel("点赞中位数（log 轴）")
    ax.set_title("标题 AI 得分与点赞（无单调梯度）")
    ax.annotate(f"总体 ρ={decile.attrs['rho']:+.3f}（正文端 {decile.attrs['body_rho']:+.3f}）",
                xy=(0.98, 0.95), xycoords="axes fraction", ha="right",
                fontsize=9, color="#555555")
    fig.tight_layout()
    save_fig(fig, INSIGHTS + "title_formulaicity.png")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    init_plot_style()
    human, aigc = load_features()
    df = pd.concat([human, aigc], ignore_index=True)
    meta_cols = ["dataset", "domain", "month", "model_family", "model",
                 "likes", "collections", "comments"]
    feats_all = [c for c in human.columns if c not in meta_cols]
    title_feats = [c for c in feats_all if c.startswith("title_")]
    body_feats = [c for c in feats_all
                  if c != "n_hashtags" and not c.startswith("title_")]

    # ---- 1. 检测器消融（与 paper_detector.py 相同的 CV） ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    col_idx = {c: i for i, c in enumerate(feats_all)}
    X_full = df[feats_all].fillna(0).values
    X_of = lambda cols: X_full[:, [col_idx[c] for c in cols]]
    ablation = {
        "body_only": cv_auc(df, body_feats, cv, X_of(body_feats)),
        "title_only": cv_auc(df, title_feats, cv, X_of(title_feats)),
        "full": cv_auc(df, feats_all, cv, X_full),
    }
    print("消融 AUC:", {k: round(v["auc"], 4) for k, v in ablation.items()})

    # ---- 2. 标题全量模型与三个子集打分 ----
    model = fit_title_model(df, title_feats, X_of(title_feats))
    coefs = pd.Series(model["coef"], index=title_feats).sort_values()
    exploring = pd.read_csv(BASE + "paper_features_exploring.csv.gz")

    def score(frame: pd.DataFrame) -> np.ndarray:
        return linear_score(frame, model["features"], model["scaler_mean"],
                            model["scaler_scale"], model["coef"], model["intercept"])

    corpus_stats = {}
    human_score = score(human)
    corpus_stats["human"] = {
        "n": int(len(human_score)),
        "mean": float(human_score.mean()),
        "median": float(np.median(human_score)),
        f"share_ge_{HIGH_CONF}": float((human_score >= HIGH_CONF).mean()),
    }
    aigc_score = score(aigc)
    corpus_stats["aigc"] = {
        "n": int(len(aigc_score)),
        "mean": float(aigc_score.mean()),
        "median": float(np.median(aigc_score)),
        f"share_ge_{HIGH_CONF}": float((aigc_score >= HIGH_CONF).mean()),
    }
    exploring["title_p_ai"] = score(exploring)
    exploring["month"] = exploring["month"].astype(str)
    detector_model = load_detector_model()
    exploring["body_p_ai"] = detector_score(exploring, detector_model, robust=True)
    corpus_stats["exploring"] = {
        "n": int(len(exploring)),
        "mean": float(exploring["title_p_ai"].mean()),
        "median": float(np.median(exploring["title_p_ai"])),
        f"share_ge_{HIGH_CONF}": float((exploring["title_p_ai"] >= HIGH_CONF).mean()),
    }

    # 17 个模型按标题得分的像人度排名
    model_rank = (aigc.assign(title_p_ai=aigc_score)
                  .groupby(["model_family", "model"])["title_p_ai"]
                  .agg(["mean", "median", "size"])
                  .reset_index()
                  .rename(columns={"mean": "title_mean", "median": "title_median",
                                   "size": "n"})
                  .sort_values("title_mean"))
    model_rank["family"] = model_rank["model_family"]
    model_rank["aigc_overall"] = float(aigc_score.mean())

    # ---- 3. 时间动态（探索集，2023.01 起，与 paper_engagement 口径一致） ----
    exp = exploring[exploring["month"] >= "202301"].copy()
    monthly = (exp.groupby("month")
               .agg(n=("title_p_ai", "size"), title_mean=("title_p_ai", "mean"),
                    body_mean=("body_p_ai", "mean"),
                    title_high=(f"title_p_ai", lambda s: float((s >= HIGH_CONF).mean())))
               .reset_index())
    rho_dyad, p_dyad = spearmanr(exp["title_p_ai"], exp["body_p_ai"])
    rho_trend, p_trend = spearmanr(range(len(monthly)), monthly["title_mean"])

    exp["half"] = exp["month"].map(half_of)
    halves = []
    for h in HALF_ORDER:
        g = exp[exp["half"] == h]
        if len(g) == 0:
            continue
        k_hi = int((g["title_p_ai"] >= HIGH_CONF).sum())
        lo, hi = wilson_ci(k_hi, len(g))
        halves.append({
            "half": h, "n": int(len(g)),
            "title_mean": float(g["title_p_ai"].mean()),
            "body_mean": float(g["body_p_ai"].mean()),
            f"title_share_ge_{HIGH_CONF}": float(k_hi / len(g)),
            "title_share_ci": [round(float(lo), 4), round(float(hi), 4)],
        })

    # ---- 4. 互动（口径同 paper_engagement.py） ----
    exp["log_likes"] = np.log10(1 + exp["likes"])
    exp["log_colls"] = np.log10(1 + exp["collections"])
    exp["log_comms"] = np.log10(1 + exp["comments"])
    eng = {}
    for label, col in (("likes", "log_likes"), ("collections", "log_colls"),
                       ("comments", "log_comms")):
        for name, score_col in (("title", "title_p_ai"), ("body", "body_p_ai")):
            rho, p = spearmanr(exp[score_col], exp[col])
            eng[f"{name}_{label}"] = {"rho": float(rho), "p": float(p)}

    cell_rhos, cell_ns = [], []
    for (_, _), g in exp.groupby(["month", "domain"]):
        if len(g) >= MIN_CELL_N:
            r, _ = spearmanr(g["title_p_ai"], g["log_likes"])
            if not np.isnan(r):
                cell_rhos.append(r)
                cell_ns.append(len(g))
    cell_rhos = np.asarray(cell_rhos)
    cell_ns = np.asarray(cell_ns)
    w_mean = float(np.average(cell_rhos, weights=cell_ns))
    title_cells = {
        "weighted_rho": w_mean,
        "neg_share": float((cell_rhos < 0).mean()),
        "n_cells": int(len(cell_rhos)),
    }

    exp["decile"] = pd.qcut(exp["title_p_ai"], 10, labels=False, duplicates="drop")
    decile = (exp.groupby("decile")
              .agg(likes_median=("likes", "median"), n=("likes", "size"))
              .reset_index())
    decile.attrs["rho"] = eng["title_likes"]["rho"]
    decile.attrs["body_rho"] = eng["body_likes"]["rho"]

    # ---- 5. 全量标题扫描：模板集中度与专属前缀 ----
    # 逐子集扫描后立即计数并释放标题列表，不再同时持有三个子集的全量标题。
    counts = {}
    for name, path in SCORES_INPUTS.items():
        titles, ids, metas = scan_titles(path)
        keep = dedup_indices(ids, metas) if name != "aigc" else range(len(titles))
        counts[name] = prefix_ngram_counts((titles[i] for i in keep),
                                           TITLE_PREFIX_LEN)
        print(f"{name}: {len(titles)} 条标题，去重后 {len(keep)}")
    tpl_stats = {}
    for k, c in counts.items():
        total = sum(c.values())
        tpl_stats[k] = {
            "n_texts": total,
            "coverage_topk": topk_coverage(c, TOPK),
            "entropy": shannon_entropy(c),
            "unique_ratio": (len(c) / total) if total else 0.0,
            "top": c.most_common(TOPK),
        }
    ai_templates = specific_templates(counts["aigc"], tpl_stats["aigc"]["n_texts"],
                                      counts["human"], tpl_stats["human"]["n_texts"],
                                      reverse=False, min_count=MIN_AIGC_TPL, k=10)
    human_templates = specific_templates(counts["aigc"], tpl_stats["aigc"]["n_texts"],
                                         counts["human"], tpl_stats["human"]["n_texts"],
                                         reverse=True, min_count=MIN_HUMAN_TPL, k=10)

    # 标题长度中位数（三子集，bootstrap CI）
    title_len_boot = {}
    for name, dfx in (("human", human), ("aigc", aigc), ("exploring", exploring)):
        title_len_boot[name] = bootstrap_median_ci(dfx["title_len"], n_boot=BOOT,
                                                   seed=BOOT_SEED)
    dl, dp = cliffs_delta(aigc["title_len"].values, human["title_len"].values)

    # ---- 6. 保存结果 ----
    out = {
        "detector_ablation": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                  for kk, vv in v.items()}
                              for k, v in ablation.items()},
        "title_model": model,
        "title_model_coef": {f: round(float(c), 4) for f, c in coefs.items()},
        "corpus_title_scores": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                    for kk, vv in v.items()}
                                for k, v in corpus_stats.items()},
        "model_title_ranking": [
            {"model": r["model"], "family": r["model_family"], "n": int(r["n"]),
             "title_mean": round(float(r["title_mean"]), 4),
             "title_median": round(float(r["title_median"]), 4)}
            for _, r in model_rank.iterrows()
        ],
        "monthly": [
            {"month": r["month"], "n": int(r["n"]),
             "title_mean": round(float(r["title_mean"]), 4),
             "body_mean": round(float(r["body_mean"]), 4),
             "title_high_share": round(float(r["title_high"]), 4)}
            for _, r in monthly.iterrows()
        ],
        "title_body_spearman": {"rho": round(float(rho_dyad), 4), "p": float(p_dyad)},
        "monthly_trend": {"rho": round(float(rho_trend), 4), "p": float(p_trend)},
        "half_years": halves,
        "engagement": {k: {"rho": round(v["rho"], 4), "p": float(v["p"])}
                       for k, v in eng.items()},
        "engagement_within_cells": {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in title_cells.items()},
        "engagement_deciles": [
            {"decile": int(r["decile"]) + 1, "n": int(r["n"]),
             "likes_median": float(r["likes_median"])}
            for _, r in decile.iterrows()
        ],
        "title_prefix_stats": {k: {"n_texts": v["n_texts"],
                                   "coverage_topk": round(v["coverage_topk"], 4),
                                   "entropy": round(v["entropy"], 3),
                                   "unique_ratio": round(v["unique_ratio"], 4)}
                               for k, v in tpl_stats.items()},
        "ai_specific_templates": ai_templates,
        "human_specific_templates": human_templates,
        "title_len_median": title_len_boot,
        "title_len_cliffs_delta_aigc_vs_human": round(float(dl), 4),
        "title_len_mannwhitney_p": float(dp),
    }
    save_json(out, BASE + "title_style_results.json")

    plot_detector(ablation, coefs)
    plot_dynamics(monthly, model_rank, rho_dyad)
    plot_formulaicity(tpl_stats, decile)

    write_markdown(out)
    print("已保存 title_style_results.json / title_style_results.md 与 insights/title_*.png")


def write_markdown(out: dict) -> None:
    ab = out["detector_ablation"]
    names = {"body_only": "正文特征（29，稳健集）", "title_only": "标题特征（7）",
             "full": "全部特征（37）"}
    lines = [
        "# 标题端「AI 味」分析结果（Title Stylometry）",
        "",
        "> 由 `title_style_analysis.py` 基于论文管线特征运行生成；检测器消融与",
        "> `paper_detector.py` 使用完全相同的 5 折分层交叉验证口径（同一随机种子）；",
        "> 标题模板统计的人类/探索集按 note_id 去重（规则同 `deep_dive_analysis.py`），",
        "> AIGC 无 note_id 逐记录处理；探索集时间/互动分析从 2023.01 起，",
        "> 与 `paper_engagement.py` 口径一致。",
        "",
        "## 1 标题单独的可分性（human vs AIGC，5 折 CV）",
        "",
        "| 特征集 | 特征数 | AUC | F1 | 准确率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in ("body_only", "title_only", "full"):
        v = ab[k]
        lines.append(f"| {names[k]} | {v['n_features']} | **{v['auc']:.3f}** | "
                     f"{v['f1']:.3f} | {v['acc']:.3f} |")
    lines += [
        "",
        f"仅 7 个标题特征即可达到正文 29 特征模型 AUC 的 "
        f"{ab['title_only']['auc']/ab['body_only']['auc']:.1%}；标题特征权重"
        "（标准化系数）见 `insights/title_detector.png`。",
        "",
        "## 2 三个子集的标题 AI 得分（全量模型）",
        "",
        "> 说明：human/aigc 为模型训练集内得分（与 paper_drift.py 的基线口径一致），",
        "> 探索集为训练分布外的跨时代得分。",
        "",
        "| 子集 | n | 均值 | 中位数 | ≥0.9 占比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, label in (("human", "人类(前LLM)"), ("aigc", "AIGC(17模型)"),
                     ("exploring", "探索集(后LLM)")):
        s = out["corpus_title_scores"][k]
        lines.append(f"| {label} | {s['n']:,} | {s['mean']:.3f} | {s['median']:.3f} | "
                     f"{s['share_ge_0.9']:.1%} |")
    lines += [
        "",
        "## 3 17 个 LLM 的标题像人度（平均标题得分升序）",
        "",
        "| 模型 | 家族 | n | 均值 | 中位数 |",
        "|---|---|---:|---:|---:|",
    ]
    for r in out["model_title_ranking"]:
        lines.append(f"| {r['model']} | {FAMILY_NAME.get(r['family'], r['family'])} | "
                     f"{r['n']} | {r['title_mean']:.3f} | {r['title_median']:.3f} |")
    dyad = out["title_body_spearman"]
    trend = out["monthly_trend"]
    lines += [
        "",
        "## 4 时间动态与互动（探索集 2023.01–2025.06）",
        "",
        f"- 标题得分与正文得分的秩相关仅 ρ={dyad['rho']:.3f}（p={dyad['p']:.2e}）——"
        "标题端与正文端的 AI 风格几乎是两个独立的信号，只看正文会漏掉标题端信息。",
        f"- 月度标题得分无显著时间趋势（ρ={trend['rho']:.3f}，p={trend['p']:.3f}）；"
        "半年度标题≥0.9 占比仅 0.6%–1.2%，远低于正文端的高置信台阶"
        "（前 LLM 基线 0.65% → 后 LLM 17%–18%），说明标题端的风格跃升明显弱于正文端。",
        "",
        "**半年度对比：**",
        "",
        "| 时期 | n | 标题均值 | 正文均值 | 标题≥0.9 占比（95%CI） |",
        "|---|---:|---:|---:|---:|",
    ]
    for h in out["half_years"]:
        ci = h["title_share_ci"]
        lines.append(f"| {h['half']} | {h['n']:,} | {h['title_mean']:.3f} | "
                     f"{h['body_mean']:.3f} | {h['title_share_ge_0.9']:.1%} "
                     f"({ci[0]:.1%}–{ci[1]:.1%}) |")
    eng = out["engagement"]
    cells = out["engagement_within_cells"]
    lines += [
        "",
        f"- 标题得分与点赞：总体 ρ={eng['title_likes']['rho']:+.3f}，"
        f"控制月份×领域后加权 ρ={cells['weighted_rho']:+.3f}、负单元占比 "
        f"{cells['neg_share']:.0%}（{cells['n_cells']} 个单元）——标题端**不存在**"
        f"正文端那样的『AI 味惩罚』（正文对照 ρ={eng['body_likes']['rho']:+.3f}）。"
        "这说明互动惩罚主要由正文端的风格与信息结构承载，标题的营销式写法"
        "（感叹号/emoji）并不直接压低互动。",
        f"- 收藏：标题 ρ={eng['title_collections']['rho']:+.3f} / "
        f"正文 ρ={eng['body_collections']['rho']:+.3f}；"
        f"评论：标题 ρ={eng['title_comments']['rho']:+.3f} / "
        f"正文 ρ={eng['body_comments']['rho']:+.3f}。",
        "",
        "**标题得分十分位 vs 点赞中位数：**",
        "",
        "| 十分位 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    dec = out["engagement_deciles"]
    lines.append("| 点赞中位数 | " + " | ".join(
        f"{int(d['likes_median'])}" for d in dec) + " |")
    lines += [
        "",
        "## 5 标题模板化（首 4 字 Top-20）",
        "",
        "| 子集 | 参与统计 | 覆盖率 | 前缀熵(bit) | 唯一率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, label in (("human", "人类(前LLM)"), ("aigc", "AIGC(17模型)"),
                     ("exploring", "探索集(后LLM)")):
        s = out["title_prefix_stats"][k]
        lines.append(f"| {label} | {s['n_texts']:,} | {s['coverage_topk']:.1%} | "
                     f"{s['entropy']:.2f} | {s['unique_ratio']:.3f} |")
    lines += [
        "",
        "**AI 专属标题前缀（按 AIGC 每帖发生率之差）：**",
        "",
    ]
    for t in out["ai_specific_templates"][:5]:
        lines.append(f"- `{t['word']}`：AIGC {t['aigc_per1k']:.1f}/千帖 vs "
                     f"人类 {t['human_per1k']:.1f}/千帖（差 {t['rate_diff_ppt']:+.2f}pp）")
    lines += ["", "**人类专属标题前缀：**", ""]
    for t in out["human_specific_templates"][:5]:
        lines.append(f"- `{t['word']}`：人类 {t['human_per1k']:.1f}/千帖 vs "
                     f"AIGC {t['aigc_per1k']:.1f}/千帖（差 {t['rate_diff_ppt']:+.2f}pp）")
    tl = out["title_len_median"]
    lines += [
        "",
        "对照：正文开篇（framing_analysis）AIGC 首 4 字 Top-20 覆盖率 14.8%、"
        "熵 11.86 bit，远高于标题端（3.8% / 12.46 bit）——AI 生成的模板化主要"
        "集中在正文框架，标题反而相对多样。",
        "",
        f"标题长度中位数：人类 {tl['human']['median']:.0f} 字、"
        f"AIGC {tl['aigc']['median']:.0f} 字、探索集 {tl['exploring']['median']:.0f} 字"
        f"（AIGC vs 人类 Cliff's δ={out['title_len_cliffs_delta_aigc_vs_human']:+.3f}，"
        f"p={out['title_len_mannwhitney_p']:.1e}）。",
        "",
        "图表见 `insights/`：`title_detector.png`、`title_dynamics.png`、"
        "`title_formulaicity.png`。",
    ]
    with open(BASE + "title_style_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
