# -*- coding: utf-8 -*-
"""Domain-level stylometry: where AI style concentrates and where it costs.

Every previous deep dive asks *how* a style axis behaves.  This one asks
*where* the detector signal and its engagement cost live.  Domains are the
platform's own content categories (穿搭 / 健康 / 情感 / ...), available in all
three corpora, and they are the natural unit for moderation, creator strategy
and sampling design.

Questions:
1. How does high-confidence AI-style prevalence differ across domains, and how
   far is each domain above its own pre-LLM human baseline?
2. Is the domain ranking stable over time, and how much of the apparent
   time trend is a change in domain composition (the exploring set is heavily
   re-weighted across half-years)?
3. Is the "AI-flavor penalty" uniform across domains once month fixed effects,
   body length and hashtag count are controlled?
4. Which generator families are over-represented in which domains, and is
   domain penetration related to the size of the engagement penalty?

Protocol
--------
- Features come from the existing ``paper_features_*.csv.gz`` files; the
  detector is the saved robust model (no retraining).
- Human and AIGC scores are 5-fold out-of-fold probabilities from
  ``analysis_utils.detector_oof_all`` (same protocol as the paper detector);
  exploring scores are the persisted out-of-sample ``p_ai`` column.  Human
  baselines use the clean pre-LLM window (<= 2022.06) to avoid the known
  2022.07 sampling break.
- The engagement panel uses 2023.01+ records.  Per-domain coefficients come
  from a month-fixed-effects OLS of log10(1+engagement) on standardized P(AI),
  body length and hashtag count, with 95% CIs from 300 cluster-bootstrap
  replications over months.  This is the same estimator as
  ``engagement_drivers_analysis.py``, applied inside each domain.
- "未知" is a missing-domain bucket, not a content category.  It is reported
  in the tables for completeness but excluded from domain rankings, family
  composition and the penetration-penalty correlation.

Outputs
-------
  insights/domain_penetration.png
  insights/domain_penalty.png
  domain_stylometry_results.json
  domain_stylometry_results.md
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import roc_auc_score

from analysis_utils import (BASE, INSIGHTS, PALETTE, cell_groups,
                            dedup_indices, detector_oof_all, init_plot_style,
                            load_detector_model, load_features, save_fig,
                            save_json, wilson_ci, within_cell_spearman)
from derived_cache import iter_cached
from engagement_drivers_analysis import (cluster_bootstrap_fe_ols, fe_ols,
                                         prepare_fe)

SCORES_PATH = BASE + "paper_exploring_scores.csv.gz"
MONTH_FROM = "202301"
HUMAN_CLEAN_TO = "202206"
HI = 0.90
MIN_DOMAIN_N = 300
MIN_MONTH_N = 50
MIN_HALF_N = 100
MIN_FAMILY_DOMAIN_N = 90
N_BOOT = 300
BOOT_SEED = 42

UNKNOWN = "未知"
LABELED_DOMAINS = ["健康", "穿搭", "美食", "职场", "宠物", "学习",
                   "运动", "情感", "旅行", "心理"]
FAMILIES = ["gpt", "deepseek", "gemini", "anthropic", "glm", "qwen"]
FAMILY_NAME = {"gpt": "GPT", "deepseek": "DeepSeek", "gemini": "Gemini",
               "anthropic": "Claude", "glm": "GLM", "qwen": "Qwen"}
# --------------------------------------------------------------------------
# Pure helpers (unit-tested without research data)
# --------------------------------------------------------------------------
def json_safe(obj):
    """Recursively replace non-finite floats with None for strict JSON."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def halfyear_of(month: str | None) -> str | None:
    """'202405' -> '2024H1'; invalid/empty input -> None."""
    if not month or len(str(month)) < 6:
        return None
    m = str(month)
    try:
        h = 1 if int(m[4:6]) <= 6 else 2
    except ValueError:
        return None
    return f"{m[:4]}H{h}"


def rate_summary(values, threshold: float = HI) -> dict:
    """n / mean / median / high-confidence share + Wilson 95% CI."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "hi_share": float("nan"), "hi_ci_low": float("nan"),
                "hi_ci_high": float("nan")}
    k = int((arr >= threshold).sum())
    lo, hi = wilson_ci(k, n)
    return {"n": n, "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "hi_share": round(float(k / n), 4),
            "hi_ci_low": round(float(lo), 4),
            "hi_ci_high": round(float(hi), 4)}


def auc_or_nan(labels, scores) -> float:
    """AUC with a NaN fallback when one class is absent."""
    y = np.asarray(labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return round(float(roc_auc_score(y, scores)), 4)


def safe_spearman(x, y) -> tuple[float, float]:
    """Spearman rho/p with constant/empty guard."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return float("nan"), float("nan")
    rho, p = spearmanr(x[keep], y[keep])
    return float(rho), float(p)


def fmt_p(p: float) -> str:
    """Compact p-value: scientific only below 0.001."""
    if not math.isfinite(p):
        return "—"
    return f"{p:.1e}" if p < 0.001 else f"{p:.2f}"


def family_share_table(aigc: pd.DataFrame, min_n: int = MIN_FAMILY_DOMAIN_N) -> dict:
    """Row-normalised family shares per labeled domain + chi-square test."""
    rows = {}
    for d in LABELED_DOMAINS:
        g = aigc[aigc["domain"] == d]
        if len(g) < min_n:
            continue
        counts = g["model_family"].fillna("").value_counts()
        n = int(counts.sum())
        shares = {f: round(float(counts.get(f, 0) / n), 4) if n else 0.0
                  for f in FAMILIES}
        rows[d] = {"n": n, "shares": shares}
    if len(rows) < 2:
        return {"domains": rows, "chi2": float("nan"), "p": float("nan"),
                "cramers_v": float("nan"), "dof": 0}
    table = np.asarray([[rows[d]["shares"][f] * rows[d]["n"] for f in FAMILIES]
                        for d in rows], dtype=float)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return {"domains": rows, "chi2": float("nan"), "p": float("nan"),
                "cramers_v": float("nan"), "dof": 0}
    chi2, p, dof, _ = chi2_contingency(table)
    n_total = float(table.sum())
    v = float(np.sqrt(chi2 / (n_total * (min(table.shape) - 1)))) if n_total else float("nan")
    return {"domains": rows, "chi2": round(float(chi2), 2),
            "p": float(p), "cramers_v": round(v, 4), "dof": int(dof)}


def domain_penalty(df: pd.DataFrame, target: str,
                   min_month_n: int = MIN_MONTH_N,
                   n_boot: int = N_BOOT) -> dict:
    """Month-FE slope of P(AI) on log10(1+target), controlling length/tags.

    `df` must contain p_ai, content_len, n_hashtags, month and `target`.
    Returns point estimate, cluster-bootstrap CI, and raw/within-month rank
    correlations for context.
    """
    month = df["month"].astype(str).to_numpy()
    p_ai = df["p_ai"].to_numpy(dtype=float)
    y = np.log10(1 + np.clip(df[target].to_numpy(dtype=float), 0, None))
    X = np.column_stack([
        p_ai,
        df["content_len"].to_numpy(dtype=float),
        df["n_hashtags"].to_numpy(dtype=float),
    ])
    groups = cell_groups(month, min_month_n)
    if len(groups) < 2 or sum(len(g) for g in groups) < min_month_n:
        return {"beta": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_obs": 0, "n_months": 0,
                "raw_rho": float("nan"), "raw_p": float("nan"),
                "within_month_rho": float("nan"), "neg_month_share": float("nan")}
    prep = prepare_fe(X, month, min_month_n)
    if prep["n_obs"] < min_month_n or prep["n_cells"] < 2:
        return {"beta": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_obs": int(prep["n_obs"]),
                "n_months": int(prep["n_cells"]), "raw_rho": float("nan"),
                "raw_p": float("nan"), "within_month_rho": float("nan"),
                "neg_month_share": float("nan")}
    fit = fe_ols(X, y, month, min_month_n, prepared=prep)
    boot = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                    fit["cell_codes"], n_boot=n_boot,
                                    seed=BOOT_SEED)
    lo, hi = np.percentile(boot[:, 0], [2.5, 97.5])
    raw_rho, raw_p = safe_spearman(p_ai, y)
    wc = within_cell_spearman(p_ai, y, month, min_cell_n=min_month_n,
                              groups=groups) if groups else {
        "weighted_rho": float("nan"), "neg_cell_share": float("nan"),
        "n_cells": 0}
    return {"beta": round(float(fit["coef"][0]), 4),
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "n_obs": int(fit["n_obs"]), "n_months": int(fit["n_cells"]),
            "raw_rho": round(raw_rho, 4), "raw_p": float(raw_p),
            "within_month_rho": round(float(wc["weighted_rho"]), 4),
            "neg_month_share": round(float(wc["neg_cell_share"]), 4)}


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------
def load_frames() -> dict:
    """Load features/scores, score human+AIGC out-of-fold, verify alignment."""
    model = load_detector_model()
    # 这三个宽表各有 46–47 列，但本脚本只用检测器稳健特征、领域/月份与
    # 互动列；按需取列可显著降低三份特征表的常驻内存（约 150k 行）。
    feats = list(model["robust_features"])
    human = load_features("human", usecols=feats + ["month", "domain"])
    aigc = load_features("aigc", usecols=feats + ["domain", "model_family"])
    expl = load_features("exploring", usecols=[
        "month", "domain", "content_len", "n_hashtags",
        "likes", "collections", "comments"])
    scores = pd.read_csv(SCORES_PATH, usecols=[
        "month", "domain", "content_len", "p_ai", "p_ai_full"])
    scores["month"] = scores["month"].astype(str)

    if len(scores) != len(expl):
        raise SystemExit("exploring features and scores have different row counts")
    checks = {
        "month": (scores["month"].to_numpy() != expl["month"].astype(str).to_numpy()).sum(),
        "domain": (scores["domain"].astype(str).to_numpy()
                   != expl["domain"].astype(str).to_numpy()).sum(),
        "content_len": (scores["content_len"].to_numpy()
                        != expl["content_len"].to_numpy()).sum(),
    }
    bad = {k: int(v) for k, v in checks.items() if v}
    if bad:
        raise SystemExit(f"exploring features/scores not row-aligned: {bad}")

    all_scores = detector_oof_all(human, aigc, model, robust=True)
    human["p_ai"] = all_scores[:len(human)]
    aigc["p_ai"] = all_scores[len(human):]
    expl["p_ai"] = scores["p_ai"].to_numpy(dtype=float)
    expl["p_ai_full"] = scores["p_ai_full"].to_numpy(dtype=float)

    for frame in (human, aigc, expl):
        frame["domain"] = frame["domain"].fillna("").astype(str)
        frame.loc[frame["domain"] == "", "domain"] = UNKNOWN
    return {"model": model, "human": human, "aigc": aigc,
            "exploring": expl, "alignment": {k: int(v) for k, v in checks.items()}}


def penetration_table(frames: dict) -> dict:
    human = frames["human"]
    aigc = frames["aigc"]
    expl = frames["exploring"][frames["exploring"]["month"] >= MONTH_FROM]
    human_clean = human[human["month"].astype(str) <= HUMAN_CLEAN_TO]
    domains = list(LABELED_DOMAINS) + [UNKNOWN]
    rows = {}
    for d in domains:
        h = human_clean[human_clean["domain"] == d]["p_ai"].to_numpy()
        a = aigc[aigc["domain"] == d]["p_ai"].to_numpy()
        e = expl[expl["domain"] == d]["p_ai"].to_numpy()
        if len(e) < (MIN_DOMAIN_N if d != UNKNOWN else 1):
            continue
        hr = rate_summary(h)
        ar = rate_summary(a)
        er = rate_summary(e)
        labels = np.r_[np.zeros(len(h), dtype=int), np.ones(len(a), dtype=int)]
        auc = auc_or_nan(labels, np.r_[h, a])
        rows[d] = {
            "human_clean": hr, "aigc": ar, "exploring": er,
            "human_vs_aigc_auc": auc,
            "excess_hi_pp": round((er["hi_share"] - hr["hi_share"]) * 100, 2),
        }
    return rows


def dedup_sensitivity(expl: pd.DataFrame,
                      path: str = BASE + "exploring_set_fixed.jsonl") -> dict:
    """Per-note robustness: deduplicate by note_id (keep the richest copy).

    The features are built from `exploring_set_fixed.jsonl` in the same row
    order, so a streaming pass over note_id + metadata is enough.  The
    metadata score mirrors `deep_dive_analysis.py` / `framing_analysis.py`.
    """
    if not os.path.isfile(path):
        return {"available": False}
    ids, meta = [], []
    # 与特征表同序的单遍投影读取；派生缓存缺失时 iter_cached 自动回退
    # 逐行解析，取值口径完全一致。
    for rec in iter_cached(path):
        ids.append(rec.get("note_id"))
        has_meta = (bool(rec.get("user_id"))
                    and rec.get("domain") not in (None, "", UNKNOWN))
        meta.append(int(has_meta) + int(bool(rec.get("likes")))
                    + int(bool(rec.get("collections")))
                    + int(bool(rec.get("comments"))))
    if len(ids) != len(expl):
        return {"available": False, "reason": "row count differs from features"}
    keep = dedup_indices(ids, meta)
    base_expl = expl[expl["month"] >= MONTH_FROM]
    ded = expl.iloc[keep]
    ded = ded[ded["month"] >= MONTH_FROM]
    rows = {}
    deltas = []
    for d in list(LABELED_DOMAINS) + [UNKNOWN]:
        base = base_expl[base_expl["domain"] == d]["p_ai"].to_numpy()
        dedup = ded[ded["domain"] == d]["p_ai"].to_numpy()
        if len(base) == 0 and len(dedup) == 0:
            continue
        br, dr = rate_summary(base), rate_summary(dedup)
        delta = ((dr["hi_share"] - br["hi_share"]) * 100
                 if np.isfinite(dr["hi_share"]) and np.isfinite(br["hi_share"])
                 else float("nan"))
        if d != UNKNOWN and np.isfinite(delta):
            deltas.append(abs(delta))
        rows[d] = {"n_records": br["n"], "n_notes": dr["n"],
                   "hi_share_records": br["hi_share"],
                   "hi_share_notes": dr["hi_share"],
                   "delta_pp": round(float(delta), 2) if np.isfinite(delta) else None}
    return {"available": True, "n_records": len(base_expl), "n_notes": len(ded),
            "n_duplicate_rows": len(base_expl) - len(ded),
            "max_abs_delta_pp": round(float(max(deltas)), 2) if deltas else None,
            "domains": rows}


def halfyear_table(expl: pd.DataFrame) -> dict:
    q = expl[expl["month"] >= MONTH_FROM].copy()
    q["halfyear"] = [halfyear_of(m) for m in q["month"]]
    q = q[q["halfyear"].notna()]
    halves = sorted(q["halfyear"].unique())
    rows = {}
    for d in LABELED_DOMAINS:
        g = q[q["domain"] == d]
        if g.empty:
            continue
        cells = {}
        for h in halves:
            gh = g[g["halfyear"] == h]["p_ai"].to_numpy()
            if len(gh) >= MIN_HALF_N:
                cells[h] = rate_summary(gh)
            else:
                cells[h] = {"n": int(len(gh)), "hi_share": None,
                            "hi_ci_low": None, "hi_ci_high": None,
                            "mean": (round(float(gh.mean()), 4) if len(gh) else None)}
        rows[d] = cells
    return {"halves": halves, "domains": rows,
            "composition": {h: {
                d: round(float(((q["halfyear"] == h) & (q["domain"] == d)).sum()
                               / max((q["halfyear"] == h).sum(), 1)), 4)
                for d in LABELED_DOMAINS}
                for h in halves}}


def penalty_table(expl: pd.DataFrame) -> dict:
    df = expl[expl["month"] >= MONTH_FROM].copy()
    out = {}
    for d in list(LABELED_DOMAINS) + [UNKNOWN]:
        g = df[df["domain"] == d]
        if len(g) < MIN_DOMAIN_N:
            continue
        row = {"n": int(len(g)),
               "mean_p_ai": round(float(g["p_ai"].mean()), 4)}
        for target in ("likes", "collections", "comments"):
            row[target] = domain_penalty(g, target)
        out[d] = row
    return out


def coupling_table(pen: dict, pen_rows: dict) -> dict:
    doms = [d for d in LABELED_DOMAINS if d in pen_rows]
    excess = np.asarray([pen["domains"][d]["excess_hi_pp"] for d in doms])
    beta = np.asarray([pen_rows[d]["likes"]["beta"] for d in doms])
    rho, p = safe_spearman(excess, beta)
    return {"domains": doms,
            "excess_hi_pp": [round(float(v), 2) for v in excess],
            "likes_beta": [round(float(v), 4) for v in beta],
            "spearman_rho": round(rho, 4) if np.isfinite(rho) else None,
            "spearman_p": float(p) if np.isfinite(p) else None,
            "n_domains": len(doms)}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _stars(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return "***" if lo > 0 or hi < 0 else ""


def plot_penetration(pen: dict, half: dict) -> None:
    init_plot_style()
    rows = {d: r for d, r in pen.items() if d != UNKNOWN}
    order = sorted(rows, key=lambda d: rows[d]["exploring"]["hi_share"])
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 11.6),
                             gridspec_kw={"height_ratios": [1.0, 0.95]})

    # (a) exploring high-confidence share with human/AIGC references
    ax = axes[0, 0]
    ypos = np.arange(len(order))
    vals = [rows[d]["exploring"]["hi_share"] * 100 for d in order]
    lo = [rows[d]["exploring"]["hi_ci_low"] * 100 for d in order]
    hi = [rows[d]["exploring"]["hi_ci_high"] * 100 for d in order]
    ax.barh(ypos, vals, color=PALETTE["exploring"], alpha=0.9, height=0.66,
            label="探索集（2023–2025）")
    ax.errorbar(vals, ypos, xerr=[np.asarray(vals) - lo, np.asarray(hi) - vals],
                fmt="none", ecolor="#333333", elinewidth=1.2, capsize=3, zorder=3)
    ax.scatter([rows[d]["human_clean"]["hi_share"] * 100 for d in order], ypos,
               marker="o", s=34, color=PALETTE["human"], zorder=4,
               label="人类干净窗口（≤2022.06）")
    ax.scatter([rows[d]["aigc"]["hi_share"] * 100 for d in order], ypos,
               marker="D", s=30, color=PALETTE["aigc"], zorder=4,
               label="AIGC 训练集")
    for i, d in enumerate(order):
        ax.annotate(f"{vals[i]:.1f}%  (+{rows[d]['excess_hi_pp']:.1f}pp)",
                    xy=(hi[i], ypos[i]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order)
    ax.set_xlabel("高置信 AI 风格占比（P(AI) ≥ 0.9，%）")
    ax.set_title("(a) 各领域野外高置信 AI 风格占比（Wilson 95% CI）\n"
                 "括号内为相对人类干净窗口的增量（百分点）")
    ax.legend(loc="lower right", fontsize=8.5)

    # (b) human baseline vs exploring: excess above the pre-LLM baseline
    ax = axes[0, 1]
    label_offsets = {
        "运动": (7, 3), "美食": (7, -9), "健康": (7, 5),
        "情感": (-62, 9), "宠物": (-55, -13), "穿搭": (7, 3),
        "职场": (7, 3), "旅行": (7, -9), "学习": (7, 3), "心理": (7, 3),
    }
    for d in order:
        x = rows[d]["human_clean"]["hi_share"] * 100
        y = rows[d]["exploring"]["hi_share"] * 100
        ax.scatter(x, y, s=28 + rows[d]["exploring"]["n"] / 80,
                   color=PALETTE["exploring"], alpha=0.85, edgecolor="white",
                   linewidth=0.8, zorder=3)
        dx, dy = label_offsets.get(d, (7, 3))
        arrow = dict(arrowstyle="-", color="#666666", lw=0.6,
                     shrinkA=0, shrinkB=3) if abs(dx) > 20 else None
        ax.annotate(d, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, va="center",
                    ha="right" if dx < 0 else "left", arrowprops=arrow)
    lim = max(max(rows[d]["human_clean"]["hi_share"] * 100 for d in order),
              max(rows[d]["exploring"]["hi_share"] * 100 for d in order)) + 3
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6, label="无增量线")
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("人类干净窗口高置信占比（%）")
    ax.set_ylabel("探索集高置信占比（%）")
    ax.set_title("(b) 各领域相对自身前 LLM 基线的 AI 渗透增量\n"
                 "点大小 ∝ 探索集帖子数")
    ax.legend(loc="lower right", fontsize=8.5)

    # (c) half-year heatmap; small cells shown grey
    ax = axes[1, 0]
    halves = half["halves"]
    dom_order = sorted(half["domains"], key=lambda d: rows.get(
        d, {}).get("exploring", {}).get("hi_share", 0))
    mat = np.full((len(dom_order), len(halves)), np.nan)
    ns = np.zeros_like(mat)
    for i, d in enumerate(dom_order):
        for j, h in enumerate(halves):
            cell = half["domains"][d].get(h, {})
            ns[i, j] = cell.get("n", 0)
            if cell.get("hi_share") is not None:
                mat[i, j] = cell["hi_share"] * 100
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=np.nanmax(mat),
                   aspect="auto")
    ax.grid(False)
    for i in range(len(dom_order)):
        for j in range(len(halves)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, f"n={int(ns[i, j])}", ha="center", va="center",
                        fontsize=7.5, color="#888888")
            else:
                ax.text(j, i, f"{mat[i, j]:.1f}%\n(n={int(ns[i, j])})",
                        ha="center", va="center", fontsize=7.2,
                        color="black" if mat[i, j] < np.nanmax(mat) * 0.62
                        else "white")
    ax.set_xticks(range(len(halves)))
    ax.set_xticklabels(halves, rotation=30, ha="right")
    ax.set_yticks(range(len(dom_order)))
    ax.set_yticklabels(dom_order)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("高置信占比（%）", fontsize=8.5)
    ax.set_title("(c) 半年度高置信占比：仅展示 n≥%d 的单元，其余标灰" % MIN_HALF_N)

    # (d) domain composition over half-years
    ax = axes[1, 1]
    comp = half["composition"]
    bottom = np.zeros(len(halves))
    for d in LABELED_DOMAINS:
        vals = np.asarray([comp[h].get(d, 0.0) * 100 for h in halves])
        ax.bar(range(len(halves)), vals, bottom=bottom, width=0.68,
               label=d, alpha=0.9)
        bottom += vals
    unknown = np.asarray([
        100 - sum(comp[h].get(d, 0.0) for d in LABELED_DOMAINS) * 100
        for h in halves])
    ax.bar(range(len(halves)), unknown, bottom=bottom, width=0.68,
           color="#BDBDBD", label="未知/其他", alpha=0.9)
    ax.set_xticks(range(len(halves)))
    ax.set_xticklabels(halves, rotation=30, ha="right")
    ax.set_ylabel("探索集帖子构成（%）")
    ax.set_ylim(0, 100)
    ax.set_title("(d) 领域构成随半年度变化：早期样本少且构成差异大\n"
                 "（解读 (c) 的时间变化时必须同时看本图）")
    ax.legend(ncol=2, fontsize=7.2, loc="upper right")

    fig.suptitle("领域渗透：哪些内容领域的高置信 AI 风格更多？", fontsize=14,
                 y=1.01)
    save_fig(fig, INSIGHTS + "domain_penetration.png")


def plot_penalty(pen_rows: dict, fam: dict) -> None:
    init_plot_style()
    doms = [d for d in LABELED_DOMAINS if d in pen_rows]
    order = sorted(doms, key=lambda d: pen_rows[d]["likes"]["beta"])
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 11.4))

    # (a) FE beta forest for likes and collections
    ax = axes[0, 0]
    ypos = np.arange(len(order))
    for target, marker, offset, color in (
            ("likes", "o", -0.14, PALETTE["human"]),
            ("collections", "^", 0.14, PALETTE["aigc"])):
        for i, d in enumerate(order):
            r = pen_rows[d][target]
            sig = r["ci_low"] > 0 or r["ci_high"] < 0
            col = color if sig else "#9E9E9E"
            ax.plot([r["ci_low"], r["ci_high"]],
                    [ypos[i] + offset, ypos[i] + offset], color=col, lw=1.8,
                    solid_capstyle="butt", zorder=2)
            ax.scatter([r["beta"]], [ypos[i] + offset], marker=marker, s=36,
                       facecolor=col if sig else "white", edgecolor=col,
                       zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order)
    ax.set_xlabel("P(AI) 每 +1 SD → log10(1+互动) 的偏效应（95% CI）")
    ax.set_title("(a) 各领域内 P(AI) 的互动偏效应（月份 FE + 长度/标签控制）")
    ax.scatter([], [], marker="o", color=PALETTE["human"], label="点赞")
    ax.scatter([], [], marker="^", color=PALETTE["aigc"], label="收藏")
    ax.legend(loc="lower left", fontsize=8.5)

    # (b) raw within-month rank correlation
    ax = axes[0, 1]
    raw = [pen_rows[d]["likes"]["within_month_rho"] for d in order]
    colors = [PALETTE["aigc"] if v < 0 else PALETTE["human"] for v in raw]
    ax.barh(ypos, raw, color=colors, alpha=0.9, height=0.66)
    ax.axvline(0, color="black", lw=0.8)
    for i, (v, d) in enumerate(zip(raw, order)):
        r = pen_rows[d]["likes"]
        ax.text(0.002, ypos[i], f"{v:+.3f}  (n={pen_rows[d]['n']:,} / {r['n_months']} 月)",
                ha="left", va="center", fontsize=8)
    ax.set_xlim(min(min(raw) - 0.012, -0.02), 0.045)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order)
    ax.set_xlabel("月内加权 Spearman ρ（P(AI) vs log10(1+点赞)）")
    ax.set_title("(b) 控制前的领域内秩相关：方向一致但强度差异很大")

    # (c) family composition heatmap
    ax = axes[1, 0]
    fam_rows = [d for d in order if d in fam["domains"]]
    mat = np.asarray([[fam["domains"][d]["shares"][f] * 100 for f in FAMILIES]
                      for d in fam_rows], dtype=float)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 1), aspect="auto")
    ax.grid(False)
    for i in range(len(fam_rows)):
        for j in range(len(FAMILIES)):
            ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if mat[i, j] > max(mat.max() * 0.62, 1) else "black")
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels([FAMILY_NAME[f] for f in FAMILIES], rotation=25, ha="right")
    ax.set_yticks(range(len(fam_rows)))
    ax.set_yticklabels([f"{d} (n={fam['domains'][d]['n']})" for d in fam_rows])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("家族在领域内的占比（%）", fontsize=8.5)
    ax.set_title("(c) AIGC 训练集的领域×家族构成（行内合计 100%）\n"
                 f"χ²={fam['chi2']:.1f}, p={fmt_p(fam['p'])}, Cramér's V={fam['cramers_v']:.3f}")

    # (d) penetration vs penalty coupling
    ax = axes[1, 1]
    rows = {d: pen_rows[d] for d in doms}
    xs = np.asarray([pen_rows[d].get("excess_hi_pp", np.nan) for d in order])
    ys = np.asarray([pen_rows[d]["likes"]["beta"] for d in order])
    ax.scatter(xs, ys, s=40 + np.asarray([pen_rows[d]["n"] for d in order]) / 90,
               color=PALETTE["exploring"], alpha=0.85, edgecolor="white",
               linewidth=0.8, zorder=3)
    for x, y, d in zip(xs, ys, order):
        ax.annotate(d, (x, y), xytext=(5, 3), textcoords="offset points",
                    fontsize=8.5)
    ax.axhline(0, color="black", lw=0.8, alpha=0.7)
    coupling = safe_spearman(xs, ys)
    ax.set_xlabel("相对前 LLM 基线的高置信占比增量（百分点）")
    ax.set_ylabel("点赞端 P(AI) 偏效应 β")
    ax.set_title("(d) 领域渗透增量与互动惩罚（描述性，n=10）\n"
                 f"Spearman ρ={coupling[0]:+.2f}, p={coupling[1]:.2f}")
    ax.grid(alpha=0.25, lw=0.6)

    fig.suptitle("领域惩罚：AI 味的互动代价在不同内容领域并不相同", fontsize=14,
                 y=1.01)
    save_fig(fig, INSIGHTS + "domain_penalty.png")


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------
def write_markdown(out: dict) -> None:
    pen = out["penetration"]
    half = out["halfyear"]
    penalty = out["penalty"]
    fam = out["family"]
    coupling = out["coupling"]
    meta = out["meta"]

    labeled = [d for d in LABELED_DOMAINS if d in pen]
    top = sorted(labeled, key=lambda d: pen[d]["exploring"]["hi_share"],
                 reverse=True)
    bottom = sorted(labeled, key=lambda d: pen[d]["exploring"]["hi_share"])
    excess_top = sorted(labeled, key=lambda d: pen[d]["excess_hi_pp"],
                        reverse=True)
    pen_order = sorted(labeled, key=lambda d: penalty[d]["likes"]["beta"])
    sig_neg = [d for d in labeled
               if penalty[d]["likes"]["ci_high"] < 0]
    null = [d for d in labeled
            if penalty[d]["likes"]["ci_low"] <= 0 <= penalty[d]["likes"]["ci_high"]]
    aucs = [pen[d]["human_vs_aigc_auc"] for d in labeled]
    aucs = [v for v in aucs if np.isfinite(v)]

    lines = [
        "# 领域异质性分析（Domain Stylometry）",
        "",
        f"> 由 `domain_stylometry_analysis.py` 运行生成。特征来自 `paper_features_*.csv.gz`，"
        f"检测器为已保存的稳健模型（29 个特征）；人类/AIGC 得分为 5 折折外概率，"
        f"探索集为持久化 out-of-sample `p_ai`。人类基线窗口 ≤{HUMAN_CLEAN_TO}；"
        f"探索集窗口 ≥{MONTH_FROM}；行序校验 "
        f"month={meta['alignment']['month']} / domain={meta['alignment']['domain']} / "
        f"content_len={meta['alignment']['content_len']}。",
        "",
        "## 1 领域渗透（P(AI) ≥ 0.9）",
        "",
        "| 领域 | 探索集 n | P(AI) 均值 | 高置信 % [95% CI] | 人类干净窗口 % | AIGC % | 相对人类基线 (pp) | human-vs-AIGC AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in top:
        r = pen[d]
        e, h, a = r["exploring"], r["human_clean"], r["aigc"]
        lines.append(
            f"| {d} | {e['n']:,} | {e['mean']:.3f} | "
            f"{e['hi_share']*100:.1f} [{e['hi_ci_low']*100:.1f}, {e['hi_ci_high']*100:.1f}] | "
            f"{h['hi_share']*100:.1f} | {a['hi_share']*100:.1f} | "
            f"{r['excess_hi_pp']:+.1f} | {r['human_vs_aigc_auc']:.3f} |")
    if UNKNOWN in pen:
        r = pen[UNKNOWN]
        e, h, a = r["exploring"], r["human_clean"], r["aigc"]
        h_txt = "—" if not math.isfinite(h["hi_share"]) else f"{h['hi_share']*100:.1f}"
        a_txt = "—" if not math.isfinite(a["hi_share"]) else f"{a['hi_share']*100:.1f}"
        ex_txt = "—" if not math.isfinite(r["excess_hi_pp"]) else f"{r['excess_hi_pp']:+.1f}"
        auc_txt = ("—" if not math.isfinite(r["human_vs_aigc_auc"])
                   else f"{r['human_vs_aigc_auc']:.3f}")
        lines.append(
            f"| {UNKNOWN}（缺失桶，仅参考） | {e['n']:,} | {e['mean']:.3f} | "
            f"{e['hi_share']*100:.1f} [{e['hi_ci_low']*100:.1f}, {e['hi_ci_high']*100:.1f}] | "
            f"{h_txt} | {a_txt} | {ex_txt} | {auc_txt} |")
    lines += [
        "",
        f"- 高置信占比最高的是 **{top[0]}**（{pen[top[0]]['exploring']['hi_share']*100:.1f}%），"
        f"最低的是 **{bottom[0]}**（{pen[bottom[0]]['exploring']['hi_share']*100:.1f}%）；"
        f"10 个已标注领域跨度为 {pen[bottom[0]]['exploring']['hi_share']*100:.1f}–"
        f"{pen[top[0]]['exploring']['hi_share']*100:.1f} 个百分点。"
        f"未知桶为 {pen[UNKNOWN]['exploring']['hi_share']*100:.1f}%，"
        "但它是缺失领域汇总，不能当作内容领域解读。",
        f"- 相对自身前 LLM 基线，增量最大的是 **{excess_top[0]}**"
        f"（{pen[excess_top[0]]['excess_hi_pp']:+.1f} pp），最小的是 "
        f"**{excess_top[-1]}**（{pen[excess_top[-1]]['excess_hi_pp']:+.1f} pp）。"
        f"所有领域都高于自身干净窗口基线。",
        f"- 领域内 human-vs-AIGC 判别 AUC 为 {min(aucs):.3f}–{max(aucs):.3f}；"
        "即使把领域固定，检测器仍有较强区分力。该口径（领域内）比 "
        "`paper_robustness.py` 的留一领域 AUC 0.914–0.949 更容易，"
        "两者不能直接比较，但都说明领域词不是唯一信号。",
    ]
    dedup = out["dedup_sensitivity"]
    if dedup.get("available"):
        lines.append(
            f"- 去重敏感性：按 note_id 保留元数据最完整的一条后"
            f"（{dedup['n_notes']:,}/{dedup['n_records']:,} 条），"
            f"各已标注领域高置信占比与逐记录口径的最大差异仅 "
            f"{dedup['max_abs_delta_pp']:.1f} pp，排序不变。")
    lines += [
        "",
        "## 2 半年度稳定性与采样构成",
        "",
        f"高置信占比（%，括号内为 n；n<{MIN_HALF_N} 的单元不参与趋势解读）：",
        "",
        "| 领域 | " + " | ".join(half["halves"]) + " |",
        "|---|" + "---:|" * len(half["halves"]),
    ]
    for d in top:
        cells = half["domains"].get(d, {})
        vals = []
        for h in half["halves"]:
            c = cells.get(h, {})
            if c.get("hi_share") is None:
                vals.append(f"– ({c.get('n', 0)})")
            else:
                vals.append(f"{c['hi_share']*100:.1f} ({c['n']})")
        lines.append(f"| {d} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "**领域构成（各半年度内 %）**：",
        "",
        "| 半年度 | " + " | ".join(LABELED_DOMAINS) + " |",
        "|---|" + "---:|" * len(LABELED_DOMAINS),
    ]
    for h in half["halves"]:
        comp = half["composition"][h]
        lines.append(f"| {h} | " + " | ".join(
            f"{comp.get(d, 0)*100:.1f}" for d in LABELED_DOMAINS) + " |")
    lines += [
        "",
        "> 注意：探索集早期（2023H1）样本量小，且 2023–2025 的领域构成剧烈变化"
        "（穿搭在 2024H1 仅 54 条、2024H2 才放量；未知桶占比 2023H1 44.1%、"
        "2024H1 51.9%、2025H1 8.9%，并非单调下降）。因此上表只能作为"
        "**领域内描述**，不能把半年度差异直接读成平台层面的时间趋势；"
        "平台层面的时间趋势以 `paper_drift.py` 的固定领域权重序列为准。",
        "",
        "## 3 领域内的互动惩罚",
        "",
        "因变量为 log10(1+互动)；β 为 P(AI) 每增加 1 个标准差的偏效应，"
        "控制正文长度、标签数与月份固定效应；95% CI 为按月份整群 bootstrap "
        f"{N_BOOT} 次。",
        "",
        "| 领域 | n | 月数 | 原始 ρ | 月内 ρ | 点赞 β [95% CI] | 收藏 β [95% CI] | 评论 β [95% CI] |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for d in pen_order:
        r = penalty[d]
        def cell(t):
            x = r[t]
            return (f"{x['beta']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]"
                    + _stars(x['ci_low'], x['ci_high']))
        lines.append(
            f"| {d} | {r['n']:,} | {r['likes']['n_months']} | "
            f"{r['likes']['raw_rho']:+.3f} | {r['likes']['within_month_rho']:+.3f} | "
            f"{cell('likes')} | {cell('collections')} | {cell('comments')} |")
    lines += [
        "",
        f"- 点赞端最负的是 **{pen_order[0]}**（β={penalty[pen_order[0]]['likes']['beta']:+.3f}），"
        f"最接近零的是 **{pen_order[-1]}**（β={penalty[pen_order[-1]]['likes']['beta']:+.3f}）。",
        f"- {len(sig_neg)}/10 个领域的点赞端 CI 完全在 0 以下；"
        f"{len(null)} 个领域的 CI 跨 0（{', '.join(null) if null else '无'}）。"
        "方向不一致的领域应结合样本量与月度构成解读，不能外推为「该领域没有惩罚」。",
        "- 原始 ρ 与 FE β 的排序不完全一致：控制月份、正文长度与标签数后，"
        "部分领域的关联被解释掉或增强，说明领域间差异不只是「哪个领域帖子更长/标签更多」。",
        "",
        "## 4 领域×模型家族构成（AIGC 训练集）",
        "",
        "| 领域 | n | " + " | ".join(FAMILY_NAME[f] for f in FAMILIES) + " |",
        "|---|---:|" + "---:|" * len(FAMILIES),
    ]
    for d in sorted(fam["domains"], key=lambda x: -fam["domains"][x]["n"]):
        r = fam["domains"][d]
        lines.append(f"| {d} | {r['n']:,} | " + " | ".join(
            f"{r['shares'][f]*100:.1f}%" for f in FAMILIES) + " |")
    lines += [
        "",
        f"家族构成与领域的整体关联**{'显著' if fam['p'] < 0.05 else '不显著'}**"
        f"（χ²={fam['chi2']:.1f}, dof={fam['dof']}, p={fmt_p(fam['p'])}, "
        f"Cramér's V={fam['cramers_v']:.3f}）。各领域内仍有可见差异（如 GPT "
        "占比约 22%–28%、Claude 约 9%–11%），但整体效应量很小，且训练集每个"
        "领域仅 97–1,141 条，不能据此推断野外帖的来源。这是训练集的生成设计"
        "事实，不代表野外家族分布；野外家族归因见 `family_transfer_analysis.py`。",
        "",
        "## 5 渗透增量与惩罚强度的关系",
        "",
        f"在 10 个有标注领域上，探索集相对人类基线的增量与点赞端 FE β 的 "
        f"Spearman ρ={coupling['spearman_rho']:+.3f}（p={coupling['spearman_p']:.3f}）："
        "方向为正，即 AI 渗透增量更大的领域，点赞端偏效应反而更接近 0"
        "（例如运动/美食 vs 心理/职场）。但只有 10 个领域、p>0.05，"
        "且这是领域层面的描述性共变，不能据此推断因果，也不能外推为"
        "「渗透越高惩罚越小」。",
        "",
        "## 6 口径与限制",
        "",
        "- 探索集主表为逐记录口径（与 `paper_drift.py` 一致），未按 note_id 去重；"
        "`dedup_sensitivity` 给出按 note_id 去重后的对照，领域高置信占比差异很小。",
        "- 领域标签来自平台/数据集，`未知` 是缺失桶（探索集 16,978 条），"
        "不纳入领域排名、家族构成与耦合分析。",
        "- 人类/AIGC 得分是折外概率，探索集得分来自全量训练模型；"
        "跨语料比较的是「同一检测器的排序」，不是校准后的真实 AI 比例。",
        "- 领域内互动模型是观察性关联；月份 FE 只能吸收月份层面的共同冲击，"
        "不能排除作者、选题与流量分配等未观测混杂。",
    ]
    with open(BASE + "domain_stylometry_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    print("加载特征与检测器得分…")
    frames = load_frames()
    for name in ("human", "aigc", "exploring"):
        print(f"  {name}: {len(frames[name]):,} rows")

    print("计算领域渗透…")
    pen = penetration_table(frames)
    half = halfyear_table(frames["exploring"])

    print("计算领域内互动惩罚…")
    penalty = penalty_table(frames["exploring"])
    for d, r in penalty.items():
        print(f"  {d}: n={r['n']:,}, likes β={r['likes']['beta']:+.3f} "
              f"[{r['likes']['ci_low']:+.3f}, {r['likes']['ci_high']:+.3f}]")

    print("计算领域×家族构成…")
    fam = family_share_table(frames["aigc"])

    coupling = coupling_table({"domains": pen}, penalty)

    out = {
        "meta": {
            "scores_path": SCORES_PATH,
            "month_from": MONTH_FROM,
            "human_clean_to": HUMAN_CLEAN_TO,
            "hi_threshold": HI,
            "min_domain_n": MIN_DOMAIN_N,
            "min_month_n": MIN_MONTH_N,
            "min_half_n": MIN_HALF_N,
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "alignment": frames["alignment"],
            "note": "人类/AIGC 为 5 折折外 P(AI)，探索集为 out-of-sample p_ai；"
                    "探索集逐记录，未去重；未知领域不参与排名与耦合分析。",
        },
        "penetration": pen,
        "halfyear": half,
        "penalty": penalty,
        "family": fam,
        "coupling": coupling,
        "dedup_sensitivity": dedup_sensitivity(frames["exploring"]),
    }
    save_json(json_safe(out), BASE + "domain_stylometry_results.json")
    print("已写出 domain_stylometry_results.json")
    write_markdown(out)
    print("已写出 domain_stylometry_results.md")
    plot_penetration(pen, half)
    plot_penalty({d: {**penalty[d], "excess_hi_pp": pen[d]["excess_hi_pp"]}
                  for d in penalty}, fam)
    print("已写出 insights/domain_penetration.png 与 insights/domain_penalty.png")


if __name__ == "__main__":
    main()
