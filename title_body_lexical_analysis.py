# -*- coding: utf-8 -*-
"""标题-正文词汇耦合（title-body lexical coupling）深潜分析。

既有标题分析有两类：`title_style_analysis.py` 看标题端的 AI 风格得分，
`title_body_congruence_analysis.py` 看标题端/正文端 AI 得分的四象限组合。
两者都没有回答一个更基础的问题：**标题里的词，有多少真的出现在正文里？**

本脚本用确定性的轻量分词（汉字字符 2-gram + 英文/数字整词，不依赖 jieba
词表）度量每篇笔记的标题-正文词汇耦合，并回答四组问题：

1. 跨子集差异：人类、AIGC、后 LLM 探索集的标题有多「贴合」正文？给出
   覆盖率（标题词元被正文覆盖的比例）、Jaccard、标题原样摘录率，并做
   领域标准化、标题长度匹配与正文长度匹配；
2. 野外梯度：探索集内 P(AI) 越高的帖子，标题是否越脱离正文？月份×领域
   单元内秩相关、标题长度/正文长度分层稳健性、首尾十分位 Cliff's δ；
3. 互动回报：控制 P(AI)、标题长度、正文长度与标签数后，标题-正文耦合
   与点赞、收藏、评论的月份×领域固定效应偏效应（整群 bootstrap CI）；
4. 生成端指纹与时间演化：6 个模型家族/17 个模型的耦合画像；探索集季度
   趋势（原始 + 固定领域权重）。

口径说明：
- 标题/正文清洗复用 analysis_utils.clean_body（与 paper_features.py 一致），
  自动剥离 `#话题#` 与 `[话题]` 残留；
- 人类/探索集按 note_id 去重（保留元数据最完整一条，规则同 framing_analysis），
  AIGC 无 note_id，逐记录处理；
- 探索集 p_ai 按行序对齐 paper_exploring_scores.csv.gz；行序用原始
  month/domain/likes/collections/comments 逐条校验，content_len 仅作
  清洗版本漂移的信息性指标（不参与错位判定）；
- 正文不足 MIN_BODY_LEN=50 字、标题切不出词元的帖子不参与统计，数量单独报告；
- 互动回归与 engagement_drivers_analysis.py 完全同协议（特征先 z-score
  再单元内去均值、单元 ≥100 条、整群 bootstrap 300 次）。

指标（全部先验定义，不做数据驱动调参）：
- coverage：|标题词元 ∩ 正文词元| / |标题词元|，标题被正文「兑现」的比例；
- jaccard：|交| / |并|，对标题/正文长度都敏感的对称重合度；
- excerpt：去空白后标题（≥6 字）作为连续子串出现在正文中，即「标题是正文
  摘录」的二值形态；
- title_tokens：标题词元数（作为标题长度的稳健代理）。

输出：
  insights/title_body_lexicon.png   跨子集分布 / 长度曲线 / 匹配差异
  insights/title_body_gradient.png  P(AI) 梯度 / 单元内 ρ / 互动偏效应 /
                                    家族指纹 / 季度趋势 / 判别力
  title_body_lexical_results.json   全部数值结果
  title_body_lexical_results.md     结果摘要
"""
from __future__ import annotations

from collections import defaultdict
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from analysis_utils import (BASE, INSIGHTS, LENGTH_BINS, PALETTE, body_cache,
                            bootstrap_auc, cell_groups, cell_rank_cache,
                            cliffs_delta, clean_body, content_of, cv_auc,
                            decile_table, dedup_indices, domain_standardized_series,
                            engagement_of, init_plot_style, length_bin_codes,
                            length_matched_mean_gap, length_residualize,
                            lexical_overlap, lexical_tokens, parse_local_time,
                            quarter_of, require_row_alignment, run_parallel,
                            save_fig, save_json, title_is_excerpt, title_of,
                            wilson_ci, within_cell_spearman)
from derived_cache import iter_cached
from engagement_drivers_analysis import (cluster_bootstrap_fe_ols, fe_ols,
                                         prepare_fe)

DATASETS = {
    "human": BASE + "training_set_human_fixed.jsonl",
    "aigc": BASE + "training_set_aigc.jsonl",
    "exploring": BASE + "exploring_set_fixed.jsonl",
}
SCORES_PATH = BASE + "paper_exploring_scores.csv.gz"
FEATURES_PATH = BASE + "paper_features_exploring.csv.gz"

MIN_BODY_LEN = 50
MIN_CELL_N = 100
MONTH_FROM = "202301"
N_BOOT = 300
AUC_BOOT = 300
BOOT_SEED = 42
MIN_QUARTER_N = 300
MIN_FAMILY_N = 100

SUBSET_ORDER = ["human", "aigc", "exploring"]
SUBSET_LABELS = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
                 "exploring": "探索集(后LLM)"}
METRIC_ORDER = ["coverage", "jaccard", "excerpt"]
METRIC_LABELS = {"coverage": "标题覆盖率", "jaccard": "标题-正文 Jaccard",
                 "excerpt": "标题原样摘录率"}
METRIC_COLORS = {"coverage": "#4C72B0", "jaccard": "#DD8452",
                 "excerpt": "#C44E52"}
TITLE_BINS = [(1, 6, "1–5"), (6, 10, "6–9"), (10, 14, "10–13"),
              (14, 20, "14–19"), (20, 10 ** 9, "≥20")]
TITLE_BIN_LABELS = [lab for _, _, lab in TITLE_BINS]

FAMILY_NAME = {"anthropic": "Claude", "deepseek": "DeepSeek", "gemini": "Gemini",
               "glm": "GLM", "gpt": "GPT", "qwen": "Qwen"}
FAMILIES = ["anthropic", "deepseek", "gemini", "glm", "gpt", "qwen"]
TARGETS = (("likes", "点赞"), ("collections", "收藏"), ("comments", "评论"))


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
def title_bin_codes(n_tokens) -> np.ndarray:
    """标题词元数映射到 TITLE_BINS 的分箱编号。"""
    arr = np.asarray(n_tokens, dtype=float)
    codes = np.full(len(arr), -1, dtype=int)
    for i, (lo, hi, _) in enumerate(TITLE_BINS):
        codes[(arr >= lo) & (arr < hi)] = i
    return codes


def scan_corpus(path: str) -> dict:
    """流式扫描一个语料，保存耦合指标与对齐/互动所需的原始字段。"""
    body_len, n_title, n_common = [], [], []
    coverage, jaccard, excerpt = [], [], []
    note_id, meta, month, domain = [], [], [], []
    likes, collections, comments = [], [], []
    family, model = [], []
    n_total = 0
    # 探索集约 45% 的正文重复导出，缓存正文词元集合即可复用。
    body_tokens = body_cache(lexical_tokens, max_entries=60_000)
    for rec in iter_cached(path):
        n_total += 1
        raw_content = content_of(rec)
        body = clean_body(raw_content)
        title = clean_body(title_of(rec))
        btok = body_tokens(body)
        ttok = lexical_tokens(title)
        inter = len(ttok & btok)
        union = len(ttok | btok)
        body_len.append(len(body))
        n_title.append(len(ttok))
        n_common.append(inter)
        coverage.append(inter / len(ttok) if ttok else np.nan)
        jaccard.append(inter / union if union else np.nan)
        excerpt.append(int(title_is_excerpt(title, body)))
        note_id.append(rec.get("note_id"))
        dom = rec.get("domain")
        domain.append(dom if dom not in (None, "") else "未知")
        lk, cl, cm = engagement_of(rec)
        likes.append(lk)
        collections.append(cl)
        comments.append(cm)
        has_meta = bool(rec.get("user_id")) and dom not in (None, "", "未知")
        dt = parse_local_time(rec.get("local_time"))
        month.append(f"{dt.year:04d}{dt.month:02d}" if dt else None)
        meta.append(int(has_meta) + int(bool(lk)) + int(bool(cl))
                    + int(bool(cm)) + int(dt is not None)
                    + int(bool(raw_content) and not raw_content.isspace()))
        family.append(rec.get("model_family") or "")
        model.append(rec.get("model") or "")
    return {
        "path": path, "n_total": n_total,
        "body_len": np.asarray(body_len, dtype=np.int32),
        "n_title": np.asarray(n_title, dtype=np.int32),
        "n_common": np.asarray(n_common, dtype=np.int32),
        "coverage": np.asarray(coverage, dtype=float),
        "jaccard": np.asarray(jaccard, dtype=float),
        "excerpt": np.asarray(excerpt, dtype=np.int8),
        "note_id": note_id, "meta": np.asarray(meta, dtype=np.int32),
        "month": month, "domain": domain,
        "likes": np.asarray(likes, dtype=float),
        "collections": np.asarray(collections, dtype=float),
        "comments": np.asarray(comments, dtype=float),
        "family": family, "model": model,
    }


def dedup_keep(scan: dict) -> np.ndarray:
    """按 note_id 去重（保留元数据最完整一条），返回排序后的保留索引。"""
    return np.asarray(sorted(dedup_indices(scan["note_id"], scan["meta"])), dtype=int)


def valid_sel(scan: dict, keep: np.ndarray, min_len: int = MIN_BODY_LEN) -> np.ndarray:
    """去重 + 正文长度门槛 + 标题/正文都可切出词元的共同样本。"""
    sel = keep[scan["body_len"][keep] >= min_len]
    sel = sel[(scan["n_title"][sel] > 0) & np.isfinite(scan["coverage"][sel])]
    return sel


def mean_ci(values) -> dict:
    """均值与解析 95% CI（截断到指标自然范围由调用方处理）。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan")}
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"n": int(n), "mean": float(arr.mean()),
            "ci_low": float(arr.mean() - 1.96 * se),
            "ci_high": float(arr.mean() + 1.96 * se)}


def subset_block(scan: dict, keep: np.ndarray) -> dict:
    """一个子集的耦合指标摘要（去重 + 有效正文/标题）。"""
    sel = valid_sel(scan, keep)
    cov = scan["coverage"][sel]
    jac = scan["jaccard"][sel]
    exc = scan["excerpt"][sel]
    high = int((cov >= 0.5).sum())
    ex_hit = int(exc.sum())
    return {
        "n_raw": int(scan["n_total"]), "n_dedup": int(len(keep)),
        "n": int(len(sel)),
        "coverage": mean_ci(cov),
        "coverage_median": float(np.median(cov)) if len(cov) else float("nan"),
        "coverage_high_share": high / len(sel) if len(sel) else 0.0,
        "coverage_high_ci": list(wilson_ci(high, len(sel))) if len(sel) else [0, 0],
        "jaccard": mean_ci(jac),
        "jaccard_median": float(np.median(jac)) if len(jac) else float("nan"),
        "excerpt": {"n": int(len(sel)), "share": ex_hit / len(sel) if len(sel) else 0.0,
                    "ci": list(wilson_ci(ex_hit, len(sel))) if len(sel) else [0, 0]},
        "title_tokens_mean": float(scan["n_title"][sel].mean()) if len(sel) else 0.0,
        "title_tokens_median": float(np.median(scan["n_title"][sel])) if len(sel) else 0.0,
        "body_len_mean": float(scan["body_len"][sel].mean()) if len(sel) else 0.0,
        "body_len_median": float(np.median(scan["body_len"][sel])) if len(sel) else 0.0,
        "dropped_short_body": int((scan["body_len"][keep] < MIN_BODY_LEN).sum()),
        "dropped_no_title_tokens": int(((scan["body_len"][keep] >= MIN_BODY_LEN)
                                        & (scan["n_title"][keep] <= 0)).sum()),
    }


def domain_standardized_block(scans: dict, keeps: dict) -> dict:
    """固定为人类领域权重的耦合均值与摘录率。"""
    human_sel = valid_sel(scans["human"], keeps["human"])
    human_share = pd.Series(
        [scans["human"]["domain"][i] for i in human_sel]).value_counts(
        normalize=True).to_dict()
    out = {}
    for name in SUBSET_ORDER:
        sc = scans[name]
        sel = valid_sel(sc, keeps[name])
        df = pd.DataFrame({"domain": [sc["domain"][i] for i in sel],
                           "coverage": sc["coverage"][sel],
                           "jaccard": sc["jaccard"][sel],
                           "excerpt": sc["excerpt"][sel].astype(float)})
        block = {}
        for col in METRIC_ORDER:
            means = df.groupby("domain")[col].mean().to_dict()
            w = {d: human_share.get(d, 0.0) for d in means}
            total = sum(w.values())
            w = ({k: v / total for k, v in w.items()} if total > 0
                 else {k: 1.0 / max(len(means), 1) for k in means})
            block[col] = float(sum(means[d] * w[d] for d in means))
        out[name] = block
    return out


def length_control_block(scans: dict, keeps: dict) -> dict:
    """标题长度匹配 / 正文长度匹配的人类−其他子集差异。"""
    data = {}
    for name in SUBSET_ORDER:
        sc = scans[name]
        sel = valid_sel(sc, keeps[name])
        data[name] = {
            "coverage": sc["coverage"][sel],
            "jaccard": sc["jaccard"][sel],
            "excerpt": sc["excerpt"][sel].astype(float),
            "title_tokens": sc["n_title"][sel].astype(float),
            "body_len": sc["body_len"][sel].astype(float),
        }
    out = {"title_matched": {}, "body_matched": {}, "title_bins": {},
           "body_bins": {}}
    for other in ("aigc", "exploring"):
        out["title_matched"][other] = {}
        out["body_matched"][other] = {}
        for metric in METRIC_ORDER:
            out["title_matched"][other][metric] = length_matched_mean_gap(
                data["human"][metric], data["human"]["title_tokens"],
                data[other][metric], data[other]["title_tokens"], TITLE_BINS)
            out["body_matched"][other][metric] = length_matched_mean_gap(
                data["human"][metric], data["human"]["body_len"],
                data[other][metric], data[other]["body_len"], LENGTH_BINS)
    # 正文长度分箱的均值曲线（图 c）
    for name in SUBSET_ORDER:
        codes = length_bin_codes(data[name]["body_len"])
        rows = []
        for b, (_, _, label) in enumerate(LENGTH_BINS):
            m = codes == b
            rows.append({"bin": b, "label": label, "n": int(m.sum()),
                         "coverage": float(data[name]["coverage"][m].mean()) if m.any() else float("nan"),
                         "jaccard": float(data[name]["jaccard"][m].mean()) if m.any() else float("nan"),
                         "excerpt": float(data[name]["excerpt"][m].mean()) if m.any() else float("nan")})
        out["body_bins"][name] = rows
    # 标题长度分箱的均值（结果文件用）
    for name in SUBSET_ORDER:
        codes = title_bin_codes(data[name]["title_tokens"])
        rows = []
        for b, label in enumerate(TITLE_BIN_LABELS):
            m = codes == b
            rows.append({"bin": b, "label": label, "n": int(m.sum()),
                         "coverage": float(data[name]["coverage"][m].mean()) if m.any() else float("nan"),
                         "jaccard": float(data[name]["jaccard"][m].mean()) if m.any() else float("nan"),
                         "excerpt": float(data[name]["excerpt"][m].mean()) if m.any() else float("nan")})
        out["title_bins"][name] = rows
    return out


def excerpt_sensitivity_block(scans: dict, keeps: dict) -> dict:
    """剔除「标题原样摘录正文」的帖子后，跨子集耦合差异是否仍在。

    摘录帖的 coverage 必然接近 1，会把人类集均值抬高；这里把 excerpt=1 的
    帖子整体排除，再按标题/正文长度匹配重算 coverage 与 Jaccard 差异。
    """
    data = {}
    for name in SUBSET_ORDER:
        sc = scans[name]
        sel = valid_sel(sc, keeps[name])
        keep = sel[sc["excerpt"][sel] == 0]
        data[name] = {
            "n": int(len(keep)),
            "coverage_mean": float(sc["coverage"][keep].mean()) if len(keep) else float("nan"),
            "jaccard_mean": float(sc["jaccard"][keep].mean()) if len(keep) else float("nan"),
            "coverage": sc["coverage"][keep],
            "jaccard": sc["jaccard"][keep],
            "title_tokens": sc["n_title"][keep].astype(float),
            "body_len": sc["body_len"][keep].astype(float),
        }
    out = {"means": {name: {k: v for k, v in data[name].items()
                            if k not in ("coverage", "jaccard",
                                         "title_tokens", "body_len")}
                      for name in SUBSET_ORDER},
           "title_matched": {}, "body_matched": {}}
    for other in ("aigc", "exploring"):
        out["title_matched"][other] = {}
        out["body_matched"][other] = {}
        for metric in ("coverage", "jaccard"):
            out["title_matched"][other][metric] = length_matched_mean_gap(
                data["human"][metric], data["human"]["title_tokens"],
                data[other][metric], data[other]["title_tokens"], TITLE_BINS)
            out["body_matched"][other][metric] = length_matched_mean_gap(
                data["human"][metric], data["human"]["body_len"],
                data[other][metric], data[other]["body_len"], LENGTH_BINS)
    return out


# --------------------------------------------------------------------------
# 探索集：梯度 / 互动 / 时间
# --------------------------------------------------------------------------
def gradient_block(scans: dict, keeps: dict, scores: pd.DataFrame) -> dict:
    """探索集内耦合指标随 P(AI) 的十分位梯度与单元内秩相关。"""
    expl = scans["exploring"]
    keep = keeps["exploring"]
    sel = valid_sel(expl, keep)
    p_ai = scores["p_ai"].to_numpy(dtype=float)[sel]
    months = np.asarray([expl["month"][i] or "未知" for i in sel])
    domains = np.asarray([expl["domain"][i] for i in sel])
    cells = pd.factorize(pd.Series([f"{m}|{d}" for m, d in zip(months, domains)]),
                         sort=True)[0]
    tbins = title_bin_codes(expl["n_title"][sel])
    bbins = length_bin_codes(expl["body_len"][sel])
    cells_title = pd.factorize(pd.Series(
        [f"{m}|{d}|{b}" for m, d, b in zip(months, domains, tbins)]),
        sort=True)[0]
    cells_body = pd.factorize(pd.Series(
        [f"{m}|{d}|{b}" for m, d, b in zip(months, domains, bbins)]),
        sort=True)[0]

    groups = cell_groups(cells, MIN_CELL_N)
    p_ai_ranks = cell_rank_cache(p_ai, groups)
    groups_title = cell_groups(cells_title, MIN_CELL_N)
    p_ai_ranks_title = cell_rank_cache(p_ai, groups_title)
    groups_body = cell_groups(cells_body, MIN_CELL_N)
    p_ai_ranks_body = cell_rank_cache(p_ai, groups_body)
    p_hi, p_lo = float(np.quantile(p_ai, 0.9)), float(np.quantile(p_ai, 0.1))
    hi_p, lo_p = p_ai >= p_hi, p_ai <= p_lo

    out = {"n": int(len(sel)), "n_cells": int(len(np.unique(cells))),
           "title_cells": int(len(np.unique(cells_title))),
           "body_cells": int(len(np.unique(cells_body))), "metrics": {}}
    for metric in METRIC_ORDER:
        vals = expl[metric][sel].astype(float)
        dec = decile_table(vals, p_ai)
        wc = within_cell_spearman(vals, p_ai, cells, min_cell_n=MIN_CELL_N,
                                  groups=groups, y_ranks=p_ai_ranks)
        wt = within_cell_spearman(vals, p_ai, cells_title, min_cell_n=MIN_CELL_N,
                                  groups=groups_title,
                                  y_ranks=p_ai_ranks_title)
        wb = within_cell_spearman(vals, p_ai, cells_body, min_cell_n=MIN_CELL_N,
                                  groups=groups_body, y_ranks=p_ai_ranks_body)
        delta, dp = cliffs_delta(vals[hi_p], vals[lo_p])
        # 高耦合（coverage ≥ 0.5）/ 摘录率 的十分位普及率
        dec_high = []
        if metric == "coverage":
            hit = vals >= 0.5
            for row in dec:
                m = (p_ai >= row["p_range"][0]) & (p_ai <= row["p_range"][1])
                k = int(hit[m].sum())
                lo, hi = wilson_ci(k, int(m.sum()))
                dec_high.append({"decile": row["decile"], "n": int(m.sum()),
                                 "share": k / int(m.sum()) if m.sum() else 0.0,
                                 "ci_low": float(lo), "ci_high": float(hi)})
        elif metric == "excerpt":
            hit = vals > 0.5
            for row in dec:
                m = (p_ai >= row["p_range"][0]) & (p_ai <= row["p_range"][1])
                k = int(hit[m].sum())
                lo, hi = wilson_ci(k, int(m.sum()))
                dec_high.append({"decile": row["decile"], "n": int(m.sum()),
                                 "share": k / int(m.sum()) if m.sum() else 0.0,
                                 "ci_low": float(lo), "ci_high": float(hi)})
        out["metrics"][metric] = {
            "deciles": dec,
            "deciles_high": dec_high,
            "weighted_rho": round(wc["weighted_rho"], 4),
            "neg_cell_share": round(wc["neg_cell_share"], 4),
            "n_cells": wc["n_cells"],
            "title_stratified_rho": round(wt["weighted_rho"], 4),
            "title_stratified_cells": wt["n_cells"],
            "body_stratified_rho": round(wb["weighted_rho"], 4),
            "body_stratified_cells": wb["n_cells"],
            "cliff_delta_top_vs_bottom": round(delta, 4),
            "cliff_p": float(dp),
            "p_range": [float(p_ai.min()), float(p_ai.max())],
        }
    return out, sel, p_ai


def engagement_block(scans: dict, keeps: dict, scores: pd.DataFrame,
                     features: pd.DataFrame, sel: np.ndarray, p_ai: np.ndarray) -> dict:
    """耦合指标对三端互动的月份×领域 FE 偏效应（整群 bootstrap）。"""
    expl = scans["exploring"]
    X = np.column_stack([
        expl["coverage"][sel], expl["jaccard"][sel],
        expl["excerpt"][sel].astype(float),
        expl["n_title"][sel].astype(float),
        p_ai,
        features["content_len"].to_numpy(dtype=float)[sel],
        features["n_hashtags"].to_numpy(dtype=float)[sel],
    ])
    names = ["coverage", "jaccard", "excerpt", "title_tokens", "p_ai",
             "content_len", "n_hashtags"]
    months = features["month"].astype(str).to_numpy()[sel]
    domains = features["domain"].astype(str).to_numpy()[sel]
    keep_eng = months >= MONTH_FROM
    cells = np.asarray([f"{m}|{d}" for m, d in zip(months, domains)])
    out = {"features": names}
    for target, label in TARGETS:
        y = np.log10(1 + np.clip(
            features[target].to_numpy(dtype=float)[sel], 0, None))
        prep = prepare_fe(X[keep_eng], cells[keep_eng], MIN_CELL_N)
        fit = fe_ols(X[keep_eng], y[keep_eng], cells[keep_eng], prepared=prep)
        boot = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                        fit["cell_codes"], n_boot=N_BOOT,
                                        seed=BOOT_SEED)
        ci = np.percentile(boot, [2.5, 97.5], axis=0)
        out[target] = {
            "coef": {n: round(float(v), 4) for n, v in zip(names, fit["coef"])},
            "ci_low": {n: round(float(v), 4) for n, v in zip(names, ci[0])},
            "ci_high": {n: round(float(v), 4) for n, v in zip(names, ci[1])},
            "n_obs": fit["n_obs"], "n_cells": fit["n_cells"], "label": label,
        }
    return out


def quarterly_block(scans: dict, keeps: dict, sel: np.ndarray) -> dict:
    """探索集季度趋势：原始均值 + 固定领域权重均值。"""
    sc = scans["exploring"]
    months = np.asarray([sc["month"][i] or "" for i in sel])
    ok = (months >= MONTH_FROM) & np.array([quarter_of(m) is not None for m in months])
    df = pd.DataFrame({
        "quarter": [quarter_of(m) for m in months[ok]],
        "domain": [sc["domain"][i] for i in sel[ok]],
        "coverage": sc["coverage"][sel[ok]],
        "jaccard": sc["jaccard"][sel[ok]],
        "excerpt": sc["excerpt"][sel[ok]].astype(float),
    })
    counts = df.groupby("quarter").size()
    quarters = [q for q in sorted(counts.index) if counts[q] >= MIN_QUARTER_N]
    df = df[df["quarter"].isin(quarters)]
    out = {"quarters": quarters, "n": [int((df["quarter"] == q).sum()) for q in quarters],
           "raw": {}, "domain_std": {}, "trend": {}}
    for col in METRIC_ORDER:
        raw = df.groupby("quarter")[col].mean().reindex(quarters).to_numpy()
        std = domain_standardized_series(df, col, "quarter").reindex(quarters).to_numpy()
        rho_raw, p_raw = spearmanr(np.arange(len(quarters)), raw)
        rho_std, p_std = spearmanr(np.arange(len(quarters)), std)
        out["raw"][col] = [float(v) for v in raw]
        out["domain_std"][col] = [float(v) for v in std]
        out["trend"][col] = {
            "rho_raw": round(float(rho_raw), 4), "p_raw": float(p_raw),
            "rho_domain_std": round(float(rho_std), 4), "p_domain_std": float(p_std),
        }
    return out


def family_block(scans: dict, keeps: dict) -> dict:
    """AIGC 子集的家族 / 单模型耦合指纹（含正文长度标准化）。"""
    sc = scans["aigc"]
    sel = valid_sel(sc, np.arange(sc["n_total"], dtype=int))
    cov = sc["coverage"][sel]
    res = length_residualize(cov, sc["body_len"][sel])
    fam = np.asarray([sc["family"][i] or "unknown" for i in sel])
    model = np.asarray([sc["model"][i] or "unknown" for i in sel])
    families = {}
    for f in sorted(set(fam)):
        m = fam == f
        if int(m.sum()) < MIN_FAMILY_N:
            continue
        families[f] = {
            "n": int(m.sum()),
            "coverage_mean": float(cov[m].mean()),
            "coverage_median": float(np.median(cov[m])),
            "coverage_length_std": float(res[m].mean()),
            "jaccard_mean": float(sc["jaccard"][sel][m].mean()),
            "excerpt_share": float(sc["excerpt"][sel][m].mean()),
            "title_tokens_mean": float(sc["n_title"][sel][m].mean()),
            "body_len_mean": float(sc["body_len"][sel][m].mean()),
        }
    models = []
    for md in sorted(set(model)):
        m = model == md
        if int(m.sum()) < 20:
            continue
        models.append({
            "model": md,
            "family": str(fam[m][0]),
            "n": int(m.sum()),
            "coverage_mean": float(cov[m].mean()),
            "coverage_length_std": float(res[m].mean()),
            "excerpt_share": float(sc["excerpt"][sel][m].mean()),
            "title_tokens_mean": float(sc["n_title"][sel][m].mean()),
        })
    models.sort(key=lambda r: (-r["coverage_mean"], r["model"]))
    return {"families": families, "models": models, "n": int(len(sel))}


def discrimination_block(scans: dict, keeps: dict) -> dict:
    """human vs AIGC：单指标 AUC 与多指标 5 折 CV。"""
    h_sel = valid_sel(scans["human"], keeps["human"])
    a_sel = valid_sel(scans["aigc"], np.arange(scans["aigc"]["n_total"], dtype=int))
    y = np.r_[np.ones(len(h_sel), dtype=int), np.zeros(len(a_sel), dtype=int)]
    metrics = {}
    for metric in METRIC_ORDER:
        score = np.r_[scans["human"][metric][h_sel],
                      scans["aigc"][metric][a_sel]].astype(float)
        metrics[metric] = bootstrap_auc(y, score, n_boot=AUC_BOOT, seed=BOOT_SEED)
    extra = {}
    for metric in ("n_title", "body_len"):
        score = np.r_[scans["human"][metric][h_sel],
                      scans["aigc"][metric][a_sel]].astype(float)
        key = "title_tokens" if metric == "n_title" else metric
        extra[key] = bootstrap_auc(y, score, n_boot=AUC_BOOT, seed=BOOT_SEED)
    df = pd.DataFrame({"dataset": ["human"] * len(h_sel) + ["aigc"] * len(a_sel)})
    for metric in METRIC_ORDER:
        df[metric] = np.r_[scans["human"][metric][h_sel],
                           scans["aigc"][metric][a_sel]].astype(float)
    df["title_tokens"] = np.r_[scans["human"]["n_title"][h_sel],
                               scans["aigc"]["n_title"][a_sel]].astype(float)
    df["body_len"] = np.r_[scans["human"]["body_len"][h_sel],
                           scans["aigc"]["body_len"][a_sel]].astype(float)
    cv_coupling = cv_auc(df, METRIC_ORDER, seed=BOOT_SEED)
    cv_context = cv_auc(df, ["title_tokens", "body_len"], seed=BOOT_SEED)
    cv = cv_auc(df, METRIC_ORDER + ["title_tokens", "body_len"], seed=BOOT_SEED)
    return {"markers": metrics, "context": extra, "cv_coupling": cv_coupling,
            "cv_context": cv_context, "cv_lr": cv,
            "n_human": int(len(h_sel)), "n_aigc": int(len(a_sel))}


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_lexicon(subsets: dict, controls: dict) -> None:
    init_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))

    # ---- (a) 标题覆盖率 ECDF ----
    ax = axes[0, 0]
    for name in SUBSET_ORDER:
        vals = np.sort(np.asarray(subsets[name]["_coverage_values"], dtype=float))
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size + 1) / vals.size
        ax.plot(vals, y, color=PALETTE[name], lw=1.8,
                label=f"{SUBSET_LABELS[name]}（中位 {np.median(vals):.3f}）")
    ax.set_xlim(0, 1)
    ax.set_xlabel("标题覆盖率（标题词元被正文覆盖的比例）")
    ax.set_ylabel("累计帖子占比")
    ax.set_title("(a) 标题-正文词汇覆盖率的累计分布")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")

    # ---- (b) 覆盖率 / Jaccard / 摘录率 ----
    ax = axes[0, 1]
    x = np.arange(3)
    width = 0.24
    for j, name in enumerate(SUBSET_ORDER):
        vals = [subsets[name]["coverage"]["mean"],
                subsets[name]["jaccard"]["mean"],
                subsets[name]["excerpt"]["share"]]
        ax.bar(x + (j - 1) * width, vals, width=width, color=PALETTE[name],
               label=SUBSET_LABELS[name])
        for xx, v in zip(x + (j - 1) * width, vals):
            ax.annotate(f"{v:.3f}", xy=(xx, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(["标题覆盖率", "Jaccard", "标题原样摘录率"])
    ax.set_ylabel("均值 / 占比")
    ax.set_title("(b) 三个耦合指标的跨子集水平")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5)

    # ---- (c) 覆盖率随正文长度的变化 ----
    ax = axes[1, 0]
    for name in SUBSET_ORDER:
        rows = controls["body_bins"][name]
        xx = [r["bin"] for r in rows if r["n"] > 0]
        yy = [r["coverage"] for r in rows if r["n"] > 0]
        ax.plot(xx, yy, "o-", color=PALETTE[name], lw=1.8, ms=5,
                label=SUBSET_LABELS[name])
    ax.set_xticks(range(len(LENGTH_BINS)))
    ax.set_xticklabels([lab for _, _, lab in LENGTH_BINS])
    ax.set_xlabel("正文长度分箱")
    ax.set_ylabel("标题覆盖率均值")
    ax.set_title("(c) 覆盖率随正文长度的变化（长度混杂诊断）")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)

    # ---- (d) 标题长度 / 正文长度匹配后的人类−其他差异 ----
    ax = axes[1, 1]
    rows, labels, metrics = [], [], []
    for metric in METRIC_ORDER:
        for other in ("aigc", "exploring"):
            for mode, tag in (("title_matched", "标题长度匹配"),
                              ("body_matched", "正文长度匹配")):
                g = controls[mode][other][metric]
                rows.append(g)
                labels.append(f"{METRIC_LABELS[metric]} · {tag} · 人类−{SUBSET_LABELS[other]}")
                metrics.append(metric)
    ypos = np.arange(len(rows))
    colors = {"coverage": "#4C72B0", "jaccard": "#DD8452", "excerpt": "#C44E52"}
    for i, (g, metric) in enumerate(zip(rows, metrics)):
        ax.plot([g["ci_low"], g["ci_high"]], [i, i], color=colors[metric],
                lw=1.8, solid_capstyle="butt")
        ax.scatter([g["gap"]], [i], color=colors[metric], s=34, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("人类 − 其他子集的均值差（95% CI）")
    ax.set_title("(d) 长度匹配后的耦合差异（>0 = 人类标题更贴合正文）")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    save_fig(fig, INSIGHTS + "title_body_lexicon.png")


def plot_gradient(grad: dict, eng: dict, families: dict,
                  quarterly: dict, disc: dict) -> None:
    init_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(19.0, 11.8))

    # ---- (a) P(AI) 十分位：覆盖率与摘录率 ----
    ax = axes[0, 0]
    dec = grad["metrics"]["coverage"]["deciles"]
    x = [d["decile"] for d in dec]
    y = [d["mean"] for d in dec]
    lo = [d["ci_low"] for d in dec]
    hi = [d["ci_high"] for d in dec]
    ax.plot(x, y, "o-", color=METRIC_COLORS["coverage"], lw=1.9, ms=5,
            label=f"标题覆盖率（ρ={grad['metrics']['coverage']['weighted_rho']:+.3f}）")
    ax.fill_between(x, lo, hi, color=METRIC_COLORS["coverage"], alpha=0.16)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("检测器 P(AI) 十分位（1=最像人，10=最像 AI）")
    ax.set_ylabel("标题覆盖率均值", color=METRIC_COLORS["coverage"])
    ax.tick_params(axis="y", labelcolor=METRIC_COLORS["coverage"])
    ax2 = ax.twinx()
    dh = grad["metrics"]["excerpt"]["deciles_high"]
    xh = [d["decile"] for d in dh]
    yh = [d["share"] for d in dh]
    loh = [d["ci_low"] for d in dh]
    hih = [d["ci_high"] for d in dh]
    ax2.plot(xh, yh, "s--", color=METRIC_COLORS["excerpt"], lw=1.6, ms=4.5,
             label=f"标题原样摘录率（ρ={grad['metrics']['excerpt']['weighted_rho']:+.3f}）")
    ax2.fill_between(xh, loh, hih, color=METRIC_COLORS["excerpt"], alpha=0.13)
    ax2.set_ylabel("标题原样摘录率", color=METRIC_COLORS["excerpt"])
    ax2.tick_params(axis="y", labelcolor=METRIC_COLORS["excerpt"])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.set_title("(a) 野外 P(AI) 梯度（含 95% CI）")

    # ---- (b) 单元内 ρ：原始 / 标题分层 / 正文分层 ----
    ax = axes[0, 1]
    ypos = np.arange(len(METRIC_ORDER))
    h = 0.26
    raw = [grad["metrics"][m]["weighted_rho"] for m in METRIC_ORDER]
    tit = [grad["metrics"][m]["title_stratified_rho"] for m in METRIC_ORDER]
    bod = [grad["metrics"][m]["body_stratified_rho"] for m in METRIC_ORDER]
    ax.barh(ypos - h, raw, height=h, color=[METRIC_COLORS[m] for m in METRIC_ORDER],
            alpha=0.95, label="月份×领域单元内 ρ")
    ax.barh(ypos, tit, height=h, color=[METRIC_COLORS[m] for m in METRIC_ORDER],
            alpha=0.6, label="再加标题长度分层")
    ax.barh(ypos + h, bod, height=h, color=[METRIC_COLORS[m] for m in METRIC_ORDER],
            alpha=0.3, label="再加正文长度分层")
    for i, (r, t, b) in enumerate(zip(raw, tit, bod)):
        ax.annotate(f"{r:+.3f}", xy=(r, ypos[i] - h), xytext=(5 if r >= 0 else -5, -3),
                    textcoords="offset points", fontsize=8,
                    ha="left" if r >= 0 else "right")
        ax.annotate(f"{t:+.3f}", xy=(t, ypos[i]), xytext=(5 if t >= 0 else -5, -3),
                    textcoords="offset points", fontsize=8,
                    ha="left" if t >= 0 else "right")
        ax.annotate(f"{b:+.3f}", xy=(b, ypos[i] + h), xytext=(5 if b >= 0 else -5, -3),
                    textcoords="offset points", fontsize=8,
                    ha="left" if b >= 0 else "right")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
    ax.set_xlabel("加权 Spearman ρ（耦合指标 vs P(AI)）")
    ax.set_title("(b) 梯度稳健性：单元内 + 长度分层")
    ax.legend(fontsize=7.5, loc="lower left")

    # ---- (c) 互动偏效应森林图 ----
    ax = axes[0, 2]
    names = ["coverage", "jaccard", "excerpt", "title_tokens", "p_ai"]
    labels = {"coverage": "标题覆盖率", "jaccard": "Jaccard",
              "excerpt": "标题原样摘录", "title_tokens": "标题词元数",
              "p_ai": "P(AI)"}
    order = list(reversed(names))
    ypos = np.arange(len(order))
    style = {"likes": ("o", -0.18, "#4C72B0", "点赞"),
             "collections": ("^", 0.0, "#DD8452", "收藏"),
             "comments": ("s", 0.18, "#55A868", "评论")}
    for target, (marker, offset, color, label) in style.items():
        fe = eng[target]
        for i, n in enumerate(order):
            b = fe["coef"][n]
            lo, hi = fe["ci_low"][n], fe["ci_high"][n]
            sig = lo > 0 or hi < 0
            col = color if sig else "#9E9E9E"
            ax.plot([lo, hi], [ypos[i] + offset, ypos[i] + offset], color=col,
                    lw=1.6, solid_capstyle="butt", zorder=2)
            ax.scatter([b], [ypos[i] + offset], marker=marker, s=28,
                       facecolor=col if sig else "white", edgecolor=col,
                       zorder=3, label=label if i == 0 else None)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[n] for n in order])
    ax.set_xlabel("单元内 FE β（每 1 SD → log10(1+互动)）")
    ax.set_title(f"(c) 耦合指标的互动偏效应（n={eng['likes']['n_obs']:,}，"
                 f"{eng['likes']['n_cells']} 单元）")
    ax.legend(fontsize=8, loc="lower right")

    # ---- (d) 家族指纹热力图 ----
    ax = axes[1, 0]
    fams = [f for f in FAMILIES if f in families["families"]]
    fams.sort(key=lambda f: -families["families"][f]["coverage_mean"])
    cols = ["coverage_mean", "jaccard_mean", "excerpt_share"]
    col_labels = ["标题覆盖率", "Jaccard", "摘录率"]
    mat = np.asarray([[families["families"][f][c] for c in cols] for f in fams])
    z = (mat - mat.mean(axis=0)) / np.where(mat.std(axis=0) > 0,
                                            mat.std(axis=0), 1.0)
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    ax.grid(False)
    for i in range(len(fams)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(z[i, j]) > 1.15 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f"{FAMILY_NAME.get(f, f)}（n={families['families'][f]['n']}）"
                        for f in fams])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("列内 z-score", fontsize=8)
    ax.set_title("(d) AIGC 家族的标题-正文耦合指纹")

    # ---- (e) 季度趋势（固定领域权重） ----
    ax = axes[1, 1]
    qs = quarterly["quarters"]
    xx = np.arange(len(qs))
    tr = quarterly["trend"]["coverage"]
    ax.plot(xx, quarterly["domain_std"]["coverage"], marker="o", ls="-",
            color=METRIC_COLORS["coverage"], lw=1.7, ms=4.5,
            label=f"标题覆盖率（ρ={tr['rho_domain_std']:+.2f}）")
    ax.set_ylabel("标题覆盖率（固定领域权重）", color=METRIC_COLORS["coverage"])
    ax.tick_params(axis="y", labelcolor=METRIC_COLORS["coverage"])
    ax2 = ax.twinx()
    tr2 = quarterly["trend"]["excerpt"]
    ax2.plot(xx, quarterly["domain_std"]["excerpt"], marker="s", ls="--",
             color=METRIC_COLORS["excerpt"], lw=1.7, ms=4.5,
             label=f"标题原样摘录率（ρ={tr2['rho_domain_std']:+.2f}）")
    ax2.set_ylabel("标题原样摘录率", color=METRIC_COLORS["excerpt"])
    ax2.tick_params(axis="y", labelcolor=METRIC_COLORS["excerpt"])
    ax.set_xticks(xx)
    ax.set_xticklabels(qs, rotation=35, fontsize=8.5, ha="right")
    ax.set_title("(e) 探索集季度趋势（领域标准化）")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    # ---- (f) 判别力 ----
    ax = axes[1, 2]
    keys = METRIC_ORDER + ["title_tokens", "body_len", "cv_lr"]
    labels2 = dict(METRIC_LABELS)
    labels2.update({"title_tokens": "标题词元数", "body_len": "正文长度",
                    "cv_lr": "五项特征 LR（5 折 CV）"})
    vals = [disc["markers"][m]["auc"] for m in METRIC_ORDER]
    vals += [disc["context"]["title_tokens"]["auc"],
             disc["context"]["body_len"]["auc"], disc["cv_lr"]["auc"]]
    los = [disc["markers"][m]["ci_low"] for m in METRIC_ORDER]
    los += [disc["context"]["title_tokens"]["ci_low"],
            disc["context"]["body_len"]["ci_low"], float("nan")]
    his = [disc["markers"][m]["ci_high"] for m in METRIC_ORDER]
    his += [disc["context"]["title_tokens"]["ci_high"],
            disc["context"]["body_len"]["ci_high"], float("nan")]
    colors = [METRIC_COLORS[m] for m in METRIC_ORDER] + ["#8172B3", "#937860", "#333333"]
    ypos = np.arange(len(keys))
    for i, (v, lo, hi, c) in enumerate(zip(vals, los, his, colors)):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [ypos[i], ypos[i]], color=c, lw=2.0,
                    solid_capstyle="butt")
        ax.scatter([v], [ypos[i]], s=48, color=c, zorder=3)
        ax.annotate(f"{v:.3f}", xy=(v, ypos[i]), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels2[k] for k in keys])
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC（human vs AIGC；越高=该指标越能区分）")
    ax.set_title("(f) 标题-正文耦合的判别力")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    save_fig(fig, INSIGHTS + "title_body_gradient.png")


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------
def fmt_ci(rec: dict, key: str = "gap", pct: bool = False) -> str:
    if pct:
        return (f"{rec[key] * 100:+.1f}pp "
                f"[{rec['ci_low'] * 100:+.1f}, {rec['ci_high'] * 100:+.1f}]")
    return (f"{rec[key]:+.3f} "
            f"[{rec['ci_low']:+.3f}, {rec['ci_high']:+.3f}]")


def write_markdown(payload: dict) -> None:
    subsets = payload["subsets"]
    ctrl = payload["length_control"]
    grad = payload["gradient"]
    eng = payload["engagement"]
    fam = payload["family"]
    q = payload["quarterly"]
    disc = payload["discrimination"]
    lines = [
        "# 标题-正文词汇耦合（title-body lexical coupling）分析结果",
        "",
        "> 由 `title_body_lexical_analysis.py` 全量运行生成；标题/正文清洗与 "
        "`paper_features.py` 一致，人类/探索集按 note_id 去重；正文 ≥50 字；"
        "探索集 p_ai 按行序对齐 `paper_exploring_scores.csv.gz`"
        f"（月份/领域/互动逐条一致；content_len mismatch={payload['meta']['content_len_mismatch']}，"
        f"原始文本长度 drift={payload['meta']['content_len_drift']}）。",
        "",
        "指标：`coverage` = 标题词元被正文覆盖的比例；`jaccard` = 标题与正文"
        "词元集合的 Jaccard；`excerpt` = 去空白后标题（≥6 字）原样出现在正文中。"
        "词元为汉字字符 2-gram + 英文/数字整词（确定性切分，不依赖 jieba）。",
        "",
        "## 1 跨子集耦合水平",
        "",
        "| 子集 | 有效帖数 | 标题覆盖率均值 | 覆盖率中位数 | 高覆盖(≥0.5)占比 | Jaccard 均值 | 标题原样摘录率 | 标题词元数 | 正文长度 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SUBSET_ORDER:
        b = subsets[name]
        lines.append(
            f"| {SUBSET_LABELS[name]} | {b['n']:,} | {b['coverage']['mean']:.3f} | "
            f"{b['coverage_median']:.3f} | {b['coverage_high_share']:.1%} | "
            f"{b['jaccard']['mean']:.3f} | {b['excerpt']['share']:.2%} | "
            f"{b['title_tokens_mean']:.1f} | {b['body_len_median']:.0f} |")
    lines += [
        "",
        "领域标准化（固定为人类领域权重）：",
        "",
        "| 子集 | 标题覆盖率 | Jaccard | 标题原样摘录率 |",
        "|---|---:|---:|---:|",
    ]
    for name in SUBSET_ORDER:
        b = payload["domain_standardized"][name]
        lines.append(f"| {SUBSET_LABELS[name]} | {b['coverage']:.3f} | "
                     f"{b['jaccard']:.3f} | {b['excerpt']:.2%} |")
    lines += [
        "",
        "长度匹配差异（人类−其他子集；标题长度分箱 / 正文长度分箱分别加权）：",
        "",
        "| 指标 | 匹配维度 | 人类−AIGC [95% CI] | 人类−探索集 [95% CI] |",
        "|---|---|---:|---:|",
    ]
    for metric in METRIC_ORDER:
        for mode, label in (("title_matched", "标题长度"), ("body_matched", "正文长度")):
            g1 = ctrl[mode]["aigc"][metric]
            g2 = ctrl[mode]["exploring"][metric]
            lines.append(f"| {METRIC_LABELS[metric]} | {label} | "
                         f"{fmt_ci(g1)} | {fmt_ci(g2)} |")
    sens = payload["excerpt_sensitivity"]
    lines += [
        "",
        "剔除标题原样摘录帖（excerpt=1）后的敏感性（仍按标题/正文长度匹配）：",
        "",
        "| 指标 | 匹配维度 | 人类−AIGC [95% CI] | 人类−探索集 [95% CI] |",
        "|---|---|---:|---:|",
    ]
    for metric in ("coverage", "jaccard"):
        for mode, label in (("title_matched", "标题长度"), ("body_matched", "正文长度")):
            g1 = sens[mode]["aigc"][metric]
            g2 = sens[mode]["exploring"][metric]
            lines.append(f"| {METRIC_LABELS[metric]} | {label} | "
                         f"{fmt_ci(g1)} | {fmt_ci(g2)} |")
    lines += [
        "",
        f"（剔除后人类/AIGC/探索集 coverage 均值 "
        f"{sens['means']['human']['coverage_mean']:.3f} / "
        f"{sens['means']['aigc']['coverage_mean']:.3f} / "
        f"{sens['means']['exploring']['coverage_mean']:.3f}；n="
        f"{sens['means']['human']['n']:,} / {sens['means']['aigc']['n']:,} / "
        f"{sens['means']['exploring']['n']:,}。）",
        "",
        "## 2 探索集内的 P(AI) 梯度",
        "",
        "| 指标 | n | 单元内加权 ρ | 负单元占比 | 标题长度分层 ρ | 正文长度分层 ρ | 首尾十分位 δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRIC_ORDER:
        g = grad["metrics"][metric]
        lines.append(f"| {METRIC_LABELS[metric]} | {grad['n']:,} | "
                     f"{g['weighted_rho']:+.4f} | {g['neg_cell_share']:.1%} | "
                     f"{g['title_stratified_rho']:+.4f} | {g['body_stratified_rho']:+.4f} | "
                     f"{g['cliff_delta_top_vs_bottom']:+.3f} |")
    lines += [
        "",
        f"（{grad['n']:,} 条 / {grad['n_cells']} 个「月份×领域」单元；"
        f"标题分层 {grad['title_cells']} 个、正文分层 {grad['body_cells']} 个单元；"
        "δ = P(AI) 前 10% vs 后 10% 的 Cliff's δ。）",
        "",
        "## 3 互动偏效应（月份×领域 FE + 整群 bootstrap 300 次）",
        "",
        "| 特征 | 点赞 β [95% CI] | 收藏 β [95% CI] | 评论 β [95% CI] |",
        "|---|---:|---:|---:|",
    ]

    def cell(fe: dict, name: str) -> str:
        return (f"{fe['coef'][name]:+.4f} [{fe['ci_low'][name]:+.4f}, "
                f"{fe['ci_high'][name]:+.4f}]")
    names = ["coverage", "jaccard", "excerpt", "title_tokens", "p_ai",
             "content_len", "n_hashtags"]
    labels = {"coverage": "标题覆盖率", "jaccard": "Jaccard",
              "excerpt": "标题原样摘录", "title_tokens": "标题词元数",
              "p_ai": "P(AI)", "content_len": "正文长度", "n_hashtags": "标签数"}
    for n in names:
        lines.append(f"| {labels[n]} | {cell(eng['likes'], n)} | "
                     f"{cell(eng['collections'], n)} | {cell(eng['comments'], n)} |")
    lines += [
        "",
        f"（n={eng['likes']['n_obs']:,}，{eng['likes']['n_cells']} 个单元；"
        "特征先 z-score 再单元内去均值，β 为每 1 SD 的偏效应；202301 起、正文 ≥50 字。）",
        "",
        "## 4 AIGC 家族指纹",
        "",
        "| 家族 | n | 标题覆盖率 | 覆盖率中位数 | 正文长度标准化覆盖率 | Jaccard | 标题原样摘录率 | 标题词元数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in sorted(fam["families"], key=lambda x: -fam["families"][x]["coverage_mean"]):
        b = fam["families"][f]
        lines.append(f"| {FAMILY_NAME.get(f, f)} | {b['n']:,} | {b['coverage_mean']:.3f} | "
                     f"{b['coverage_median']:.3f} | {b['coverage_length_std']:+.4f} | "
                     f"{b['jaccard_mean']:.3f} | {b['excerpt_share']:.2%} | "
                     f"{b['title_tokens_mean']:.1f} |")
    lines += [
        "",
        "## 5 探索集季度趋势（固定领域权重）",
        "",
        "| 季度 | n | 标题覆盖率 | 摘录率 | 覆盖率（领域标准化） | 摘录率（领域标准化） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i, qq in enumerate(q["quarters"]):
        lines.append(f"| {qq} | {q['n'][i]:,} | {q['raw']['coverage'][i]:.3f} | "
                     f"{q['raw']['excerpt'][i]:.2%} | {q['domain_std']['coverage'][i]:.3f} | "
                     f"{q['domain_std']['excerpt'][i]:.2%} |")
    lines += [
        "",
        "| 指标 | 原始 ρ (p) | 领域标准化 ρ (p) |",
        "|---|---:|---:|",
    ]
    for metric in ("coverage", "jaccard", "excerpt"):
        tr = q["trend"][metric]
        lines.append(f"| {METRIC_LABELS[metric]} | {tr['rho_raw']:+.3f} ({tr['p_raw']:.3g}) | "
                     f"{tr['rho_domain_std']:+.3f} ({tr['p_domain_std']:.3g}) |")
    lines += [
        "",
        "## 6 human vs AIGC 判别力",
        "",
        "| 指标 | AUC [95% CI] |",
        "|---|---:|",
    ]
    for metric in METRIC_ORDER:
        a = disc["markers"][metric]
        lines.append(f"| {METRIC_LABELS[metric]} | {a['auc']:.3f} "
                     f"[{a['ci_low']:.3f}, {a['ci_high']:.3f}] |")
    for metric, label in (("title_tokens", "标题词元数"), ("body_len", "正文长度")):
        a = disc["context"][metric]
        lines.append(f"| {label} | {a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}] |")
    lines.append(f"| 三项耦合 LR（5 折 CV） | {disc['cv_coupling']['auc']:.3f} |")
    lines.append(f"| 标题/正文长度 LR（5 折 CV） | {disc['cv_context']['auc']:.3f} |")
    lines.append(f"| 五项特征 LR（5 折 CV） | {disc['cv_lr']['auc']:.3f} |")
    lines += [
        "",
        f"（human n={disc['n_human']:,}，AIGC n={disc['n_aigc']:,}；"
        "单指标 AUC 为 human=1 方向的得分，bootstrap 300 次。）",
        "",
        "## 7 方法注记",
        "",
        "- 标题原样摘录率对标题字段的截断方式敏感：探索集有部分标题本身就是正文开头"
        "（标题被平台截断/复用），因此该指标应理解为「标题是否为正文片段」，"
        "而不是「标题写得好不好」。",
        "- 覆盖率与 Jaccard 受标题/正文长度影响，正文已给出标题长度分箱、正文长度"
        "分箱两套匹配差异；梯度分析同时报告标题长度与正文长度分层后的单元内 ρ。",
        "- 所有结果均为观察性关联，不对标题写法与互动做因果解读。",
    ]
    with open(BASE + "title_body_lexical_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    print("扫描三个语料（全量，进程池并行）…")
    scans = dict(zip(DATASETS, run_parallel(
        [partial(scan_corpus, path) for path in DATASETS.values()])))
    for name, sc in scans.items():
        print(f"  {name}: {sc['n_total']:,} 条记录")

    scores = pd.read_csv(SCORES_PATH, usecols=["p_ai", "content_len"])
    features = pd.read_csv(FEATURES_PATH)
    if len(scores) != scans["exploring"]["n_total"]:
        raise SystemExit("scores 与 exploring JSONL 行数不一致")
    if len(features) != scans["exploring"]["n_total"]:
        raise SystemExit("features 与 exploring JSONL 行数不一致")
    require_row_alignment(scans["exploring"], features, "features",
                          fields=("month", "domain", "likes", "collections",
                                  "comments"))
    require_row_alignment(scans["exploring"], scores, "scores",
                          fields=("month", "domain", "likes", "collections",
                                  "comments"))
    content_len_mismatch = int((scores["content_len"].to_numpy()
                                != features["content_len"].to_numpy()).sum())
    content_len_drift = int((features["content_len"].to_numpy()
                             != scans["exploring"]["body_len"]).sum())

    keeps = {name: dedup_keep(sc) for name, sc in scans.items()}
    subsets = {name: subset_block(scans[name], keeps[name]) for name in SUBSET_ORDER}
    # 供 ECDF 绘图用的覆盖率向量（不写入 JSON）
    for name in SUBSET_ORDER:
        sel = valid_sel(scans[name], keeps[name])
        subsets[name]["_coverage_values"] = scans[name]["coverage"][sel]
        print(f"  {name}: 去重后 {len(keeps[name]):,} 条，有效 {len(sel):,} 条，"
              f"覆盖率均值 {subsets[name]['coverage']['mean']:.4f}，"
              f"摘录率 {subsets[name]['excerpt']['share']:.2%}")

    domain_std = domain_standardized_block(scans, keeps)
    ctrl = length_control_block(scans, keeps)
    grad, sel_expl, p_ai = gradient_block(scans, keeps, scores)
    eng = engagement_block(scans, keeps, scores, features, sel_expl, p_ai)
    fam = family_block(scans, keeps)
    q = quarterly_block(scans, keeps, sel_expl)
    disc = discrimination_block(scans, keeps)

    payload = {
        "meta": {
            "min_body_len": MIN_BODY_LEN, "min_cell_n": MIN_CELL_N,
            "month_from": MONTH_FROM, "n_boot": N_BOOT, "auc_boot": AUC_BOOT,
            "seed": BOOT_SEED, "title_bins": [lab for _, _, lab in TITLE_BINS],
            "content_len_mismatch": content_len_mismatch,
            "content_len_drift": content_len_drift,
        },
        "subsets": {name: {k: v for k, v in subsets[name].items()
                           if not k.startswith("_")} for name in SUBSET_ORDER},
        "domain_standardized": domain_std,
        "length_control": {k: v for k, v in ctrl.items() if k != "title_bins"},
        "title_length_bins": ctrl["title_bins"],
        "excerpt_sensitivity": excerpt_sensitivity_block(scans, keeps),
        "gradient": grad,
        "engagement": eng,
        "family": fam,
        "quarterly": q,
        "discrimination": disc,
    }
    save_json(payload, BASE + "title_body_lexical_results.json")
    write_markdown(payload)
    plot_lexicon(subsets, ctrl)
    plot_gradient(grad, eng, fam, q, disc)
    print("标题-正文词汇耦合分析完成：insights/title_body_lexicon.png、"
          "insights/title_body_gradient.png、title_body_lexical_results.{json,md}")


if __name__ == "__main__":
    main()
