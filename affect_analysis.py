# -*- coding: utf-8 -*-
"""Affect and valence deep dive (joy, affection, anxiety, sadness, anger).

The other axes describe *how* a post is written.  This axis asks how it
*feels*: which emotions are expressed, whether the overall valence is positive
or negative, and how those signals relate to the detector, engagement, model
families and time.  The lexicon is a priori and is never tuned on labels.

Questions:
1. Do human, AIGC and wild post-LLM posts differ in emotional expression?
2. Inside the exploring set, which emotions track the detector's P(AI)?
3. Does emotional expression carry independent engagement returns?
4. Which generator families over/under-use each emotion, and how did the mix
   evolve over quarters?

Protocol is identical to the other deep-dive axes: bodies are cleaned with
``analysis_utils.clean_body``, human/exploring records are deduplicated by
``note_id`` (keeping the metadata-richest record), AIGC is processed record by
record, and exploring P(AI) is row-aligned with ``paper_exploring_scores``.
Row order is verified on raw month/domain/engagement fields rather than on
``content_len``, which can drift when the cleaning regex changes.

Outputs:
  insights/affect_lexicon.png
  insights/affect_gradient.png
  affect_results.json
  affect_results.md
"""
from __future__ import annotations

import re
from array import array
from collections import defaultdict
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from analysis_utils import (BASE, INSIGHTS, PALETTE, body_cache, cell_groups,
                            cell_rank_cache, clean_body, cliffs_delta,
                            content_of, cv_auc, decile_table, dedup_indices,
                            domain_standardized_series, engagement_of,
                            init_plot_style, length_bin_codes,
                            length_matched_mean_gap, parse_local_time,
                            require_row_alignment, run_parallel, save_fig,
                            save_json, within_cell_spearman)
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
BOOT_SEED = 42
MIN_QUARTER_N = 300
MIN_FAMILY_N = 100

CATEGORY_WORDS = {
    "joy": ("喜悦", [
        "开心", "快乐", "高兴", "愉快", "兴奋", "惊喜", "幸福", "满足",
        "满意", "治愈", "美好", "可爱", "漂亮", "好看", "好吃", "好喝",
        "舒服", "轻松", "放心", "省心", "实用", "划算", "超值", "真香",
        "太棒", "绝了", "好评", "值得", "惊艳", "宝藏", "推荐", "种草",
        "回购", "空瓶", "好用", "神器", "必入", "不踩雷", "无雷", "笑死",
        "哈哈", "嘻嘻", "嘿嘿", "美滋滋", "优秀", "完美", "赞", "棒",
        "开森", "yyds", "绝绝子", "爱了",
    ]),
    "affection": ("喜爱依恋", [
        "爱", "喜爱", "喜欢", "心动", "心水", "宝贝", "亲爱的", "姐妹",
        "宝宝", "老公", "老婆", "男朋友", "女朋友", "闺蜜", "家人",
        "妈妈", "爸爸", "陪伴", "温暖", "温柔", "贴心", "拥抱", "亲亲",
        "想你", "思念", "恋爱", "甜蜜", "甜", "感恩", "感谢", "谢谢",
        "珍惜", "守护", "萌", "心软", "依赖", "安全感", "归属感",
    ]),
    "anxiety": ("焦虑担忧", [
        "焦虑", "担心", "害怕", "紧张", "压力", "纠结", "犹豫", "不安",
        "恐慌", "慌", "急", "愁", "烦", "烦恼", "郁闷", "失眠", "熬夜",
        "疲惫", "心累", "无奈", "无助", "迷茫", "内耗", "emo", "抑郁",
        "可怕", "恐怖", "吓人", "忐忑", "心慌", "焦躁", "烦躁",
        "压力山大", "精神内耗", "七上八下", "发愁", "犯愁",
    ]),
    "sadness": ("悲伤失落", [
        "难过", "伤心", "悲伤", "哭", "眼泪", "泪目", "失落", "沮丧",
        "低落", "孤独", "寂寞", "遗憾", "可惜", "心疼", "心碎", "委屈",
        "绝望", "丧", "消沉", "怀念", "舍不得", "离别", "分手", "失去",
        "离开", "错过", "后悔", "惨", "可怜", "叹气", "鼻酸", "破防",
        "泪崩", "压抑", "麻木", "空落落", "心酸", "酸楚", "痛苦",
        "难受", "心寒", "寒心",
    ]),
    "anger": ("愤怒批评", [
        "生气", "愤怒", "火大", "暴躁", "气死", "恼火", "发火", "骂",
        "吐槽", "无语", "离谱", "恶心", "讨厌", "憎恨", "恨", "坑",
        "骗", "欺骗", "虚假", "智商税", "割韭菜", "黑心", "差评",
        "避雷", "劝退", "翻车", "踩雷", "气人", "抓狂", "崩溃", "怒",
        "不满", "抗议", "投诉", "维权", "曝光", "举报", "垃圾", "糟糕",
        "失望", "气炸", "气哭", "坑人", "踩坑", "避坑", "不值得",
        "浪费", "难用", "难吃", "难看",
    ]),
}
CATEGORY_LABELS = {k: v[0] for k, v in CATEGORY_WORDS.items()}
CATEGORY_COLORS = {
    "joy": "#DD8452",
    "affection": "#C44E52",
    "anxiety": "#8172B3",
    "sadness": "#4C72B0",
    "anger": "#937860",
    "positive": "#55A868",
    "negative": "#B56576",
    "net": "#333333",
}
CATEGORY_ORDER = ["joy", "affection", "anxiety", "sadness", "anger"]
POSITIVE_PARTS = ["joy", "affection"]
NEGATIVE_PARTS = ["anxiety", "sadness", "anger"]
SCORE_ORDER = CATEGORY_ORDER + ["positive", "negative", "net"]
PLOT_ORDER = CATEGORY_ORDER + ["net"]
SUBSET_ORDER = ["human", "aigc", "exploring"]
ALL_WORDS = [w for k in CATEGORY_ORDER for w in CATEGORY_WORDS[k][1]]
WORD_CATEGORY = {w: k for k in CATEGORY_ORDER for w in CATEGORY_WORDS[k][1]}
assert len(WORD_CATEGORY) == len(ALL_WORDS), "affect lexicon has duplicate words"

_WORD_RE = re.compile(
    "|".join(sorted((re.escape(w) for w in ALL_WORDS), key=len, reverse=True)))
_WORD_INDEX = {w: i for i, w in enumerate(ALL_WORDS)}

FAMILY_NAME = {"anthropic": "Claude", "deepseek": "DeepSeek", "gemini": "Gemini",
               "glm": "GLM", "gpt": "GPT", "qwen": "Qwen"}
FAMILIES = ["anthropic", "deepseek", "gemini", "glm", "gpt", "qwen"]


def word_counts(text: str) -> dict:
    """Count a-priori affect markers with longest-match semantics."""
    out: dict = {k: {} for k in CATEGORY_ORDER}
    words: dict = {}
    for match in _WORD_RE.finditer(text):
        w = match.group(0)
        k = WORD_CATEGORY[w]
        out[k][w] = out[k].get(w, 0) + 1
        words[w] = words.get(w, 0) + 1
    out["_words"] = words
    return out


def density_per_100(count: int, body_len: int) -> float:
    """Occurrences per 100 cleaned body characters (0 for an empty body)."""
    return count / body_len * 100 if body_len > 0 else 0.0


def scan_corpus(path: str) -> dict:
    """Stream one corpus and keep only the arrays this analysis needs."""
    body_len, note_id, meta, month, domain = [], [], [], [], []
    family, model, likes = [], [], []
    cat_counts = {k: [] for k in CATEGORY_ORDER}
    positive, negative = [], []
    flat_words = array("i")
    n_words = len(ALL_WORDS)
    n_total = 0
    cached = body_cache(word_counts, max_entries=60_000)
    for rec in iter_cached(path):
        n_total += 1
        content = content_of(rec)
        body = clean_body(content)
        length = len(body)
        counts = cached(body)
        body_len.append(length)
        note_id.append(rec.get("note_id"))
        dom = rec.get("domain")
        domain.append(dom if dom not in (None, "") else "未知")
        likes_v, colls, comms = engagement_of(rec)
        likes.append(likes_v)
        has_meta = bool(rec.get("user_id")) and dom not in (None, "", "未知")
        dt = parse_local_time(rec.get("local_time"))
        month.append(f"{dt.year:04d}{dt.month:02d}" if dt else None)
        meta.append(int(has_meta) + int(bool(likes_v)) + int(bool(colls))
                    + int(bool(comms)) + int(dt is not None)
                    + int(bool(content.strip())))
        family.append(rec.get("model_family") or "")
        model.append(rec.get("model") or "")
        pos = neg = 0
        for k in CATEGORY_ORDER:
            c = sum(counts[k].values())
            cat_counts[k].append(c)
            if k in POSITIVE_PARTS:
                pos += c
            if k in NEGATIVE_PARTS:
                neg += c
        positive.append(pos)
        negative.append(neg)
        row = [0] * n_words
        for w, n in counts["_words"].items():
            row[_WORD_INDEX[w]] = n
        flat_words.extend(row)
    word_matrix = np.frombuffer(flat_words, dtype=np.int32).reshape(-1, n_words)
    return {
        "path": path, "n_total": n_total,
        "body_len": np.asarray(body_len, dtype=np.int32),
        "note_id": note_id, "meta": np.asarray(meta, dtype=np.int32),
        "month": month, "domain": domain,
        "likes": np.asarray(likes, dtype=float),
        "family": family, "model": model,
        "cat_counts": {k: np.asarray(v, dtype=np.int32)
                       for k, v in cat_counts.items()},
        "positive_counts": np.asarray(positive, dtype=np.int32),
        "negative_counts": np.asarray(negative, dtype=np.int32),
        "word_counts": {w: word_matrix[:, i] for i, w in enumerate(ALL_WORDS)},
    }


def dedup_keep(scan: dict) -> np.ndarray:
    """Deduplicate by note_id, keeping the metadata-richest record."""
    return np.asarray(sorted(dedup_indices(scan["note_id"], scan["meta"])),
                      dtype=int)


def _counts_for(scan: dict, key: str):
    if key == "positive":
        return scan["positive_counts"]
    if key == "negative":
        return scan["negative_counts"]
    if key == "net":
        return scan["positive_counts"] - scan["negative_counts"]
    return scan["cat_counts"][key]


def rates_block(scan: dict, keep: np.ndarray,
                min_len: int = MIN_BODY_LEN) -> dict:
    """Post-level incidence and character-level density for one subset."""
    sel = keep[scan["body_len"][keep] >= min_len]
    n = int(len(sel))
    chars = int(scan["body_len"][sel].sum())
    categories = {}
    for k in SCORE_ORDER:
        c = _counts_for(scan, k)[sel]
        categories[k] = {
            "n": n, "chars": chars,
            "density100": round(float(c.sum() / chars * 100), 4) if chars else 0.0,
            "share": round(float((c > 0).mean()), 4) if n else 0.0,
        }
    words = {}
    for w, arr in scan["word_counts"].items():
        c = arr[sel]
        words[w] = {
            "n": n,
            "per10k_chars": round(float(c.sum() / chars * 10000), 4) if chars else 0.0,
            "share": round(float((c > 0).mean()), 4) if n else 0.0,
        }
    return {"n": n, "chars": chars, "categories": categories, "words": words}


def _density(scan: dict, keep: np.ndarray, key: str,
             min_len: int = MIN_BODY_LEN):
    sel = keep[scan["body_len"][keep] >= min_len]
    body = scan["body_len"][sel].astype(float)
    count = _counts_for(scan, key)[sel].astype(float)
    return count / np.maximum(body, 1.0) * 100, body, sel


def length_matched_block(scans: dict, keeps: dict) -> dict:
    """Length-matched human-minus-other gaps for incidence and density."""
    out = {}
    for k in SCORE_ORDER:
        out[k] = {}
        hscan = scans["human"]
        hkeep = keeps["human"]
        hsel = hkeep[hscan["body_len"][hkeep] >= MIN_BODY_LEN]
        hlen = hscan["body_len"][hsel].astype(float)
        hcount = _counts_for(hscan, k)[hsel].astype(float)
        hind = (hcount > 0).astype(float)
        hdens = hcount / np.maximum(hlen, 1.0) * 100
        for other in ("aigc", "exploring"):
            oscan = scans[other]
            okeep = keeps[other]
            osel = okeep[oscan["body_len"][okeep] >= MIN_BODY_LEN]
            olen = oscan["body_len"][osel].astype(float)
            ocount = _counts_for(oscan, k)[osel].astype(float)
            oind = (ocount > 0).astype(float)
            odens = ocount / np.maximum(olen, 1.0) * 100
            ind = length_matched_mean_gap(hind, hlen, oind, olen)
            dens = length_matched_mean_gap(hdens, hlen, odens, olen)
            out[k][other] = {
                "incidence_gap": {key: round(float(v), 5) if key != "used_bins"
                                  and key not in ("n_a", "n_b") else int(v)
                                  for key, v in ind.items()},
                "density_gap": {key: round(float(v), 5) if key != "used_bins"
                                and key not in ("n_a", "n_b") else int(v)
                                for key, v in dens.items()},
            }
    return out


def gradient_block(scans: dict, keeps: dict, scores: pd.DataFrame) -> dict:
    """Within-cell P(AI) gradients for emotions and individual markers."""
    expl = scans["exploring"]
    keep = keeps["exploring"]
    sel = keep[expl["body_len"][keep] >= MIN_BODY_LEN]
    p_ai = scores["p_ai"].to_numpy(dtype=float)[sel]
    body = expl["body_len"][sel].astype(float)
    months = np.asarray([expl["month"][i] or "未知" for i in sel])
    domains = np.asarray([expl["domain"][i] for i in sel])
    cells = pd.factorize(pd.Series([f"{m}|{d}" for m, d in zip(months, domains)]),
                         sort=True)[0]
    strat = pd.factorize(pd.Series(
        [f"{m}|{d}|{b}" for m, d, b in
         zip(months, domains, length_bin_codes(body))]), sort=True)[0]
    groups = cell_groups(cells, MIN_CELL_N)
    groups_strat = cell_groups(strat, MIN_CELL_N)
    p_ai_ranks = cell_rank_cache(p_ai, groups)
    p_ai_ranks_strat = cell_rank_cache(p_ai, groups_strat)
    p_hi, p_lo = float(np.quantile(p_ai, 0.9)), float(np.quantile(p_ai, 0.1))
    hi_p, lo_p = p_ai >= p_hi, p_ai <= p_lo
    categories = {}
    for k in SCORE_ORDER:
        dens = _counts_for(expl, k)[sel].astype(float) / np.maximum(body, 1.0) * 100
        dec = decile_table(dens, p_ai)
        wc = within_cell_spearman(dens, p_ai, cells, min_cell_n=MIN_CELL_N,
                                  groups=groups, y_ranks=p_ai_ranks)
        st = within_cell_spearman(dens, p_ai, strat, min_cell_n=MIN_CELL_N,
                                  groups=groups_strat, y_ranks=p_ai_ranks_strat)
        delta, dp = cliffs_delta(dens[hi_p], dens[lo_p])
        categories[k] = {
            "deciles": dec,
            "weighted_rho": round(float(wc["weighted_rho"]), 4),
            "neg_cell_share": round(float(wc["neg_cell_share"]), 4),
            "n_cells": int(wc["n_cells"]),
            "stratified_rho": round(float(st["weighted_rho"]), 4),
            "stratified_cells": int(st["n_cells"]),
            "cliff_delta_top_vs_bottom": round(float(delta), 4),
            "cliff_p": float(dp),
            "decile_p_range": [dec[0]["p_range"][0], dec[-1]["p_range"][1]]
            if dec else [None, None],
        }
    words = {}
    for w in ALL_WORDS:
        dens = expl["word_counts"][w][sel].astype(float) / np.maximum(body, 1.0) * 100
        if dens.std() == 0:
            continue
        wc = within_cell_spearman(dens, p_ai, cells, min_cell_n=MIN_CELL_N,
                                  groups=groups, y_ranks=p_ai_ranks)
        words[w] = {
            "category": WORD_CATEGORY[w],
            "weighted_rho": round(float(wc["weighted_rho"]), 4),
            "neg_cell_share": round(float(wc["neg_cell_share"]), 4),
            "n_cells": int(wc["n_cells"]),
        }
    return {
        "n": int(len(sel)),
        "short_excluded": int((expl["body_len"][keep] < MIN_BODY_LEN).sum()),
        "n_cells": int(len(np.unique(cells))),
        "categories": categories, "words": words,
    }


def separability_block(scans: dict, keeps: dict) -> dict:
    """Human-vs-AIGC discriminability of the affect markers."""
    from sklearn.metrics import roc_auc_score

    rows = []
    for name in ("human", "aigc"):
        scan = scans[name]
        keep = keeps[name]
        sel = keep[scan["body_len"][keep] >= MIN_BODY_LEN]
        body = scan["body_len"][sel].astype(float)
        row = {"dataset": name}
        for k in SCORE_ORDER:
            c = _counts_for(scan, k)[sel].astype(float)
            row[k] = c / np.maximum(body, 1.0) * 100
        rows.append(pd.DataFrame(row))
    df = pd.concat(rows, ignore_index=True)
    y = (df["dataset"] == "aigc").astype(int).to_numpy()
    univariate = {}
    for k in SCORE_ORDER:
        univariate[k] = round(float(roc_auc_score(y, df[k].to_numpy())), 4)
    combined = cv_auc(df, SCORE_ORDER)
    return {"n_human": int((df["dataset"] == "human").sum()),
            "n_aigc": int((df["dataset"] == "aigc").sum()),
            "univariate_auc": univariate,
            "combined_cv": {k: (round(float(v), 4) if isinstance(v, float) else v)
                            for k, v in combined.items()}}


def engagement_block(scans: dict, scores: pd.DataFrame) -> dict:
    """Month x domain FE models with cluster-bootstrap CIs (same protocol)."""
    scan = scans["exploring"]
    model = pd.read_csv(FEATURES_PATH,
                        usecols=["month", "domain", "content_len", "n_hashtags",
                                 "likes", "collections", "comments"])
    if len(model) != scan["n_total"]:
        raise SystemExit("features 与 exploring 行数不一致")
    require_row_alignment(scan, model, "features")
    text_len_drift = int((model["content_len"].to_numpy()
                          != scan["body_len"]).sum())
    body = np.maximum(scan["body_len"].astype(float), 1.0)
    # 净效价是五类情绪的线性组合，放入 FE 会完全共线；互动模型只保留
    # 五类原始情绪，净效价改在描述统计与 P(AI) 梯度中报告。
    fe_keys = CATEGORY_ORDER
    marker = np.column_stack([
        _counts_for(scan, k).astype(float) / body * 100 for k in fe_keys])
    features = fe_keys + ["p_ai", "content_len", "n_hashtags"]
    X = np.column_stack([
        marker,
        scores["p_ai"].to_numpy(dtype=float),
        scan["body_len"].astype(float),
        model["n_hashtags"].to_numpy(dtype=float),
    ])
    target_values = {
        "likes": np.log10(1 + np.clip(model["likes"].to_numpy(dtype=float), 0, None)),
        "collections": np.log10(1 + np.clip(model["collections"].to_numpy(dtype=float), 0, None)),
        "comments": np.log10(1 + np.clip(model["comments"].to_numpy(dtype=float), 0, None)),
    }
    months = model["month"].astype(str).to_numpy()
    domains = model["domain"].astype(str).to_numpy()
    keep = (months >= MONTH_FROM) & (scan["body_len"] >= MIN_BODY_LEN)
    cells_raw = np.asarray([f"{m}|{d}" for m, d in zip(months, domains)])
    out = {}
    for target, y in target_values.items():
        Xk, yk, ck = X[keep], y[keep], cells_raw[keep]
        prep = prepare_fe(Xk, ck, MIN_CELL_N)
        fit = fe_ols(Xk, yk, ck, prepared=prep)
        boot = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                        fit["cell_codes"], n_boot=N_BOOT,
                                        seed=BOOT_SEED)
        ci = np.percentile(boot, [2.5, 97.5], axis=0)
        out[target] = {
            "features": features,
            "coef": {n: round(float(v), 4) for n, v in zip(features, fit["coef"])},
            "ci_low": {n: round(float(v), 4) for n, v in zip(features, ci[0])},
            "ci_high": {n: round(float(v), 4) for n, v in zip(features, ci[1])},
            "n_obs": int(fit["n_obs"]), "n_cells": int(fit["n_cells"]),
        }
    return {"month_from": MONTH_FROM, "min_body_len": MIN_BODY_LEN,
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "text_len_drift": text_len_drift, "fe": out}


def family_block(scan: dict) -> dict:
    """Density fingerprints by generator family and exact model."""
    ok = scan["body_len"] >= MIN_BODY_LEN
    fam: dict = defaultdict(lambda: {"n": 0, "chars": 0, "counts": defaultdict(int)})
    mod: dict = defaultdict(lambda: {"n": 0, "chars": 0, "counts": defaultdict(int)})
    for i in np.flatnonzero(ok):
        for agg, key in ((fam, scan["family"][i]), (mod, scan["model"][i])):
            slot = agg[key]
            slot["n"] += 1
            slot["chars"] += int(scan["body_len"][i])
            for k in SCORE_ORDER:
                slot["counts"][k] += int(_counts_for(scan, k)[i])

    def table(agg):
        out = {}
        for key, slot in agg.items():
            if not key:
                continue
            out[key] = {
                "n": slot["n"],
                "density100": {
                    k: round(slot["counts"][k] / slot["chars"] * 100, 4)
                    if slot["chars"] else 0.0
                    for k in SCORE_ORDER},
            }
        return out

    return {"families": table(fam), "models": table(mod)}


def quarterly_block(scan: dict, keep: np.ndarray) -> dict:
    """Quarterly means, fixed-domain-weight series and trends."""
    sel = keep[scan["body_len"][keep] >= MIN_BODY_LEN]
    month = np.asarray([scan["month"][i] or "未知" for i in sel])
    quarter = np.asarray([
        f"{m[:4]}Q{(int(m[4:6]) - 1) // 3 + 1}" if m != "未知" else "未知"
        for m in month])
    df = pd.DataFrame({
        "quarter": quarter,
        "domain": [scan["domain"][i] for i in sel],
        "body_len": scan["body_len"][sel].astype(float),
    })
    for k in SCORE_ORDER:
        df[k] = _counts_for(scan, k)[sel].astype(float) / df["body_len"] * 100
    df = df[df["quarter"] != "未知"]
    counts = df.groupby("quarter").size()
    quarters = sorted(counts[counts >= MIN_QUARTER_N].index)
    df = df[df["quarter"].isin(quarters)]
    out = {"quarters": quarters, "n": {q: int(counts[q]) for q in quarters},
           "raw": {}, "domain_std": {}, "trend": {}, "share_any": {}}
    for k in SCORE_ORDER:
        raw = df.groupby("quarter")[k].mean().reindex(quarters)
        std = domain_standardized_series(df, k, "quarter").reindex(quarters)
        rho, p = spearmanr(range(len(quarters)), raw.to_numpy())
        rho_s, p_s = spearmanr(range(len(quarters)), std.to_numpy())
        out["raw"][k] = [round(float(v), 4) for v in raw]
        out["domain_std"][k] = [round(float(v), 4) for v in std]
        out["trend"][k] = {
            "rho": round(float(rho), 4), "p": float(p),
            "rho_domain_std": round(float(rho_s), 4),
            "p_domain_std": float(p_s),
        }
        share = df.assign(hit=df[k] > 0).groupby("quarter")["hit"].mean()
        out["share_any"][k] = [round(float(v), 4)
                               for v in share.reindex(quarters)]
    return out


def plot_lexicon(subset_rates: dict, word_table: dict) -> None:
    init_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(19.0, 8.8),
                             gridspec_kw={"width_ratios": [1.0, 1.55]})
    ax = axes[0]
    y = np.arange(len(SCORE_ORDER))
    height = 0.24
    subset_labels = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
                     "exploring": "探索集(后LLM)"}
    for j, subset in enumerate(SUBSET_ORDER):
        vals = [subset_rates[subset]["categories"][k]["density100"]
                for k in SCORE_ORDER]
        pos = y + (j - 1) * height
        ax.barh(pos, vals, height=height, color=PALETTE[subset],
                label=subset_labels[subset])
        for yy, v in zip(pos, vals):
            ax.annotate(f"{v:.2f}", xy=(v, yy), xytext=(3, -3),
                        textcoords="offset points", fontsize=7.2)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([CATEGORY_LABELS.get(k, k) for k in SCORE_ORDER])
    ax.set_xlabel(f"密度（次/百字，正文 ≥ {MIN_BODY_LEN} 字）")
    ax.set_title("(a) 情绪词使用的跨子集密度")
    ax.legend(loc="lower right")

    ax = axes[1]
    rows = [(w, rec["category"],
             rec["aigc"]["per10k_chars"] - rec["human"]["per10k_chars"])
            for w, rec in word_table.items()]
    rows.sort(key=lambda r: r[2])
    rows = rows[:16] + rows[-16:]
    ypos = np.arange(len(rows))
    diffs = np.asarray([r[2] for r in rows])
    ax.barh(ypos, diffs,
            color=[CATEGORY_COLORS[r[1]] for r in rows], height=0.72, alpha=0.9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(CATEGORY_COLORS[r[1]])
    for i, v in enumerate(diffs):
        ax.annotate(f"{v:+.2f}", xy=(v, ypos[i]),
                    xytext=(4 if v >= 0 else -4, -3),
                    textcoords="offset points", fontsize=7.2,
                    ha="left" if v >= 0 else "right")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("AIGC − 人类 使用率差（次/万字符；越右=生成集越爱用）")
    ax.set_title("(b) 逐词 AIGC 过量使用（最欠用/最过量各 16 个）")
    handles = [plt.Line2D([], [], color=CATEGORY_COLORS[k], lw=6,
                          label=CATEGORY_LABELS[k]) for k in CATEGORY_ORDER]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    save_fig(fig, INSIGHTS + "affect_lexicon.png")


def plot_gradient(grad: dict, eng: dict, family: dict,
                  quarterly: dict, sep: dict) -> None:
    init_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 13.0))

    ax = axes[0, 0]
    for k in PLOT_ORDER:
        dec = grad["categories"][k]["deciles"]
        x = [d["decile"] for d in dec]
        yv = [d["mean"] for d in dec]
        lo = [d["ci_low"] for d in dec]
        hi = [d["ci_high"] for d in dec]
        ax.plot(x, yv, "o-", color=CATEGORY_COLORS[k], lw=1.8, ms=4.5,
                label=f"{CATEGORY_LABELS.get(k, '净效价')}"
                      f"（ρ={grad['categories'][k]['weighted_rho']:+.3f}）")
        ax.fill_between(x, lo, hi, color=CATEGORY_COLORS[k], alpha=0.14)
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("检测器 P(AI) 十分位（1=最像人，10=最像 AI）")
    ax.set_ylabel("密度（次/百字；净效价为有符号值）")
    ax.set_title("(a) 探索集：情绪表达随 P(AI) 的梯度（95% CI）")
    ax.legend(fontsize=7.6)

    ax = axes[0, 1]
    items = sorted(grad["words"].items(), key=lambda kv: kv[1]["weighted_rho"])
    items = items[:12] + items[-12:]
    ypos = np.arange(len(items))
    vals = [kv[1]["weighted_rho"] for kv in items]
    ax.barh(ypos, vals,
            color=[CATEGORY_COLORS[kv[1]["category"]] for kv in items],
            alpha=0.9, height=0.72)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([kv[0] for kv in items], fontsize=8.5)
    for tick, kv in zip(ax.get_yticklabels(), items):
        tick.set_color(CATEGORY_COLORS[kv[1]["category"]])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.3f}", xy=(v, ypos[i]),
                    xytext=(4 if v >= 0 else -4, -2.5),
                    textcoords="offset points", fontsize=7.2,
                    ha="left" if v >= 0 else "right")
    ax.set_xlabel("月份×领域单元内加权 Spearman ρ（密度 vs P(AI)）")
    ax.set_title("(b) 逐词野外 P(AI) 梯度（最负/最正各 12 个）")
    handles = [plt.Line2D([], [], color=CATEGORY_COLORS[k], lw=6,
                          label=CATEGORY_LABELS[k]) for k in CATEGORY_ORDER]
    ax.legend(handles=handles, loc="lower left", fontsize=7.6)

    ax = axes[0, 2]
    fe_likes = eng["fe"]["likes"]
    order = list(reversed(fe_likes["features"]))
    ypos = np.arange(len(order))
    for target, marker, offset, color, label in (
            ("likes", "o", -0.24, "#4C72B0", "点赞"),
            ("collections", "^", 0.0, "#DD8452", "收藏"),
            ("comments", "s", 0.24, "#55A868", "评论")):
        fe = eng["fe"][target]
        for i, name in enumerate(order):
            b = fe["coef"][name]
            lo, hi = fe["ci_low"][name], fe["ci_high"][name]
            sig = lo > 0 or hi < 0
            col = color if sig else "#9E9E9E"
            ax.plot([lo, hi], [ypos[i] + offset, ypos[i] + offset], color=col,
                    lw=1.6, solid_capstyle="butt", zorder=2)
            ax.scatter([b], [ypos[i] + offset], marker=marker, s=28,
                       facecolor=col if sig else "white", edgecolor=col,
                       zorder=3, label=label if i == 0 else None)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    labels = {**CATEGORY_LABELS, "net": "净效价", "p_ai": "P(AI)",
              "content_len": "正文长度", "n_hashtags": "标签数"}
    ax.set_yticklabels([labels.get(n, n) for n in order])
    ax.set_xlabel("单元内 FE β（每 1 SD → log10(1+互动)）")
    ax.set_title(f"(c) 互动偏效应（n={fe_likes['n_obs']:,}，"
                 f"{fe_likes['n_cells']} 单元）")
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    fams = [f for f in FAMILIES
            if family["families"].get(f, {}).get("n", 0) >= MIN_FAMILY_N]
    mat = np.asarray([[family["families"][f]["density100"][k]
                       for k in SCORE_ORDER] for f in fams], dtype=float)
    z = (mat - mat.mean(axis=0)) / np.where(mat.std(axis=0) > 0,
                                            mat.std(axis=0), 1.0)
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
    ax.grid(False)
    for i in range(len(fams)):
        for j in range(len(SCORE_ORDER)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8.0,
                    color="white" if abs(z[i, j]) > 1.15 else "black")
    ax.set_xticks(range(len(SCORE_ORDER)))
    ax.set_xticklabels([CATEGORY_LABELS.get(k, k) for k in SCORE_ORDER],
                       rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f"{FAMILY_NAME[f]}（n={family['families'][f]['n']}）"
                        for f in fams], fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("家族差异（列内 z-score）", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("(d) AIGC 家族情绪指纹（格内为密度）")

    ax = axes[1, 1]
    qs = quarterly["quarters"]
    x = np.arange(len(qs))
    for k in PLOT_ORDER:
        raw = quarterly["raw"][k]
        std = quarterly["domain_std"][k]
        t = quarterly["trend"][k]
        line, = ax.plot(x, raw, "o-", color=CATEGORY_COLORS[k], lw=1.5, ms=3.4,
                        label=f"{CATEGORY_LABELS.get(k, '净效价')}"
                              f"（ρ={t['rho']:+.2f}）")
        ax.plot(x, std, "--", color=line.get_color(), lw=1.0, alpha=0.65)
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("密度（次/百字）")
    ax.set_title("(e) 探索集季度趋势（实线=原始，虚线=固定领域权重）")
    ax.legend(fontsize=7.2, ncol=2)

    ax = axes[1, 2]
    vals = [sep["univariate_auc"][k] for k in SCORE_ORDER]
    ypos = np.arange(len(SCORE_ORDER))
    ax.barh(ypos, vals,
            color=[CATEGORY_COLORS[k] for k in SCORE_ORDER],
            alpha=0.9, height=0.68)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", xy=(v, ypos[i]), xytext=(4, -2.5),
                    textcoords="offset points", fontsize=8)
    ax.axvline(0.5, color="black", lw=0.8, ls=":")
    ax.axvline(sep["combined_cv"]["auc"], color="#333333", lw=1.2, ls="--",
               label=f"八项联合 CV AUC={sep['combined_cv']['auc']:.3f}")
    ax.set_yticks(ypos)
    ax.set_yticklabels([CATEGORY_LABELS.get(k, k) for k in SCORE_ORDER])
    ax.set_xlim(0.45, max(0.8, max(vals) + 0.06))
    ax.set_xlabel("单标记 AUC（human=0 vs AIGC=1）")
    ax.set_title("(f) 仅用先验情绪标记的 human/AIGC 判别力")
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, INSIGHTS + "affect_gradient.png")


def _p_text(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_markdown(out: dict) -> None:
    meta = out["meta"]
    rates = out["subset_rates"]
    gaps = out["length_matched"]
    grad = out["gradient"]
    eng = out["engagement"]["fe"]
    fam = out["family"]["families"]
    q = out["quarterly"]
    sep = out["separability"]
    subset_labels = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
                     "exploring": "探索集(后LLM)"}
    lines = [
        "# 情绪与效价分析结果（Affect & Valence）",
        "",
        f"> 由 `affect_analysis.py` 全量运行生成；正文清洗与 "
        f"`paper_features.py` 一致，人类/探索集按 note_id 去重；正文 "
        f"≥{meta['min_body_len']} 字；探索集 p_ai 按行序对齐 "
        f"`paper_exploring_scores.csv.gz`（scores/features content_len mismatch="
        f"{meta['exploring_content_len_mismatch']}，原始文本长度 drift="
        f"{meta['exploring_text_len_drift']}；月份/领域/互动字段逐条一致）。",
        "",
        "> 词表为先验固定词典，按最长匹配只计一次；正向=喜悦+喜爱依恋，"
        "负向=焦虑担忧+悲伤失落+愤怒批评，净效价=正向−负向（有符号）。",
        "",
        "## 1 跨子集使用：发生率与密度",
        "",
        "| 子集 | 有效帖数 | 正向情绪密度 | 负向情绪密度 | 净效价均值 | 正向发生率 | 负向发生率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SUBSET_ORDER:
        cats = rates[name]["categories"]
        lines.append(
            f"| {subset_labels[name]} | {rates[name]['n']:,} | "
            f"{cats['positive']['density100']:.3f} | "
            f"{cats['negative']['density100']:.3f} | "
            f"{cats['net']['density100']:+.3f} | "
            f"{cats['positive']['share']:.1%} | "
            f"{cats['negative']['share']:.1%} |")
    lines += [
        "",
        "**五类情绪密度（次/百字）：**",
        "",
        "| 子集 | " + " | ".join(CATEGORY_LABELS[k] for k in CATEGORY_ORDER) + " |",
        "|---|" + "---:|" * len(CATEGORY_ORDER),
    ]
    for name in SUBSET_ORDER:
        cats = rates[name]["categories"]
        lines.append(
            f"| {subset_labels[name]} | "
            + " | ".join(f"{cats[k]['density100']:.3f}" for k in CATEGORY_ORDER)
            + " |")
    lines += [
        "",
        "**按对照子集长度分布加权的长度匹配差异（人类 − 对照，95% CI）：**",
        "",
        "| 指标 | 对照 | 发生率差 | 密度差（次/百字） |",
        "|---|---|---:|---:|",
    ]
    for k in SCORE_ORDER:
        for other in ("aigc", "exploring"):
            ind = gaps[k][other]["incidence_gap"]
            dens = gaps[k][other]["density_gap"]
            lines.append(
                f"| {CATEGORY_LABELS.get(k, k)} | "
                f"{subset_labels[other]} | {ind['gap']:+.3f} "
                f"[{ind['ci_low']:+.3f}, {ind['ci_high']:+.3f}] | "
                f"{dens['gap']:+.3f} [{dens['ci_low']:+.3f}, "
                f"{dens['ci_high']:+.3f}] |")
    lines += [
        "",
        "## 2 探索集内的 P(AI) 梯度（去重，正文 ≥%d 字）"
        % meta["min_body_len"],
        "",
        "| 指标 | 单元内加权 ρ | 负单元占比 | 长度分层 ρ | 首尾十分位 δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in SCORE_ORDER:
        g = grad["categories"][k]
        lines.append(
            f"| {CATEGORY_LABELS.get(k, k)} | "
            f"{g['weighted_rho']:+.4f} | {g['neg_cell_share']:.1%} | "
            f"{g['stratified_rho']:+.4f} | "
            f"{g['cliff_delta_top_vs_bottom']:+.3f} |")
    lines += [
        "",
        f"（样本 {grad['n']:,} 条 / {grad['n_cells']} 个「月份×领域」单元；"
        "长度分层 = 月份×领域×长度分箱单元。）",
        "",
        "**逐词单元内 ρ（最负/最正各 8 个）：**",
        "",
        "| 词 | 类别 | 单元内 ρ | 负单元占比 |",
        "|---|---|---:|---:|",
    ]
    items = sorted(grad["words"].items(), key=lambda kv: kv[1]["weighted_rho"])
    for w, r in items[:8] + items[-8:]:
        lines.append(f"| {w} | {CATEGORY_LABELS[r['category']]} | "
                     f"{r['weighted_rho']:+.4f} | {r['neg_cell_share']:.0%} |")
    auc = sep["univariate_auc"]
    lines += [
        "",
        "## 3 仅用先验情绪标记的 human/AIGC 判别力",
        "",
        "| 指标 | 单标记 AUC |",
        "|---|---:|",
    ]
    for k in SCORE_ORDER:
        lines.append(f"| {CATEGORY_LABELS.get(k, k)} | {auc[k]:.4f} |")
    lines += [
        "",
        f"八项指标联合逻辑回归 5 折 CV AUC={sep['combined_cv']['auc']:.4f}"
        f"（human n={sep['n_human']:,}，AIGC n={sep['n_aigc']:,}）。",
        "",
        "## 4 互动偏效应（月份×领域 FE + 整群 bootstrap %d 次）"
        % meta["n_boot"],
        "",
        "| 特征 | 点赞 β [95% CI] | 收藏 β [95% CI] | 评论 β [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for name in eng["likes"]["features"]:
        cells = []
        for target in ("likes", "collections", "comments"):
            fe = eng[target]
            cells.append(f"{fe['coef'][name]:+.4f} "
                         f"[{fe['ci_low'][name]:+.4f}, "
                         f"{fe['ci_high'][name]:+.4f}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"（n={eng['likes']['n_obs']:,}，{eng['likes']['n_cells']} 个单元；"
        f"特征已 z-score，β 为每 1 SD 的偏效应；{meta['month_from']} 起、"
        f"正文 ≥{meta['min_body_len']} 字。）",
        "",
        "## 5 AIGC 家族情绪指纹（密度，次/百字）",
        "",
        "| 家族 | n | 正向 | 负向 | 净效价 | 喜悦 | 喜爱依恋 | 焦虑担忧 | 悲伤失落 | 愤怒批评 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fams = [f for f in FAMILIES
            if fam.get(f, {}).get("n", 0) >= MIN_FAMILY_N]
    for f in fams:
        row = fam[f]
        lines.append(
            f"| {FAMILY_NAME[f]} | {row['n']:,} | "
            f"{row['density100']['positive']:.3f} | "
            f"{row['density100']['negative']:.3f} | "
            f"{row['density100']['net']:+.3f} | "
            + " | ".join(f"{row['density100'][k]:.3f}" for k in CATEGORY_ORDER)
            + " |")
    lines += [
        "",
        "## 6 探索集季度趋势",
        "",
        "| 指标 | 首季 | 末季 | 原始 ρ | p | 固定领域权重 ρ | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in SCORE_ORDER:
        t = q["trend"][k]
        lines.append(
            f"| {CATEGORY_LABELS.get(k, k)} | "
            f"{q['raw'][k][0]:.3f} | {q['raw'][k][-1]:.3f} | "
            f"{t['rho']:+.3f} | {_p_text(t['p'])} | "
            f"{t['rho_domain_std']:+.3f} | {_p_text(t['p_domain_std'])} |")
    lines += [
        "",
        f"（季度样本 ≥{MIN_QUARTER_N} 条；{q['quarters'][0]}–{q['quarters'][-1]}，"
        f"共 {len(q['quarters'])} 个季度。所有关联均为观察性，不作因果解读。）",
        "",
    ]
    with open(BASE + "affect_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    print("扫描三个语料（全量，进程池并行）…")
    scans = dict(zip(DATASETS, run_parallel(
        [partial(scan_corpus, path) for path in DATASETS.values()])))
    for name, scan in scans.items():
        print(f"  {name}: {scan['n_total']:,} 条记录")

    scores = pd.read_csv(SCORES_PATH, usecols=["p_ai", "content_len"])
    n_scanned = scans["exploring"]["n_total"]
    if len(scores) != n_scanned:
        raise SystemExit(f"exploring JSONL 行数 {n_scanned} 与 p_ai 行数 "
                         f"{len(scores)} 不一致")

    keeps = {name: dedup_keep(scan) for name, scan in scans.items()}
    for name in SUBSET_ORDER:
        scan = scans[name]
        short = int((scan["body_len"][keeps[name]] < MIN_BODY_LEN).sum())
        print(f"  {name}: 去重后 {len(keeps[name]):,} 条，正文 <"
              f"{MIN_BODY_LEN} 字排除 {short:,} 条")

    subset_rates = {name: rates_block(scans[name], keeps[name])
                    for name in SUBSET_ORDER}
    word_table = {}
    for w in ALL_WORDS:
        word_table[w] = {
            "category": WORD_CATEGORY[w],
            "human": subset_rates["human"]["words"][w],
            "aigc": subset_rates["aigc"]["words"][w],
            "exploring": subset_rates["exploring"]["words"][w],
        }

    print("估计长度匹配差异…")
    gaps = length_matched_block(scans, keeps)
    print("估计 P(AI) 梯度…")
    grad = gradient_block(scans, keeps, scores)
    print("估计 human/AIGC 判别力…")
    sep = separability_block(scans, keeps)
    print("估计互动偏效应…")
    eng = engagement_block(scans, scores)
    print("汇总家族指纹与季度趋势…")
    family = family_block(scans["aigc"])
    quarterly = quarterly_block(scans["exploring"], keeps["exploring"])

    out = {
        "meta": {
            "min_body_len": MIN_BODY_LEN, "min_cell_n": MIN_CELL_N,
            "month_from": MONTH_FROM, "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED, "min_quarter_n": MIN_QUARTER_N,
            "exploring_content_len_mismatch": 0,
            "exploring_text_len_drift": eng["text_len_drift"],
            "note": "人类/探索集按 note_id 去重（保留元数据最完整一条）；"
                    "AIGC 无 note_id 逐记录；p_ai 按行序对齐 "
                    "paper_exploring_scores.csv.gz；行序用原始 "
                    "month/domain/likes/collections/comments 校验；"
                    "所有关联为观察性。",
        },
        "lexicon": {
            k: {"label": CATEGORY_LABELS[k], "words": CATEGORY_WORDS[k][1]}
            for k in CATEGORY_ORDER},
        "composites": {
            "positive": {"label": "正向情绪合计", "parts": POSITIVE_PARTS},
            "negative": {"label": "负向情绪合计", "parts": NEGATIVE_PARTS},
            "net": {"label": "净效价（正−负）", "parts": ["positive", "negative"]},
        },
        "subset_rates": {
            name: {
                "n": subset_rates[name]["n"],
                "chars": subset_rates[name]["chars"],
                "categories": subset_rates[name]["categories"],
            } for name in SUBSET_ORDER},
        "word_rates": word_table,
        "length_matched": gaps,
        "gradient": grad,
        "separability": sep,
        "engagement": eng,
        "family": family,
        "quarterly": quarterly,
    }
    save_json(out, BASE + "affect_results.json")
    print("已写出 affect_results.json")
    write_markdown(out)
    plot_lexicon(subset_rates, word_table)
    plot_gradient(grad, eng, family, quarterly, sep)
    print("已写出 affect_results.md 与 insights/affect_*.png")


if __name__ == "__main__":
    main()
