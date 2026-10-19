# -*- coding: utf-8 -*-
"""深潜分析之十八：正文自重复与句式复读（within-post redundancy）。

既有重复分析都在**跨文档**层面：`deep_dive_analysis.py` 覆盖逐字节完全
重复，`near_duplicate_analysis.py` 覆盖字符 5-gram Jaccard ≥0.5 的改写模板；
`diversity_analysis.py` 的 MTLD / burstiness 只刻画词汇量与句子长短起伏。
但生成模型（尤其指令微调模型）还有一类典型痕迹是**单篇正文内部的复读**：
同一段意思反复说、句子换词重复、每句都以同样的开头起手——这些在跨文档
去重和长短多样性指标里都看不见。本脚本第一次把「自重复」作为一条独立的
风格轴做全量扫描，用四个先验定义的指标（不依赖任何监督信号）回答：

1. 跨子集对比：生成集的正文是否比人类更「自我复读」？四个指标的帖子层
   均值、长度分箱曲线与**长度匹配**后的 AIGC−人类 / 探索集−人类差异
   （连续指标版，按人类长度分布加权）；
2. 野外梯度：探索集内 P(AI) 越高的帖子是否越复读？月份×领域单元内加权
   ρ、长度分层稳健性、首尾十分位 Cliff's δ，以及「存在近重复句」的
   十分位占比曲线（Wilson CI）；
3. 互动回报：控制 P(AI)、正文长度与标签数后，四个自重复指标对点赞、
   收藏、评论的月份×领域固定效应偏效应（整群 bootstrap CI）；
4. 生成端指纹与时间演化：17 个模型/6 个家族的指标画像（长度标准化后
   的家族热力图）；探索集季度趋势（原始 + 「领域×长度」固定权重）。

四个指标的先验定义（参数固定，不做数据驱动调参）：

- `gram6` 片段复读率：清洗后正文内**重复字符 6-gram 的出现占比**
  （1 − 唯一 6-gram / 总 6-gram）；64 位滚动哈希向量化计数，单帖内碰撞
  概率可忽略（见 `analysis_utils.ngram_repeat_share`）。
- `max_sim` 句间最大相似度：把正文切成句子后只保留 ≥`SENT_MIN_CHARS=6`
  字的实义句，两两计算字符 3-gram Jaccard，取最大值。
- `dup_sent` 近重复句占比：参与过至少一对 Jaccard ≥ `DUP_SIM_THRESHOLD=0.5`
  近重复句对的实义句比例。
- `open_rep` 复读句首占比：句首 `OPENING_CHARS=3` 字与同帖另一句完全相同
  的实义句比例（排比/模板起手的直接度量）。

后三个指标只在实义句数 ≥2 的帖子上有定义（单句帖计入「未定义」，单独
报出数量）；`dup_any`（存在近重复句对）为 `dup_sent>0` 的二值汇总。

口径说明：
- 正文清洗复用 analysis_utils.clean_body（与 paper_features.py 一致）；
- 人类/探索集按 note_id 去重（保留元数据最完整一条，规则同
  cta_analysis.py；AIGC 无 note_id 逐记录处理）；
- 探索集 p_ai 按行序对齐 paper_exploring_scores.csv.gz；行序用原始
  month/domain/likes/collections/comments 逐条校验，content_len 仅作
  清洗版本漂移的信息性指标（不参与错位判定）；
- 正文不足 `MIN_BODY_LEN=50` 字的帖子（纯标签/空正文等）不参与统计，
  数量单独报告；
- 互动回归与 engagement_drivers_analysis.py 完全同协议（特征先 z-score
  再单元内去均值、单元 ≥100 条、整群 bootstrap 300 次），被解释变量为
  log10(1+互动)，并在实义句 ≥2 的共同样本上加入四个自重复特征；
- 家族/季度均值另给出**长度标准化**版本（`length_residualize`：减去
  同一长度分箱的 AIGC/探索集池均值），避免长度构成差异污染画像；
  季度「领域×长度」固定权重序列在 domain×length 单元上重新标准化。

输出：
  insights/redundancy_lexicon.png   子集均值 / 长度匹配差异 / 长度曲线 / 句相似度分布
  insights/redundancy_gradient.png  P(AI) 梯度 / 单元内 ρ / 互动偏效应 /
                                    家族指纹 / 季度趋势 / 近重复句普及率
  redundancy_results.json           全部数值结果
  redundancy_results.md             结果摘要
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis_utils import (BASE, INSIGHTS, LENGTH_BINS, PALETTE, body_cache,
                            cell_groups, cell_rank_cache, cliffs_delta,
                            clean_body, content_of, decile_table, dedup_indices,
                            domain_standardized_series, engagement_of,
                            init_plot_style, length_bin_codes,
                            length_matched_mean_gap, length_residualize,
                            ngram_repeat_share, parse_local_time, run_parallel,
                            save_fig, save_json, split_sentences, wilson_ci,
                            require_row_alignment, within_cell_spearman)
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

MIN_BODY_LEN = 50        # 自重复指标在更短正文（纯标签/空正文）上无意义
SENT_MIN_CHARS = 6       # 实义句的最小字数（滤掉「好」「谢谢」这类短句噪点）
GRAM_N = 6               # 片段复读率的 n-gram 长度
DUP_SIM_THRESHOLD = 0.5  # 近重复句对的 Jaccard 阈值（与 near_duplicate 同口径）
OPENING_CHARS = 3        # 句首复读的字符数
MIN_CELL_N = 100         # 单元内统计的最小单元样本量（与论文口径一致）
MONTH_FROM = "202301"    # 互动分析起始月份（与 paper_engagement 一致）
N_BOOT = 300             # 整群 bootstrap 次数
BOOT_SEED = 42
MIN_QUARTER_N = 300
MIN_FAMILY_N = 100

METRIC_ORDER = ["gram6", "max_sim", "dup_sent", "open_rep"]
METRIC_LABELS = {"gram6": "片段复读率", "max_sim": "句间最大相似度",
                 "dup_sent": "近重复句占比", "open_rep": "复读句首占比"}
METRIC_COLORS = {"gram6": "#4C72B0", "max_sim": "#DD8452",
                 "dup_sent": "#C44E52", "open_rep": "#55A868"}
SENT_METRICS = ("max_sim", "dup_sent", "open_rep")   # 需实义句 >= 2
SUBSET_ORDER = ["human", "aigc", "exploring"]
SUBSET_LABELS = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
                 "exploring": "探索集(后LLM)"}
LINESTYLES = {"human": "-", "aigc": "--", "exploring": ":"}

FAMILY_NAME = {"anthropic": "Claude", "deepseek": "DeepSeek", "gemini": "Gemini",
               "glm": "GLM", "gpt": "GPT", "qwen": "Qwen"}
FAMILIES = ["anthropic", "deepseek", "gemini", "glm", "gpt", "qwen"]
TARGETS = (("likes", "点赞"), ("collections", "收藏"), ("comments", "评论"))


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def char_ngram_set(text: str, n: int = 3) -> frozenset:
    """字符 n-gram 集合（长度不足 n 返回空集）。"""
    m = len(text) - n + 1
    if m <= 0:
        return frozenset()
    return frozenset(text[i:i + n] for i in range(m))


def jaccard(a: frozenset, b: frozenset) -> float:
    """两个 n-gram 集合的 Jaccard 相似度；任一为空返回 0。"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def sentence_repetition(text: str | None) -> dict:
    """一篇正文的句子层自重复指标。

    返回 {"n_sent": 实义句数, "max_sim": 句间最大 Jaccard,
    "dup_sent": 参与近重复句对的实义句占比, "open_rep": 复读句首的实义句占比}。
    实义句不足 2 句时后三项为 NaN（未定义），由调用方按需过滤。
    """
    sentences = [s for s in split_sentences(text) if len(s) >= SENT_MIN_CHARS]
    out = {"n_sent": len(sentences), "max_sim": float("nan"),
           "dup_sent": float("nan"), "open_rep": float("nan")}
    if len(sentences) < 2:
        return out
    sets = [char_ngram_set(s) for s in sentences]
    sizes = [len(st) for st in sets]
    involved = np.zeros(len(sentences), dtype=bool)
    max_sim = 0.0
    # 上三角两两比较：句数通常 <30，包级别的集合交并比逐对正则扫描更快。
    # 集合大小预先算好（省掉每对的两次 len）；交集为空时 Jaccard 必为 0，
    # 直接跳过除法（对 max_sim/近重复判定无影响）。
    for i in range(len(sentences)):
        si = sets[i]
        li = sizes[i]
        for j in range(i + 1, len(sentences)):
            inter = len(si & sets[j])
            if not inter:
                continue
            sim = inter / (li + sizes[j] - inter)
            if sim > max_sim:
                max_sim = sim
            if sim >= DUP_SIM_THRESHOLD:
                involved[i] = involved[j] = True
    out["max_sim"] = float(max_sim)
    out["dup_sent"] = float(involved.mean())
    openings_list = [s[:OPENING_CHARS] for s in sentences]
    openings = Counter(openings_list)
    out["open_rep"] = float(
        sum(1 for o in openings_list if openings[o] >= 2)
        / len(sentences))
    return out


def redundancy_metrics(body: str) -> dict:
    """清洗后正文的四个自重复指标（片段级 + 句子级）。"""
    out = {"gram6": ngram_repeat_share(body, n=GRAM_N)}
    out.update(sentence_repetition(body))
    return out


def mean_ci(values) -> tuple[float, float, float]:
    """均值与 95% 正态近似 CI；空数组返回 (NaN, NaN, NaN)。"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
def scan_corpus(path: str) -> dict:
    """单次流式扫描一个语料，返回逐记录的四个自重复指标与元数据。

    重复正文（探索集约 45%）经 `body_cache` 直接复用上一次的指标结果，
    结果与逐帖重算逐位一致（缓存值只读）。探索集的重复正文在文件里分布
    很散，默认 2 万条 LRU 只能命中约 18%；这里每条缓存值只有几个浮点数，
    放宽到 10 万条可命中约 45%，内存增量约 20–30MB，远小于省下的计算量。
    """
    body_len, note_id, meta, month, domain = [], [], [], [], []
    family, model = [], []
    likes, collections = [], []
    metrics = {k: [] for k in METRIC_ORDER}
    n_sent = []
    n_total = 0
    cached = body_cache(redundancy_metrics, max_entries=60_000)
    for rec in iter_cached(path):
        n_total += 1
        content = content_of(rec)
        body = clean_body(content)
        m = cached(body)
        body_len.append(len(body))
        note_id.append(rec.get("note_id"))
        dom = rec.get("domain")
        domain.append(dom if dom not in (None, "") else "未知")
        likes_v, colls, comms = engagement_of(rec)
        likes.append(likes_v)
        collections.append(colls)
        has_meta = bool(rec.get("user_id")) and dom not in (None, "", "未知")
        dt = parse_local_time(rec.get("local_time"))
        month.append(f"{dt.year:04d}{dt.month:02d}" if dt else None)
        score = (int(has_meta) + int(bool(likes_v)) + int(bool(colls))
                 + int(bool(comms)) + int(dt is not None)
                 + int(bool(content.strip())))
        meta.append(score)
        family.append(rec.get("model_family") or "")
        model.append(rec.get("model") or "")
        n_sent.append(m["n_sent"])
        for k in METRIC_ORDER:
            metrics[k].append(m[k])
    return {
        "path": path, "n_total": n_total,
        "body_len": np.asarray(body_len, dtype=np.int32),
        "note_id": note_id, "meta": np.asarray(meta, dtype=np.int32),
        "month": month, "domain": domain,
        "likes": np.asarray(likes, dtype=float),
        "collections": np.asarray(collections, dtype=float),
        "family": family, "model": model,
        "n_sent": np.asarray(n_sent, dtype=np.int16),
        **{k: np.asarray(v, dtype=np.float32) for k, v in metrics.items()},
    }


def dedup_keep(scan: dict) -> np.ndarray:
    """按 note_id 去重（保留元数据最完整一条），返回排序后的保留索引。"""
    return np.asarray(sorted(dedup_indices(scan["note_id"], scan["meta"])),
                      dtype=int)


def rates_block(scan: dict, keep: np.ndarray, min_len: int = MIN_BODY_LEN) -> dict:
    """一个子集的四个指标统计（帖子层均值/中位数/CI + 长度分箱 + 近重复句普及率）。"""
    sel = keep[scan["body_len"][keep] >= min_len]
    n = int(len(sel))
    lens = scan["body_len"][sel].astype(float)
    codes = length_bin_codes(lens)
    out = {"n": n, "chars": int(lens.sum()), "metrics": {}, "length_bins": {},
           "n_multi": int((scan["n_sent"][sel] >= 2).sum())}
    for k in METRIC_ORDER:
        vals = scan[k][sel].astype(float)
        defined = np.isfinite(vals)
        v = vals[defined]
        mean, lo, hi = mean_ci(v)
        out["metrics"][k] = {
            "n_defined": int(defined.sum()),
            "share_defined": float(defined.mean()) if n else 0.0,
            "mean": mean, "ci_low": lo, "ci_high": hi,
            "median": float(np.median(v)) if v.size else float("nan"),
            "p90": float(np.percentile(v, 90)) if v.size else float("nan"),
        }
        bins = []
        lv = lens[defined]
        cv = codes[defined]
        for b in range(len(LENGTH_BINS)):
            vb = v[cv == b]
            bm, blo, bhi = mean_ci(vb)
            bins.append({"bin": b, "n": int(vb.size), "mean": bm,
                         "ci_low": blo, "ci_high": bhi})
        out["length_bins"][k] = bins
    # 近重复句普及率（仅实义句 >=2 的帖子）
    dup = scan["dup_sent"][sel].astype(float)
    ddef = np.isfinite(dup)
    hit = int((dup[ddef] > 0).sum())
    nd = int(ddef.sum())
    lo, hi = wilson_ci(hit, nd) if nd else (0.0, 0.0)
    out["dup_any"] = {"n": nd, "hits": hit,
                      "share": float(hit / nd) if nd else 0.0,
                      "ci_low": float(lo), "ci_high": float(hi)}
    return out


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_gradient(grad: dict, eng: dict, family_z: dict, family_n: dict,
                  quarterly: dict, dup_any_grad: dict) -> None:
    """六联图：P(AI) 梯度 / 单元内 ρ / 互动偏效应 / 家族指纹 / 季度趋势 /
    近重复句普及率梯度。"""
    init_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 12.2))

    # ---- (a) 标准化 P(AI) 十分位曲线 ----
    ax = axes[0, 0]
    for k in METRIC_ORDER:
        dec = grad[k]["deciles"]
        base = grad[k]["mean"]
        sd = grad[k]["sd"] if grad[k]["sd"] > 0 else 1.0
        x = [d["decile"] for d in dec]
        yv = [(d["mean"] - base) / sd for d in dec]
        lo = [(d["ci_low"] - base) / sd for d in dec]
        hi = [(d["ci_high"] - base) / sd for d in dec]
        ax.plot(x, yv, "o-", color=METRIC_COLORS[k], lw=1.8, ms=4.5,
                label=f"{METRIC_LABELS[k]}（ρ={grad[k]['weighted_rho']:+.3f}）")
        ax.fill_between(x, lo, hi, color=METRIC_COLORS[k], alpha=0.14)
    ax.axhline(0, color="black", lw=0.7, alpha=0.6)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("检测器 P(AI) 十分位（1=最像人，10=最像 AI）")
    ax.set_ylabel("指标（z-score，相对全样本）")
    ax.set_title("(a) 探索集：自重复随 P(AI) 的梯度（含 95% CI）")
    ax.legend(fontsize=8.5)

    # ---- (b) 单元内加权 ρ（实心=全样本，空心=长度分层） ----
    ax = axes[0, 1]
    ypos = np.arange(len(METRIC_ORDER))
    vals = [grad[k]["weighted_rho"] for k in METRIC_ORDER]
    strat = [grad[k]["stratified_rho"] for k in METRIC_ORDER]
    ax.barh(ypos, vals, color=[METRIC_COLORS[k] for k in METRIC_ORDER],
            alpha=0.9, height=0.6)
    ax.scatter(strat, ypos, s=44, facecolor="white", edgecolor="black",
               zorder=3, label="长度分层 ρ")
    for i, (v, s) in enumerate(zip(vals, strat)):
        # 数值标注统一放在 0 线的右侧（负向条右端即 0 线旁的空区），
        # 避免贴到轴外沿的刻度标签上。
        anchor = v if v >= 0 else 0.0
        ax.annotate(f"{v:+.3f}", xy=(anchor, ypos[i]), xytext=(6, -3),
                    textcoords="offset points", fontsize=8, ha="left")
    ax.axvline(0, color="black", lw=0.8)
    ax.margins(x=0.10)
    ax.set_yticks(ypos)
    ax.set_yticklabels([METRIC_LABELS[k] for k in METRIC_ORDER], fontsize=9)
    ax.set_xlabel("月份×领域单元内加权 Spearman ρ（指标 vs P(AI)）")
    ax.set_title("(b) 野外 P(AI) 梯度（黑点=月份×领域×长度分箱）")
    ax.legend(fontsize=8.5, loc="lower right")

    # ---- (c) 三指标互动偏效应森林图 ----
    ax = axes[0, 2]
    fe_likes = eng["fe"]["likes"]
    names = list(fe_likes["coef"].keys())
    order = list(reversed(names))
    ypos = np.arange(len(order))
    label_map = {**{k: METRIC_LABELS[k] for k in METRIC_ORDER},
                 "p_ai": "P(AI)", "content_len": "正文长度",
                 "n_hashtags": "标签数"}
    markers = (("likes", "o", -0.24, "#4C72B0", "点赞"),
               ("collections", "^", 0.0, "#DD8452", "收藏"),
               ("comments", "s", 0.24, "#55A868", "评论"))
    for target, marker, offset, color, tlabel in markers:
        fe = eng["fe"][target]
        first = True
        for i, name in enumerate(order):
            b = fe["coef"][name]
            lo, hi = fe["ci_low"][name], fe["ci_high"][name]
            sig = lo > 0 or hi < 0
            col = color if sig else "#9E9E9E"
            ax.plot([lo, hi], [ypos[i] + offset, ypos[i] + offset], color=col,
                    lw=1.6, solid_capstyle="butt", zorder=2)
            ax.scatter([b], [ypos[i] + offset], marker=marker, s=30,
                       facecolor=col if sig else "white", edgecolor=col,
                       zorder=3, label=tlabel if first else None)
            first = False
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([label_map.get(n, n) for n in order], fontsize=8.5)
    ax.set_xlabel("单元内 FE β（每 1 SD → log10(1+互动)）")
    ax.set_title(f"(c) 自重复的互动偏效应（n={fe_likes['n_obs']:,}，"
                 f"{fe_likes['n_cells']} 单元，整群 bootstrap）")
    ax.legend(loc="lower right", fontsize=8.5)

    # ---- (d) 家族指纹（长度标准化后按列 z-score；格内为标准化均值） ----
    ax = axes[1, 0]
    fams = [f for f in FAMILIES if family_n.get(f, 0) >= MIN_FAMILY_N]
    mat = np.asarray([[family_z[f][k] for k in METRIC_ORDER] for f in fams])
    z = (mat - mat.mean(axis=0)) / np.where(mat.std(axis=0) > 0,
                                            mat.std(axis=0), 1.0)
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
    ax.grid(False)
    for i in range(len(fams)):
        for j in range(len(METRIC_ORDER)):
            ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(z[i, j]) > 1.15 else "black")
    ax.set_xticks(range(len(METRIC_ORDER)))
    ax.set_xticklabels([METRIC_LABELS[k] for k in METRIC_ORDER],
                       fontsize=8.5, rotation=12, ha="right")
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f"{FAMILY_NAME[f]}（n={family_n[f]}）" for f in fams],
                       fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("列内 z-score", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("(d) AIGC 家族指纹（长度标准化均值，格内为原值）")

    # ---- (e) 季度趋势（标准化；实线=原始，虚线=领域×长度固定权重） ----
    ax = axes[1, 1]
    qs = quarterly["quarters"]
    x = np.arange(len(qs))
    for k in METRIC_ORDER:
        raw = np.asarray(quarterly["raw"][k], dtype=float)
        std = np.asarray(quarterly["domain_len_std"][k], dtype=float)
        base = quarterly["overall"][k]
        sd = quarterly["sd"][k] if quarterly["sd"][k] > 0 else 1.0
        t = quarterly["trend"][k]
        line, = ax.plot(x, (raw - base) / sd, "o-", color=METRIC_COLORS[k],
                        lw=1.7, ms=4,
                        label=f"{METRIC_LABELS[k]}（ρ={t['rho']:+.2f}，p={t['p']:.2f}）")
        ax.plot(x, (std - base) / sd, "--", color=line.get_color(),
                lw=1.15, alpha=0.65)
    ax.axhline(0, color="black", lw=0.7, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("指标（z-score，相对探索集全样本）")
    ax.set_title("(e) 探索集季度趋势（实线=原始均值，虚线=领域×长度固定权重）")
    ax.legend(fontsize=7.8)

    # ---- (f) 「存在近重复句」的 P(AI) 十分位占比（Wilson CI） ----
    ax = axes[1, 2]
    dec = dup_any_grad["deciles"]
    x = [d["decile"] for d in dec]
    share = [d["mean"] for d in dec]
    lo = [d["ci_low"] for d in dec]
    hi = [d["ci_high"] for d in dec]
    ax.errorbar(x, share, yerr=[np.asarray(share) - np.asarray(lo),
                                np.asarray(hi) - np.asarray(share)],
                fmt="o-", color="#C44E52", lw=1.8, ms=5, capsize=3)
    ax.axhline(dup_any_grad["overall_share"], color="#4C72B0", ls="--",
               lw=1.4, label=f"总体普及率 {dup_any_grad['overall_share']:.1%}")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("检测器 P(AI) 十分位")
    ax.set_ylabel("存在近重复句对的帖子占比")
    ax.set_title(f"(f) 近重复句普及率随 P(AI) 上升"
                 f"（ρ={dup_any_grad['weighted_rho']:+.3f}）")
    ax.legend(fontsize=8.5)
    save_fig(fig, INSIGHTS + "redundancy_gradient.png")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    print("扫描三个语料（全量，进程池并行）…")
    scans = dict(zip(DATASETS, run_parallel(
        [partial(scan_corpus, path) for path in DATASETS.values()])))
    for name, sc in scans.items():
        print(f"  {name}: {sc['n_total']:,} 条记录")

    # ---- 对齐探索集 p_ai ----
    scores = pd.read_csv(SCORES_PATH, usecols=["p_ai", "content_len"])
    n_score = len(scores)
    n_scanned = scans["exploring"]["n_total"]
    if n_score != n_scanned:
        raise SystemExit(f"exploring JSONL 行数 {n_scanned} 与 p_ai 行数 {n_score} 不一致")
    # 行序用 features 的原始字段（月份/领域/互动）校验；scores 与 features
    # 的 content_len 互校。正文清洗版本差异造成的长度 drift 仅作信息记录。
    mismatch = 0

    # ---- 去重 + 子集统计 ----
    keeps = {name: dedup_keep(sc) for name, sc in scans.items()}
    subset_rates = {name: rates_block(scans[name], keeps[name])
                    for name in SUBSET_ORDER}
    for name in SUBSET_ORDER:
        sc = scans[name]
        short = int((sc["body_len"][keeps[name]] < MIN_BODY_LEN).sum())
        single = subset_rates[name]["n"] - subset_rates[name]["n_multi"]
        print(f"  {name}: 去重后 {len(keeps[name]):,} 条，正文 <{MIN_BODY_LEN} 字 "
              f"排除 {short:,} 条；实义句 <2 句 {single:,} 条")

    # ---- 长度匹配差异（AIGC/探索集 − 人类，按人类长度分布加权）----
    human = scans["human"]
    h_keep = keeps["human"]
    matched = {}
    for name in ("aigc", "exploring"):
        sc = scans[name]
        sel_h = h_keep[human["body_len"][h_keep] >= MIN_BODY_LEN]
        sel_s = keeps[name][sc["body_len"][keeps[name]] >= MIN_BODY_LEN]
        rec = {}
        for k in METRIC_ORDER:
            vh = human[k][sel_h].astype(float)
            vs = sc[k][sel_s].astype(float)
            mh = np.isfinite(vh)
            ms = np.isfinite(vs)
            rec[k] = length_matched_mean_gap(
                vs[ms], sc["body_len"][sel_s][ms].astype(float),
                vh[mh], human["body_len"][sel_h][mh].astype(float))
        matched[name] = rec
        print(f"  长度匹配差异（{name}）: " + "，".join(
            f"{k} {rec[k]['gap']:+.4f}" for k in METRIC_ORDER))

    # ---- 探索集 P(AI) 梯度（去重 + 长度门槛；逐指标按定义样本） ----
    expl = scans["exploring"]
    keep = keeps["exploring"]
    sel = keep[expl["body_len"][keep] >= MIN_BODY_LEN]
    p_ai_all = scores["p_ai"].to_numpy(dtype=float)
    p_ai = p_ai_all[sel]
    body = expl["body_len"][sel].astype(float)
    months = np.asarray([expl["month"][i] or "未知" for i in sel])
    domains = np.asarray([expl["domain"][i] for i in sel])
    cells = pd.factorize(pd.Series([f"{m}|{d}" for m, d in zip(months, domains)]),
                         sort=True)[0]
    strat = pd.factorize(pd.Series(
        [f"{m}|{d}|{b}" for m, d, b in
         zip(months, domains, length_bin_codes(body))]), sort=True)[0]
    print(f"梯度样本：{len(sel):,} 条（去重 + 正文 ≥{MIN_BODY_LEN} 字），"
          f"单元 {len(np.unique(cells))} 个")

    grad = {}
    for k in METRIC_ORDER:
        vals = expl[k][sel].astype(float)
        m = np.isfinite(vals)
        v = vals[m]
        p = p_ai[m]
        dec = decile_table(v, p)
        # 单元内秩相关：掩码子样本重新建单元（保持 min_cell_n 口径）
        sub_cells = pd.factorize(
            pd.Series([f"{mm}|{dd}" for mm, dd in
                       zip(months[m], domains[m])]), sort=True)[0]
        sub_strat = pd.factorize(
            pd.Series([f"{mm}|{dd}|{b}" for mm, dd, b in
                       zip(months[m], domains[m], length_bin_codes(body[m]))]),
            sort=True)[0]
        wc = within_cell_spearman(v, p, sub_cells, min_cell_n=MIN_CELL_N)
        st = within_cell_spearman(v, p, sub_strat, min_cell_n=MIN_CELL_N)
        p_hi, p_lo = float(np.quantile(p, 0.9)), float(np.quantile(p, 0.1))
        delta, dp = cliffs_delta(v[p >= p_hi], v[p <= p_lo])
        grad[k] = {
            "n": int(m.sum()),
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "deciles": dec,
            "weighted_rho": round(wc["weighted_rho"], 4),
            "neg_cell_share": round(wc["neg_cell_share"], 4),
            "n_cells": wc["n_cells"],
            "stratified_rho": round(st["weighted_rho"], 4),
            "stratified_cells": st["n_cells"],
            "cliff_delta_top_vs_bottom": round(delta, 4),
            "cliff_p": float(dp),
        }
        print(f"  {k}: n={m.sum():,} wcell ρ={wc['weighted_rho']:+.4f} "
              f"(长度分层 {st['weighted_rho']:+.4f})，δ={delta:+.4f}")

    # 近重复句普及率的十分位曲线（二值；Wilson CI 由 decile_table 的正态
    # CI 近似，另行在结果中报总体普及率）
    dup_sel = expl["dup_sent"][sel].astype(float)
    dm = np.isfinite(dup_sel)
    dup_bin = (dup_sel[dm] > 0).astype(float)
    p_dup = p_ai[dm]
    dup_dec = decile_table(dup_bin, p_dup)
    dup_cells = pd.factorize(
        pd.Series([f"{mm}|{dd}" for mm, dd in zip(months[dm], domains[dm])]),
        sort=True)[0]
    dup_wc = within_cell_spearman(dup_bin, p_dup, dup_cells,
                                  min_cell_n=MIN_CELL_N)
    dup_any_grad = {
        "n": int(dm.sum()),
        "overall_share": float(dup_bin.mean()) if dup_bin.size else 0.0,
        "deciles": dup_dec,
        "weighted_rho": round(dup_wc["weighted_rho"], 4),
        "neg_cell_share": round(dup_wc["neg_cell_share"], 4),
        "n_cells": dup_wc["n_cells"],
    }
    print(f"  dup_any: 普及率={dup_any_grad['overall_share']:.1%}，"
          f"wcell ρ={dup_wc['weighted_rho']:+.4f}")

    # ---- 互动关联（与 engagement_drivers 同协议；实义句 ≥2 的共同样本）----
    feats = pd.read_csv(FEATURES_PATH,
                        usecols=["month", "domain", "content_len", "n_hashtags",
                                 "likes", "collections", "comments"])
    if len(feats) != n_scanned:
        raise SystemExit("features 与 scores 行数不一致")
    mismatch = int((scores["content_len"].to_numpy()
                    != feats["content_len"].to_numpy()).sum())
    if mismatch:
        raise SystemExit(f"scores 与 features 行序未对齐：{mismatch}")
    require_row_alignment(expl, feats, "features")
    likes_mismatch = colls_mismatch = 0
    text_len_drift = int((feats["content_len"].to_numpy()
                          != expl["body_len"]).sum())
    target_cols = {name: np.log10(1 + np.clip(feats[name].to_numpy(dtype=float),
                                              0, None))
                   for name, _ in TARGETS}
    month_all = feats["month"].astype(str).to_numpy()
    dom_all = feats["domain"].astype(str).to_numpy()
    metric_all = {k: expl[k].astype(float) for k in METRIC_ORDER}
    multi = expl["n_sent"] >= 2
    keep_mask = np.zeros(len(expl["body_len"]), dtype=bool)
    keep_mask[keep] = True
    keep_eng = (keep_mask & (month_all >= MONTH_FROM)
                & (expl["body_len"] >= MIN_BODY_LEN) & multi
                & np.isfinite(expl["gram6"].astype(float))
                & np.isfinite(expl["max_sim"].astype(float))
                & np.isfinite(expl["dup_sent"].astype(float))
                & np.isfinite(expl["open_rep"].astype(float)))
    eng_cells_raw = np.asarray([f"{m}|{d}" for m, d in zip(month_all, dom_all)])
    X = np.column_stack([metric_all[k] for k in METRIC_ORDER]
                        + [p_ai_all,
                           expl["body_len"].astype(float),
                           feats["n_hashtags"].to_numpy(dtype=float)])
    feature_names = METRIC_ORDER + ["p_ai", "content_len", "n_hashtags"]
    eng = {"fe": {}, "univariate": {}}
    prep = prepare_fe(X[keep_eng], eng_cells_raw[keep_eng], MIN_CELL_N)
    for target, _ in TARGETS:
        yk = target_cols[target][keep_eng]
        fit = fe_ols(X[keep_eng], yk, eng_cells_raw[keep_eng], prepared=prep)
        boot = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                        fit["cell_codes"], n_boot=N_BOOT,
                                        seed=BOOT_SEED)
        ci = np.percentile(boot, [2.5, 97.5], axis=0)
        eng["fe"][target] = {
            "features": feature_names,
            "coef": {n: round(float(v), 4) for n, v in zip(feature_names, fit["coef"])},
            "ci_low": {n: round(float(v), 4) for n, v in zip(feature_names, ci[0])},
            "ci_high": {n: round(float(v), 4) for n, v in zip(feature_names, ci[1])},
            "n_obs": fit["n_obs"], "n_cells": fit["n_cells"],
        }
        print(f"  互动 FE（{target}）: n={fit['n_obs']:,}, "
              + ", ".join(f"{k} β={fit['coef'][i]:+.4f}"
                          for i, k in enumerate(METRIC_ORDER)))

    groups_eng = cell_groups(eng_cells_raw[keep_eng], MIN_CELL_N)
    y_ranks = {t: cell_rank_cache(target_cols[t][keep_eng], groups_eng)
               for t, _ in TARGETS}
    for k in METRIC_ORDER:
        xk = metric_all[k][keep_eng]
        x_ranks = cell_rank_cache(xk, groups_eng)
        eng["univariate"][k] = {
            t: within_cell_spearman(xk, target_cols[t][keep_eng],
                                    eng_cells_raw[keep_eng],
                                    min_cell_n=MIN_CELL_N, groups=groups_eng,
                                    x_ranks=x_ranks, y_ranks=y_ranks[t])
            for t, _ in TARGETS
        }

    # ---- 指标相关矩阵（共同样本上的 Spearman 秩相关）----
    common = keep_eng
    corr = {}
    mat_vals = {k: metric_all[k][common] for k in METRIC_ORDER}
    for a in METRIC_ORDER:
        corr[a] = {}
        for b in METRIC_ORDER:
            if a == b:
                corr[a][b] = 1.0
                continue
            x, y = mat_vals[a], mat_vals[b]
            xr = pd.Series(x).rank().to_numpy()
            yr = pd.Series(y).rank().to_numpy()
            corr[a][b] = round(float(np.corrcoef(xr, yr)[0, 1]), 4)

    # ---- AIGC 家族 / 单模型指纹（实义句 >=2 的共同样本；长度标准化） ----
    aigc = scans["aigc"]
    ok = (aigc["body_len"] >= MIN_BODY_LEN) & (aigc["n_sent"] >= 2)
    ok &= np.isfinite(aigc["gram6"].astype(float))
    aigc_lens = aigc["body_len"][ok].astype(float)
    resid = {}
    for k in METRIC_ORDER:
        vals = aigc[k][ok].astype(float)
        resid[k] = length_residualize(vals, aigc_lens)
    fam_agg: dict = defaultdict(lambda: {"n": 0, "raw": defaultdict(float),
                                         "res": defaultdict(float)})
    mod_agg: dict = defaultdict(lambda: {"n": 0, "raw": defaultdict(float),
                                         "res": defaultdict(float)})
    for i, idx in enumerate(np.flatnonzero(ok)):
        for agg, key in ((fam_agg, aigc["family"][idx]), (mod_agg, aigc["model"][idx])):
            slot = agg[key]
            slot["n"] += 1
            for k in METRIC_ORDER:
                slot["raw"][k] += float(aigc[k][idx])
                slot["res"][k] += float(resid[k][i])

    def agg_to_table(agg):
        out = {}
        for key, slot in agg.items():
            if not key or not slot["n"]:
                continue
            out[key] = {
                "n": slot["n"],
                "mean": {k: round(slot["raw"][k] / slot["n"], 4)
                         for k in METRIC_ORDER},
                "mean_len_std": {k: round(slot["res"][k] / slot["n"], 4)
                                 for k in METRIC_ORDER},
            }
        return out

    family_table = agg_to_table(fam_agg)
    model_table = agg_to_table(mod_agg)
    family_z = {f: family_table[f]["mean_len_std"] for f in family_table}
    family_n = {f: family_table[f]["n"] for f in family_table}

    # ---- 季度趋势（探索集去重、实义句 >=2；原始 + 领域×长度固定权重）----
    q_mask = np.zeros(len(expl["month"]), dtype=bool)
    q_mask[keep] = True
    q_mask &= expl["body_len"] >= MIN_BODY_LEN
    q_mask &= expl["n_sent"] >= 2
    q_idx = np.flatnonzero(q_mask)
    q_df = pd.DataFrame({
        "quarter": [f"{m[:4]}Q{(int(m[4:6]) - 1) // 3 + 1}" if m else "未知"
                    for m in (expl["month"][i] for i in q_idx)],
        "domain": [expl["domain"][i] for i in q_idx],
        "len_bin": length_bin_codes(expl["body_len"][q_idx]),
    })
    for k in METRIC_ORDER:
        q_df[k] = expl[k][q_idx].astype(float)
    q_df["cell_key"] = q_df["domain"] + "|" + q_df["len_bin"].astype(str)
    q_df = q_df[q_df["quarter"] != "未知"]
    q_counts = q_df.groupby("quarter").size()
    valid_q = sorted(q_counts[q_counts >= MIN_QUARTER_N].index)
    q_df = q_df[q_df["quarter"].isin(valid_q)]
    quarterly = {"quarters": valid_q, "n": {q: int(q_counts[q]) for q in valid_q},
                 "raw": {}, "domain_len_std": {}, "trend": {},
                 "overall": {}, "sd": {}}
    for k in METRIC_ORDER:
        vals = q_df[k].to_numpy(dtype=float)
        quarterly["overall"][k] = round(float(np.nanmean(vals)), 4)
        quarterly["sd"][k] = round(float(np.nanstd(vals, ddof=1)), 4)
        raw = q_df.groupby("quarter")[k].mean().reindex(valid_q)
        # 领域×长度固定权重：把 cell_key（领域|长度分箱）当“领域”，
        # 权重来自探索集全样本的 cell 构成，修正季度间的长度/领域漂移。
        q_cell = q_df.drop(columns=["domain"]).rename(
            columns={"cell_key": "domain", k: "__v"})
        std = domain_standardized_series(q_cell, "__v", "quarter").reindex(valid_q)
        rho, p = _spearman(len(valid_q), raw.to_numpy())
        rho_s, p_s = _spearman(len(valid_q), std.to_numpy())
        quarterly["raw"][k] = [round(float(v), 4) for v in raw]
        quarterly["domain_len_std"][k] = [round(float(v), 4) for v in std]
        quarterly["trend"][k] = {"rho": round(rho, 4), "p": float(p),
                                 "rho_domain_len_std": round(rho_s, 4),
                                 "p_domain_len_std": float(p_s)}
    quarterly["cells"] = int((q_df.groupby("cell_key").size() >= 20).sum())

    # ---- 结果 ----
    out = {
        "meta": {
            "min_body_len": MIN_BODY_LEN, "sent_min_chars": SENT_MIN_CHARS,
            "gram_n": GRAM_N, "dup_sim_threshold": DUP_SIM_THRESHOLD,
            "opening_chars": OPENING_CHARS, "min_cell_n": MIN_CELL_N,
            "month_from": MONTH_FROM, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "min_quarter_n": MIN_QUARTER_N,
            "exploring_content_len_mismatch": mismatch,
            "exploring_text_len_drift": text_len_drift,
            "exploring_likes_mismatch": likes_mismatch,
            "exploring_collections_mismatch": colls_mismatch,
            "note": "人类/探索集按 note_id 去重（保留元数据最完整一条）；"
                    "AIGC 无 note_id 逐记录；p_ai 按行序对齐 "
                    "paper_exploring_scores.csv.gz；句子层指标只在实义句"
                    f"（≥{SENT_MIN_CHARS} 字）≥2 的帖子上有定义；"
                    "gram6 使用 64 位滚动哈希，单帖内碰撞概率可忽略。",
        },
        "metrics": {k: {"label": METRIC_LABELS[k],
                        "defined_on": ("all" if k not in SENT_METRICS
                                       else "multi_sentence")}
                    for k in METRIC_ORDER},
        "subset_stats": {
            name: {
                "n": subset_rates[name]["n"],
                "chars": subset_rates[name]["chars"],
                "n_multi": subset_rates[name]["n_multi"],
                "metrics": subset_rates[name]["metrics"],
                "length_bins": subset_rates[name]["length_bins"],
                "dup_any": subset_rates[name]["dup_any"],
            } for name in SUBSET_ORDER},
        "length_matched_gap_vs_human": matched,
        "exploring_gradient": {
            "n": int(len(sel)),
            "n_cells": int(len(np.unique(cells))),
            "metrics": grad,
            "dup_any": dup_any_grad,
        },
        "engagement": {
            "month_from": MONTH_FROM, "min_body_len": MIN_BODY_LEN,
            "multi_sentence_only": True,
            "univariate": {
                k: {t: {"weighted_rho": round(v[t]["weighted_rho"], 4),
                        "neg_cell_share": round(v[t]["neg_cell_share"], 4),
                        "n_cells": v[t]["n_cells"]}
                    for t, _ in TARGETS}
                for k, v in eng["univariate"].items()},
            "fe": eng["fe"],
        },
        "metric_correlation": corr,
        "family": {"families": family_table, "models": model_table},
        "quarterly": quarterly,
        "length_control": {"bins": [lab for _, _, lab in LENGTH_BINS]},
    }
    save_json(out, BASE + "redundancy_results.json")
    print("已写出 redundancy_results.json")

    # ---- 词表图的 (d) 面板需要逐帖 max_sim 分布：就地重扫一次该指标样本 ----
    ms = expl["max_sim"][sel].astype(float)
    ms = ms[np.isfinite(ms)]
    dist = {"exploring": ms}
    for name in ("human", "aigc"):
        sc = scans[name]
        kk = keeps[name][sc["body_len"][keeps[name]] >= MIN_BODY_LEN]
        v = sc["max_sim"][kk].astype(float)
        dist[name] = v[np.isfinite(v)]
    plot_lexicon(subset_rates, matched, dist)
    write_markdown(out)
    plot_gradient(grad, eng, family_z, family_n, quarterly, dup_any_grad)
    print("已写出 redundancy_results.md 与 insights/redundancy_*.png")


def _spearman(n: int, values) -> tuple[float, float]:
    """等距序号 vs 序列的 Spearman ρ（与 cta_analysis 相同的最小实现）。"""
    from scipy.stats import spearmanr

    if n < 3:
        return 0.0, 1.0
    rho, p = spearmanr(np.arange(n), np.asarray(values, dtype=float))
    return float(rho), float(p)


def plot_lexicon(subset_rates: dict, matched: dict, dist: dict) -> None:
    """四联图：子集均值 / 长度匹配差异 / 长度曲线 / 句间相似度分布。"""
    init_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.0))

    # (a) 四指标 × 三子集均值
    ax = axes[0, 0]
    x = np.arange(len(METRIC_ORDER))
    w = 0.26
    for j, subset in enumerate(SUBSET_ORDER):
        vals, los, his = [], [], []
        for k in METRIC_ORDER:
            rec = subset_rates[subset]["metrics"][k]
            vals.append(rec["mean"])
            los.append(rec["ci_low"])
            his.append(rec["ci_high"])
        vals = np.asarray(vals)
        err = np.vstack([vals - np.asarray(los), np.asarray(his) - vals])
        pos = x + (j - 1) * w
        ax.bar(pos, vals, width=w, color=PALETTE[subset],
               label=SUBSET_LABELS[subset], alpha=0.92)
        ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor="black",
                    elinewidth=0.9, capsize=2.5, zorder=3)
        for p, v in zip(pos, vals):
            ax.annotate(f"{v:.3f}", xy=(p, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[k] for k in METRIC_ORDER], fontsize=9)
    ax.set_ylabel("帖子层均值（误差棒=95% CI）")
    ax.set_title("(a) 三子集的正文自重复指标（帖子层，正文 ≥%d 字）" % MIN_BODY_LEN)
    ax.legend(fontsize=8.5, loc="upper right")

    # (b) 长度匹配差异
    ax = axes[0, 1]
    ypos = np.arange(len(METRIC_ORDER))
    for key, glabel, color, offset in (
            ("aigc", "AIGC − 人类", "#C44E52", -0.16),
            ("exploring", "探索集 − 人类", "#55A868", 0.16)):
        first = True
        for i, k in enumerate(METRIC_ORDER):
            rec = matched[key][k]
            sig = (rec["ci_low"] > 0) or (rec["ci_high"] < 0)
            ax.plot([rec["ci_low"], rec["ci_high"]],
                    [ypos[i] + offset, ypos[i] + offset], color=color,
                    lw=1.8, solid_capstyle="butt", zorder=2)
            ax.scatter([rec["gap"]], [ypos[i] + offset], s=34, zorder=3,
                       facecolor=color if sig else "white", edgecolor=color,
                       label=glabel if first else None)
            first = False
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([METRIC_LABELS[k] for k in METRIC_ORDER], fontsize=9)
    ax.set_xlabel("长度匹配均值差（按人类长度分布加权，含 95% CI）")
    ax.set_title("(b) 长度匹配后的自重复差异（>0 = 该子集更复读）")
    ax.legend(fontsize=8.5, loc="best")

    # (c) 长度曲线（相对子集均值）
    ax = axes[1, 0]
    labels = [lab for _, _, lab in LENGTH_BINS]
    xb = np.arange(len(labels))
    for k in METRIC_ORDER:
        for subset in SUBSET_ORDER:
            bins = subset_rates[subset]["length_bins"][k]
            overall = subset_rates[subset]["metrics"][k]["mean"]
            if not overall or not np.isfinite(overall):
                continue
            ys = np.asarray([b["mean"] for b in bins], dtype=float) / overall
            ns = [b["n"] for b in bins]
            ys = np.where(np.asarray(ns) >= 20, ys, np.nan)
            ax.plot(xb, ys, LINESTYLES[subset], color=METRIC_COLORS[k],
                    lw=1.6, marker="o", ms=3.4, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(xb)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel("分箱均值 / 该子集总体均值（log 刻度）")
    ax.set_title("(c) 指标的长度依赖（线型=子集，颜色=指标；<20 条的分箱不画）")
    metric_handles = [plt.Line2D([], [], color=METRIC_COLORS[k], lw=6,
                                 label=METRIC_LABELS[k]) for k in METRIC_ORDER]
    style_handles = [plt.Line2D([], [], color="#555555", lw=1.6,
                                ls=LINESTYLES[s], label=SUBSET_LABELS[s])
                     for s in SUBSET_ORDER]
    leg1 = ax.legend(handles=metric_handles, fontsize=8, loc="upper left",
                     title="指标", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, fontsize=8, loc="lower right",
              title="子集", title_fontsize=8)

    # (d) 句间最大相似度分布（多句帖）
    ax = axes[1, 1]
    edges = np.linspace(0, 1, 41)
    for subset in SUBSET_ORDER:
        v = dist[subset]
        ax.hist(v, bins=edges, histtype="step", lw=1.8, density=True,
                color=PALETTE[subset],
                label=f"{SUBSET_LABELS[subset]}（中位 {np.median(v):.2f}）")
    ax.set_xlabel("句间最大 Jaccard 相似度（多句帖）")
    ax.set_ylabel("密度")
    ax.set_title("(d) 多句帖的句间最大相似度分布")
    ax.legend(fontsize=8.5)
    save_fig(fig, INSIGHTS + "redundancy_lexicon.png")


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def write_markdown(out: dict) -> None:
    meta = out["meta"]
    rates = out["subset_stats"]
    grad = out["exploring_gradient"]
    eng = out["engagement"]
    q = out["quarterly"]
    lines = [
        "# 正文自重复与句式复读（within-post redundancy）分析结果",
        "",
        f"> 由 `redundancy_analysis.py` 全量运行生成；正文清洗与 `paper_features.py` "
        f"一致，人类/探索集按 note_id 去重；正文 ≥{meta['min_body_len']} 字；"
        f"实义句（≥{meta['sent_min_chars']} 字）≥2 句才有句子层指标；"
        f"探索集 p_ai 按行序对齐 `paper_exploring_scores.csv.gz`"
        f"（scores/features content_len mismatch="
        f"{meta['exploring_content_len_mismatch']}，原始文本长度 drift="
        f"{meta['exploring_text_len_drift']}；月份/领域/互动逐条一致）。",
        "",
        "指标：`gram6` 重复字符 6-gram 出现占比（64 位滚动哈希）；`max_sim` 句间"
        "最大字符 3-gram Jaccard；`dup_sent` 参与 Jaccard ≥ "
        f"{meta['dup_sim_threshold']} 近重复句对的实义句占比；`open_rep` 句首 "
        f"{meta['opening_chars']} 字复读的实义句占比。",
        "",
        "## 1 跨子集自重复水平（帖子层均值，含 95% CI）",
        "",
        "| 子集 | 有效帖数 | 实义句≥2 | " + " | ".join(
            f"{METRIC_LABELS[k]}（n）" for k in METRIC_ORDER) + " | 近重复句普及率 |",
        "|---|---:|---:|" + "---:|" * len(METRIC_ORDER) + "---:|",
    ]
    for name in SUBSET_ORDER:
        rec = rates[name]
        cells = []
        for k in METRIC_ORDER:
            m = rec["metrics"][k]
            cells.append(f"{m['mean']:.4f}（{m['n_defined']:,}）")
        lines.append(
            f"| {SUBSET_LABELS[name]} | {rec['n']:,} | {rec['n_multi']:,} | "
            + " | ".join(cells)
            + f" | {rec['dup_any']['share']:.1%} |")
    lines += [
        "",
        "> 注：n 为该指标有定义的帖子数（句子层指标只在实义句 ≥2 时定义）；"
        "近重复句普及率 = 至少存在一对 Jaccard ≥ 0.5 句对的帖子占比。",
        "",
        "**长度匹配差异（按人类长度分布加权，>0 = 该子集比人类更复读）：**",
        "",
        "| 指标 | AIGC−人类 [95% CI] | 探索集−人类 [95% CI] |",
        "|---|---:|---:|",
    ]
    for k in METRIC_ORDER:
        a = out["length_matched_gap_vs_human"]["aigc"][k]
        e = out["length_matched_gap_vs_human"]["exploring"][k]
        lines.append(
            f"| {METRIC_LABELS[k]} | {a['gap']:+.4f} "
            f"[{a['ci_low']:+.4f}, {a['ci_high']:+.4f}] | "
            f"{e['gap']:+.4f} [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}] |")
    lines += [
        "",
        "## 2 探索集内的 P(AI) 梯度（去重，正文 ≥%d 字）" % meta["min_body_len"],
        "",
        "| 指标 | n | 单元内加权 ρ | 负单元占比 | 长度分层 ρ | 首尾十分位 δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in METRIC_ORDER:
        g = grad["metrics"][k]
        lines.append(
            f"| {METRIC_LABELS[k]} | {g['n']:,} | {g['weighted_rho']:+.4f} | "
            f"{g['neg_cell_share']:.1%} | {g['stratified_rho']:+.4f} | "
            f"{g['cliff_delta_top_vs_bottom']:+.3f} |")
    d = grad["dup_any"]
    lines += [
        f"| 存在近重复句（二值） | {d['n']:,} | {d['weighted_rho']:+.4f} | "
        f"{d['neg_cell_share']:.1%} | — | — |",
        "",
        f"（总体普及率 {d['overall_share']:.1%}；长度分层 = 月份×领域×长度分箱单元；"
        f"首尾十分位 δ = P(AI) 前 10% vs 后 10% 的 Cliff's δ。）",
        "",
        "## 3 互动偏效应（月份×领域 FE + 整群 bootstrap %d 次）" % meta["n_boot"],
        "",
        "| 特征 | 点赞 β [95% CI] | 收藏 β [95% CI] | 评论 β [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for name in eng["fe"]["likes"]["features"]:
        cells_txt = []
        for t, _ in TARGETS:
            fe = eng["fe"][t]
            cells_txt.append(
                f"{fe['coef'][name]:+.4f} "
                f"[{fe['ci_low'][name]:+.4f}, {fe['ci_high'][name]:+.4f}]")
        lines.append(f"| {name} | " + " | ".join(cells_txt) + " |")
    lines += [
        "",
        f"（n={eng['fe']['likes']['n_obs']:,}，{eng['fe']['likes']['n_cells']} "
        f"个单元；实义句 ≥2 的共同样本；特征 z-score，β 为每 1 SD 的偏效应；"
        f"{meta['month_from']} 起、正文 ≥{meta['min_body_len']} 字。）",
        "",
        "## 4 AIGC 家族/模型指纹（长度标准化均值，共同样本）",
        "",
        "| 家族 | n | " + " | ".join(METRIC_LABELS[k] for k in METRIC_ORDER) + " |",
        "|---|---:|" + "---:|" * len(METRIC_ORDER),
    ]
    for fam in FAMILIES:
        slot = out["family"]["families"].get(fam)
        if not slot or slot["n"] < MIN_FAMILY_N:
            continue
        lines.append(
            f"| {FAMILY_NAME.get(fam, fam)} | {slot['n']} | "
            + " | ".join(f"{slot['mean_len_std'][k]:+.4f}"
                         for k in METRIC_ORDER)
            + " |")
    lines += [
        "",
        "## 5 季度趋势（探索集去重，实义句 ≥2 的共同样本）",
        "",
        "| 季度 | n | " + " | ".join(METRIC_LABELS[k] for k in METRIC_ORDER) + " |",
        "|---|---:|" + "---:|" * len(METRIC_ORDER),
    ]
    for i, qq in enumerate(q["quarters"]):
        lines.append(
            f"| {qq} | {q['n'][qq]:,} | "
            + " | ".join(f"{q['raw'][k][i]:.4f}" for k in METRIC_ORDER)
            + " |")
    lines.append("")
    for k in METRIC_ORDER:
        t = q["trend"][k]
        lines.append(f"- {METRIC_LABELS[k]}：季度 ρ={t['rho']:+.3f}（p={t['p']:.3f}），"
                     f"领域×长度固定权重后 ρ={t['rho_domain_len_std']:+.3f}"
                     f"（p={t['p_domain_len_std']:.3f}）。")
    lines += [
        "",
        "**四指标共同样本的 Spearman 相关（探索集）：**",
        "",
        "| | " + " | ".join(METRIC_LABELS[k] for k in METRIC_ORDER) + " |",
        "|---|" + "---:|" * len(METRIC_ORDER),
    ]
    for a in METRIC_ORDER:
        lines.append(f"| {METRIC_LABELS[a]} | " + " | ".join(
            f"{out['metric_correlation'][a][b]:+.3f}" for b in METRIC_ORDER) + " |")
    lines += [
        "",
        "图表见 `insights/redundancy_lexicon.png`、`insights/redundancy_gradient.png`。",
    ]
    with open(BASE + "redundancy_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
