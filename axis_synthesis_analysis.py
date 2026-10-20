# -*- coding: utf-8 -*-
"""深潜分析之二十：跨轴综合（五轴风格指标的共现结构、独立信号与互动堆叠）。

前十九个深潜分析把「AI 味」拆成了五条独立的风格轴（语气情态 / 具体性 /
互动引导 / 标点排版 / 篇章连接），每条轴只在自身类别内回答「跨子集使用
差异、野外 P(AI) 梯度、互动偏效应」。但轴与轴之间**从未被放在同一张
桌子上**：这些风格指标是彼此独立的信号，还是同一套「AI 腔」的不同侧面？
哪条轴在控制其余轴后仍然携带 P(AI) 信息？它们对互动的影响会互相抵消还是
层层叠加？本脚本用五个轴各自的 30 个先验类别密度（不引入任何新的监督
信号）回答：

1. **共现结构**：探索集内 30 个类别密度与 5 个轴复合指标的「月份×领域」
   单元内 Spearman 相关矩阵——AI 腔是一束还是多束？
2. **独立信号**：五轴复合指标联合进入「月份×领域」固定效应回归时，各自
   对 P(AI) 的偏效应 β（整群 bootstrap CI）；30 类别全模型作为附录。
3. **轴空间可分性**：仅用轴词表密度（30 个特征）做人 vs AIGC 的 5 折
   交叉验证 AUC，并与各轴单独 AUC 对照——先验风格轴联合起来能还原多少
   检测器的判别力。
4. **轴堆叠**：按「与 P(AI) 同向的轴数」（0–5）分层看 P(AI) 与互动——
   多轴叠加是否比单轴更「AI」、互动惩罚是否随堆叠加深。
5. **联合互动模型**：五轴复合指标 + P(AI) + 长度 + 标签数同时进入互动
   固定效应回归（此前每条轴各自控制 P(AI)，从未五轴同场），以及「AI 向
   轴数」作为单一指数的互动偏效应。

口径说明：
- 正文清洗复用 analysis_utils.clean_body（与 paper_features.py 一致）；
- 人类/探索集按 note_id 去重（保留元数据最完整一条，规则同 deep_dive_analysis），
  AIGC 无 note_id 逐记录处理；
- 探索集 p_ai 按行序对齐 paper_exploring_scores.csv.gz；行序用原始
  month/domain/likes/collections/comments 逐条校验，content_len 仅作
  清洗版本漂移的信息性指标（不参与错位判定）；
- 类别计数直接调用五条轴分析脚本的纯计数函数（stance.word_counts /
  concreteness.classify / cta.cta_counts / typography.typography_counts /
  discourse.word_counts），保证与各轴结果同口径；
- 轴复合密度 = 该轴各类别密度（次/百字）之和；共现相关与堆叠用「单元内
  平均秩 z-score」（月份×领域单元 ≥100 条，与既有轴分析同协议）；
- 互动回归与 engagement_drivers_analysis.py 完全同协议（特征先 z-score
  再单元内去均值、整群 bootstrap 300 次）。

输出：
  insights/axis_synthesis_structure.png  类别共现热力图 + 轴复合相关矩阵 + 跨子集画像
  insights/axis_synthesis_signal.png     P(AI) 独立偏效应 / 轴空间 AUC / 堆叠分层 / 联合互动森林
  axis_synthesis_results.json            全部数值结果
  axis_synthesis_results.md              结果摘要
"""
from __future__ import annotations

from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis_utils import (BASE, INSIGHTS, average_ranks, body_cache,
                            init_plot_style,
                            save_fig, save_json, cell_groups, cell_rank_cache,
                            clean_body, content_of, cv_auc, dedup_indices,
                            engagement_of, parse_local_time,
                            require_row_alignment, run_parallel,
                            within_cell_spearman)
from derived_cache import iter_cached
from cta_analysis import CATEGORY_LABELS as CTA_LABELS
from cta_analysis import CATEGORY_ORDER as CTA_ORDER
from cta_analysis import cta_counts
from concreteness_analysis import CATEGORY_LABELS as CONCR_LABELS
from concreteness_analysis import CATEGORY_ORDER as CONCR_ORDER
from concreteness_analysis import classify as concrete_classify
from discourse_analysis import CATEGORY_LABELS as DISC_LABELS
from discourse_analysis import CATEGORY_ORDER as DISC_ORDER
from discourse_analysis import word_counts as discourse_word_counts
from engagement_drivers_analysis import (cluster_bootstrap_fe_ols, fe_ols,
                                         prepare_fe)
from stance_analysis import CATEGORY_LABELS as STANCE_LABELS
from stance_analysis import CATEGORY_ORDER as STANCE_ORDER
from stance_analysis import word_counts as stance_word_counts
from typography_analysis import CATEGORY_LABELS as TYPO_LABELS
from typography_analysis import CATEGORY_ORDER as TYPO_ORDER
from typography_analysis import typography_counts

DATASETS = {
    "human": BASE + "training_set_human_fixed.jsonl",
    "aigc": BASE + "training_set_aigc.jsonl",
    "exploring": BASE + "exploring_set_fixed.jsonl",
}
SCORES_PATH = BASE + "paper_exploring_scores.csv.gz"
FEATURES_PATH = BASE + "paper_features_exploring.csv.gz"

MIN_BODY_LEN = 50        # 风格密度在更短正文（纯标签/空正文）上无意义
MIN_CELL_N = 100         # 单元内统计的最小单元样本量（与论文口径一致）
MONTH_FROM = "202301"    # 互动分析起始月份（与 paper_engagement 一致）
N_BOOT = 300             # 整群 bootstrap 次数
BOOT_SEED = 42
STACK_BOOT = 1000        # 堆叠分层 P(AI) 均值的描述性重抽次数
MAX_STACK = 5

SUBSET_ORDER = ["human", "aigc", "exploring"]

# 五条轴与各自的类别（顺序即图中块顺序，类别名全局唯一）
AXES = [
    ("stance", "语气情态", STANCE_ORDER, STANCE_LABELS),
    ("concreteness", "具体性", CONCR_ORDER, CONCR_LABELS),
    ("cta", "互动引导", CTA_ORDER, CTA_LABELS),
    ("typography", "标点排版", TYPO_ORDER, TYPO_LABELS),
    ("discourse", "篇章连接", DISC_ORDER, DISC_LABELS),
]
AXIS_ORDER = [a[0] for a in AXES]
AXIS_NAME = {a[0]: a[1] for a in AXES}
AXIS_COLORS = {"stance": "#4C72B0", "concreteness": "#55A868",
               "cta": "#C44E52", "typography": "#8172B3",
               "discourse": "#DD8452"}
ALL_CATEGORIES = [c for _, _, cats, _ in AXES for c in cats]
CATEGORY_AXIS = {c: a for a, _, cats, _ in AXES for c in cats}
CATEGORY_LABEL = {c: lab for _, _, cats, labs in AXES for c in cats
                  for lab in [labs[c]]}
CAT_INDEX = {c: i for i, c in enumerate(ALL_CATEGORIES)}
AXIS_INDICES = {a: [CAT_INDEX[c] for c in cats] for a, _, cats, _ in AXES}
N_CAT = len(ALL_CATEGORIES)
EPS_RATIO = 0.01         # log2 比值中的加性平滑（次/百字）


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def category_vector(body: str) -> np.ndarray:
    """一段正文的 30 个先验类别命中次数（按 ALL_CATEGORIES 顺序）。

    直接调用五条轴分析脚本的纯计数函数，返回 int32 向量；可测试且与各轴
    结果完全同口径。
    """
    v = np.zeros(N_CAT, dtype=np.int32)
    sc = stance_word_counts(body)
    for k in STANCE_ORDER:
        v[CAT_INDEX[k]] = sum(sc[k].values())
    cc, *_ = concrete_classify(body)
    for k in CONCR_ORDER:
        v[CAT_INDEX[k]] = cc[k]
    for k, n in cta_counts(body).items():
        if k in CAT_INDEX:
            v[CAT_INDEX[k]] = n
    for k, n in typography_counts(body).items():
        if k in CAT_INDEX:
            v[CAT_INDEX[k]] = n
    dc = discourse_word_counts(body)
    for k in DISC_ORDER:
        v[CAT_INDEX[k]] = sum(dc[k].values())
    return v


def within_cell_rank_z(values, groups) -> np.ndarray:
    """按单元分组把数值转成「单元内平均秩 z-score」。

    单元内常量（ptp=0）、含缺失值或不在任何单元（`groups` 未覆盖）的记录
    返回 NaN；其余为 (rank − mean) / std（ddof=0）。等价于先做单元内
    Spearman 的秩变换再标准化，供跨类别的共现相关与轴堆叠复用。
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    for idx in groups:
        vc = v[idx]
        if vc.size == 0 or bool(np.isnan(vc).any()) \
                or float(np.ptp(vc)) == 0.0:
            continue
        r = average_ranks(vc)
        out[idx] = (r - r.mean()) / r.std()
    return out


def _average_ranks_matrix(values: np.ndarray) -> np.ndarray:
    """按列计算并列平均秩（1-based），与逐列 `average_ranks` 逐位一致。"""
    n = values.shape[0]
    sorter = np.argsort(values, axis=0)
    inv = np.argsort(sorter, axis=0)
    sorted_values = np.take_along_axis(values, sorter, axis=0)
    obs = np.ones(values.shape, dtype=bool)
    if n > 1:
        obs[1:] = sorted_values[1:] != sorted_values[:-1]
    rows = np.arange(n, dtype=np.intp)[:, None]
    group_start = np.maximum.accumulate(np.where(obs, rows, 0), axis=0)
    # 每行所属并列段之后的第一个起点（不存在则为 n），即段的右开边界。
    start_pos = np.where(obs, rows, n)
    suffix_next = np.empty_like(start_pos)
    suffix_next[-1] = n
    if n > 1:
        suffix_next[:-1] = np.minimum.accumulate(
            start_pos[:0:-1], axis=0)[::-1]
    ranks_sorted = 0.5 * (group_start + suffix_next + 1)
    return np.take_along_axis(ranks_sorted, inv, axis=0)


def within_cell_rank_z_matrix(values, groups) -> np.ndarray:
    """对多列数值一次性做单元内平均秩 z-score，等价于逐列调用
    `within_cell_rank_z`，但把每单元 30 次一维排序合并为一次二维排序。

    常量列、含缺失值的列以及未覆盖的记录保持 NaN，与单列版本口径一致。
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 2:
        raise ValueError("values must be 2-D")
    out = np.full(v.shape, np.nan)
    for idx in groups:
        vc = v[idx]
        if vc.size == 0:
            continue
        valid = ~np.isnan(vc).any(axis=0) & (np.ptp(vc, axis=0) != 0.0)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            continue
        z = _average_ranks_matrix(vc[:, valid_idx])
        z = (z - z.mean(axis=0)) / z.std(axis=0)
        out[np.ix_(idx, valid_idx)] = z
    return out


def composite_z(z_matrix: np.ndarray) -> pd.DataFrame:
    """把 30 列类别 z-score 按轴取均值（自动跳过 NaN 列）。

    返回列顺序为 AXIS_ORDER 的 DataFrame；某行若该轴全部类别为 NaN 则
    该轴为 NaN。
    """
    df = pd.DataFrame(z_matrix, columns=ALL_CATEGORIES)
    comp = df.T.groupby(pd.Series(CATEGORY_AXIS)).mean().T
    return comp[[a for a in AXIS_ORDER if a in comp.columns]]


def composite_density(dens_matrix: np.ndarray) -> np.ndarray:
    """把 30 列类别密度（次/百字）按轴求和成 5 列轴复合密度。"""
    arr = np.asarray(dens_matrix, dtype=float)
    return np.column_stack([arr[:, AXIS_INDICES[a]].sum(axis=1)
                            for a in AXIS_ORDER])


def log2_ratio(a: float, b: float, eps: float = EPS_RATIO) -> float:
    """log2((a+eps)/(b+eps))，用于跨子集使用率比值（加性平滑防零）。"""
    return float(np.log2((a + eps) / (b + eps)))


def stack_hits(z_matrix: np.ndarray, signs) -> tuple[np.ndarray, np.ndarray]:
    """「与 P(AI) 同向的轴数」与「有效轴数」。

    `signs` 为每个轴在联合 P(AI) 模型中的偏效应方向；某轴的 z 与 sign 乘积
    > 0 记 1 次命中（并列于 0 不算）。轴 z 为 NaN（单元过小/常量）时不
    参与命中，计入有效轴数用于报告。
    """
    z = np.asarray(z_matrix, dtype=float)
    s = np.asarray(signs, dtype=float)
    valid = np.isfinite(z)
    hit = np.where(valid, (z * s[None, :]) > 0, False)
    return hit.sum(axis=1).astype(np.int32), valid.sum(axis=1).astype(np.int32)


def fe_r2(fit: dict) -> float:
    """FE-OLS 的组内 R² = 1 − SSres/SStot（基于单元内去均值后的 y）。"""
    resid = fit["y_demeaned"] - fit["X_demeaned"] @ fit["coef"]
    sst = float((fit["y_demeaned"] ** 2).sum())
    return float(1 - (resid ** 2).sum() / sst) if sst > 0 else float("nan")


def stack_shares(stack_arr, ok_stack) -> dict:
    """各「与 P(AI) 同向的轴数」在有效帖子中的占比。

    `stack_arr` 用 -1 标记有效轴数不足 5 的帖子；这些帖子不进入分布，
    分母必须是有效帖子数（`ok_stack` 之和），否则占比之和小于 1。
    """
    arr = np.asarray(stack_arr)
    n = int(np.asarray(ok_stack, dtype=bool).sum())
    ks = sorted(int(k) for k in np.unique(arr[arr >= 0]))
    if n <= 0:
        return {str(k): 0.0 for k in ks}
    return {str(k): float((arr == k).sum() / n) for k in ks}


def mean_ci(values, n_boot: int = STACK_BOOT, seed: int = BOOT_SEED) -> dict:
    """均值 + 对帖子重抽的描述性 95% bootstrap CI（不处理单元聚类）。"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = v[rng.integers(0, len(v), len(v))].mean()
    return {"mean": float(v.mean()),
            "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
            "n": int(len(v))}


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
def scan_corpus(path: str) -> dict:
    """单次流式扫描一个语料，返回逐记录数组（30 类别计数 + 互动/元数据）。

    每条记录保存：正文长度、note_id、元数据完整度（用于去重取舍）、月份、
    领域、模型家族/模型（AIGC）、点赞数，以及 30 个类别的命中次数矩阵。
    """
    body_len, note_id, meta, month, domain = [], [], [], [], []
    family, model, likes = [], [], []
    vectors = []
    n_total = 0
    # 重复正文（探索集约 45%）直接复用上一次的 30 类向量；缓存值是只读的
    # int32 数组（下游只做 vstack 拷贝），复用同一对象不影响结果。
    cached = body_cache(category_vector, max_entries=60_000)
    for rec in iter_cached(path):
        n_total += 1
        content = content_of(rec)
        body = clean_body(content)
        L = len(body)
        body_len.append(L)
        note_id.append(rec.get("note_id"))
        dom = rec.get("domain")
        domain.append(dom if dom not in (None, "") else "未知")
        likes_v, colls, comms = engagement_of(rec)
        likes.append(likes_v)
        has_meta = bool(rec.get("user_id")) and dom not in (None, "", "未知")
        dt = parse_local_time(rec.get("local_time"))
        month.append(f"{dt.year:04d}{dt.month:02d}" if dt else None)
        score = (int(has_meta) + int(bool(likes_v)) + int(bool(colls))
                 + int(bool(comms)) + int(dt is not None)
                 + int(bool(content.strip())))
        meta.append(score)
        family.append(rec.get("model_family") or "")
        model.append(rec.get("model") or "")
        vectors.append(cached(body))
    return {
        "path": path, "n_total": n_total,
        "body_len": np.asarray(body_len, dtype=np.int32),
        "note_id": note_id, "meta": np.asarray(meta, dtype=np.int32),
        "month": month, "domain": domain,
        "likes": np.asarray(likes, dtype=float),
        "family": family, "model": model,
        "counts": np.vstack(vectors) if vectors else np.zeros((0, N_CAT),
                                                              dtype=np.int32),
    }


def dedup_keep(scan: dict) -> np.ndarray:
    """按 note_id 去重（保留元数据最完整一条），返回排序后的保留索引。"""
    keep = dedup_indices(scan["note_id"], scan["meta"])
    return np.asarray(sorted(keep), dtype=int)


def rates_block(scan: dict, keep: np.ndarray,
                min_len: int = MIN_BODY_LEN) -> dict:
    """一个子集的类别/轴复合使用率（仅正文 >= min_len 的记录）。"""
    sel = keep[scan["body_len"][keep] >= min_len]
    n = int(len(sel))
    chars = int(scan["body_len"][sel].sum())
    counts = scan["counts"][sel].astype(float)
    cat = {}
    for i, c in enumerate(ALL_CATEGORIES):
        col = counts[:, i]
        cat[c] = {
            "density100": float(col.sum() / chars * 100) if chars else 0.0,
            "share": float((col > 0).mean()) if n else 0.0,
        }
    comp = {}
    for a in AXIS_ORDER:
        col = counts[:, AXIS_INDICES[a]].sum(axis=1)
        comp[a] = {
            "density100": float(col.sum() / chars * 100) if chars else 0.0,
            "share": float((col > 0).mean()) if n else 0.0,
        }
    return {"n": n, "chars": chars, "categories": cat, "composites": comp}


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_structure(corr_cat: pd.DataFrame, corr_comp: pd.DataFrame,
                   profile: dict) -> None:
    """三联图：30×30 类别共现热力图 / 5×5 轴相关矩阵 / 跨子集画像。"""
    init_plot_style()
    fig = plt.figure(figsize=(18.5, 10.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.32, 1.0],
                          height_ratios=[1.0, 1.06], wspace=0.16, hspace=0.30)

    # ---- (a) 30×30 类别共现热力图 ----
    ax = fig.add_subplot(gs[:, 0])
    mat = corr_cat.loc[ALL_CATEGORIES, ALL_CATEGORIES].to_numpy()
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.grid(False)
    pos = 0
    for a in AXIS_ORDER:
        k = len(AXIS_INDICES[a])
        if pos:
            ax.axhline(pos - 0.5, color="black", lw=1.2)
            ax.axvline(pos - 0.5, color="black", lw=1.2)
        pos += k
    ax.set_xticks(range(N_CAT))
    ax.set_xticklabels([CATEGORY_LABEL[c] for c in ALL_CATEGORIES],
                       rotation=90, fontsize=8)
    ax.set_yticks(range(N_CAT))
    ax.set_yticklabels([CATEGORY_LABEL[c] for c in ALL_CATEGORIES], fontsize=8)
    for lab, c in zip(ax.get_xticklabels(), ALL_CATEGORIES):
        lab.set_color(AXIS_COLORS[CATEGORY_AXIS[c]])
    for lab, c in zip(ax.get_yticklabels(), ALL_CATEGORIES):
        lab.set_color(AXIS_COLORS[CATEGORY_AXIS[c]])
    ax.set_title("(a) 探索集类别密度的单元内 Spearman 共现（30 类别，按轴分块）")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("单元内 ρ（月份×领域 ≥100 条）", fontsize=9)

    # ---- (b) 5×5 轴复合相关矩阵 ----
    ax = fig.add_subplot(gs[0, 1])
    cm = corr_comp.loc[AXIS_ORDER, AXIS_ORDER].to_numpy()
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.grid(False)
    for i in range(len(AXIS_ORDER)):
        for j in range(len(AXIS_ORDER)):
            ax.text(j, i, f"{cm[i, j]:+.2f}", ha="center", va="center",
                    fontsize=11, color="white" if abs(cm[i, j]) > 0.30 else "black")
    ax.set_xticks(range(len(AXIS_ORDER)))
    ax.set_xticklabels([AXIS_NAME[a] for a in AXIS_ORDER], fontsize=9.5)
    ax.set_yticks(range(len(AXIS_ORDER)))
    ax.set_yticklabels([AXIS_NAME[a] for a in AXIS_ORDER], fontsize=9.5)
    ax.set_title("(b) 五个轴复合密度的共现相关")

    # ---- (c) 跨子集画像：AIGC−人类 / 探索集−人类 的 log2 使用率比 ----
    ax = fig.add_subplot(gs[1, 1])
    rows = ALL_CATEGORIES
    ypos = np.arange(len(rows))
    for a in AXIS_ORDER:
        idx = [i for i, c in enumerate(rows) if CATEGORY_AXIS[c] == a]
        if AXIS_ORDER.index(a) % 2 == 1:
            ax.axhspan(min(idx) - 0.5, max(idx) + 0.5, color="#EEEEEE", zorder=0)
    vals = [profile[c]["log2_aigc_human"] for c in rows]
    ax.barh(ypos, vals, height=0.66,
            color=[AXIS_COLORS[CATEGORY_AXIS[c]] for c in rows], alpha=0.92,
            zorder=2)
    ax.scatter([profile[c]["log2_exploring_human"] for c in rows], ypos,
               s=20, color="black", zorder=4, edgecolor="white", linewidth=0.4,
               label="探索集 vs 人类")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([CATEGORY_LABEL[c] for c in rows], fontsize=8)
    for i, c in enumerate(rows):
        v = profile[c]["log2_aigc_human"]
        ax.annotate(f"{v:+.1f}", xy=(v, ypos[i]),
                    xytext=(4 if v >= 0 else -4, -3),
                    textcoords="offset points", fontsize=6.6,
                    ha="left" if v >= 0 else "right")
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel("log2(子集密度 / 人类密度)（加性平滑 0.01）")
    ax.set_title("(c) 每个类别在 AIGC 与探索集的过量/欠量使用（黑点=探索集）")
    handles = [plt.Line2D([], [], color=AXIS_COLORS[a], lw=6, label=AXIS_NAME[a])
               for a in AXIS_ORDER]
    handles.append(plt.Line2D([], [], color="black", marker="o", ls="",
                              ms=4, label="探索集 vs 人类"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    save_fig(fig, INSIGHTS + "axis_synthesis_structure.png")


def plot_signal(p_ai_axis: dict, p_ai_full: dict, sep: dict, stack: dict,
                eng: dict) -> None:
    """四联图：P(AI) 联合偏效应 / 轴空间 AUC / 堆叠分层 / 联合互动森林。"""
    init_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12.0))

    # ---- (a) P(AI) 联合偏效应（轴复合 + 控制变量） ----
    ax = axes[0, 0]
    feats = p_ai_axis["features"]
    order = list(reversed(feats))
    ypos = np.arange(len(order))
    for i, name in enumerate(order):
        b = p_ai_axis["coef"][name]
        lo, hi = p_ai_axis["ci_low"][name], p_ai_axis["ci_high"][name]
        sig = lo > 0 or hi < 0
        col = AXIS_COLORS.get(name, "#4C72B0")
        ax.plot([lo, hi], [ypos[i], ypos[i]], color=col if sig else "#9E9E9E",
                lw=2.0, solid_capstyle="butt", zorder=2)
        ax.scatter([b], [ypos[i]], marker="o", s=42,
                   facecolor=col if sig else "white",
                   edgecolor=col if sig else "#9E9E9E", zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    labels = {a: AXIS_NAME[a] for a in AXIS_ORDER}
    labels.update({"content_len": "正文长度", "n_hashtags": "标签数"})
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels.get(n, n) for n in order])
    ax.set_xlabel("单元内 FE β（每 1 SD → P(AI)）")
    ax.set_title(f"(a) 五轴联合的 P(AI) 偏效应（n={p_ai_axis['n_obs']:,}，"
                 f"{p_ai_axis['n_cells']} 单元；组内 R²={p_ai_axis['r2']:.3f}，"
                 f"30 类别全模型 {p_ai_full['r2']:.3f}）")

    # ---- (b) 轴空间可分性 ----
    ax = axes[0, 1]
    items = AXIS_ORDER + ["all_axes"]
    vals = [sep["per_axis"][a]["auc"] for a in AXIS_ORDER] + \
           [sep["all_axes"]["auc"]]
    cols = [AXIS_COLORS[a] for a in AXIS_ORDER] + ["#555555"]
    names = [AXIS_NAME[a] for a in AXIS_ORDER] + \
            [f"五轴合计（{sep['all_axes']['n_features']} 特征）"]
    xpos = np.arange(len(items))
    ax.bar(xpos, vals, color=cols, alpha=0.9, width=0.62)
    ax.axhline(sep["all_axes"]["auc"], color="#555555", lw=1.0, ls="--")
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    for x, v in zip(xpos, vals):
        ax.annotate(f"{v:.3f}", xy=(x, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("5 折交叉验证 AUC（人类 vs AIGC）")
    ax.set_title(f"(b) 轴词表密度的判别力（人类 {sep['n_human']:,} vs "
                 f"AIGC {sep['n_aigc']:,}，逻辑回归）")

    # ---- (c) 轴堆叠分层：占比与平均 P(AI) ----
    ax = axes[1, 0]
    ks = stack["order"]
    shares = [stack["share_by_stack"].get(str(k), 0.0) for k in ks]
    xpos = np.arange(len(ks))
    ax.bar(xpos, [s * 100 for s in shares], color="#B8C4E0", width=0.62,
           label="帖子占比（%）")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{k} 轴" for k in ks])
    ax.set_xlabel("与 P(AI) 同向的轴数（有效轴数=5 的帖子）")
    ax.set_ylabel("帖子占比（%）")
    ax2 = ax.twinx()
    means = [stack["mean_p_ai_by_stack"].get(str(k), {}).get("mean", np.nan)
             for k in ks]
    lo = [stack["mean_p_ai_by_stack"].get(str(k), {}).get("ci_low", np.nan)
          for k in ks]
    hi = [stack["mean_p_ai_by_stack"].get(str(k), {}).get("ci_high", np.nan)
          for k in ks]
    ax2.plot(xpos, means, "o-", color="#C44E52", lw=2.0, ms=6,
             label="平均 P(AI)")
    ax2.fill_between(xpos, lo, hi, color="#C44E52", alpha=0.16)
    ax2.set_ylabel("平均 P(AI)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(0, 1.05)
    for x, (m, s) in enumerate(zip(means, shares)):
        if s > 0.005:
            ax2.annotate(f"{m:.3f}", xy=(x, m), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=8.5,
                         color="#C44E52")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    rho = stack["rho_p_ai"]["weighted_rho"]
    ax.set_title(f"(c) 轴堆叠分层：堆叠数 vs P(AI)"
                 f"（单元内 ρ={rho:+.3f}，n={stack['n']:,}）")

    # ---- (d) 联合互动森林图 ----
    ax = axes[1, 1]
    fe_l, fe_c = eng["joint_composites"]["likes"], eng["joint_composites"]["collections"]
    names = list(fe_l["coef"].keys())
    order = list(reversed(names))
    ypos = np.arange(len(order))
    for target, marker, offset, color in (
            ("likes", "o", -0.14, "#4C72B0"),
            ("collections", "^", 0.14, "#DD8452")):
        fe = fe_l if target == "likes" else fe_c
        for i, name in enumerate(order):
            b = fe["coef"][name]
            lo, hi = fe["ci_low"][name], fe["ci_high"][name]
            sig = lo > 0 or hi < 0
            col = color if sig else "#9E9E9E"
            ax.plot([lo, hi], [ypos[i] + offset, ypos[i] + offset], color=col,
                    lw=1.8, solid_capstyle="butt", zorder=2)
            ax.scatter([b], [ypos[i] + offset], marker=marker, s=34,
                       facecolor=col if sig else "white", edgecolor=col,
                       zorder=3,
                       label=("点赞" if target == "likes" else "收藏")
                       if i == 0 else None)
    labels = {a: AXIS_NAME[a] for a in AXIS_ORDER}
    labels.update({"p_ai": "P(AI)", "content_len": "正文长度",
                   "n_hashtags": "标签数"})
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels.get(n, n) for n in order])
    ax.set_xlabel("单元内 FE β（每 1 SD → log10(1+互动)）")
    st = eng["stack_model"]["likes"]
    st_c = eng["stack_model"]["collections"]
    ax.set_title(f"(d) 五轴联合的互动偏效应（n={fe_l['n_obs']:,}）；"
                 f"每多 1 个同向轴：点赞 β={st['coef']['stack']:+.4f}"
                 f"[{st['ci_low']['stack']:+.4f}, {st['ci_high']['stack']:+.4f}]，"
                 f"收藏 β={st_c['coef']['stack']:+.4f}"
                 f"[{st_c['ci_low']['stack']:+.4f}, {st_c['ci_high']['stack']:+.4f}]",
                 fontsize=9.2)
    ax.legend(loc="lower right")
    save_fig(fig, INSIGHTS + "axis_synthesis_signal.png")


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
    # scores 与 features 由同一份 JSONL 同序生成；先用两者的 content_len
    # 互校，再用原始未清洗字段（月份/领域/互动）校验 features 与本次扫描
    # 的行序。正文清洗版本差异只会改变 content_len，不是行序错位。
    scores = pd.read_csv(SCORES_PATH, usecols=["p_ai", "content_len"])
    n_score = len(scores)
    n_scanned = scans["exploring"]["n_total"]
    if n_score != n_scanned:
        raise SystemExit(f"exploring JSONL 行数 {n_scanned} 与 p_ai 行数 {n_score} 不一致")

    model = pd.read_csv(FEATURES_PATH,
                        usecols=["month", "domain", "content_len", "n_hashtags",
                                 "likes", "collections"])
    if len(model) != n_scanned:
        raise SystemExit("features 与 scores 行数不一致")
    mismatch = int((scores["content_len"].to_numpy()
                    != model["content_len"].to_numpy()).sum())
    if mismatch:
        raise SystemExit(f"scores 与 features 行序未对齐：{mismatch}")
    require_row_alignment(scans["exploring"], model, "features")
    likes_mismatch = 0
    text_len_drift = int((model["content_len"].to_numpy()
                          != scans["exploring"]["body_len"]).sum())

    # ---- 去重 + 子集使用率 ----
    keeps = {name: dedup_keep(sc) for name, sc in scans.items()}
    subset_rates = {name: rates_block(scans[name], keeps[name])
                    for name in SUBSET_ORDER}
    for name in SUBSET_ORDER:
        sc = scans[name]
        short = int((sc["body_len"][keeps[name]] < MIN_BODY_LEN).sum())
        print(f"  {name}: 去重后 {len(keeps[name]):,} 条，正文 <{MIN_BODY_LEN} 字 "
              f"排除 {short:,} 条")

    profile = {}
    for c in ALL_CATEGORIES:
        h = subset_rates["human"]["categories"][c]["density100"]
        a = subset_rates["aigc"]["categories"][c]["density100"]
        e = subset_rates["exploring"]["categories"][c]["density100"]
        profile[c] = {"human": round(h, 4), "aigc": round(a, 4),
                      "exploring": round(e, 4),
                      "log2_aigc_human": round(log2_ratio(a, h), 4),
                      "log2_exploring_human": round(log2_ratio(e, h), 4)}

    # ---- 探索集单元内共现结构 ----
    expl = scans["exploring"]
    keep = keeps["exploring"]
    sel = keep[expl["body_len"][keep] >= MIN_BODY_LEN]
    body = expl["body_len"][sel].astype(float)
    months = np.asarray([expl["month"][i] or "未知" for i in sel])
    domains = np.asarray([expl["domain"][i] for i in sel])
    cells = pd.factorize(pd.Series([f"{m}|{d}" for m, d in zip(months, domains)]),
                         sort=True)[0]
    groups = cell_groups(cells, MIN_CELL_N)
    p_ai = scores["p_ai"].to_numpy(dtype=float)[sel]
    print(f"共现样本：{len(sel):,} 条（去重 + 正文 ≥{MIN_BODY_LEN} 字），"
          f"单元 {len(np.unique(cells))} 个（≥{MIN_CELL_N} 条的 {len(groups)} 个）")

    body_all = expl["body_len"].astype(float)
    dens_all = expl["counts"].astype(float) / np.maximum(body_all, 1)[:, None] * 100
    dens = dens_all[sel]
    z = within_cell_rank_z_matrix(dens, groups)
    z_df = pd.DataFrame(z, columns=ALL_CATEGORIES)
    corr_cat = z_df.corr()
    comp_df = composite_z(z)
    corr_comp = comp_df.corr()
    print("轴复合相关矩阵：")
    for a in AXIS_ORDER:
        row = "  ".join(f"{corr_comp.loc[a, b]:+.2f}" for b in AXIS_ORDER)
        print(f"  {AXIS_NAME[a]:<6} {row}")

    # ---- 人类 vs AIGC 的轴空间可分性 ----
    sep_parts = []
    for name, label in (("human", "human"), ("aigc", "aigc")):
        sc = scans[name]
        k = keeps[name]
        s = k[sc["body_len"][k] >= MIN_BODY_LEN]
        d = sc["counts"][s].astype(float) / sc["body_len"][s][:, None] * 100
        part = pd.DataFrame(d, columns=ALL_CATEGORIES)
        part["dataset"] = label
        sep_parts.append(part)
    sep_df = pd.concat(sep_parts, ignore_index=True)
    sep_all = cv_auc(sep_df, ALL_CATEGORIES)
    sep_axis = {}
    for a in AXIS_ORDER:
        cats = [c for c in ALL_CATEGORIES if CATEGORY_AXIS[c] == a]
        sep_axis[a] = cv_auc(sep_df, cats)
    print(f"轴空间 AUC：五轴合计 {sep_all['auc']:.4f}；" + "；".join(
        f"{AXIS_NAME[a]} {sep_axis[a]['auc']:.4f}" for a in AXIS_ORDER))

    # ---- P(AI) 联合 FE 模型（轴复合 + 30 类别全模型） ----
    n_hashtags = model["n_hashtags"].to_numpy(dtype=float)
    comp_dens = composite_density(dens)
    ctrl = np.column_stack([body, n_hashtags[sel]])

    def fit_with_ci(X, y, cells_like, names):
        prep = prepare_fe(X, cells_like, MIN_CELL_N)
        fit = fe_ols(X, y, cells_like, prepared=prep)
        boot = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                        fit["cell_codes"], n_boot=N_BOOT,
                                        seed=BOOT_SEED)
        ci = np.percentile(boot, [2.5, 97.5], axis=0)
        return {
            "features": names,
            "coef": {n: round(float(v), 4) for n, v in zip(names, fit["coef"])},
            "ci_low": {n: round(float(v), 4) for n, v in zip(names, ci[0])},
            "ci_high": {n: round(float(v), 4) for n, v in zip(names, ci[1])},
            "n_obs": fit["n_obs"], "n_cells": fit["n_cells"],
            "r2": round(fe_r2(fit), 4),
        }

    axis_names = AXIS_ORDER + ["content_len", "n_hashtags"]
    X_axis = np.column_stack([comp_dens, ctrl])
    p_ai_axis = fit_with_ci(X_axis, p_ai, cells, axis_names)
    print("P(AI) 联合偏效应（轴复合）：" + "；".join(
        f"{AXIS_NAME[a]} {p_ai_axis['coef'][a]:+.4f}"
        f"[{p_ai_axis['ci_low'][a]:+.4f},{p_ai_axis['ci_high'][a]:+.4f}]"
        for a in AXIS_ORDER) + f"；组内 R²={p_ai_axis['r2']:.3f}")

    cat_names = ALL_CATEGORIES + ["content_len", "n_hashtags"]
    X_full = np.column_stack([dens, ctrl])
    p_ai_full = fit_with_ci(X_full, p_ai, cells, cat_names)

    # ---- 轴堆叠 ----
    signs = [np.sign(p_ai_axis["coef"][a]) for a in AXIS_ORDER]
    hits, n_valid = stack_hits(comp_df.to_numpy(), signs)
    ok_stack = n_valid == len(AXIS_ORDER)
    stack_arr = np.where(ok_stack, hits, -1)
    stack_all = np.full(len(dens_all), np.nan)
    stack_all[sel] = np.where(ok_stack, hits.astype(float), np.nan)
    order_ks = sorted(int(k) for k in np.unique(stack_arr[stack_arr >= 0]))
    share_by_stack = stack_shares(stack_arr, ok_stack)
    mean_p_ai = {str(k): mean_ci(p_ai[stack_arr == k], seed=BOOT_SEED)
                 for k in order_ks}
    stack_rho = within_cell_spearman(
        stack_arr.astype(float), p_ai, cells, min_cell_n=MIN_CELL_N,
        groups=groups, y_ranks=cell_rank_cache(p_ai, groups))
    print("堆叠分布：" + "；".join(
        f"{k} 轴 {share_by_stack[str(k)]:.1%}（P(AI)={mean_p_ai[str(k)]['mean']:.3f}）"
        for k in order_ks))
    print(f"堆叠数 vs P(AI) 单元内 ρ={stack_rho['weighted_rho']:+.4f}")

    # ---- 联合互动模型（五轴复合 + 堆叠指数） ----
    target_likes = np.log10(1 + np.clip(model["likes"].to_numpy(dtype=float), 0, None))
    y_coll = np.log10(1 + np.clip(model["collections"].to_numpy(dtype=float), 0, None))
    month_all = model["month"].astype(str).to_numpy()
    dom_all = model["domain"].astype(str).to_numpy()
    eng_cells_raw = np.asarray([f"{m}|{d}" for m, d in zip(month_all, dom_all)])
    p_ai_all = scores["p_ai"].to_numpy(dtype=float)
    keep_eng = (month_all >= MONTH_FROM) & (body_all >= MIN_BODY_LEN) & \
        np.isfinite(stack_all)
    X_joint = np.column_stack([
        composite_density(dens_all), p_ai_all, body_all, n_hashtags])
    joint_names = AXIS_ORDER + ["p_ai", "content_len", "n_hashtags"]
    X_stack = np.column_stack([stack_all, p_ai_all, body_all, n_hashtags])
    stack_names = ["stack", "p_ai", "content_len", "n_hashtags"]
    eng = {"joint_composites": {}, "stack_model": {}, "univariate": {}}
    for target, y in (("likes", target_likes), ("collections", y_coll)):
        eng["joint_composites"][target] = fit_with_ci(
            X_joint[keep_eng], y[keep_eng], eng_cells_raw[keep_eng], joint_names)
        eng["stack_model"][target] = fit_with_ci(
            X_stack[keep_eng], y[keep_eng], eng_cells_raw[keep_eng], stack_names)
        print(f"  互动 FE（{target}）：stack β="
              f"{eng['stack_model'][target]['coef']['stack']:+.4f}"
              f"[{eng['stack_model'][target]['ci_low']['stack']:+.4f},"
              f"{eng['stack_model'][target]['ci_high']['stack']:+.4f}]"
              f"（n={eng['stack_model'][target]['n_obs']:,}）")

    groups_eng = cell_groups(eng_cells_raw[keep_eng], MIN_CELL_N)
    y_ranks = cell_rank_cache(target_likes[keep_eng], groups_eng)
    yc_ranks = cell_rank_cache(y_coll[keep_eng], groups_eng)
    for i, a in enumerate(AXIS_ORDER):
        eng["univariate"][a] = {
            "likes": within_cell_spearman(
                X_joint[keep_eng, i], target_likes[keep_eng],
                eng_cells_raw[keep_eng], min_cell_n=MIN_CELL_N,
                groups=groups_eng, y_ranks=y_ranks),
            "collections": within_cell_spearman(
                X_joint[keep_eng, i], y_coll[keep_eng],
                eng_cells_raw[keep_eng], min_cell_n=MIN_CELL_N,
                groups=groups_eng, y_ranks=yc_ranks),
        }

    # ---- 结果 ----
    out = {
        "meta": {
            "min_body_len": MIN_BODY_LEN, "min_cell_n": MIN_CELL_N,
            "month_from": MONTH_FROM, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "stack_boot": STACK_BOOT,
            "exploring_content_len_mismatch": mismatch,
            "exploring_text_len_drift": text_len_drift,
            "exploring_likes_mismatch": likes_mismatch,
            "note": "人类/探索集按 note_id 去重（保留元数据最完整一条）；"
                    "AIGC 无 note_id 逐记录；p_ai 按行序对齐 "
                    "paper_exploring_scores.csv.gz；类别计数直接调用五条轴"
                    "脚本的纯计数函数；轴复合密度=该轴各类别密度之和",
        },
        "axes": {a: {"name": AXIS_NAME[a], "categories": cats,
                     "labels": [labs[c] for c in cats]}
                 for a, _, cats, labs in AXES},
        "subset_rates": {
            name: {
                "n": subset_rates[name]["n"],
                "chars": subset_rates[name]["chars"],
                "categories": subset_rates[name]["categories"],
                "composites": subset_rates[name]["composites"],
            } for name in SUBSET_ORDER},
        "profile": profile,
        "correlation": {
            "n": int(np.isfinite(z).any(axis=1).sum()),
            "n_cells": int(len(groups)),
            "categories": corr_cat.round(4).to_dict(),
            "composites": corr_comp.round(4).to_dict(),
        },
        "separability": {
            "n_human": int((sep_df["dataset"] == "human").sum()),
            "n_aigc": int((sep_df["dataset"] == "aigc").sum()),
            "all_axes": sep_all,
            "per_axis": sep_axis,
        },
        "p_ai_model": {
            "composite": p_ai_axis,
            "full": p_ai_full,
        },
        "stack": {
            "direction": {a: int(s) for a, s in zip(AXIS_ORDER, signs)},
            "order": order_ks,
            "n": int(ok_stack.sum()),
            "share_by_stack": share_by_stack,
            "mean_p_ai_by_stack": mean_p_ai,
            "rho_p_ai": {
                "weighted_rho": round(stack_rho["weighted_rho"], 4),
                "neg_cell_share": round(stack_rho["neg_cell_share"], 4),
                "n_cells": stack_rho["n_cells"],
            },
        },
        "engagement": {
            "month_from": MONTH_FROM, "min_body_len": MIN_BODY_LEN,
            "joint_composites": eng["joint_composites"],
            "stack_model": eng["stack_model"],
            "univariate": {
                a: {t: {"weighted_rho": round(v[t]["weighted_rho"], 4),
                        "neg_cell_share": round(v[t]["neg_cell_share"], 4),
                        "n_cells": v[t]["n_cells"]}
                    for t in ("likes", "collections")}
                for a, v in eng["univariate"].items()},
        },
    }
    save_json(out, BASE + "axis_synthesis_results.json")
    print("已写出 axis_synthesis_results.json")
    write_markdown(out)
    plot_structure(corr_cat, corr_comp, profile)
    plot_signal(p_ai_axis, p_ai_full, out["separability"], out["stack"], eng)
    print("已写出 axis_synthesis_results.md 与 insights/axis_synthesis_*.png")


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def write_markdown(out: dict) -> None:
    meta = out["meta"]
    rates = out["subset_rates"]
    corr = out["correlation"]["composites"]
    sep = out["separability"]
    pm = out["p_ai_model"]
    st = out["stack"]
    eng = out["engagement"]
    labels = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
              "exploring": "探索集(后LLM)"}
    lines = [
        "# 跨轴综合分析结果（Axis Synthesis）",
        "",
        f"> 由 `axis_synthesis_analysis.py` 全量运行生成；类别计数直接调用五条轴"
        f"脚本的纯计数函数；人类/探索集按 note_id 去重；正文 ≥"
        f"{meta['min_body_len']} 字；探索集 p_ai 按行序对齐 "
        f"`paper_exploring_scores.csv.gz`（scores/features content_len "
        f"mismatch={meta['exploring_content_len_mismatch']}，原始文本长度"
        f"因清洗版本差异 drift={meta['exploring_text_len_drift']}，"
        f"月份/领域/互动字段逐条一致）。",
        "",
        "## 1 五轴复合密度（各类别密度之和，次/百字）",
        "",
        "| 子集 | 有效帖数 | " + " | ".join(AXIS_NAME[a] for a in AXIS_ORDER)
        + " |",
        "|---|---:|" + "---:|" * len(AXIS_ORDER),
    ]
    for name in SUBSET_ORDER:
        comp = rates[name]["composites"]
        lines.append(
            f"| {labels[name]} | {rates[name]['n']:,} | "
            + " | ".join(f"{comp[a]['density100']:.2f}" for a in AXIS_ORDER)
            + " |")
    lines += [
        "",
        "> 单类别级别的 AIGC/探索集 vs 人类 log2 使用率比见结果 JSON 的 "
        "`profile` 与图 `axis_synthesis_structure.png` (c)。",
        "",
        "## 2 轴内共现（探索集单元内 Spearman）",
        "",
        "| 轴 | " + " | ".join(AXIS_NAME[b] for b in AXIS_ORDER) + " |",
        "|---|" + "---:|" * len(AXIS_ORDER),
    ]
    for a in AXIS_ORDER:
        lines.append(
            f"| {AXIS_NAME[a]} | "
            + " | ".join(f"{corr[a][b]:+.3f}" for b in AXIS_ORDER) + " |")
    lines += [
        "",
        f"（n={out['correlation']['n']:,}，{out['correlation']['n_cells']} 个"
        f"样本量 ≥{meta['min_cell_n']} 的「月份×领域」单元；30 类别全矩阵见 "
        f"JSON `correlation.categories`。）",
        "",
        "## 3 五轴联合的 P(AI) 独立信号（单元内 FE + 整群 bootstrap）",
        "",
        "| 特征 | β [95% CI] |",
        "|---|---:|",
    ]
    for name in pm["composite"]["features"]:
        if name in AXIS_NAME:
            name_show = AXIS_NAME[name]
        else:
            name_show = {"content_len": "正文长度", "n_hashtags": "标签数"}[name]
        lines.append(
            f"| {name_show} | {pm['composite']['coef'][name]:+.4f} "
            f"[{pm['composite']['ci_low'][name]:+.4f}, "
            f"{pm['composite']['ci_high'][name]:+.4f}] |")
    lines += [
        "",
        f"（n={pm['composite']['n_obs']:,}，{pm['composite']['n_cells']} 个单元，"
        f"组内 R²={pm['composite']['r2']:.3f}；30 类别全模型 "
        f"R²={pm['full']['r2']:.3f}，系数见 JSON `p_ai_model.full`。）",
        "",
        "## 4 人类 vs AIGC 的轴空间可分性（5 折 CV AUC）",
        "",
        "| 特征集 | 特征数 | AUC | F1 | Acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for a in AXIS_ORDER + ["all_axes"]:
        rec = sep["all_axes"] if a == "all_axes" else sep["per_axis"][a]
        name = "五轴合计" if a == "all_axes" else AXIS_NAME[a]
        lines.append(f"| {name} | {rec['n_features']} | {rec['auc']:.4f} | "
                     f"{rec['f1']:.4f} | {rec['acc']:.4f} |")
    lines += [
        "",
        f"（人类 {sep['n_human']:,} vs AIGC {sep['n_aigc']:,}，逻辑回归，"
        f"与 `paper_detector.py` 同折同协议；仅供轴词表判别力对照，"
        f"不是检测器消融。）",
        "",
        "## 5 轴堆叠：与 P(AI) 同向的轴数",
        "",
        f"方向定义（五轴联合 P(AI) 模型的 β 符号）："
        + "；".join(f"{AXIS_NAME[a]} {'+' if st['direction'][a] > 0 else '−'}"
                    for a in AXIS_ORDER) + "。",
        "",
        "| 同向轴数 | 帖子占比 | 平均 P(AI) [95% CI] |",
        "|---:|---:|---:|",
    ]
    for k in st["order"]:
        rec = st["mean_p_ai_by_stack"][str(k)]
        lines.append(
            f"| {k} | {st['share_by_stack'][str(k)]:.1%} | {rec['mean']:.4f} "
            f"[{rec['ci_low']:.4f}, {rec['ci_high']:.4f}] |")
    lines += [
        "",
        f"（有效轴数=5 的帖子 n={st['n']:,}；堆叠数与 P(AI) 的单元内 "
        f"ρ={st['rho_p_ai']['weighted_rho']:+.4f}，"
        f"{st['rho_p_ai']['n_cells']} 个单元；CI 为对帖子重抽的描述性区间。）",
        "",
        "## 6 五轴联合的互动偏效应（月份×领域 FE + 整群 bootstrap）",
        "",
        "| 特征 | 点赞 β [95% CI] | 收藏 β [95% CI] |",
        "|---|---:|---:|",
    ]
    jl, jc = eng["joint_composites"]["likes"], eng["joint_composites"]["collections"]
    for name in jl["features"]:
        name_show = AXIS_NAME.get(name) or \
            {"p_ai": "P(AI)", "content_len": "正文长度",
             "n_hashtags": "标签数"}[name]
        lines.append(
            f"| {name_show} | {jl['coef'][name]:+.4f} "
            f"[{jl['ci_low'][name]:+.4f}, {jl['ci_high'][name]:+.4f}] | "
            f"{jc['coef'][name]:+.4f} "
            f"[{jc['ci_low'][name]:+.4f}, {jc['ci_high'][name]:+.4f}] |")
    sl = eng["stack_model"]["likes"]
    sc_ = eng["stack_model"]["collections"]
    lines += [
        "",
        f"堆叠指数（每多 1 个同向轴，控制 P(AI)/长度/标签数）：点赞 β="
        f"{sl['coef']['stack']:+.4f} [{sl['ci_low']['stack']:+.4f}, "
        f"{sl['ci_high']['stack']:+.4f}]；收藏 β={sc_['coef']['stack']:+.4f} "
        f"[{sc_['ci_low']['stack']:+.4f}, {sc_['ci_high']['stack']:+.4f}]"
        f"（n={sl['n_obs']:,}，{sl['n_cells']} 个单元）。",
        "",
        "图表见 `insights/axis_synthesis_structure.png`、"
        "`insights/axis_synthesis_signal.png`。",
    ]
    with open(BASE + "axis_synthesis_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
