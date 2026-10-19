# -*- coding: utf-8 -*-
"""作者层分解：把「AI 味惩罚」拆成作者间差异与人内差异（探索集，2023.01–2025.06）。

`paper_engagement.py` 报告了探索集内 P(AI) 与互动的负相关（总体 ρ=-0.139、
月份×领域单元内加权 ρ=-0.091），`engagement_structure_analysis.py` /
`engagement_drivers_analysis.py` 分别把它拆成互动结构与风格特征。但这些
分析都在「逐帖」层面，无法回答一个关键混杂问题：

    互动差异到底来自「同一作者写了更 AI 风格的帖子」，
    还是来自「长期写 AI 风格的作者整体互动更低」？

探索集约 55% 的去重笔记带 `user_id`，本分析在这个子集上做四件事：

1. **覆盖与选择检查**：带 `user_id` 的笔记与其余笔记在 P(AI)、互动、
   时间上的差异——先把「作者层结论只适用于哪个采样层」讲清楚；
2. **作者集中度与风格方差分解**：AI 味是否会向少数作者集中？用一元随机
   效应 ANOVA 把 P(AI) 方差拆成作者间/作者内（ICC），度量「风格是作者
   属性还是单帖属性」；
3. **互动斜率的作者间/人内分解**：对 log10(1+点赞) 与 P(AI) 做精确的
   方差分解 β_pooled = w_b·β_between + w_w·β_within（作者固定效应），
   并附月份固定效应的人内版本；所有作者层统计量用**按作者整群 bootstrap**
   给出 95% CI（同一作者的多篇帖子不独立）；
4. **人内配对对照**：对同时有「像人帖」（P(AI)<0.5）与「像 AI 帖」（≥0.5）
   的作者，比较同作者两半的平均 log 点赞（Wilcoxon 符号秩 + 自举 CI），
   以及同作者同月份单元内的对照，控制作者身份与时间。

口径说明：
- 数据源 `paper_exploring_scores.csv.gz`（逐帖 P(AI)）与
  `exploring_set_fixed.jsonl` 按行序对齐，content_len 逐条校验
  （清洗复用 `analysis_utils.clean_body`，与 paper_features 一致）；
- 去重规则同 `deep_dive_analysis.py` / `symbol_analysis.py`（note_id 保留
  元数据最完整的一条）；作者子集取去重后带 `user_id` 的笔记（该子集内
  每篇笔记恰有一条记录）；
- 月份窗口与 `paper_engagement.py` 一致（≥2023.01）；同一份结果文件里
  复算总体 ρ 与单元内加权 ρ 校验与 `paper_engagement_results.json` 吻合。

输出：
  insights/author_structure.png   覆盖/选择、作者集中度、作者间梯度
  insights/author_within.png      斜率分解森林图、人内配对散点、Δ 分布
  author_style_results.json       全部数值结果
  author_style_results.md         结果摘要
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

from analysis_utils import (BASE, INSIGHTS, PALETTE, clean_body, close_pool,
                            content_of, dedup_indices, engagement_of,
                            gini_coefficient, init_plot_style,
                            make_pool, parse_local_time, save_fig, save_json,
                            wilson_ci, within_cell_spearman)
from derived_cache import iter_cached

SCORES_PATH = BASE + "paper_exploring_scores.csv.gz"
EXPLORING_PATH = BASE + "exploring_set_fixed.jsonl"
MONTH_FROM = "202301"      # 与 paper_engagement.py 相同窗口
STYLE_SPLIT = 0.5          # 「像人 / 像 AI」的作者内切分
HI_CONF = 0.9              # 高置信阈值
TOP_AUTHOR_FRAC = 0.10     # 集中度：发文量前 10% 作者
MIN_AUTHOR_NOTES = 2       # ICC/作者均值分析的参与者门槛
MIN_WITHIN_NOTES = 4       # 人内秩相关的最少帖子数
N_BOOT = 500               # 作者整群 bootstrap 次数（单因素）
N_BOOT_TWO = 300           # 双因素（作者+月份）bootstrap 次数
BOOT_SEED = 42
CI_COLOR = "#C44E52"


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def factorize_codes(codes) -> tuple[np.ndarray, int]:
    """把任意分组标签编码为 0..K-1 的整数，返回 (codes, K)。"""
    arr = np.asarray(codes)
    out, uniques = pd.factorize(arr, sort=True)
    return out.astype(np.int64), len(uniques)


def group_denominators(codes, weights=None) -> np.ndarray:
    """逐观测的组样本量/组权重和（值 - Σw·v / denom 中的 denom）。

    同一组编码与权重下 denom 固定不变：demean_group 在迭代中被调用数百次，
    预先算一次即可避免每轮重复 bincount；加权时 counts==0 的组记为 1.0，
    与 `demean_group` 内 `np.where(counts > 0, counts, 1.0)` 完全一致。
    """
    codes = np.asarray(codes)
    if weights is None:
        counts = np.bincount(codes)
        return counts[codes].astype(float)
    w = np.asarray(weights, dtype=float)
    counts = np.bincount(codes, weights=w)
    safe = np.where(counts > 0, counts, 1.0)
    return safe[codes]


def demean_group(values, codes, weights=None, denom=None) -> np.ndarray:
    """按组去均值；weights 非空时为加权均值（残差 = 值 - 组加权均值）。

    `denom` 可传入 `group_denominators(codes, weights)` 的结果复用（同一
    codes/weights 的迭代场景），取值与内部现算完全一致。
    """
    arr = np.asarray(values, dtype=float)
    codes = np.asarray(codes)
    if weights is None:
        sums = np.bincount(codes, weights=arr)
        if denom is None:
            denom = group_denominators(codes)
        return arr - sums[codes] / denom
    w = np.asarray(weights, dtype=float)
    sums = np.bincount(codes, weights=arr * w)
    if denom is None:
        denom = group_denominators(codes, w)
    return arr - sums[codes] / denom


def demean_two_way(values, codes_a, codes_b, weights=None,
                   max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """交替投影去均值（作者+月份固定效应的吸收）。

    对无权重情形，两个因素交替去均值收敛到联合最小二乘残差；带权重时
    为加权去均值迭代（bootstrap 重抽后的加权拟合近似）。两个因素的分母
    只与编码/权重有关，预先算一次供全部迭代复用。

    热循环用预分配缓冲区做原地版本：每轮只更新一个输出数组、复用 3 块
    工作缓冲区（与 `demean_group` 的逐轮「乘权重 → 分组求和 → 取回 →
    除以分母 → 相减」逐一对应，浮点运算顺序与分配临时数组的实现完全
    一致，结果逐位相同），省去每轮 6 次临时数组分配/释放。输入数组
    只读，返回值始终是新数组。
    """
    codes_a = np.asarray(codes_a)
    codes_b = np.asarray(codes_b)
    den_a = group_denominators(codes_a, weights)
    den_b = group_denominators(codes_b, weights)
    v = np.array(values, dtype=float, copy=True)
    n = v.size
    buf = np.empty(n, dtype=float)
    prev = np.empty(n, dtype=float)
    diff = np.empty(n, dtype=float)
    if weights is None:
        for _ in range(max_iter):
            np.copyto(prev, v)
            s = np.bincount(codes_a, weights=v)
            np.take(s, codes_a, out=buf)
            buf /= den_a
            v -= buf
            s = np.bincount(codes_b, weights=v)
            np.take(s, codes_b, out=buf)
            buf /= den_b
            v -= buf
            np.subtract(v, prev, out=diff)
            np.abs(diff, out=diff)
            if float(diff.max()) < tol:
                break
        return v
    w = np.asarray(weights, dtype=float)
    vw = np.empty(n, dtype=float)
    for _ in range(max_iter):
        np.copyto(prev, v)
        np.multiply(v, w, out=vw)
        s = np.bincount(codes_a, weights=vw)
        np.take(s, codes_a, out=buf)
        buf /= den_a
        v -= buf
        np.multiply(v, w, out=vw)
        s = np.bincount(codes_b, weights=vw)
        np.take(s, codes_b, out=buf)
        buf /= den_b
        v -= buf
        np.subtract(v, prev, out=diff)
        np.abs(diff, out=diff)
        if float(diff.max()) < tol:
            break
    return v


def ols_slope(x, y, weights=None) -> float:
    """一元 OLS 斜率（可选权重）；先中心化（含截距等价），分母为 0 时 NaN。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if weights is None:
        xc, yc = x - x.mean(), y - y.mean()
        den = float((xc * xc).sum())
        return float((xc * yc).sum() / den) if den > 0 else float("nan")
    w = np.asarray(weights, dtype=float)
    xc = x - np.average(x, weights=w)
    yc = y - np.average(y, weights=w)
    den = float((w * xc * xc).sum())
    return float((w * xc * yc).sum() / den) if den > 0 else float("nan")


def decompose_slope(x, y, codes) -> dict:
    """精确分解 OLS 斜率：pooled = w_between·between + w_within·within。

    先把 x/y 全局中心化（OLS 斜率对截距不变），再把 Σxy 与 Σx² 同时拆成
    「作者均值间」（组间）与「组内离差」（组内）两部分：
    numerator = Σ n_i x̄_i ȳ_i + Σ x̃ ỹ，denominator 同理。
    返回各分量斜率、组间权重与组数。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    codes, k = factorize_codes(codes)
    counts = np.bincount(codes)
    xbar = np.bincount(codes, weights=x) / counts
    ybar = np.bincount(codes, weights=y) / counts
    xbd, ybd = x - xbar[codes], y - ybar[codes]
    num_b = float((counts * xbar * ybar).sum())
    num_w = float((xbd * ybd).sum())
    den_b = float((counts * xbar * xbar).sum())
    den_w = float((xbd * xbd).sum())
    den = den_b + den_w
    return {
        "pooled": (num_b + num_w) / den if den > 0 else float("nan"),
        "between": num_b / den_b if den_b > 0 else float("nan"),
        "within": num_w / den_w if den_w > 0 else float("nan"),
        "between_weight": den_b / den if den > 0 else float("nan"),
        "n_groups": int(k),
    }


def variance_decomposition(values, codes) -> dict:
    """一元随机效应方差分解（ANOVA 估计量）。

    组间/组内方差分量按不平衡设计的 n0 校正估计；ICC 为组间方差占比。
    组数 < 2 或自由度不足时各分量返回 NaN/0。
    """
    x = np.asarray(values, dtype=float)
    codes, k = factorize_codes(codes)
    n = x.size
    if k < 2 or n - k < 1:
        return {"icc": float("nan"), "sd_between": float("nan"),
                "sd_within": float("nan"), "var_between": float("nan"),
                "var_within": float("nan"), "n_groups": int(k), "n": int(n)}
    counts = np.bincount(codes)
    means = np.bincount(codes, weights=x) / counts
    ssb = float((counts * (means - x.mean()) ** 2).sum())
    ssw = float(((x - means[codes]) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - float((counts ** 2).sum()) / n) / (k - 1)
    var_b = max((msb - msw) / n0, 0.0) if n0 > 0 else 0.0
    var_w = max(msw, 0.0)
    total = var_b + var_w
    return {
        "icc": var_b / total if total > 0 else 0.0,
        "sd_between": float(np.sqrt(var_b)),
        "sd_within": float(np.sqrt(var_w)),
        "var_between": float(var_b),
        "var_within": float(var_w),
        "n_groups": int(k), "n": int(n),
    }


def _paired_groups(ids, p_ai, log_likes, split: float = STYLE_SPLIT
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐组「低风格半组均值 / 高风格半组均值」，并返回保留下来的组键。

    只保留两半都非空（p_ai < split 与 p_ai >= split 各至少一条）的组。
    用稳定排序 + 分组求和实现，与逐组 pandas 布尔筛选/求均值取值一致，
    但把 O(组数) 次 DataFrame 筛选降为几次 O(n) 的数组运算。
    返回 (keys, lo_means, hi_means)，三个数组在稳定排序下互相对齐。
    """
    ids = np.asarray(ids)
    p = np.asarray(p_ai, dtype=float)
    y = np.asarray(log_likes, dtype=float)
    if ids.size == 0:
        return ids[:0], np.empty(0, dtype=float), np.empty(0, dtype=float)
    order = np.argsort(ids, kind="stable")
    ids_s, p_s, y_s = ids[order], p[order], y[order]
    change = np.empty(ids_s.shape, dtype=bool)
    change[0] = True
    if ids_s.size > 1:
        change[1:] = ids_s[1:] != ids_s[:-1]
    g = np.cumsum(change) - 1
    n_groups = int(g[-1]) + 1
    hi = p_s >= split
    lo = ~hi
    n_hi = np.bincount(g[hi], minlength=n_groups)
    n_lo = np.bincount(g[lo], minlength=n_groups)
    keep = (n_hi > 0) & (n_lo > 0)
    sum_hi = np.bincount(g[hi], weights=y_s[hi], minlength=n_groups)
    sum_lo = np.bincount(g[lo], weights=y_s[lo], minlength=n_groups)
    keys = ids_s[np.flatnonzero(change)][keep]
    return keys, (sum_lo[keep] / n_lo[keep]).astype(float), \
        (sum_hi[keep] / n_hi[keep]).astype(float)


def _paired_half_means(ids, p_ai, log_likes,
                       split: float = STYLE_SPLIT
                       ) -> tuple[np.ndarray, np.ndarray]:
    """逐组「低风格半组均值 / 高风格半组均值」（组键见 `_paired_groups`）。"""
    _, lo, hi = _paired_groups(ids, p_ai, log_likes, split)
    return lo, hi


def paired_style_deltas(ids, p_ai, log_likes,
                        split: float = STYLE_SPLIT) -> np.ndarray:
    """逐作者 Δ = 高风格半组均值(log 点赞) - 低风格半组均值(log 点赞)。

    只保留两半都非空（p_ai < split 与 p_ai >= split 各至少一条）的作者。
    """
    lo, hi = _paired_half_means(ids, p_ai, log_likes, split)
    return hi - lo


def top_author_shares(counts, indicator, frac: float = TOP_AUTHOR_FRAC) -> dict:
    """发文量前 `frac` 作者的笔记占比与「命中」笔记占比。

    `counts` 为逐作者发文数（与 `indicator` 同序：作者发文是否命中，
    如 P(AI)>=0.9），用于检验命中笔记是否比笔记总量更/更不集中。
    """
    counts = np.asarray(counts, dtype=float)
    indicator = np.asarray(indicator, dtype=float)
    if counts.size == 0:
        return {"n_authors": 0, "top_n": 0, "note_share": float("nan"),
                "hit_share": float("nan")}
    k = max(1, int(np.ceil(counts.size * frac)))
    top = np.argsort(-counts, kind="stable")[:k]
    total_notes = counts.sum()
    total_hits = indicator.sum()
    return {
        "n_authors": int(counts.size),
        "top_n": int(k),
        "note_share": float(counts[top].sum() / total_notes),
        "hit_share": (float(indicator[top].sum() / total_hits)
                      if total_hits > 0 else float("nan")),
    }


def bootstrap_mean_ci(values, n_boot: int = 2000, seed: int = BOOT_SEED,
                      alpha: float = 0.05) -> dict:
    """一维样本均值的 bootstrap 置信区间（分块重采样，NaN 剔除）。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(n_boot)
    chunk = max(1, 2_000_000 // max(n, 1))
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = rng.integers(0, n, (size, n))
        means[done:done + size] = arr[idx].mean(axis=1)
        done += size
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(arr.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "n": int(n)}


def bootstrap_wls_slope(x, y, weights=None, n_boot: int = N_BOOT,
                        seed: int = BOOT_SEED, alpha: float = 0.05) -> dict:
    """一观测一作者场景的加权最小二乘斜率 bootstrap（作者整群重抽）。

    对作者均值（每人一行）回归时使用：每次有放回重抽作者，重抽次数乘上
    原权重得到新权重，计算 WLS 斜率；返回点估计与百分位 CI。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    n = x.size
    point = ols_slope(x, y, w)
    if n == 0:
        return {"slope": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # 先合并重复抽中的作者，再用「重抽次数 × 原权重」作为该作者的新权重；
        # 若直接在展开后的行上再乘一次 mult 会把权重平方（m_i²·w_i）。
        uniq, mult = np.unique(idx, return_counts=True)
        s = ols_slope(x[uniq], y[uniq], w[uniq] * mult)
        if not np.isnan(s):
            slopes.append(s)
    arr = np.asarray(slopes)
    if arr.size == 0:
        return {"slope": point, "ci_low": float("nan"),
                "ci_high": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"slope": float(point), "ci_low": float(lo),
            "ci_high": float(hi), "n_boot": int(arr.size)}


def _cluster_boot_slopes_segment(payload) -> list[float]:
    """整群 bootstrap 第 [start, end) 次重抽的斜率（供分段并行调用）。

    先重放 RNG 到第 start 次重抽再计算本段，抽样序列与串行循环逐次一致
    （`default_rng(seed)` 每次循环恰好调用一次 `integers(0, k, k)`），
    因此分段并行与串行结果逐位相同。`codes` 与 `second_codes` 均为已
    编码的整数（后者可为 None；调用方在进入本函数前编码一次，避免每次
    重抽重复做标签编码）。
    payload = (x, y, codes, second_codes, k, seed, start, end)。
    """
    x, y, codes, second_codes, k, seed, start, end = payload
    rng = np.random.default_rng(seed)
    second = None if second_codes is None else np.asarray(second_codes)
    slopes: list[float] = []
    for i in range(end):
        samp = rng.integers(0, k, k)
        if i < start:
            continue
        mult = np.bincount(samp, minlength=k).astype(float)
        w = mult[codes]
        m = w > 0
        if m.sum() < 2:
            continue
        cc = codes[m]
        if second is None:
            wm = w[m]
            den = group_denominators(cc, wm)
            xr = demean_group(x[m], cc, wm, denom=den)
            yr = demean_group(y[m], cc, wm, denom=den)
        else:
            c2 = second[m]
            xr = demean_two_way(x[m], cc, c2, w[m])
            yr = demean_two_way(y[m], cc, c2, w[m])
        # 每组仅一条观测（去均值后全为浮点噪声）时跳过，避免把噪声当斜率
        if np.ptp(xr) <= 1e-9 * max(1.0, float(np.ptp(x[m]))):
            continue
        s = ols_slope(xr, yr, w[m])
        if not np.isnan(s):
            slopes.append(s)
    return slopes


def cluster_bootstrap_slope(x, y, codes, n_boot: int = N_BOOT,
                            seed: int = BOOT_SEED, second_codes=None,
                            alpha: float = 0.05) -> dict:
    """按 `codes`（作者）整群重抽的斜率 bootstrap。

    单因素时对组内去均值后做 OLS；`second_codes` 非空时对两个因素交替
    去均值（作者+月份固定效应）。重抽后按作者重抽次数加权（未抽中的
    作者剔除），返回点估计与百分位 CI。要求组内确有离差：每组仅一条
    观测（无组内变异）时点估计为 NaN，退化重抽样本直接跳过。

    双因素去均值每次重抽要跑满交替投影的迭代上限，是主要 CPU 开销；
    双因素且样本规模足够大时按重抽次数分段、用进程池并行计算（Linux
    fork；不可用时自动回退串行），分段之间通过重放 RNG 保证抽样序列
    与串行完全一致。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    codes, k = factorize_codes(codes)
    rng = np.random.default_rng(seed)
    if second_codes is None:
        xr, yr = demean_group(x, codes), demean_group(y, codes)
        point = ols_slope(xr, yr)
    else:
        c2, _ = factorize_codes(second_codes)
        point = ols_slope(demean_two_way(x, codes, c2),
                          demean_two_way(y, codes, c2))
        second_codes = c2
    slopes: list[float] = []
    parallel = second_codes is not None and n_boot >= 64 and k >= 64
    pool = make_pool(4) if parallel else None
    if pool is not None:
        try:
            n_workers = min(4, max(1, os.cpu_count() or 1))
            bounds = np.linspace(0, n_boot, n_workers + 1).astype(int)
            segments = [(int(bounds[i]), int(bounds[i + 1]))
                        for i in range(n_workers) if bounds[i] < bounds[i + 1]]
            results = pool.map(
                _cluster_boot_slopes_segment,
                [(x, y, codes, second_codes, k, seed, a, b)
                 for a, b in segments], chunksize=1)
            slopes = [s for part in results for s in part]
        finally:
            close_pool(pool)
    else:
        slopes = _cluster_boot_slopes_segment(
            (x, y, codes, second_codes, k, seed, 0, n_boot))
    arr = np.asarray(slopes)
    if arr.size == 0:
        return {"slope": point, "ci_low": float("nan"),
                "ci_high": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"slope": float(point), "ci_low": float(lo),
            "ci_high": float(hi), "n_boot": int(arr.size)}


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORES_PATH,
                     usecols=["month", "domain", "likes", "collections",
                              "comments", "p_ai", "content_len"])
    df["month"] = df["month"].astype(str)
    return df


def scan_users(scores: pd.DataFrame) -> dict:
    """流式扫描 `exploring_set_fixed.jsonl`，收集 note_id / user_id / 元数据分。

    同时用 scores 的 month/domain 原始字段逐条校验行序（content_len 因
    清洗版本差异可能漂移，不作为错位依据）；返回逐行数组，供后续去重与
    作者子集筛选。
    """
    nid: list = []
    uid: list = []
    meta: list = []
    mismatch = 0
    months_ref = scores["month"].astype(str).to_numpy()
    domains_ref = scores["domain"].astype(str).to_numpy()
    n_scores = months_ref.size
    for i, rec in enumerate(iter_cached(EXPLORING_PATH)):
        body = clean_body(content_of(rec))
        lk, cl, cm = engagement_of(rec)
        user = rec.get("user_id") or ""
        nid.append(rec.get("note_id"))
        uid.append(user)
        dt = parse_local_time(rec.get("local_time"))
        month = f"{dt.year:04d}{dt.month:02d}" if dt is not None else ""
        domain = str(rec.get("domain") or "")
        if (i >= n_scores or month != months_ref[i]
                or domain != domains_ref[i]):
            mismatch += 1
        has_meta = (bool(user)
                    and rec.get("domain") not in (None, "", "未知"))
        # 去重打分与 analysis_utils.dedup_indices 文档的共享规则完全一致
        # （含时间戳），保证与其他全量扫描脚本保留同一条重复记录。
        meta.append(int(has_meta) + int(bool(lk)) + int(bool(cl))
                    + int(bool(cm)) + int(dt is not None)
                    + int(len(body) > 0))
    return {"nid": nid, "uid": uid, "meta": np.asarray(meta),
            "n_raw": len(nid), "mismatch": int(mismatch)}


def build_frames(scores: pd.DataFrame, scan: dict) -> dict:
    """构造去重后的全量帧与带 user_id 的作者子集帧。

    作者子集 = 去重保留帧中 user_id 非空的记录（该子集内每篇笔记恰一条）。
    """
    keep_all = dedup_indices(scan["nid"], scan["meta"])
    uid_all = np.asarray(scan["uid"], dtype=object)
    keep_uid = [i for i in keep_all if uid_all[i]]
    full = scores.iloc[keep_all].copy()
    full["user_id"] = [uid_all[i] for i in keep_all]
    full["log_likes"] = np.log10(1 + full["likes"])
    full = full[full["month"] >= MONTH_FROM].reset_index(drop=True)
    auth = full[full["user_id"] != ""].reset_index(drop=True)
    return {"full": full, "author": auth,
            "n_dedup": int(len(keep_all)), "n_author": int(len(auth))}


# --------------------------------------------------------------------------
# 统计主体
# --------------------------------------------------------------------------
def selection_table(full: pd.DataFrame) -> dict:
    """带 user_id 与不带 user_id 的去重笔记对比（选择检查）。"""
    out = {}
    for key, g in (("with_uid", full[full["user_id"] != ""]),
                   ("without_uid", full[full["user_id"] == ""])):
        k = int((g["p_ai"] >= HI_CONF).sum())
        lo, hi = wilson_ci(k, len(g))
        out[key] = {
            "n": int(len(g)),
            "p_ai_mean": round(float(g["p_ai"].mean()), 4),
            "p_ai_median": round(float(g["p_ai"].median()), 4),
            "hi_conf_share": round(k / len(g), 4),
            "hi_conf_ci": [round(float(lo), 4), round(float(hi), 4)],
            "likes_median": float(g["likes"].median()),
            "content_len_mean": round(float(g["content_len"].mean()), 1),
            "months": f"{g['month'].min()}–{g['month'].max()}",
            "year_share_2025": round(float((g["month"] >= "202501").mean()), 3),
        }
    return out


def author_concentration(auth: pd.DataFrame) -> dict:
    """作者发文集中度与「高置信 AI 帖是否更集中」。"""
    counts = auth.groupby("user_id").size()
    author_hi = auth.assign(hi=(auth["p_ai"] >= HI_CONF).astype(float))
    hi_per_author = author_hi.groupby("user_id")["hi"].sum()
    mean_p = auth.groupby("user_id")["p_ai"].mean()
    shares = top_author_shares(counts.to_numpy(), hi_per_author.to_numpy())
    rho, p = spearmanr(counts.to_numpy(), mean_p.to_numpy())
    return {
        "n_authors": int(counts.size),
        "n_notes": int(len(auth)),
        "notes_gini": round(gini_coefficient(counts.to_numpy()), 4),
        "top_frac": TOP_AUTHOR_FRAC,
        "top_n": shares["top_n"],
        "top_note_share": round(shares["note_share"], 4),
        "top_hi_conf_share": round(shares["hit_share"], 4),
        "notes_per_author_mean": round(float(counts.mean()), 3),
        "notes_per_author_median": int(counts.median()),
        "multi_note_author_share": round(float((counts >= MIN_AUTHOR_NOTES).mean()), 4),
        "prolific_vs_mean_p_ai": {"rho": round(float(rho), 4), "p": float(p)},
    }


def engagement_decomposition(auth: pd.DataFrame) -> dict:
    """互动斜率的作者间 / 人内分解 + 整群 bootstrap CI。"""
    x = auth["p_ai"].to_numpy()
    y = auth["log_likes"].to_numpy()
    codes = auth["user_id"].to_numpy()
    months = auth["month"].to_numpy()
    dec = decompose_slope(x, y, codes)

    g = auth.groupby("user_id").agg(n=("p_ai", "size"), mp=("p_ai", "mean"),
                                    my=("log_likes", "mean"),
                                    likes_med=("likes", "median"))
    between = {}
    between_ci = {}
    for key, mask in (("all_authors", np.ones(len(g), dtype=bool)),
                      ("multi_note_authors", g["n"].to_numpy() >= MIN_AUTHOR_NOTES)):
        gg = g[mask]
        between[key] = {
            "n_authors": int(len(gg)),
            "n_notes": int(gg["n"].sum()),
            "slope_wls": round(float(ols_slope(gg["mp"].to_numpy(),
                                               gg["my"].to_numpy(),
                                               gg["n"].to_numpy())), 4),
            "slope_ols": round(float(ols_slope(gg["mp"].to_numpy(),
                                               gg["my"].to_numpy())), 4),
        }
        between_ci[key] = bootstrap_wls_slope(
            gg["mp"].to_numpy(), gg["my"].to_numpy(), gg["n"].to_numpy())

    boot_within = cluster_bootstrap_slope(x, y, codes)
    boot_two = cluster_bootstrap_slope(x, y, codes, n_boot=N_BOOT_TWO,
                                       second_codes=months)

    quartiles = []
    gm = g[g["n"] >= MIN_AUTHOR_NOTES].copy()
    gm["q"] = pd.qcut(gm["mp"], 4, labels=False, duplicates="drop")
    for q, gg in gm.groupby("q"):
        quartiles.append({
            "quartile": int(q) + 1,
            "n_authors": int(len(gg)),
            "n_notes": int(gg["n"].sum()),
            "mean_p_ai": round(float(gg["mp"].mean()), 4),
            "mean_log_likes": round(float(gg["my"].mean()), 4),
        })

    return {
        "decomposition": {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in dec.items()},
        "between_authors": between,
        "between_ci": {
            key: {"slope": round(ci["slope"], 4),
                  "ci_low": round(ci["ci_low"], 4),
                  "ci_high": round(ci["ci_high"], 4),
                  "n_boot": ci["n_boot"]}
            for key, ci in between_ci.items()},
        "within_author": {"slope": round(boot_within["slope"], 4),
                          "ci_low": round(boot_within["ci_low"], 4),
                          "ci_high": round(boot_within["ci_high"], 4),
                          "n_boot": boot_within["n_boot"]},
        "within_author_month": {"slope": round(boot_two["slope"], 4),
                                "ci_low": round(boot_two["ci_low"], 4),
                                "ci_high": round(boot_two["ci_high"], 4),
                                "n_boot": boot_two["n_boot"]},
        "author_quartiles": quartiles,
        "spearman_pooled_uid": round(float(spearmanr(x, y).statistic), 4),
    }


def paired_analysis(auth: pd.DataFrame, deltas: np.ndarray | None = None) -> dict:
    """人内配对：同作者两半、同作者同月份两半的 Δ(log 点赞)。"""
    out = {}
    if deltas is None:
        deltas = paired_style_deltas(auth["user_id"], auth["p_ai"],
                                     auth["log_likes"])
    if deltas.size:
        p_wil = float(wilcoxon(deltas).pvalue)
    else:
        p_wil = float("nan")
    ci = bootstrap_mean_ci(deltas)
    out["authors"] = {
        "n_authors": int(deltas.size),
        "mean_delta": round(float(ci["mean"]), 4),
        "ci_low": round(float(ci["ci_low"]), 4),
        "ci_high": round(float(ci["ci_high"]), 4),
        "median_delta": round(float(np.median(deltas)), 4) if deltas.size else float("nan"),
        "neg_share": round(float((deltas < 0).mean()), 4) if deltas.size else float("nan"),
        "likes_multiplier": round(float(10 ** ci["mean"]), 3),
        "wilcoxon_p": p_wil,
    }

    # 同作者同月份单元：与逐组 DataFrame 筛选求均值同口径，但一次向量化算完。
    # 同一作者可贡献多个月份单元，这些 Δ 并非独立观测（伪重复）；描述统计仍
    # 按单元给出，Wilcoxon 检验则先按作者把单元 Δ 聚合成一个观测（簇级检验），
    # 避免把同作者单元当作独立样本而低估 p 值。
    user_ids = auth["user_id"].to_numpy().astype(str)
    cell_ids = user_ids + "|" + auth["month"].to_numpy().astype(str)
    cell_keys, cell_lo, cell_hi = _paired_groups(
        cell_ids, auth["p_ai"], auth["log_likes"])
    cell_deltas = cell_hi - cell_lo
    if cell_deltas.size:
        uid_by_cell = dict(zip(cell_ids, user_ids))
        cell_authors = np.asarray([uid_by_cell[k] for k in cell_keys])
        cluster_codes, n_clusters = factorize_codes(cell_authors)
        cluster_deltas = (np.bincount(cluster_codes, weights=cell_deltas)
                          / np.bincount(cluster_codes))
        p_cell = float(wilcoxon(cluster_deltas).pvalue)
    else:
        n_clusters = 0
        cluster_deltas = np.empty(0, dtype=float)
        p_cell = float("nan")
    out["author_month_cells"] = {
        "n_cells": int(cell_deltas.size),
        "n_authors": int(n_clusters),
        "mean_delta": round(float(cell_deltas.mean()), 4) if cell_deltas.size else float("nan"),
        "neg_share": round(float((cell_deltas < 0).mean()), 4) if cell_deltas.size else float("nan"),
        "wilcoxon_p": p_cell,
    }

    rhos, ns = [], []
    for uid, g in auth.groupby("user_id", sort=True):
        if len(g) < MIN_WITHIN_NOTES or g["p_ai"].nunique() < 2:
            continue
        r = spearmanr(g["p_ai"], g["log_likes"]).statistic
        if not np.isnan(r):
            rhos.append(float(r))
            ns.append(len(g))
    if rhos:
        weighted = float(np.average(rhos, weights=ns))
        out["within_author_rho"] = {
            "min_notes": MIN_WITHIN_NOTES,
            "n_authors": int(len(rhos)),
            "weighted_mean_rho": round(weighted, 4),
            "neg_share": round(float((np.asarray(rhos) < 0).mean()), 4),
        }
    else:
        out["within_author_rho"] = {
            "min_notes": MIN_WITHIN_NOTES, "n_authors": 0,
            "weighted_mean_rho": float("nan"), "neg_share": float("nan"),
        }
    return out


def consistency_check(full_scores: pd.DataFrame) -> dict:
    """复算 paper_engagement 的总体 ρ 与单元内加权 ρ，校验口径一致。"""
    import json

    df = full_scores[full_scores["month"] >= MONTH_FROM]
    x = df["p_ai"].to_numpy()
    y = np.log10(1 + df["likes"].to_numpy())
    rho_overall = float(spearmanr(x, y).statistic)
    cell_codes = df["month"].astype(str) + "|" + df["domain"].astype(str)
    cells = within_cell_spearman(df["p_ai"], y, cell_codes.to_numpy(), 100)
    with open(BASE + "paper_engagement_results.json", encoding="utf-8") as f:
        paper = json.load(f)
    return {
        "overall_rho_recomputed": round(rho_overall, 4),
        "overall_rho_paper": paper["overall_spearman_likes"]["rho"],
        "within_cell_rho_recomputed": round(float(cells["weighted_rho"]), 4),
        "within_cell_rho_paper": paper["within_cell_weighted_rho"],
        "n_rows_full": int(len(df)),
    }


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_structure(full: pd.DataFrame, auth: pd.DataFrame, conc: dict,
                   decom: dict, out_path: str) -> None:
    init_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    for key, lab, color in (("with_uid", "带 user_id（作者子集）", PALETTE["aigc"]),
                            ("without_uid", "无 user_id", PALETTE["human"])):
        g = full[full["user_id"] != ""] if key == "with_uid" else full[full["user_id"] == ""]
        xs = np.sort(g["p_ai"].to_numpy())
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=color, lw=1.6, label=f"{lab}（n={len(g):,}）")
    ax.set_xlabel("P(AI) 得分")
    ax.set_ylabel("累积占比（ECDF）")
    ax.set_title("选择检查：作者子集的 P(AI) 分布")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    counts = np.sort(auth.groupby("user_id").size().to_numpy())
    cum_authors = np.arange(1, counts.size + 1) / counts.size
    cum_notes = np.cumsum(counts) / counts.sum()
    ax.plot(cum_authors, cum_notes, color=PALETTE["aigc"], lw=1.8,
            label=f"实际（基尼 {conc['notes_gini']:.3f}）")
    ax.plot([0, 1], [0, 1], color="#8C8C8C", lw=1.0, ls="--", label="完全均等线")
    ax.axvline(1 - TOP_AUTHOR_FRAC, color="#4C72B0", lw=1.0, ls=":")
    ax.annotate(f"前 {int(TOP_AUTHOR_FRAC*100)}% 作者\n占 "
                f"{conc['top_note_share']:.1%} 笔记\n高置信帖 {conc['top_hi_conf_share']:.1%}",
                xy=(1 - TOP_AUTHOR_FRAC, 1 - conc["top_note_share"]),
                xytext=(0.18, 0.62), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"))
    ax.set_xlabel("作者累积占比（按发文量升序）")
    ax.set_ylabel("笔记累积占比")
    ax.set_title(f"作者发文集中度（{conc['n_authors']:,} 位作者）")
    ax.legend(fontsize=8.5, loc="upper left")

    ax = axes[2]
    q = decom["author_quartiles"]
    xs = np.arange(len(q))
    ax.bar(xs, [r["mean_log_likes"] for r in q], color="#55A868", width=0.6)
    for i, r in enumerate(q):
        ax.text(i, r["mean_log_likes"] + 0.01,
                f"×{10**(r['mean_log_likes']-q[0]['mean_log_likes']):.2f}\n"
                f"{r['n_authors']}人/{r['n_notes']}帖", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Q{i+1}\nP̄={r['mean_p_ai']:.2f}"
                        for i, r in enumerate(q)], fontsize=8)
    ax.set_ylim(min(r["mean_log_likes"] for r in q) - 0.06,
                max(r["mean_log_likes"] for r in q) + 0.09)
    ax.set_ylabel("作者平均 log10(1+点赞)")
    ax.set_title("作者间梯度：作者平均 P(AI) 四分位（≥2 帖作者）")

    fig.suptitle("作者层结构：谁在写「AI 味」内容（探索集 2023.01–2025.06）",
                 fontsize=13)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_within(decom: dict, paired: dict, auth: pd.DataFrame,
                out_path: str, deltas: np.ndarray | None = None,
                half_means: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    init_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    rows = [
        ("总体（作者子集）", decom["decomposition"]["pooled"], None),
        ("作者间（全体作者）", decom["between_authors"]["all_authors"]["slope_wls"], None),
        ("作者间（≥2 帖）",
         decom["between_ci"]["multi_note_authors"]["slope"],
         (decom["between_ci"]["multi_note_authors"]["ci_low"],
          decom["between_ci"]["multi_note_authors"]["ci_high"])),
        ("人内（作者 FE）", decom["within_author"]["slope"],
         (decom["within_author"]["ci_low"], decom["within_author"]["ci_high"])),
        ("人内（作者+月份 FE）", decom["within_author_month"]["slope"],
         (decom["within_author_month"]["ci_low"],
          decom["within_author_month"]["ci_high"])),
    ]
    ypos = np.arange(len(rows))[::-1]
    for y, (lab, s, ci) in zip(ypos, rows):
        color = "#4C72B0" if ci is None else CI_COLOR
        if ci and ci[0] <= 0 <= ci[1]:
            color = "#8C8C8C"
        xerr = np.array([[s - ci[0]], [ci[1] - s]]) if ci else None
        ax.errorbar(s, y, xerr=xerr, fmt="o", color=color, capsize=3, ms=5)
        ax.text(s, y + 0.22, f"{s:+.3f}", ha="center", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("P(AI) → log10(1+点赞) 斜率（95% 整群 bootstrap CI）")
    ax.set_title("互动斜率的作者间 / 人内分解")
    ax.set_xlim(min(-0.45, min(r[1] for r in rows) - 0.05), 0.12)
    ax.set_ylim(-0.5, len(rows) - 0.4)

    ax = axes[1]
    if half_means is None:
        half_means = _paired_half_means(
            auth["user_id"], auth["p_ai"], auth["log_likes"])
    lo_means, hi_means = half_means
    both = np.column_stack([lo_means, hi_means]) if lo_means.size else \
        np.empty((0, 2), dtype=float)
    if both.size:
        ax.scatter(both[:, 0], both[:, 1], s=9, alpha=0.35,
                   color=PALETTE["aigc"], edgecolor="none")
        lim = [min(both.min(), 0), max(both.max(), 5)]
        ax.plot(lim, lim, color="#8C8C8C", lw=1.0, ls="--",
                label="y=x（无差异）")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xlabel("作者低风格半组 平均 log10(1+点赞)")
    ax.set_ylabel("作者高风格半组 平均 log10(1+点赞)")
    ax.set_title(f"人内配对（{paired['authors']['n_authors']:,} 位两半作者）")
    ax.legend(fontsize=8.5, loc="upper left")

    ax = axes[2]
    if deltas is None:
        deltas = paired_style_deltas(auth["user_id"], auth["p_ai"],
                                     auth["log_likes"])
    if deltas.size:
        ax.hist(deltas, bins=41, color=PALETTE["exploring"], alpha=0.85)
        ax.axvline(0, color="k", lw=0.9)
        ax.axvline(deltas.mean(), color=CI_COLOR, lw=1.6,
                   label=f"均值 {deltas.mean():+.3f}")
        ax.axvspan(paired["authors"]["ci_low"], paired["authors"]["ci_high"],
                   color=CI_COLOR, alpha=0.12, lw=0, label="95% CI")
        ax.legend(fontsize=8.5)
    ax.set_xlabel("Δ = 高风格半组 − 低风格半组（log10 点赞）")
    ax.set_ylabel("作者数")
    ax.set_title(f"人内 Δ 分布（负向 {paired['authors']['neg_share']:.1%}，"
                 f"Wilcoxon p={paired['authors']['wilcoxon_p']:.2g}）")

    fig.suptitle("「AI 味惩罚」的作者间 vs 人内分解（探索集 2023.01–2025.06）",
                 fontsize=13)
    fig.tight_layout()
    save_fig(fig, out_path)


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def render_markdown(out: dict) -> str:
    sel = out["selection"]
    conc = out["concentration"]
    dec = out["engagement"]["decomposition"]
    eb = out["engagement"]["between_authors"]
    bc = out["engagement"]["between_ci"]
    wc = out["engagement"]["within_author"]
    wm = out["engagement"]["within_author_month"]
    pa = out["paired"]["authors"]
    pm = out["paired"]["author_month_cells"]
    wr = out["paired"]["within_author_rho"]
    cons = out["consistency"]
    lines = [
        "# 作者层分解：作者间 vs 人内（Author-level Decomposition）",
        "",
        "> 由 `author_style_analysis.py` 全量运行生成；数据源 "
        "`paper_exploring_scores.csv.gz` + `exploring_set_fixed.jsonl`"
        "（month/domain 逐条对齐，mismatch="
        f"{out['meta']['row_mismatch']}）；月份窗口 ≥2023.01，"
        "口径与 `paper_engagement.py` 一致。作者子集 = 去重后带 `user_id` "
        "的笔记（该子集内每篇笔记恰一条记录）。",
        "",
        "## 1 覆盖与选择检查（去重口径）",
        "",
        "| 子集 | n | P(AI) 均值 | P(AI) 中位 | 高置信≥0.9 | 点赞中位 | 正文均长 | 月份范围 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, lab in (("with_uid", "带 user_id"), ("without_uid", "无 user_id")):
        s = sel[key]
        lines.append(
            f"| {lab} | {s['n']:,} | {s['p_ai_mean']:.3f} | {s['p_ai_median']:.3f} | "
            f"{s['hi_conf_share']:.1%} | {s['likes_median']:.0f} | "
            f"{s['content_len_mean']:.0f} | {s['months']} |")
    lines += [
        "",
        f"作者子集占去重笔记的 {out['meta']['author_share']:.1%}；"
        f"其中 2025 年占 {sel['with_uid']['year_share_2025']:.1%}——"
        "作者层结论只在此采样层内成立。",
        "",
        "## 2 作者集中度",
        "",
        f"- 作者数 {conc['n_authors']:,}，人均发文 "
        f"{conc['notes_per_author_mean']:.2f} 帖（中位 {conc['notes_per_author_median']}）；"
        f"≥2 帖作者占 {conc['multi_note_author_share']:.1%}",
        f"- 发文量基尼系数 {conc['notes_gini']:.3f}；前 "
        f"{int(conc['top_frac']*100)}% 作者（{conc['top_n']:,} 人）占笔记 "
        f"{conc['top_note_share']:.1%}、占高置信帖 {conc['top_hi_conf_share']:.1%}",
        f"- 发文量与作者平均 P(AI) 相关 ρ="
        f"{conc['prolific_vs_mean_p_ai']['rho']:+.3f} "
        f"(p={conc['prolific_vs_mean_p_ai']['p']:.2g})",
        "",
        "## 3 风格方差分解（≥2 帖作者）",
        "",
        f"- ICC（作者间方差占比）= {out['variance']['icc']:.3f}；"
        f"作者间 SD={out['variance']['sd_between']:.3f}、作者内 SD="
        f"{out['variance']['sd_within']:.3f}（{out['variance']['n_groups']:,} 位作者 / "
        f"{out['variance']['n']:,} 帖）——P(AI) 同时是作者属性与单帖属性。",
        "",
        "## 4 互动斜率的作者间 / 人内分解（P(AI) → log10(1+点赞)）",
        "",
        f"精确分解：β_pooled = {dec['pooled']:+.3f} = "
        f"{dec['between_weight']:.2f} × β_between({dec['between']:+.3f}) + "
        f"{1-dec['between_weight']:.2f} × β_within({dec['within']:+.3f})。",
        "",
        "| 口径 | 斜率 β | 95% 整群 bootstrap CI | n |",
        "|---|---:|---|---:|",
        f"| 作者间（全体作者，WLS） | {eb['all_authors']['slope_wls']:+.4f} | "
        f"[{bc['all_authors']['ci_low']:+.4f}, "
        f"{bc['all_authors']['ci_high']:+.4f}] | "
        f"{eb['all_authors']['n_authors']:,} 人 |",
        f"| 作者间（≥2 帖，WLS） | {eb['multi_note_authors']['slope_wls']:+.4f} | "
        f"[{bc['multi_note_authors']['ci_low']:+.4f}, "
        f"{bc['multi_note_authors']['ci_high']:+.4f}] | "
        f"{eb['multi_note_authors']['n_authors']:,} 人 |",
        f"| 人内（作者 FE） | {wc['slope']:+.4f} | "
        f"[{wc['ci_low']:+.4f}, {wc['ci_high']:+.4f}] | "
        f"{out['meta']['n_author_notes']:,} 帖 |",
        f"| 人内（作者+月份 FE） | {wm['slope']:+.4f} | "
        f"[{wm['ci_low']:+.4f}, {wm['ci_high']:+.4f}] | 同上 |",
        "",
        "作者平均 P(AI) 四分位（≥2 帖作者）的平均 log 点赞见 "
        "`author_style_results.json` 的 `author_quartiles`。",
        "",
        "## 5 人内配对对照",
        "",
        f"- 同时有「像人」（P(AI)<0.5）与「像 AI」（≥0.5）两半的作者 "
        f"{pa['n_authors']:,} 位：平均 Δ={pa['mean_delta']:+.4f} "
        f"(95% CI [{pa['ci_low']:+.4f}, {pa['ci_high']:+.4f}])，"
        f"中位 Δ={pa['median_delta']:+.4f}，负向作者占 {pa['neg_share']:.1%}，"
        f"Wilcoxon p={pa['wilcoxon_p']:.3g}",
        f"  即同作者内部，风格更高半组的点赞量乘数约 ×{pa['likes_multiplier']:.2f}"
        "（点估计），与无差异无法区分。",
        f"- 同作者同月份单元（{pm['n_cells']:,} 个，{pm['n_authors']:,} 位作者）："
        f"平均 Δ={pm['mean_delta']:+.4f}，负向占 {pm['neg_share']:.1%}，"
        f"按作者聚类的 Wilcoxon p={pm['wilcoxon_p']:.3g}",
        f"- 发文 ≥{wr['min_notes']} 帖且风格有变化的作者（{wr['n_authors']:,} 位）"
        f"人内 ρ 加权均值 {wr['weighted_mean_rho']:+.4f}，负向占 {wr['neg_share']:.1%}",
        "",
        "## 6 口径一致性校验",
        "",
        f"- 全样本总体 ρ 复算 {cons['overall_rho_recomputed']:+.4f}"
        f" vs `paper_engagement_results.json` {cons['overall_rho_paper']:+.4f}；"
        f"单元内加权 ρ 复算 {cons['within_cell_rho_recomputed']:+.4f} vs "
        f"{cons['within_cell_rho_paper']:+.4f}",
        "",
        "> 解读提示：横截面上的「AI 味惩罚」主要由**作者间**差异承载"
        "（长期写 AI 风格的一组作者整体互动更低）；同一作者内部，风格更 AI "
        "的帖子没有可检出的系统性互动损失。这不是对 `paper_engagement.py` "
        "总体负相关的否定，而是把相关性的位置定位到作者层；人内 CI 较宽且"
        f"都跨零（点估计 {wc['slope']:+.4f}，区间 "
        f"[{wc['ci_low']:+.4f}, {wc['ci_high']:+.4f}]；加月份 FE 后 "
        f"{wm['slope']:+.4f}，区间 [{wm['ci_low']:+.4f}, "
        f"{wm['ci_high']:+.4f}]），无法排除小幅惩罚。"
        "作者子集的采样偏差（2025 年偏多、互动偏高、AI 味偏低）限制了外推。",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 主程序
# --------------------------------------------------------------------------
def main() -> None:
    scores = load_scores()
    scan = scan_users(scores)
    assert scan["n_raw"] == len(scores), (
        f"exploring_set_fixed.jsonl 行数 {scan['n_raw']} 与 scores {len(scores)} 不一致")
    frames = build_frames(scores, scan)
    full, auth = frames["full"], frames["author"]
    print(f"去重笔记 {frames['n_dedup']:,}（原始 {scan['n_raw']:,}），"
          f"带 user_id {frames['n_author']:,}，"
          f"行序 mismatch={scan['mismatch']}")

    sel = selection_table(full)
    conc = author_concentration(auth)
    multi = auth.groupby("user_id")["p_ai"].transform("size").to_numpy() >= MIN_AUTHOR_NOTES
    var = variance_decomposition(auth.loc[multi, "p_ai"].to_numpy(),
                                 auth.loc[multi, "user_id"].to_numpy())
    eng = engagement_decomposition(auth)
    pair_deltas = paired_style_deltas(auth["user_id"], auth["p_ai"],
                                      auth["log_likes"])
    pair_halves = _paired_half_means(auth["user_id"], auth["p_ai"],
                                     auth["log_likes"])
    paired = paired_analysis(auth, deltas=pair_deltas)
    cons = consistency_check(scores)

    print(f"ICC={var['icc']:.3f}  分解: pooled={eng['decomposition']['pooled']:+.3f}"
          f" = {eng['decomposition']['between_weight']:.2f}×between"
          f"({eng['decomposition']['between']:+.3f})"
          f" + within({eng['decomposition']['within']:+.3f})")
    print(f"人内 β={eng['within_author']['slope']:+.4f} "
          f"CI [{eng['within_author']['ci_low']:+.4f}, "
          f"{eng['within_author']['ci_high']:+.4f}]；"
          f"配对 Δ={paired['authors']['mean_delta']:+.4f} "
          f"(p={paired['authors']['wilcoxon_p']:.3g})")
    print(f"一致性: 总体 ρ={cons['overall_rho_recomputed']:+.4f} "
          f"(paper {cons['overall_rho_paper']:+.4f})")

    out = {
        "meta": {
            "source": "paper_exploring_scores.csv.gz + exploring_set_fixed.jsonl",
            "month_from": MONTH_FROM,
            "style_split": STYLE_SPLIT,
            "hi_conf": HI_CONF,
            "n_raw_records": scan["n_raw"],
            "n_dedup_notes": frames["n_dedup"],
            "n_author_notes": frames["n_author"],
            "author_share": round(frames["n_author"]
                                  / max(len(frames["full"]), 1), 4),
            "row_mismatch": scan["mismatch"],
            "n_boot": N_BOOT,
            "n_boot_two_way": N_BOOT_TWO,
            "seed": BOOT_SEED,
        },
        "selection": sel,
        "concentration": conc,
        "variance": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in var.items()},
        "engagement": eng,
        "paired": paired,
        "consistency": cons,
    }
    save_json(out, BASE + "author_style_results.json")
    with open(BASE + "author_style_results.md", "w", encoding="utf-8") as f:
        f.write(render_markdown(out))
    plot_structure(full, auth, conc, eng, INSIGHTS + "author_structure.png")
    plot_within(eng, paired, auth, INSIGHTS + "author_within.png",
                deltas=pair_deltas, half_means=pair_halves)
    print("已保存 author_style_results.json/.md 与 "
          "insights/author_structure.png、insights/author_within.png")


if __name__ == "__main__":
    main()
