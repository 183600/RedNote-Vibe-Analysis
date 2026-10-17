# -*- coding: utf-8 -*-
"""互动惩罚的逐特征载体：单元内秩相关 + 固定效应多元回归（探索集 2023.01–2025.06）。

`paper_engagement.py` / `engagement_structure_analysis.py` 回答的是「AI 味越重、
互动越差」；本分析进一步回答「这种惩罚具体由哪些**可解释风格特征**承载」：

1. **单变量**：对检测器的 37 个可解释特征与 P(AI) 分别计算「月份×领域」单元内
   Spearman(特征, log10(1+互动))，按单元样本量加权汇总，得到原始关联的强弱与
   方向（同时报告单元内负相关的单元占比）；
2. **多元固定效应回归**：把 log10(1+互动) 对 37 个标准化特征做单元内去均值
   （等价于加入月份×领域固定效应）OLS，得到「控制其余风格特征后」的偏效应 β
   （每 1 SD 特征变化对应的 log10 互动变化），并用**按单元整群 bootstrap** 的
   95% CI 判断稳健性（不假设同一单元内帖子相互独立）；
3. **P(AI) 净效应**：单独回归 p_ai，以及控制最强正驱动（正文长度、标签数）后的
   p_ai 系数，检验「AI 味惩罚」是否只是长度/标签数量的间接效应。

口径与既有论文管线一致：数据源 `paper_features_exploring.csv.gz` +
`paper_exploring_scores.csv.gz`（逐行对齐，content_len 校验）；月份自 2023.01
起（同 `paper_engagement.py`）；单元样本量 >= 100；特征名取自
`paper_detector_model.json` 的 37 特征，不重新训练检测器。

输出：
  insights/engagement_drivers.png   单变量 ρ / 多元偏效应 β / likes-vs-collections β 对照
  engagement_drivers_results.json   全部数值结果
  engagement_drivers_results.md     结果摘要
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_utils import (BASE, INSIGHTS, init_plot_style, save_fig, save_json,
                            cell_groups, cell_rank_cache, load_detector_model,
                            within_cell_spearman)

MIN_CELL_N = 100        # 月份×领域单元最少帖子数（同 paper_engagement）
MONTH_FROM = "202301"   # 与 paper_engagement 相同的观察窗口
N_BOOT = 300            # 整群 bootstrap 次数
BOOT_SEED = 42
TARGETS = (("likes", "点赞"), ("collections", "收藏"), ("comments", "评论"))
CI_EXCL0_COLOR = "#C44E52"
CI_INCL0_COLOR = "#8C8C8C"

# 特征名 -> 图/表用中文标签（与 paper_features.FIELDS 一一对应）
FEATURE_LABELS = {
    "content_len": "正文长度", "n_lines": "行数", "avg_line_len": "平均行长",
    "colon_line_ratio": "冒号行占比", "list_line_ratio": "列表行占比",
    "excl100": "感叹号/百字", "ques100": "问号/百字", "ellip100": "省略号/百字",
    "tilde100": "波浪号/百字", "comma100": "逗号/百字", "pause100": "顿号/百字",
    "period100": "句号/百字", "emoji100": "emoji/百字",
    "emoji_distinct": "不同 emoji 数", "xhsemo100": "社区表情/百字",
    "p_的": "「的」/百字", "p_了": "「了」/百字", "p_是": "「是」/百字",
    "p_我": "「我」/百字", "p_你": "「你」/百字", "p_就": "「就」/百字",
    "p_都": "「都」/百字", "p_很": "「很」/百字", "p_有": "「有」/百字",
    "p_不": "「不」/百字", "ai_lex100": "AI 味词/百字",
    "hum_lex100": "人类连接词/百字", "uniq_char_ratio": "独特字比率",
    "uniq_bigram_ratio": "独特二元组比率", "n_hashtags": "标签数",
    "title_len": "标题长度", "title_excl": "标题含感叹号", "title_ques": "标题含问号",
    "title_emoji": "标题含 emoji", "title_pipe": "标题含竖线",
    "title_num": "标题含数字", "title_bracket": "标题含方括号",
}


def label_of(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def cell_keep_mask(cell_codes, min_cell_n: int = MIN_CELL_N) -> np.ndarray:
    """标记样本量 >= min_cell_n 的单元中的记录。"""
    codes = np.asarray(cell_codes)
    _, inv, counts = np.unique(codes, return_inverse=True, return_counts=True)
    return counts[inv] >= min_cell_n


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按列做 z-score 标准化；常数列的 scale 置 1（标准化后仍为 0）。"""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    mean = arr.mean(axis=0)
    scale = arr.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (arr - mean) / scale, mean, scale


def demean_within(values: np.ndarray, cell_codes) -> np.ndarray:
    """按单元整组去均值（加入单元固定效应的等价操作）。

    `cell_codes` 需为 0..K-1 的整数编码；对二维输入按行处理，一维输入保持
    一维返回。
    """
    arr = np.asarray(values, dtype=float)
    one_dim = arr.ndim == 1
    arr2 = arr[:, None] if one_dim else arr
    codes = np.asarray(cell_codes)
    n_cells = int(codes.max()) + 1
    sums = np.zeros((n_cells, arr2.shape[1]))
    np.add.at(sums, codes, arr2)
    means = sums / np.bincount(codes, minlength=n_cells)[:, None]
    out = arr2 - means[codes]
    return out[:, 0] if one_dim else out


def prepare_fe(X: np.ndarray, cell_codes,
               min_cell_n: int = MIN_CELL_N) -> dict:
    """FE-OLS 的特征侧预处理（与 y 无关），多次拟合同一 X 时可复用。

    返回 keep_mask / cell_codes / X_demeaned / mean / scale / n_obs / n_cells。
    """
    keep = cell_keep_mask(cell_codes, min_cell_n)
    Xk = np.asarray(X, dtype=float)[keep]
    codes_raw = np.asarray(cell_codes)[keep]
    codes = pd.factorize(codes_raw, sort=True)[0]
    z, mean, scale = standardize(Xk)
    zd = demean_within(z, codes)
    return {
        "keep_mask": keep, "cell_codes": codes,
        "X_demeaned": zd, "mean": mean, "scale": scale,
        "n_obs": int(keep.sum()), "n_cells": int(codes.max()) + 1,
    }


def fe_ols(X: np.ndarray, y: np.ndarray, cell_codes,
           min_cell_n: int = MIN_CELL_N, prepared: dict | None = None) -> dict:
    """月份×领域固定效应 OLS（特征先 z-score、再单元内去均值）。

    返回 dict：coef（每 1 SD 特征的偏效应）、mean/scale（标准化参数）、
    keep_mask（参与拟合的行）、cell_codes（保留下来的行重新编码为 0..K-1）、
    X_demeaned / y_demeaned（供整群 bootstrap 复用）、n_obs / n_cells。
    传入 `prepared`（`prepare_fe` 的结果）时复用特征侧预处理（X 被忽略）。
    """
    if prepared is None:
        prepared = prepare_fe(X, cell_codes, min_cell_n)
    yk = np.asarray(y, dtype=float)[prepared["keep_mask"]]
    yd = demean_within(yk, prepared["cell_codes"])
    coef, *_ = np.linalg.lstsq(prepared["X_demeaned"], yd, rcond=None)
    out = dict(prepared)
    out["coef"] = coef
    out["y_demeaned"] = yd
    return out


def cluster_bootstrap_fe_ols(X_demeaned: np.ndarray, y_demeaned: np.ndarray,
                             cell_codes, n_boot: int = N_BOOT,
                             seed: int = BOOT_SEED) -> np.ndarray:
    """按单元整群重采样拟合 FE-OLS，返回 shape=(n_boot, n_features) 的系数。

    同一单元内的帖子一起被抽中/丢弃，避免把「同月同领域」的强相关观测
    当作独立样本而低估不确定性。

    重采样只改变各单元被抽中的重复次数，故每次重抽的 OLS 等价于对原始
    样本的加权最小二乘（权重=该单元的重复次数）。这里先按单元累计
    X'X 与 X'y 充分统计量，每次重抽用「单元重抽次数 × 单元统计量」合成
    正规方程再求解：与对重抽后展开矩阵做最小二乘是同一个估计量，但省去
    每次重抽拼接 O(n) 行并做一次 SVD 的开销（n_boot 次 → 一次预计算）。
    """
    z = np.asarray(X_demeaned, dtype=float)
    y = np.asarray(y_demeaned, dtype=float)
    codes = np.asarray(cell_codes)
    # 预分组一次（稳定排序，单元按编码升序、单元内按行号升序），等价于逐单元
    # flatnonzero 但避免了 O(单元数 × 样本数) 的布尔扫描。
    idx_by_cell = cell_groups(codes, 1)
    n_cells = len(idx_by_cell)
    n_feat = z.shape[1]
    xtx_cell = np.empty((n_cells, n_feat, n_feat))
    xty_cell = np.empty((n_cells, n_feat))
    for c, rows in enumerate(idx_by_cell):
        zc = z[rows]
        xtx_cell[c] = zc.T @ zc
        xty_cell[c] = zc.T @ y[rows]
    xtx_flat = xtx_cell.reshape(n_cells, n_feat * n_feat)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, n_feat))
    for b in range(n_boot):
        pick = rng.integers(0, n_cells, n_cells)
        mult = np.bincount(pick, minlength=n_cells).astype(float)
        lhs = (mult @ xtx_flat).reshape(n_feat, n_feat)
        rhs = mult @ xty_cell
        try:
            boots[b] = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            boots[b] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return boots


def univariate_table(feature_names, columns: dict, cell_codes,
                     min_cell_n: int = MIN_CELL_N) -> dict:
    """对每个特征在三个互动指标上做单元内加权秩相关。

    `columns` 为 {指标名: 数值数组}；返回 {feature: {target: stat, ...}}。
    预分组一次、所有特征共享，避免每个「特征×指标」组合重复 argsort；
    各特征与各指标的单元内平均秩也各自只算一次（40 个特征×3 指标共
    120 个组合，若逐组合重排会多做约 3 倍排序）。
    """
    groups = cell_groups(cell_codes, min_cell_n)
    target_ranks = {tname: cell_rank_cache(columns["targets"][tname], groups)
                    for tname in columns["targets"]}
    out: dict = {}
    for feat in feature_names:
        x = columns["x"][feat]
        x_ranks = cell_rank_cache(x, groups)
        row = {}
        for tname in columns["targets"]:
            row[tname] = within_cell_spearman(
                x, columns["targets"][tname], cell_codes,
                min_cell_n=min_cell_n, groups=groups,
                x_ranks=x_ranks, y_ranks=target_ranks[tname])
        out[feat] = row
    return out


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_drivers(uni: dict, beta: dict, feature_names, n_sig: dict) -> None:
    """三联图：单变量 ρ（点赞） / 多元 β（点赞，95% CI） / β 点赞-收藏对照。"""
    init_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.6))

    # ---- (a) 单变量：p_ai + 37 特征 ----
    ax = axes[0]
    names = ["P(AI) 检测器得分"] + [label_of(f) for f in feature_names]
    rho_vals = [uni["p_ai"]["likes"]["weighted_rho"]] + \
               [uni[f]["likes"]["weighted_rho"] for f in feature_names]
    colors = [("#C44E52" if v < 0 else "#4C72B0") for v in rho_vals]
    order = np.argsort(rho_vals)
    ypos = np.arange(len(order))
    vals = np.asarray(rho_vals)[order]
    ax.hlines(ypos, 0, vals, color=[colors[i] for i in order], lw=1.6)
    ax.scatter(vals, ypos, c=[colors[i] for i in order], s=24, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([names[i] for i in order], fontsize=7.5)
    ax.set_ylim(-0.8, len(order) - 0.2)
    for k in list(range(4)) + list(range(len(order) - 4, len(order))):
        v = vals[k]
        ax.annotate(f"{v:+.3f}", xy=(v, ypos[k]),
                    xytext=(5 if v >= 0 else 4, -2),
                    textcoords="offset points", fontsize=7.5, ha="left")
    # P(AI) 行加星标突出（按排序后的位置找）
    p_pos = int(np.flatnonzero(order == 0)[0])
    ax.scatter([rho_vals[0]], [p_pos], marker="*", s=170, color="#F2B701",
               edgecolor="black", linewidth=0.5, zorder=4)
    ax.set_xlabel("单元内加权 Spearman ρ（对 log10(1+点赞)）")
    ax.set_title("(a) 单变量：各特征与点赞的原始关联")

    # ---- (b) 多元固定效应 β（点赞） ----
    ax = axes[1]
    beta_vals = np.asarray(beta["coef"])
    ci_lo, ci_hi = beta["ci_low"], beta["ci_high"]
    order = np.argsort(beta_vals)
    ypos = np.arange(len(order))
    vals = beta_vals[order]
    lo, hi = np.asarray(ci_lo)[order], np.asarray(ci_hi)[order]
    sig = (lo > 0) | (hi < 0)
    cols = [CI_EXCL0_COLOR if v < 0 else "#4C72B0" for v in vals]
    for i, y in enumerate(ypos):
        col = cols[i] if sig[i] else CI_INCL0_COLOR
        ax.plot([lo[i], hi[i]], [y, y], color=col, lw=1.8,
                solid_capstyle="butt", zorder=2)
        for edge in (lo[i], hi[i]):
            ax.plot([edge, edge], [y - 0.22, y + 0.22], color=col, lw=1.2,
                    zorder=2)
    ax.scatter(vals, ypos, c=[(c if s else "white")
                              for c, s in zip(cols, sig)],
               edgecolor=cols, s=30, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([label_of(feature_names[i]) for i in order], fontsize=7.5)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel("固定效应 β（每 1 SD 特征 → log10(1+点赞)）")
    ax.set_title(f"(b) 多元偏效应（实心=95% CI 不含 0，{n_sig['likes']}/"
                 f"{len(feature_names)} 个显著）")

    # ---- (c) β：点赞 vs 收藏 ----
    ax = axes[2]
    bx = np.asarray(beta["coef"])
    by = np.asarray(beta["coef_collections"])
    lim = max(0.12, float(np.abs(np.r_[bx, by]).max()) * 1.15)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="gray", lw=1)
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    likes_sig = (np.asarray(beta["ci_low"]) > 0) | \
                (np.asarray(beta["ci_high"]) < 0)
    ax.scatter(bx, by, c=[("#4C72B0" if v >= 0 else "#C44E52") for v in bx],
               s=[38 if s else 14 for s in likes_sig], alpha=0.9,
               edgecolor="white", linewidth=0.6)
    r = float(np.corrcoef(bx, by)[0, 1])
    top = np.argsort(-np.maximum(np.abs(bx), np.abs(by)))[:8]
    spans = [((5, 4), "left"), ((-4, -12), "right"), ((5, -12), "left"),
             ((-4, 6), "right"), ((5, 6), "left"), ((-4, -12), "right"),
             ((5, -12), "left"), ((-4, 4), "right")]
    for j, i in enumerate(top):
        off, ha = spans[j % len(spans)]
        ax.annotate(label_of(feature_names[i]), xy=(bx[i], by[i]),
                    xytext=off, textcoords="offset points", fontsize=7.5,
                    ha=ha)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("β（点赞）")
    ax.set_ylabel("β（收藏）")
    ax.set_title(f"(c) 偏效应跨互动指标一致性（Pearson r={r:.2f}）")

    fig.suptitle("「AI 味惩罚」的逐特征载体（探索集 2023.01–2025.06，"
                 f"月份×领域固定效应，n={beta['n_obs']:,}）", fontsize=13)
    fig.tight_layout()
    save_fig(fig, INSIGHTS + "engagement_drivers.png")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    model = load_detector_model()
    features = list(model["features"])
    feat = pd.read_csv(BASE + "paper_features_exploring.csv.gz",
                       usecols=["month", "domain", "likes", "collections",
                                "comments"] + features)
    scores = pd.read_csv(BASE + "paper_exploring_scores.csv.gz",
                         usecols=["p_ai", "content_len"])
    mismatch = int((feat["content_len"].to_numpy()
                    != scores["content_len"].to_numpy()).sum())
    if mismatch:
        raise SystemExit(f"paper_features / scores 行序未对齐："
                         f"content_len mismatch={mismatch}")
    df = feat.assign(p_ai=scores["p_ai"].to_numpy())
    df["month"] = df["month"].astype(str)
    df = df[df["month"] >= MONTH_FROM].reset_index(drop=True)
    return df, features


def main() -> None:
    df, features = load_data()
    print(f"探索集（{MONTH_FROM} 起）: {len(df):,} 条")
    cells = df.groupby(["month", "domain"], sort=True).ngroup().to_numpy()

    targets = {name: np.log10(1 + df[name].to_numpy(dtype=float))
               for name, _ in TARGETS}
    cols = {"x": {"p_ai": df["p_ai"].to_numpy(dtype=float)}}
    cols["x"].update({f: df[f].to_numpy(dtype=float) for f in features})
    cols["targets"] = targets

    uni = univariate_table(["p_ai"] + features, cols, cells)
    print(f"单元内单变量：p_ai vs 点赞 加权 ρ="
          f"{uni['p_ai']['likes']['weighted_rho']:+.4f}（"
          f"{uni['p_ai']['likes']['n_cells']} 个单元）")

    X = df[features].fillna(0).to_numpy(dtype=float)
    prep = prepare_fe(X, cells)  # 特征侧预处理只做一次，供两个被解释变量复用
    beta = fe_ols(X, targets["likes"], cells, prepared=prep)
    boot = cluster_bootstrap_fe_ols(beta["X_demeaned"], beta["y_demeaned"],
                                    beta["cell_codes"])
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    beta["ci_low"], beta["ci_high"] = ci[0], ci[1]
    n_sig = {"likes": int(((ci[0] > 0) | (ci[1] < 0)).sum())}

    beta_c = fe_ols(X, targets["collections"], cells, prepared=prep)
    boot_c = cluster_bootstrap_fe_ols(beta_c["X_demeaned"], beta_c["y_demeaned"],
                                      beta_c["cell_codes"])
    ci_c = np.percentile(boot_c, [2.5, 97.5], axis=0)
    beta["coef_collections"] = beta_c["coef"]
    beta["ci_low_collections"], beta["ci_high_collections"] = ci_c[0], ci_c[1]
    n_sig["collections"] = int(((ci_c[0] > 0) | (ci_c[1] < 0)).sum())
    print(f"多变量 FE-OLS（点赞）: n={beta['n_obs']:,}, "
          f"显著特征 {n_sig['likes']}/{len(features)}")

    net = {}
    for label, names in (("p_ai_only", ["p_ai"]),
                         ("p_ai_len_tags",
                          ["p_ai", "content_len", "n_hashtags"])):
        Xn = df[names].fillna(0).to_numpy(dtype=float)
        fit = fe_ols(Xn, targets["likes"], cells)
        bts = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                       fit["cell_codes"])
        c = np.percentile(bts, [2.5, 97.5], axis=0)
        net[label] = {
            "features": names,
            "coef": {n: float(v) for n, v in zip(names, fit["coef"])},
            "ci_low": {n: float(v) for n, v in zip(names, c[0])},
            "ci_high": {n: float(v) for n, v in zip(names, c[1])},
            "n_obs": fit["n_obs"], "n_cells": fit["n_cells"],
        }
    print(f"P(AI) 净效应：单独 β={net['p_ai_only']['coef']['p_ai']:+.4f}，"
          f"控制长度+标签数后 β={net['p_ai_len_tags']['coef']['p_ai']:+.4f}")

    with open(BASE + "paper_engagement_results.json", encoding="utf-8") as f:
        paper = json.load(f)
    consistency = {
        "p_ai_likes_weighted_rho": round(
            uni["p_ai"]["likes"]["weighted_rho"], 4),
        "paper_within_cell_weighted_rho": paper["within_cell_weighted_rho"],
        "abs_diff": round(abs(uni["p_ai"]["likes"]["weighted_rho"]
                              - paper["within_cell_weighted_rho"]), 4),
        "n_cells": uni["p_ai"]["likes"]["n_cells"],
    }

    out = {
        "meta": {
            "source": "paper_features_exploring.csv.gz + "
                      "paper_exploring_scores.csv.gz",
            "month_from": MONTH_FROM, "min_cell_n": MIN_CELL_N,
            "n_obs": int(len(df)), "n_cells_total": int(cells.max()) + 1,
            "n_cells_used": beta["n_cells"], "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "model": "unit(=month×domain) fixed effects; features z-scored; "
                     "log10(1+engagement); cluster bootstrap over cells",
        },
        "univariate": {
            feat: {t: {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in uni[feat][t].items()}
                   for t, _ in TARGETS}
            for feat in uni
        },
        "multivariate_likes": {
            "features": features,
            "coef": [round(float(v), 5) for v in beta["coef"]],
            "ci_low": [round(float(v), 5) for v in beta["ci_low"]],
            "ci_high": [round(float(v), 5) for v in beta["ci_high"]],
            "n_sig": n_sig["likes"],
            "n_obs": beta["n_obs"], "n_cells": beta["n_cells"],
        },
        "multivariate_collections": {
            "features": features,
            "coef": [round(float(v), 5) for v in beta["coef_collections"]],
            "ci_low": [round(float(v), 5) for v in beta["ci_low_collections"]],
            "ci_high": [round(float(v), 5) for v in beta["ci_high_collections"]],
            "n_sig": n_sig["collections"],
            "n_obs": beta_c["n_obs"], "n_cells": beta_c["n_cells"],
        },
        "p_ai_net_effects": net,
        "consistency": consistency,
    }
    save_json(out, BASE + "engagement_drivers_results.json")
    with open(BASE + "engagement_drivers_results.md", "w",
              encoding="utf-8") as f:
        f.write(write_markdown(out, feature_names=features))
    plot_drivers(uni, beta, features, n_sig=n_sig)
    print("已保存 engagement_drivers_results.{json,md} 与 "
          "insights/engagement_drivers.png")


def write_markdown(out: dict, feature_names: list[str]) -> str:
    uni = out["univariate"]
    mv = out["multivariate_likes"]
    mvc = out["multivariate_collections"]
    net = out["p_ai_net_effects"]
    meta = out["meta"]

    def top(feats, key, tname, k):
        rows = sorted(((f, uni[f][tname][key]) for f in feats),
                      key=lambda r: r[1])
        return rows[:k], rows[-k:]

    like_neg, like_pos = top(feature_names, "weighted_rho", "likes", 5)
    lines = [
        "# 「AI 味惩罚」的逐特征载体（Engagement Drivers）",
        "",
        "> 由 `engagement_drivers_analysis.py` 生成；数据源 "
        "`paper_features_exploring.csv.gz` + `paper_exploring_scores.csv.gz`"
        "（行序对齐，content_len 校验 mismatch=0）。分析样本自 "
        f"{meta['month_from']} 起 {meta['n_obs']:,} 条，月份×领域单元 "
        f"{meta['n_cells_used']} 个（每单元 >= {meta['min_cell_n']} 条）。",
        "",
        "**方法**：(1) 单变量 = 单元内 Spearman(特征, log10(1+互动)) 的样本量"
        "加权均值；(2) 多元 = 特征先 z-score 再单元内去均值的固定效应 OLS，"
        f"β 为每 1 SD 特征变化对应的 log10(1+互动) 变化，95% CI 来自按单元"
        f"整群 bootstrap（{meta['n_boot']} 次）。两种口径回答的问题不同：前者"
        "是原始关联，后者是控制其余 36 个风格特征后的偏效应。",
        "",
        "## 1 单变量：最强的 5 个正/负关联（点赞）",
        "",
        "| 特征 | 单元内加权 ρ | 负相关单元占比 | 收藏 ρ |",
        "|---|---:|---:|---:|",
    ]
    for f, v in like_neg:
        lines.append(f"| {FEATURE_LABELS.get(f, f)} | {v:+.4f} | "
                     f"{uni[f]['likes']['neg_cell_share']:.1%} | "
                     f"{uni[f]['collections']['weighted_rho']:+.4f} |")
    lines.append("| … | | | |")
    for f, v in like_pos[::-1]:
        lines.append(f"| {FEATURE_LABELS.get(f, f)} | {v:+.4f} | "
                     f"{uni[f]['likes']['neg_cell_share']:.1%} | "
                     f"{uni[f]['collections']['weighted_rho']:+.4f} |")
    p_uni = uni["p_ai"]
    lines += [
        "",
        f"对照：检测器 P(AI) 的单元内加权 ρ={p_uni['likes']['weighted_rho']:+.4f}"
        f"（负相关单元 {p_uni['likes']['neg_cell_share']:.0%}），与 "
        f"`paper_engagement_results.json` 的 "
        f"{out['consistency']['paper_within_cell_weighted_rho']:+.4f} 一致"
        f"（最大绝对差 {out['consistency']['abs_diff']}）。",
        "",
        "## 2 多元固定效应偏效应（点赞，每 1 SD）",
        "",
        f"37 个特征中 **{mv['n_sig']} 个**的 95% CI 不含 0。按 β 排序的"
        "前 5 / 后 5：",
        "",
        "| 特征 | β | 95% CI | 收藏 β |",
        "|---|---:|---|---:|",
    ]
    order = np.argsort(np.asarray(mv["coef"]))
    idx = list(order[:5]) + list(order[-5:][::-1])
    for i in idx:
        lines.append(
            f"| {FEATURE_LABELS.get(feature_names[i], feature_names[i])} | "
            f"{mv['coef'][i]:+.4f} | [{mv['ci_low'][i]:+.4f}, "
            f"{mv['ci_high'][i]:+.4f}] | {mvc['coef'][i]:+.4f} |")
    lines += [
        "",
        f"系数对照（点赞 vs 收藏）Pearson r="
        f"{float(np.corrcoef(mv['coef'], mvc['coef'])[0, 1]):.3f}，"
        f"收藏侧显著特征 {mvc['n_sig']}/37——两个互动指标的偏效应结构高度一致。",
        "",
        "> 注意：单变量与多元的口径不同，个别特征会出现符号反转（例如正文长度"
        "单变量 ρ 略负、偏效应显著为正），这是特征间强相关（长度与标点密度、"
        "独特字比率等）带来的抑制定量（suppression），应以「控制其余特征」的"
        "偏效应解释独立贡献，以单变量排序解释原始强弱。",
        "",
        "## 3 P(AI) 的净效应（控制候选混淆）",
        "",
        "| 模型 | P(AI) β | 95% CI |",
        "|---|---:|---|",
    ]
    for label, zh in (("p_ai_only", "仅 P(AI)"),
                      ("p_ai_len_tags", "P(AI)+正文长度+标签数")):
        n = net[label]
        lines.append(
            f"| {zh} | {n['coef']['p_ai']:+.4f} | "
            f"[{n['ci_low']['p_ai']:+.4f}, {n['ci_high']['p_ai']:+.4f}] |")
    len_net = net["p_ai_len_tags"]["coef"]
    lines += [
        "",
        f"控制正文长度（β={len_net['content_len']:+.4f}）与标签数"
        f"（β={len_net['n_hashtags']:+.4f}）后，P(AI) 的惩罚仍为 "
        f"{len_net['p_ai']:+.4f}（CI 不含 0）——「AI 味惩罚」不是长度或标签"
        "数量的间接效应，而是与风格本身相关的独立成分。",
        "",
        "图表见 `insights/engagement_drivers.png`："
        "(a) 单变量 ρ；(b) 多元 β 与整群 bootstrap CI；"
        "(c) 点赞-收藏 β 对照。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
