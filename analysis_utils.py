# -*- coding: utf-8 -*-
"""公共工具模块：数据路径、JSONL 流式读取、中文字体、检测器打分与统计辅助。

所有分析脚本共用同一份实现，避免每个脚本各自复制粘贴加载/字体/打分逻辑。
"""
from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator
from datetime import datetime

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__)) + "/"
FIGS = BASE + "paper_figs/"
INSIGHTS = BASE + "insights/"

FEATURE_META = ["dataset", "domain", "month", "model_family", "model",
                "likes", "collections", "comments"]

CONTENT_KEYS = ("note_content", "desc")

# 三个子集统一的绘图配色（Human / AIGC / 后 LLM 探索集）
PALETTE = {"human": "#4C72B0", "aigc": "#C44E52", "exploring": "#55A868"}

_CJK_FALLBACKS = ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
                  "Source Han Sans SC", "WenQuanYi Micro Hei", "SimHei")


def setup_chinese_fonts(font_dir: str = "/usr/share/fonts/noto-cjk") -> str | None:
    """让 matplotlib 能渲染中文；返回注册成功的字体名，找不到时返回 None。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    if os.path.isdir(font_dir):
        for f in sorted(os.listdir(font_dir)):
            if "NotoSansCJK" in f and f.endswith(".ttc"):
                path = os.path.join(font_dir, f)
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                plt.rcParams["font.family"] = name
                return name
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FALLBACKS:
        if name in installed:
            plt.rcParams["font.family"] = name
            return name
    return None


def init_plot_style() -> str | None:
    """初始化统一的绘图风格（中文字体 + 一致的 rcParams）；返回字体名。"""
    import matplotlib.pyplot as plt

    font = setup_chinese_fonts()
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    return font


def save_fig(fig, path: str, dpi: int = 150) -> str:
    """紧凑布局保存图片并关闭；自动创建输出目录，返回路径。"""
    import matplotlib.pyplot as plt

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def iter_jsonl(path: str) -> Iterator[dict]:
    """流式读取 JSONL，逐行解析，避免把大文件一次性载入内存。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def post_text(rec: dict, with_title: bool = True,
              content_keys: tuple[str, ...] = CONTENT_KEYS) -> str:
    """从一条记录中取正文（可选拼接标题），兼容 note_content / desc 两种字段名。"""
    content = ""
    for k in content_keys:
        v = rec.get(k)
        if v:
            content = v
            break
    if with_title:
        return (rec.get("note_title") or "") + "\n" + content
    return content


def load_posts(path: str, with_title: bool = True, sample: int | None = None,
               seed: int = 42) -> list[str]:
    """读取一个 JSONL 数据集为文本列表；可复现地随机抽样。"""
    texts = [post_text(rec, with_title) for rec in iter_jsonl(path)]
    if sample is not None and len(texts) > sample:
        texts = random.Random(seed).sample(texts, sample)
    return texts


def load_records(path: str, sample: int | None = None,
                 seed: int = 42) -> list[dict]:
    """读取一个 JSONL 数据集为记录列表；可复现地随机抽样（同 load_posts 的抽样口径）。"""
    recs = list(iter_jsonl(path))
    if sample is not None and len(recs) > sample:
        recs = random.Random(seed).sample(recs, sample)
    return recs


def to_int(v, default: int = 0) -> int:
    """把 '123' / 123 / None 等混用的计数字段安全地转为整数。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def engagement_of(rec: dict) -> tuple[int, int, int]:
    """读取一条记录的（点赞, 收藏, 评论），兼容两种字段命名。"""
    return (
        to_int(rec.get("liked_count") if rec.get("liked_count") is not None
               else rec.get("likes")),
        to_int(rec.get("collected_count") if rec.get("collected_count") is not None
               else rec.get("collections")),
        to_int(rec.get("comments_count") if rec.get("comments_count") is not None
               else rec.get("comments")),
    )


def parse_local_time(value) -> datetime | None:
    """解析数据集的本地时间字段（'YYYYMMDDHH'，兼容 8 位日期）；失败返回 None。"""
    s = str(value or "").strip()
    if not s.isdigit():
        return None
    try:
        if len(s) >= 10:
            return datetime.strptime(s[:10], "%Y%m%d%H")
        if len(s) == 8:
            return datetime.strptime(s, "%Y%m%d")
    except ValueError:
        return None
    return None


def hour_of(value) -> int | None:
    """本地小时（0–23），解析失败返回 None。"""
    dt = parse_local_time(value)
    return dt.hour if dt is not None else None


def weekday_of(value) -> int | None:
    """本地星期（0=周一 … 6=周日），解析失败返回 None。"""
    dt = parse_local_time(value)
    return dt.weekday() if dt is not None else None


def load_features(dataset: str, base: str = BASE) -> pd.DataFrame:
    """读取 paper_features_{dataset}.csv.gz 并规范化 month 列。"""
    df = pd.read_csv(base + f"paper_features_{dataset}.csv.gz")
    if "month" in df.columns:
        df["month"] = df["month"].astype(str)
    return df


def load_detector_model(base: str = BASE) -> dict:
    with open(base + "paper_detector_model.json", encoding="utf-8") as f:
        return json.load(f)


def detector_score(df: pd.DataFrame, model: dict, robust: bool = True) -> np.ndarray:
    """用保存的逻辑回归参数对 DataFrame 打分；robust=True 用稳健特征模型。"""
    if robust:
        feats = model["robust_features"]
        mean = model["robust_scaler_mean"]
        scale = model["robust_scaler_scale"]
        coef = model["robust_coef"]
        b = model["robust_intercept"]
    else:
        feats = model["features"]
        mean = model["scaler_mean"]
        scale = model["scaler_scale"]
        coef = model["coef"]
        b = model["intercept"]
    z = (df[feats].fillna(0).values - np.asarray(mean)) / np.asarray(scale)
    logit = z @ np.asarray(coef) + b
    return 1.0 / (1.0 + np.exp(-logit))


def cliffs_delta_from_u(u: float, n1: int, n2: int) -> float:
    return 2.0 * u / (n1 * n2) - 1.0


def cliffs_delta(a, b) -> tuple[float, float]:
    """返回 (Cliff's delta, Mann-Whitney U 双侧 p 值)。"""
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return cliffs_delta_from_u(u, len(a), len(b)), float(p)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(center - half, 0.0)), float(min(center + half, 1.0)))


def bootstrap_auc(y, proba, n_boot: int = 1000, seed: int = 42) -> dict:
    """对已有点预测做 bootstrap，返回 AUC 的 95% 置信区间。"""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    proba = np.asarray(proba)
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        pos_val = yb.max()
        if yb.min() == pos_val:
            continue
        pb = proba[idx]
        order = np.argsort(pb)
        ps = pb[order]
        ys = yb[order]
        starts = np.flatnonzero(ps[1:] != ps[:-1]) + 1
        starts = np.concatenate(([0], starts))
        sizes = np.diff(np.append(starts, n))
        avg_ranks = starts + (sizes + 1) / 2.0
        pos_cnt = np.add.reduceat((ys == pos_val).astype(np.int64), starts)
        n1 = float(pos_cnt.sum())
        n0 = n - n1
        aucs.append(
            (float(np.dot(pos_cnt, avg_ranks)) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        )
    aucs = np.asarray(aucs)
    return {
        "auc": float(roc_auc_score(y, proba)),
        "ci_low": float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)),
        "n_boot": int(len(aucs)),
    }


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
