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


def iter_jsonl_relaxed(path: str) -> Iterator[dict]:
    """流式读取 JSONL，跳过空行与无法解析的行（历史探索脚本的容错口径）。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_jsonl_codecs(path: str) -> Iterator[dict]:
    """流式读取 JSONL，保持 `codecs.open` 的按行切分语义（历史脚本口径）。

    codecs.StreamReader.readline 会在 \\u2028/\\u2029/\\x0b 等 Unicode 行边界
    切分，行为与内置 open 的通用换行不同；为保持既有统计结果，这里沿用
    codecs 读取并丢弃无法解析的片段。
    """
    import codecs

    with codecs.open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line.strip())
            except Exception:
                continue


def _stream_sample(factory, sample: int | None, seed: int):
    """两遍流式抽样，结果与 `random.Random(seed).sample(list(items), sample)` 完全一致。

    第一遍只计数（O(1) 内存），第二遍仅解析被抽中的记录。索引抽样使用
    `sample(range(n), k)`，与对等长列表抽样走完全相同的随机数路径。
    """
    n = 0
    if sample is None:
        items = list(factory())
        return items, len(items)
    for _ in factory():
        n += 1
    if n <= sample:
        return list(factory()), n
    idx = random.Random(seed).sample(range(n), sample)
    wanted = set(idx)
    picked: dict[int, dict] = {}
    for i, item in enumerate(factory()):
        if i in wanted:
            picked[i] = item
            if len(picked) == len(wanted):
                break
    return [picked[i] for i in idx], n


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


def content_of(rec: dict) -> str:
    """统一取正文：兼容 note_content / desc 两种字段（任一非空即可）。"""
    return rec.get("note_content") or rec.get("desc") or ""


def clean_body(content: str | None) -> str:
    """与 paper_features.py 相同的正文清洗口径。

    删除 `[话题]` 字面标记（导出残留）、把 `#话题` 标签替换为空格并去掉首尾空白。
    复用 `paper_features` 中的同一组正则，保证所有分析跨脚本口径一致。
    """
    from paper_features import HASHTAG, HUATI

    body = HUATI.sub("", content or "")
    return HASHTAG.sub(" ", body).strip()


def shannon_entropy(counts, miller_madow: bool = False) -> float:
    """离散分布的香农熵（bit）。miller_madow=True 时加小样本校正。

    经验熵随样本量增大而系统性偏高（能观测到更多稀有变体），跨不同样本量的
    比较应使用 Miller–Madow 校正：H_MM = H + (K-1)/(2N ln2)。
    """
    arr = np.asarray(list(counts.values()) if isinstance(counts, dict)
                     else list(counts), dtype=float)
    arr = arr[arr > 0]
    n = float(arr.sum())
    if n <= 0:
        return 0.0
    p = arr / n
    h = float(-(p * np.log2(p)).sum())
    if miller_madow and n > 1:
        h += (len(arr) - 1) / (2.0 * n * np.log(2.0))
    return h


def topk_coverage(counts, k: int) -> float:
    """前 k 个最高频项覆盖的计数占比（0~1）；衡量分布集中度/模板化程度。"""
    arr = np.asarray(list(counts.values()) if isinstance(counts, dict)
                     else list(counts), dtype=float)
    arr = arr[arr > 0]
    n = float(arr.sum())
    if n <= 0 or k <= 0:
        return 0.0
    return float(np.sort(arr)[::-1][:k].sum() / n)


def load_posts(path: str, with_title: bool = True, sample: int | None = None,
               seed: int = 42) -> list[str]:
    """读取一个 JSONL 数据集为文本列表；可复现地随机抽样（流式，仅解析抽中记录）。"""
    recs, _ = _stream_sample(lambda: iter_jsonl(path), sample, seed)
    return [post_text(rec, with_title) for rec in recs]


def load_records(path: str, sample: int | None = None,
                 seed: int = 42) -> list[dict]:
    """读取一个 JSONL 数据集为记录列表；可复现地随机抽样（同 load_posts 的抽样口径）。"""
    return _stream_sample(lambda: iter_jsonl(path), sample, seed)[0]


def load_records_with_total(path: str, sample: int | None = None,
                            seed: int = 42, iterator=iter_jsonl_relaxed
                            ) -> tuple[list[dict], int]:
    """流式读取 JSONL 并返回 (记录列表, 总记录数)。

    容错口径与历史探索脚本一致：跳过空行与无法解析的行。抽样结果与
    `random.Random(seed).sample(完整记录列表, sample)` 完全一致，但内存占用
    与 sample 成正比而非与文件大小成正比。
    """
    return _stream_sample(lambda: iterator(path), sample, seed)


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
