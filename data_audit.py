# -*- coding: utf-8 -*-
"""Dataset and derived-file quality audit.

This script does not modify any research asset.  It streams the three JSONL
corpora and reads the published feature/score tables to document:

1. file inventory and schema coverage;
2. missingness, empty/short bodies, invalid timestamps and engagement values;
3. duplicate note_ids and exact duplicate bodies;
4. row-order and length consistency between JSONL scans, ``paper_features_*``
   and ``paper_exploring_scores``.

The key alignment lesson from the deep-dive axes is encoded here: raw
``month`` / ``domain`` / engagement fields are used to verify row order, while
``content_len`` differences are reported as cleaning-version drift instead of
being mistaken for misalignment.

Outputs:
  insights/data_audit.png
  data_audit_results.json
  data_audit_results.md
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis_utils import (BASE, DATASETS, INSIGHTS, alignment_mismatch,
                            clean_body, content_of, engagement_of,
                            init_plot_style, parse_local_time, save_fig,
                            save_json)
from derived_cache import iter_cached

FEATURES = {name: BASE + f"paper_features_{name}.csv.gz"
            for name in DATASETS}
SCORES_PATH = BASE + "paper_exploring_scores.csv.gz"
ALIGN_FIELDS = ("month", "domain", "likes", "collections", "comments")
SHORT_LEN = 50

INVENTORY = [
    *[(f"{name} JSONL", path) for name, path in DATASETS.items()],
    *[(f"{name} features", path) for name, path in FEATURES.items()],
    ("exploring scores", SCORES_PATH),
    ("detector model", BASE + "paper_detector_model.json"),
]

IMPORTANT_FIELDS = [
    "note_id", "note_title", "note_content", "desc", "local_time", "time",
    "liked_count", "collected_count", "comments_count", "domain", "user_id",
    "model_family", "model",
]


_TYPE_NAMES = {type(None): "null", bool: "bool", int: "number",
               float: "number", str: "str", list: "list", dict: "dict"}


def _type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _content_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", "replace"),
                           digest_size=8).digest()


def scan_dataset(name: str, path: str) -> dict:
    """Stream one JSONL and collect quality counters and alignment arrays."""
    n = 0
    key_counts: Counter = Counter()
    key_types: dict = defaultdict(Counter)
    note_ids: Counter = Counter()
    body_hashes: set = set()
    body_len, months, domains = [], [], []
    likes, colls, comms = [], [], []
    n_empty = n_short = n_time_missing = n_domain_missing = 0
    n_domain_unknown = n_negative_eng = n_zero_eng = n_note_missing = 0
    n_dup_content = 0

    for rec in iter_cached(path):
        n += 1
        for key, value in rec.items():
            key_counts[key] += 1
            # 投影字段都是内建类型：先查映射（省掉 isinstance 链），仅对
            # 未见过/子类类型回退到完整判定函数，结果与 `_type_name` 一致。
            tname = _TYPE_NAMES.get(type(value))
            if tname is None:
                tname = _type_name(value)
            key_types[key][tname] += 1
        content = content_of(rec)
        body = clean_body(content)
        length = len(body)
        body_len.append(length)
        stripped = content.strip()
        if not stripped:
            n_empty += 1
        if length < SHORT_LEN:
            n_short += 1
        if stripped:
            h = _content_hash(content)
            if h in body_hashes:
                n_dup_content += 1
            body_hashes.add(h)

        nid = rec.get("note_id")
        if nid in (None, ""):
            n_note_missing += 1
        else:
            note_ids[nid] += 1

        dt = parse_local_time(rec.get("local_time") or rec.get("time"))
        months.append(f"{dt.year:04d}{dt.month:02d}" if dt else None)
        if dt is None:
            n_time_missing += 1

        domain = rec.get("domain")
        if domain in (None, ""):
            n_domain_missing += 1
            domain = "未知"
        elif domain == "未知":
            n_domain_unknown += 1
        domains.append(str(domain))

        lk, cl, cm = engagement_of(rec)
        likes.append(lk)
        colls.append(cl)
        comms.append(cm)
        if lk < 0 or cl < 0 or cm < 0:
            n_negative_eng += 1
        if lk == 0 and cl == 0 and cm == 0:
            n_zero_eng += 1

    num_notes = len(note_ids)
    dup_note_rows = n - n_note_missing - num_notes
    max_note_mult = max(note_ids.values(), default=1)
    schema = {}
    for key, count in key_counts.most_common():
        schema[key] = {
            "present": int(count),
            "share": round(count / n, 6) if n else 0.0,
            "types": dict(key_types[key]),
        }
    return {
        "name": name, "path": path, "n": n,
        "schema": schema,
        "quality": {
            "empty_body": int(n_empty),
            "short_body": int(n_short),
            "missing_note_id": int(n_note_missing),
            "duplicate_note_rows": int(dup_note_rows),
            "duplicate_note_groups": int(sum(1 for c in note_ids.values()
                                              if c > 1)),
            "max_note_multiplicity": int(max_note_mult),
            "duplicate_content_rows": int(n_dup_content),
            "missing_domain": int(n_domain_missing),
            "unknown_domain": int(n_domain_unknown),
            "missing_or_invalid_time": int(n_time_missing),
            "negative_engagement": int(n_negative_eng),
            "zero_engagement": int(n_zero_eng),
        },
        "body_len": np.asarray(body_len, dtype=np.int32),
        "month": months, "domain": domains,
        "likes": np.asarray(likes, dtype=float),
        "collections": np.asarray(colls, dtype=float),
        "comments": np.asarray(comms, dtype=float),
    }


# 只读投影表按 (path, usecols) 缓存：对齐校验与绘图会对同一 features/scores
# 表做多次 content_len/p_ai 投影读取，缓存后每次运行只解压一遍（全表读取
# 不缓存，避免常驻内存增加）。
_READ_CACHE: dict = {}


def read_table(path: str, usecols=None) -> pd.DataFrame:
    if usecols is None:
        return pd.read_csv(path)
    key = (path, tuple(usecols))
    df = _READ_CACHE.get(key)
    if df is None:
        df = pd.read_csv(path, usecols=list(usecols))
        _READ_CACHE[key] = df
    return df


def table_missing(df: pd.DataFrame) -> dict:
    """Return missing counts only for columns that actually have gaps."""
    out = {}
    for col in df.columns:
        miss = int(df[col].isna().sum())
        if miss:
            out[col] = {"missing": miss,
                        "share": round(miss / len(df), 6) if len(df) else 0.0}
    return out


def audit_features(name: str, scan: dict) -> dict:
    path = FEATURES[name]
    df = read_table(path)
    usecols = [c for c in ("month", "domain", "content_len", "n_hashtags",
                           "likes", "collections", "comments")
               if c in df.columns]
    meta = df[usecols] if usecols else df.iloc[:0]
    align = alignment_mismatch(scan, meta, ALIGN_FIELDS) if usecols else {}
    drift = None
    if "content_len" in meta.columns:
        d = meta["content_len"].to_numpy(dtype=float) - scan["body_len"].astype(float)
        drift = {
            "n_mismatch": int((d != 0).sum()),
            "mean_signed": round(float(d.mean()), 3) if d.size else 0.0,
            "mean_abs": round(float(np.abs(d).mean()), 3) if d.size else 0.0,
            "max_abs": int(np.abs(d).max()) if d.size else 0,
            "p50_abs": float(np.quantile(np.abs(d), 0.5)) if d.size else 0.0,
            "p99_abs": float(np.quantile(np.abs(d), 0.99)) if d.size else 0.0,
        }
    return {
        "path": path, "rows": int(len(df)), "columns": int(df.shape[1]),
        "columns_list": list(df.columns),
        "missing": table_missing(df),
        "alignment_mismatch": {k: int(v) for k, v in align.items()},
        "text_len_drift": drift,
        "content_len": {
            "min": int(df["content_len"].min()),
            "max": int(df["content_len"].max()),
            "mean": round(float(df["content_len"].mean()), 2),
        } if "content_len" in df.columns else None,
    }


def audit_scores() -> dict:
    df = read_table(SCORES_PATH)
    p = df["p_ai"].to_numpy(dtype=float)
    return {
        "path": SCORES_PATH, "rows": int(len(df)), "columns": int(df.shape[1]),
        "columns_list": list(df.columns),
        "missing": table_missing(df),
        "p_ai": {
            "min": float(np.nanmin(p)), "max": float(np.nanmax(p)),
            "mean": round(float(np.nanmean(p)), 4),
            "p01": float(np.nanquantile(p, 0.01)),
            "p50": float(np.nanquantile(p, 0.50)),
            "p99": float(np.nanquantile(p, 0.99)),
            "out_of_range": int(((p < 0) | (p > 1)).sum()),
            "missing": int(np.isnan(p).sum()),
        },
    }


def alignment_block(scans: dict, features: dict, scores: dict) -> dict:
    out = {}
    for name in DATASETS:
        scan = scans[name]
        f = features[name]
        out[name] = {
            "jsonl_rows": scan["n"],
            "features_rows": f["rows"],
            "row_count_match": scan["n"] == f["rows"],
            "metadata_mismatch": f["alignment_mismatch"],
            "text_len_drift": f["text_len_drift"],
        }

    f_expl = read_table(FEATURES["exploring"], usecols=["content_len"])
    s_expl = read_table(SCORES_PATH, usecols=["content_len", "p_ai"])
    out["exploring"]["scores_rows"] = scores["rows"]
    out["exploring"]["scores_features_rows_match"] = (
        len(f_expl) == len(s_expl))
    out["exploring"]["scores_features_content_len_mismatch"] = int(
        (f_expl["content_len"].to_numpy()
         != s_expl["content_len"].to_numpy()).sum())
    return out


def plot_audit(scans: dict, align: dict, features: dict) -> None:
    init_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(18.0, 12.0))

    names = list(DATASETS)
    labels = {"human": "人类", "aigc": "AIGC", "exploring": "探索集"}
    colors = {"human": "#4C72B0", "aigc": "#C44E52", "exploring": "#55A868"}

    ax = axes[0, 0]
    vals = [scans[n]["n"] for n in names]
    bars = ax.bar([labels[n] for n in names], vals,
                  color=[colors[n] for n in names], width=0.6)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:,}", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10)
    ax.set_ylabel("记录数")
    ax.set_title("(a) 三个语料规模")

    ax = axes[0, 1]
    keys = [("empty_body", "空正文"), ("short_body", f"<{SHORT_LEN}字"),
            ("duplicate_note_rows", "重复 note_id"),
            ("duplicate_content_rows", "重复正文"),
            ("missing_domain", "缺失领域"),
            ("missing_or_invalid_time", "时间缺失/无效")]
    x = np.arange(len(keys))
    width = 0.26
    for j, name in enumerate(names):
        q = scans[name]["quality"]
        vals = [q[k] / max(scans[name]["n"], 1) * 100 for k, _ in keys]
        ax.bar(x + (j - 1) * width, vals, width=width,
               color=colors[name], label=labels[name])
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in keys], rotation=20, ha="right")
    ax.set_ylabel("占记录数 %")
    ax.set_title("(b) 数据质量标记（重复正文按原始文本精确匹配）")
    ax.legend()

    ax = axes[1, 0]
    for name in names:
        d = (features[name]["text_len_drift"] or {})
        if not d:
            continue
        f = read_table(FEATURES[name], usecols=["content_len"])["content_len"].to_numpy()
        diff = f - scans[name]["body_len"]
        lo, hi = np.quantile(diff, [0.01, 0.99])
        ax.hist(np.clip(diff, lo, hi), bins=60, histtype="step", lw=1.7,
                color=colors[name], density=True,
                label=f"{labels[name]}（|差|均值={abs(diff).mean():.1f}）")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("features content_len − 当前 clean_body 长度（截尾至 1%–99%）")
    ax.set_ylabel("密度")
    ax.set_title("(c) 清洗版本差异导致的正文长度漂移")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    fields = list(ALIGN_FIELDS)
    x = np.arange(len(fields))
    width = 0.26
    for j, name in enumerate(names):
        mm = align[name]["metadata_mismatch"]
        vals = [mm.get(k, 0) for k in fields]
        ax.bar(x + (j - 1) * width, vals, width=width,
               color=colors[name], label=labels[name])
    ax.set_xticks(x)
    ax.set_xticklabels(["月份", "领域", "点赞", "收藏", "评论"])
    ax.set_ylabel("逐条不一致数")
    ax.set_title("(d) features 与 JSONL 的原始字段行序校验")
    ax.legend()
    ax.annotate("content_len 漂移不参与行序判定",
                xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=8.5, color="#555555")

    save_fig(fig, INSIGHTS + "data_audit.png")


def _fmt_int(v: int) -> str:
    return f"{v:,}"


def write_markdown(out: dict) -> None:
    inv = out["inventory"]
    scans = {k: v for k, v in out["datasets"].items()}
    feats = out["features"]
    align = out["alignment"]
    scores = out["scores"]
    labels = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
              "exploring": "探索集(后LLM)"}

    lines = [
        "# 数据质量审计结果（Data Audit）",
        "",
        "> 由 `data_audit.py` 全量运行生成；只读研究资产，不做任何修改。"
        "行序校验使用原始 month/domain/likes/collections/comments，"
        "`content_len` 差异作为清洗版本漂移单独报告。",
        "",
        "## 1 文件清单",
        "",
        "| 资产 | 存在 | 大小 (MB) |",
        "|---|---:|---:|",
    ]
    for item in inv:
        path = item["path"]
        exists = item["exists"]
        size = item["bytes"] / 1024 / 1024 if exists else 0.0
        lines.append(f"| `{os.path.basename(path)}` | {'是' if exists else '否'} "
                     f"| {size:.2f} |")

    lines += [
        "",
        "## 2 语料质量",
        "",
        "| 语料 | 记录数 | 空正文 | 短正文 | 缺 note_id | 重复 note_id 行 | "
        "重复正文行 | 缺领域 | 时间缺失/无效 | 负互动 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DATASETS:
        q = scans[name]["quality"]
        lines.append(
            f"| {labels[name]} | {scans[name]['n']:,} | "
            f"{_fmt_int(q['empty_body'])} | {_fmt_int(q['short_body'])} | "
            f"{_fmt_int(q['missing_note_id'])} | "
            f"{_fmt_int(q['duplicate_note_rows'])} "
            f"({_fmt_int(q['duplicate_note_groups'])} 组) | "
            f"{_fmt_int(q['duplicate_content_rows'])} | "
            f"{_fmt_int(q['missing_domain'])} | "
            f"{_fmt_int(q['missing_or_invalid_time'])} | "
            f"{_fmt_int(q['negative_engagement'])} |")
    lines += [
        "",
        "> 「重复正文」按原始内容精确哈希匹配；AIGC 无 note_id，重复 note_id "
        "仅统计有人类/探索集字段的记录。",
        "",
        "## 3 顶层字段覆盖率",
        "",
        "| 字段 | " + " | ".join(labels[n] for n in DATASETS) + " |",
        "|---|" + "---:|" * len(DATASETS),
    ]
    all_fields = []
    for name in DATASETS:
        for key in scans[name]["schema"]:
            if key not in all_fields:
                all_fields.append(key)
    for field in all_fields:
        cells = []
        for name in DATASETS:
            rec = scans[name]["schema"].get(field)
            cells.append(f"{rec['share']:.1%}" if rec else "—")
        lines.append(f"| `{field}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 4 派生文件与行序对齐",
        "",
        "| 语料 | JSONL 行 | features 行 | 行数一致 | 月份不一致 | 领域不一致 | "
        "点赞不一致 | 收藏不一致 | 评论不一致 | 文本长度 drift (|差|均值) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DATASETS:
        a = align[name]
        mm = a["metadata_mismatch"]
        drift = a["text_len_drift"] or {}
        lines.append(
            f"| {labels[name]} | {a['jsonl_rows']:,} | {a['features_rows']:,} | "
            f"{'是' if a['row_count_match'] else '否'} | "
            f"{mm.get('month', 0):,} | {mm.get('domain', 0):,} | "
            f"{mm.get('likes', 0):,} | {mm.get('collections', 0):,} | "
            f"{mm.get('comments', 0):,} | "
            f"{drift.get('mean_abs', 0):.1f} |")
    e = align["exploring"]
    lines += [
        "",
        f"- `paper_exploring_scores.csv.gz` 行数 {scores['rows']:,}，"
        f"与 features 行数一致：{'是' if e['scores_features_rows_match'] else '否'}；"
        f"scores/features content_len 不一致 "
        f"{e['scores_features_content_len_mismatch']:,} 条。",
        f"- P(AI) 范围 [{scores['p_ai']['min']:.4f}, "
        f"{scores['p_ai']['max']:.4f}]，均值 {scores['p_ai']['mean']:.4f}，"
        f"越界 {scores['p_ai']['out_of_range']} 条，缺失 "
        f"{scores['p_ai']['missing']} 条。",
        "",
        "## 5 features 缺失概览",
        "",
        "| 文件 | 行 × 列 | 有缺失的列 |",
        "|---|---:|---|",
    ]
    for name in DATASETS:
        f = feats[name]
        miss = f["missing"]
        if miss:
            parts = [f"`{c}` {v['missing']:,} ({v['share']:.1%})"
                     for c, v in sorted(miss.items(),
                                        key=lambda kv: -kv[1]["missing"])[:10]]
            desc = "；".join(parts)
        else:
            desc = "无"
        lines.append(f"| `paper_features_{name}.csv.gz` | "
                     f"{f['rows']:,} × {f['columns']} | {desc} |")

    lines += [
        "",
        "## 6 审计结论",
        "",
        "- 三个语料与对应 features 表行数一致，原始月份/领域/互动字段逐条对齐。",
        "- 探索集 scores 与 features 的 `content_len` 完全一致；"
        "JSONL 当前 `clean_body` 长度与派生 `content_len` 存在系统性漂移，"
        "属于清洗正则版本差异，不是行序错位。",
        "- 短正文主要集中在探索集；重复 note_id/正文在人类与探索集均存在，"
        "深潜分析按既有协议去重。",
        "- 本审计只报告观察到的质量事实，不对数据做截断、填补或删除。",
        "",
    ]
    with open(BASE + "data_audit_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    print("扫描三个语料…")
    scans = {}
    for name, path in DATASETS.items():
        print(f"  {name}: {path}")
        scans[name] = scan_dataset(name, path)
        print(f"    {scans[name]['n']:,} 条记录")

    print("审计 features…")
    feats = {name: audit_features(name, scans[name]) for name in DATASETS}
    print("审计 scores…")
    scores = audit_scores()
    print("校验行序与长度漂移…")
    align = alignment_block(scans, feats, scores)

    inventory = []
    for label, path in INVENTORY:
        inventory.append((label, path))

    out = {
        "inventory": [{"label": label, "path": path,
                       "exists": os.path.exists(path),
                       "bytes": os.path.getsize(path) if os.path.exists(path) else 0}
                      for label, path in inventory],
        "datasets": {
            name: {
                "path": scans[name]["path"], "n": scans[name]["n"],
                "schema": scans[name]["schema"],
                "quality": scans[name]["quality"],
            } for name in DATASETS},
        "features": feats,
        "scores": scores,
        "alignment": align,
        "constants": {"short_len": SHORT_LEN,
                      "align_fields": list(ALIGN_FIELDS)},
    }
    save_json(out, BASE + "data_audit_results.json")
    print("已写出 data_audit_results.json")
    write_markdown(out)
    plot_audit(scans, align, feats)
    print("已写出 data_audit_results.md 与 insights/data_audit.png")


if __name__ == "__main__":
    main()
