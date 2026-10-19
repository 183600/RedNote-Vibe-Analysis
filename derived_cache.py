# -*- coding: utf-8 -*-
"""fork 管线共享的 JSONL 派生记录缓存（只读投影，可选加速层）。

同一条 95MB 的 exploring JSONL 会被十多个分析步骤分别打开、逐行
`json.loads`；每个步骤只读取少量字段。本模块把每条记录投影为分析用到的
字段集合，一次性序列化为分块 pickle 派生物；后续步骤经 `iter_cached()`
优先读取派生物，回退路径始终是原始的 `analysis_utils.iter_jsonl`。

正确性约定：
- 派生物只做字段白名单投影，不改写任何值；投影保留字段原名，因此
  `content_of / title_of / engagement_of / parse_local_time / post_text`
  等既有取值函数对读出的记录行为与逐行解析完全一致。
- 指纹 =（文件大小, mtime_ns）。构建只由 `python3 derived_cache.py`
  显式执行（fork 管线第一步），读取方只读不建；指纹不符、清单/分块缺失
  或损坏时自动回退 `iter_jsonl`，结果与无缓存时逐位一致。
- 缓存落在系统临时目录（不进入仓库、不提交），同一工作目录同一轮
  运行内供所有步骤共享；下载/解压导致 mtime 变化只会造成一次重建。
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import tempfile
from typing import Iterator

from analysis_utils import BASE, iter_jsonl

# 各分析脚本从记录中读取的全部字段（白名单投影的并集）。
PROJECT_KEYS = (
    "note_title", "note_content", "desc", "domain", "local_time", "time",
    "liked_count", "collected_count", "comments_count",
    "likes", "collections", "comments",
    "note_id", "user_id", "model_family", "model",
)

DATASETS = (
    "training_set_human.jsonl",
    "training_set_human_fixed.jsonl",
    "training_set_aigc.jsonl",
    "exploring_set.jsonl",
    "exploring_set_fixed.jsonl",
)

CHUNK = 4096


def _fingerprint(path: str) -> str:
    st = os.stat(path)
    return f"{st.st_size:x}-{st.st_mtime_ns:x}"


def _dataset_dir(path: str) -> str:
    h = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), "rednote_derived_cache",
                        f"{os.path.basename(path)}-{h}")


def build(path: str) -> int:
    """为单个 JSONL 构建派生缓存；已有同指纹缓存时直接跳过并返回记录数。"""
    fp = _fingerprint(path)
    d = _dataset_dir(path)
    os.makedirs(d, exist_ok=True)
    manifest = os.path.join(d, f"{fp}.json")
    if os.path.exists(manifest):
        return -1

    n = 0
    k = 0
    buf: list[dict] = []
    for rec in iter_jsonl(path):
        buf.append({key: rec[key] for key in PROJECT_KEYS if key in rec})
        n += 1
        if len(buf) >= CHUNK:
            with open(os.path.join(d, f"{fp}-{k:05d}.pkl"), "wb") as f:
                pickle.dump(buf, f, protocol=pickle.HIGHEST_PROTOCOL)
            buf = []
            k += 1
    if buf or n == 0:
        with open(os.path.join(d, f"{fp}-{k:05d}.pkl"), "wb") as f:
            pickle.dump(buf, f, protocol=pickle.HIGHEST_PROTOCOL)
        k += 1

    # 清单最后原子发布：读者只看到「分块齐全」的完整缓存。
    tmp = manifest + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"chunks": k, "n": n, "size": os.stat(path).st_size}, f)
    os.replace(tmp, manifest)
    return n


def iter_cached(path: str) -> Iterator[dict]:
    """迭代单条记录的投影字段；缓存无效时回退 `iter_jsonl`（结果一致）。"""
    try:
        fp = _fingerprint(path)
        d = _dataset_dir(path)
        with open(os.path.join(d, f"{fp}.json"), encoding="utf-8") as f:
            info = json.load(f)
        files = [os.path.join(d, f"{fp}-{i:05d}.pkl")
                 for i in range(int(info["chunks"]))]
        if not all(os.path.exists(p) for p in files):
            raise FileNotFoundError(path)
    except (OSError, ValueError, KeyError, TypeError):
        yield from iter_jsonl(path)
        return
    yielded = 0
    for i, p in enumerate(files):
        try:
            with open(p, "rb") as f:
                recs = pickle.load(f)
            if not isinstance(recs, list):
                raise ValueError(p)
        except (OSError, ValueError, EOFError, AttributeError, ImportError,
                pickle.UnpicklingError):
            if yielded or i:
                raise
            yield from iter_jsonl(path)
            return
        yielded += len(recs)
        yield from recs


def main() -> int:
    for name in DATASETS:
        path = BASE + name
        if not os.path.exists(path):
            print(f"[cache] 跳过（不存在）: {name}")
            continue
        n = build(path)
        if n < 0:
            print(f"[cache] 命中 {name}")
        else:
            print(f"[cache] 已构建 {name}: {n} 条")
    print("派生记录缓存就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
