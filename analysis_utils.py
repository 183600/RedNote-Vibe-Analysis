# -*- coding: utf-8 -*-
"""公共工具模块：数据路径、JSONL 流式读取、中文字体、检测器打分与统计辅助。

所有分析脚本共用同一份实现，避免每个脚本各自复制粘贴加载/字体/打分逻辑。
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from functools import lru_cache

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

# 三个研究数据集的标准名称 -> 文件名映射（脚本统一走 dataset_path）
DATASETS = {
    "human": "training_set_human.jsonl",
    "aigc": "training_set_aigc.jsonl",
    "exploring": "exploring_set.jsonl",
}

# 三个子集的中文显示名（图表/报告统一）
DATASET_LABELS = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
                  "exploring": "探索集(后LLM)"}

_CJK_FALLBACKS = ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
                  "Noto Sans SC", "Noto Sans TC", "Source Han Sans SC",
                  "Source Han Sans CN", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                  "SimHei", "Microsoft YaHei", "PingFang SC", "LXGW WenKai")

# 中文字体文件名特征（用于在系统目录中扫描 .ttf/.otf/.ttc）
_CJK_FONT_FILE_RE = re.compile(
    r"(noto[-_ ]?sans[-_ ]?(cjk|sc|tc|jp)|noto[-_ ]?serif[-_ ]?(cjk|sc)"
    r"|source[-_ ]?han|wqy|wenquanyi|simhei|simsun|msyh|microsoft[-_ ]?yahei"
    r"|pingfang|heiti|songti|kaiti|lxgw)",
    re.IGNORECASE)

# 备选下载源（OFL 开源字体）；下载后缓存在 ~/.cache/rednote-vibe/fonts/
_FONT_DOWNLOAD_URLS = (
    "https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf",
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf",
    "https://cdn.jsdelivr.net/gh/anthonyfok/fonts-wqy-microhei@master/wqy-microhei.ttc",
    "https://raw.githubusercontent.com/anthonyfok/fonts-wqy-microhei/master/wqy-microhei.ttc",
)
# 单色 emoji 字体（matplotlib 不支持彩色 emoji，NotoColorEmoji 会被排除）
_EMOJI_FONT_DOWNLOAD_URLS = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notoemoji/NotoEmoji%5Bwght%5D.ttf",
    "https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/notoemoji/NotoEmoji%5Bwght%5D.ttf",
)
_EMOJI_FONT_FILE_RE = re.compile(
    r"(noto[-_ ]?emoji|symbola|opensymbol)", re.IGNORECASE)
_FONT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache",
                               "rednote-vibe", "fonts")
_FONT_EXTS = (".ttf", ".otf", ".ttc", ".otc")
_FONT_DOWNLOAD_TRIED = False
_EMOJI_FONT_DOWNLOAD_TRIED = False


def _font_search_dirs(extra_dir: str | None = None) -> list[str]:
    """可能包含中文字体的目录（按优先级）：环境变量 > 仓库内 > 缓存 > 系统。"""
    home = os.path.expanduser("~")
    dirs = [
        os.environ.get("REDNOTE_FONT_DIR", ""),
        BASE + "assets/fonts",
        BASE + "fonts",
        _FONT_CACHE_DIR,
        os.path.join(home, ".fonts"),
        os.path.join(home, ".local", "share", "fonts"),
        "/usr/share/fonts/noto-cjk",
        "/usr/share/fonts/opentype/noto",
        "/usr/share/fonts/truetype/noto",
        "/usr/share/fonts/truetype/wqy",
        "/usr/share/fonts/wenquanyi",
        "/usr/local/share/fonts",
        "/Library/Fonts",
        "/System/Library/Fonts",
    ]
    if extra_dir:
        dirs.insert(0, extra_dir)
    out = []
    for d in dirs:
        if d and os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def _scan_fonts(dirs, name_re, exclude_re=None) -> str | None:
    """在目录列表中按文件名特征查找字体文件（每个目录先浅层后递归）。"""
    for d in dirs:
        shallow, nested = [], []
        for root, dirnames, filenames in os.walk(d):
            dirnames.sort()
            for fn in sorted(filenames):
                if not fn.lower().endswith(_FONT_EXTS) or not name_re.search(fn):
                    continue
                if exclude_re is not None and exclude_re.search(fn):
                    continue
                path = os.path.join(root, fn)
                (shallow if root == d else nested).append(path)
        if shallow:
            return shallow[0]
        if nested:
            return nested[0]
    return None


def discover_cjk_font(extra_dir: str | None = None,
                      dirs: list[str] | None = None) -> str | None:
    """在常见目录中查找中文字体文件，返回路径；找不到返回 None。

    按目录优先级扫描（每个目录内先浅层后递归），匹配文件名特征
    （Noto Sans CJK / Source Han / 文泉驿 / SimHei / 雅黑 / 霞鹜文楷 等）。
    `dirs` 可显式指定搜索目录（测试用），默认走 `_font_search_dirs`。
    """
    search = dirs if dirs is not None else _font_search_dirs(extra_dir)
    return _scan_fonts(search, _CJK_FONT_FILE_RE)


def discover_emoji_font(extra_dir: str | None = None,
                        dirs: list[str] | None = None) -> str | None:
    """查找单色 emoji 字体（Noto Emoji / Symbola 等）；彩色 emoji 字体跳过。"""
    search = dirs if dirs is not None else _font_search_dirs(extra_dir)
    return _scan_fonts(search, _EMOJI_FONT_FILE_RE,
                       exclude_re=re.compile(r"color", re.IGNORECASE))


def _register_font(path: str) -> str | None:
    """把字体文件注册进 matplotlib，成功时返回字体名。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    try:
        font_manager.fontManager.addfont(path)
        name = font_manager.FontProperties(fname=path).get_name()
    except Exception:
        return None
    if not name:
        return None
    plt.rcParams["font.family"] = name
    return name


def _download_font_file(urls, timeout: float = 30.0) -> str | None:
    """按顺序尝试下载字体到缓存目录，返回首个成功的本地路径；失败返回 None。"""
    import urllib.parse
    import urllib.request

    os.makedirs(_FONT_CACHE_DIR, exist_ok=True)
    for url in urls:
        name = os.path.basename(urllib.parse.unquote(url.split("?")[0]))
        dest = os.path.join(_FONT_CACHE_DIR, name)
        try:
            if not os.path.exists(dest) or os.path.getsize(dest) < 1_000_000:
                tmp = dest + ".part"
                with urllib.request.urlopen(url, timeout=timeout) as resp, \
                        open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                if os.path.getsize(tmp) < 1_000_000:
                    os.remove(tmp)
                    continue
                os.replace(tmp, dest)
            return dest
        except Exception:
            continue
    return None


def _download_cjk_font(timeout: float = 30.0) -> str | None:
    """下载一个开源中文字体到缓存目录（仅当本机无任何中文字体时）。

    任一环节失败都静默返回 None（离线环境保持与旧版一致的降级行为）；
    可用环境变量 REDNOTE_FONT_DOWNLOAD=0 显式禁用。
    """
    global _FONT_DOWNLOAD_TRIED
    if _FONT_DOWNLOAD_TRIED:
        return None
    if os.environ.get("REDNOTE_FONT_DOWNLOAD", "1") == "0":
        return None
    _FONT_DOWNLOAD_TRIED = True
    path = _download_font_file(_FONT_DOWNLOAD_URLS, timeout)
    return path if path and _register_font(path) else None


def _download_emoji_font(timeout: float = 30.0) -> str | None:
    """下载单色 Noto Emoji 到缓存目录；失败静默返回 None。"""
    global _EMOJI_FONT_DOWNLOAD_TRIED
    if _EMOJI_FONT_DOWNLOAD_TRIED:
        return None
    if os.environ.get("REDNOTE_FONT_DOWNLOAD", "1") == "0":
        return None
    _EMOJI_FONT_DOWNLOAD_TRIED = True
    path = _download_font_file(_EMOJI_FONT_DOWNLOAD_URLS, timeout)
    return path if path and _register_font(path) else None


def _apply_emoji_fallback(cjk_name: str, font_dir: str | None,
                          allow_download: bool) -> str:
    """在中文主字体后追加单色 emoji 字体（matplotlib 3.6+ 逐字形回退）。

    找不到 emoji 字体时保持单一中文字体，行为与旧版一致。
    """
    import matplotlib.pyplot as plt

    path = discover_emoji_font(font_dir)
    if not path and allow_download:
        path = _download_emoji_font()
    if not path:
        return cjk_name
    emoji_name = _register_font(path)
    if not emoji_name:
        return cjk_name
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    families = [cjk_name]
    if emoji_name in installed and emoji_name != cjk_name:
        families.append(emoji_name)
    plt.rcParams["font.family"] = families
    return cjk_name


def setup_chinese_fonts(font_dir: str | None = None,
                        allow_download: bool = True,
                        with_emoji: bool = True) -> str | None:
    """让 matplotlib 能渲染中文（及 emoji）；返回注册成功的字体名，找不到返回 None。

    依次尝试：(1) `REDNOTE_FONT_PATH` 指定的字体文件；(2) 目录扫描
    （`REDNOTE_FONT_DIR` / 仓库 assets/fonts / 缓存 / 系统字体目录）；
    (3) matplotlib 已安装字体名；(4) 可选地从网络下载开源字体到用户缓存。
    成功后再追加单色 emoji 字体回退（可用 `with_emoji=False` 关闭）。
    任何一步失败都不会抛异常，绘图脚本在无字体环境下仍可运行（中文会显示
    为占位方框），只是不再静默出图。
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    name = None

    env_font = os.environ.get("REDNOTE_FONT_PATH", "")
    if env_font and os.path.isfile(env_font):
        name = _register_font(env_font)

    if not name:
        path = discover_cjk_font(font_dir)
        if path:
            name = _register_font(path)

    if not name:
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for cand in _CJK_FALLBACKS:
            if cand in installed:
                plt.rcParams["font.family"] = cand
                name = cand
                break

    if not name and allow_download:
        path = _download_cjk_font()
        if path:
            name = _register_font(path)

    if name and with_emoji:
        _apply_emoji_fallback(name, font_dir, allow_download)
    return name


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


def make_pool(max_workers: int = 4, preload=None):
    """按需创建 fork 进程池（Linux）；异常/单核环境返回 None 走串行回退。

    preload 在 fork 之前执行（例如 jieba.initialize），让子进程继承已加载
    的只读状态，避免每个 worker 重复初始化。
    """
    try:
        import multiprocessing as mp

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            return None
        n = min(max_workers, mp.cpu_count() or 1)
        if n <= 1:
            return None
        if preload is not None:
            preload()
        return ctx.Pool(n)
    except Exception:
        return None


def close_pool(pool) -> None:
    if pool is not None:
        pool.close()
        pool.join()


def iter_jsonl(path: str) -> Iterator[dict]:
    """流式读取 JSONL，逐行解析，避免把大文件一次性载入内存。

    `json.loads` 本身接受首尾空白，因此不再对每行做一次多余的 strip 拷贝，
    只用 `isspace()` 判断空行（行为与原先的 `line.strip()` 判断一致）。
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line or line.isspace():
                continue
            yield json.loads(line)


def iter_jsonl_relaxed(path: str) -> Iterator[dict]:
    """流式读取 JSONL，跳过空行与无法解析的行（历史探索脚本的容错口径）。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line or line.isspace():
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
                yield json.loads(line)
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


def dataset_path(name: str, base: str = BASE) -> str:
    """标准数据集名（human/aigc/exploring）-> 绝对路径；未知名称直接报错。"""
    try:
        filename = DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}")
    return os.path.join(base, filename)


def title_of(rec: dict) -> str:
    """取标题（缺失时返回空串），与正文口径分开。"""
    return rec.get("note_title") or ""


def compact_text(text: str | None) -> str:
    """去掉全部空白字符，得到用于前缀/模板统计的紧凑文本（与 framing 口径一致）。"""
    return re.sub(r"\s+", "", str(text or ""))


# 话题标签：`#` 后跟最多 30 个非 `#`/空白/方括号字符。排除方括号是为了
# 兼容 AIGC 与探索集的导出格式 `#标签[话题]#`（标签体在 `[话题]` 前结束）。
_TAG_RE = re.compile(r"#([^#\s\[\]]{1,30})")
# 采集残留：BOM / 零宽空格 / 方向标记，会混进标签首尾，需剥掉。
_TAG_JUNK = "\ufeff\u200b\u200e\u200f"


def extract_hashtags(text: str | None) -> list[str]:
    """从文本中提取 `#话题` 标签列表（保留原大小写与首次出现的形态）。

    兼容三种历史格式：人类的 `#标签`、AIGC 的 `#标签[话题]#`、
    探索集行内混入的 BOM/零宽字符（如 `#\\ufeff标签`）。空标签与被
    BOM 包裹出空串的标签不会返回；`#话题`（字面量）作为清洗残留剔除。
    """
    out = []
    for raw in _TAG_RE.findall(str(text or "")):
        tag = raw.strip(_TAG_JUNK + " \t\r\n")
        if tag and tag != "话题":
            out.append(tag)
    return out


def rate_diff_ranking(counts_a: dict, n_a: int, counts_h: dict, n_h: int,
                      reverse: bool = False, min_count: int = 10,
                      k: int = 30) -> list[dict]:
    """按每篇发生率之差提炼某一侧专属词条（框架模板/话题标签/表情共用）。

    reverse=False：A 侧专属（rate_a - rate_h 降序，要求 A 出现次数 >= min_count）；
    reverse=True ：H 侧专属（rate_h - rate_a 降序，要求 H 出现次数 >= min_count）。
    返回 [{"word", "rate_diff_ppt", "aigc_per1k", "human_per1k"}, ...]，
    counts 为「每篇出现次数（去重）」的 Counter，n 为对应子集篇数。
    """
    rows = []
    for w in set(counts_a) | set(counts_h):
        ca, ch = counts_a.get(w, 0), counts_h.get(w, 0)
        if (ch if reverse else ca) < min_count:
            continue
        diff = (ch / n_h - ca / n_a) if reverse else (ca / n_a - ch / n_h)
        rows.append((w, diff, ca, ch))
    rows.sort(key=lambda r: (-r[1], r[0]))
    return [{"word": w, "rate_diff_ppt": round(d * 100, 3),
             "aigc_per1k": round(ca / n_a * 1000, 2),
             "human_per1k": round(ch / n_h * 1000, 2)}
            for w, d, ca, ch in rows[:k]]


def rate_top(counts: dict, n: int, min_count: int = 10,
             k: int = 50) -> list[dict]:
    """单侧高频词条（按每篇发生率降序，不做另一侧过滤）。"""
    rows = [(w, c / n) for w, c in counts.items() if c >= min_count]
    rows.sort(key=lambda r: -r[1])
    return [{"word": w, "aigc_per1k": round(r * 1000, 2)} for w, r in rows[:k]]


def quarter_of(month: str | None) -> str | None:
    """把 `YYYYMM` 月份键转为 `YYYYQX` 季度键；非法输入返回 None。"""
    m = str(month or "")
    if len(m) < 6 or not m[:6].isdigit():
        return None
    q = (int(m[4:6]) - 1) // 3 + 1
    if not 1 <= q <= 4:
        return None
    return f"{m[:4]}Q{q}"


def dedup_indices(ids: list, meta) -> list[int]:
    """按 note_id 去重：同名笔记保留元数据最完整（meta 分数最高）的一条。

    meta 与 ids 等长（可为 numpy 数组或列表）；打分规则由调用方决定
    （见各全量扫描脚本）。无 note_id 的记录全部保留，返回值按原顺序。
    规则与 deep_dive_analysis.py / framing_analysis.py 完全一致。
    """
    best: dict = {}
    for i, nid in enumerate(ids):
        if nid is None:
            continue
        j = best.get(nid)
        if j is None or meta[i] > meta[j]:
            best[nid] = i
    keep = set(best.values())
    return [i for i, nid in enumerate(ids) if nid is None or i in keep]


_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;…]+|\n+")


def split_sentences(text: str | None) -> list[str]:
    """把正文切成句子（按中英文句末标点/换行/分号），去除空片段与首尾空白。

    正则与 `deep_dive_analysis.py` 的 SENT_SPLIT 完全一致（连续标点视为一个边界），
    保证「平均句长」等指标跨脚本可比。用 `map(str.strip, parts)` 一次性剥离首尾
    空白（C 层调用），再用结果本身做非空判断，与逐片段 strip 的结果完全一致。
    """
    parts = _SENTENCE_SPLIT_RE.split(str(text or ""))
    return [s for s in map(str.strip, parts) if s]


def burstiness(lengths) -> float:
    """句长变异系数 CV=std/mean（人类写作「有起伏」，模板化文本更均匀）。

    少于 2 个句长或均值为 0 时返回 0.0。
    """
    arr = np.asarray(list(lengths), dtype=float)
    if arr.size < 2:
        return 0.0
    mean = float(arr.mean())
    if mean <= 0:
        return 0.0
    return float(arr.std() / mean)


def _mtld_pass(tokens: list[str], threshold: float) -> float:
    types: set[str] = set()
    factors = 0.0
    start = 0
    n = len(tokens)
    for i in range(n):
        types.add(tokens[i])
        seen = i - start + 1
        if len(types) / seen <= threshold:
            factors += 1.0
            types = set()
            start = i + 1
    if start < n:
        seen = n - start
        ttr = len(types) / seen
        factors += (1.0 - ttr) / (1.0 - threshold)
    return factors


def mtld(tokens, threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity（McCarthy & Jarvis, 2010）。

    双向（正向 + 反向）因子的平均长度，值越大词汇越丰富；文本过短时不稳定，
    调用方可自行设置最小 token 数。空输入或全部重复 token 返回 0.0。
    """
    toks = [t for t in tokens if t]
    if len(toks) < 2:
        return 0.0
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")

    def direction_value(seq: list[str]) -> float:
        factors = _mtld_pass(seq, threshold)
        if factors <= 0:
            # 全程 TTR 都高于阈值（几乎全部 token 互不相同）：整段文本视为
            # 单个因子，MTLD 退化为文本长度，避免把最丰富的文本记成 0。
            return float(len(seq))
        return len(seq) / factors

    return float((direction_value(toks) + direction_value(toks[::-1])) / 2.0)


def prefix_ngram_counts(texts, n: int = 4) -> Counter:
    """统计文本前缀 n-gram 频次（先做 `compact_text`，长度不足 n 的跳过）。"""
    counts: Counter[str] = Counter()
    for t in texts:
        compact = compact_text(t)
        if len(compact) >= n:
            counts[compact[:n]] += 1
    return counts


def prefix_ngram_stats(texts, n: int = 4, k: int = 20) -> dict:
    """文本前缀 n-gram 集中度统计（模板化的直接度量）。

    先对每条文本做 `compact_text`，长度不足 n 的跳过；返回 Top-k 覆盖率、
    前缀经验熵（bit，越高越多样）与前缀唯一率。
    """
    counts = prefix_ngram_counts(texts, n)
    total = sum(counts.values())
    return {
        "n_texts": total,
        "coverage_topk": topk_coverage(counts, k),
        "entropy": shannon_entropy(counts),
        "unique_ratio": (len(counts) / total) if total else 0.0,
        "top": counts.most_common(k),
    }


_CLEAN_BODY_RES = None


def clean_body(content: str | None) -> str:
    """与 paper_features.py 相同的正文清洗口径。

    删除 `[话题]` 字面标记（导出残留）、把 `#话题` 标签替换为空格并去掉首尾空白。
    复用 `paper_features` 中的同一组正则，保证所有分析跨脚本口径一致。
    """
    global _CLEAN_BODY_RES

    if _CLEAN_BODY_RES is None:
        from paper_features import HASHTAG, HUATI

        _CLEAN_BODY_RES = (HUATI, HASHTAG)
    huati, hashtag = _CLEAN_BODY_RES
    body = huati.sub("", content or "")
    return hashtag.sub(" ", body).strip()


def _positive_counts(counts) -> np.ndarray:
    """提取正计数并转为 float64 数组。

    与旧的 `np.asarray(list(counts), dtype=float)` 结果完全一致，但当输入已是
    numpy 计数数组（如 bootstrap 中的 `np.bincount` 结果）时，直接从数组上过滤
    并原地转换，避免每轮把整条数组转成 Python list 再转回数组的开销。
    """
    if isinstance(counts, np.ndarray):
        arr = counts[counts > 0]
        return arr if arr.dtype == np.float64 else arr.astype(np.float64)
    values = list(counts.values()) if isinstance(counts, dict) else list(counts)
    arr = np.asarray(values, dtype=float)
    return arr[arr > 0]


def shannon_entropy(counts, miller_madow: bool = False) -> float:
    """离散分布的香农熵（bit）。miller_madow=True 时加小样本校正。

    经验熵随样本量增大而系统性偏高（能观测到更多稀有变体），跨不同样本量的
    比较应使用 Miller–Madow 校正：H_MM = H + (K-1)/(2N ln2)。
    """
    arr = _positive_counts(counts)
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
    arr = _positive_counts(counts)
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
    """读取一条记录的（点赞, 收藏, 评论），兼容两种字段命名。

    每个字段只做一次 `rec.get`（命中旧字段名时不再重复查字典），
    与旧实现的取值口径完全一致（新字段值为 None 时回退旧字段名）。
    """
    liked = rec.get("liked_count")
    if liked is None:
        liked = rec.get("likes")
    collected = rec.get("collected_count")
    if collected is None:
        collected = rec.get("collections")
    comments = rec.get("comments_count")
    if comments is None:
        comments = rec.get("comments")
    return to_int(liked), to_int(collected), to_int(comments)


@lru_cache(maxsize=16384)
def _parse_local_time_str(s: str) -> datetime | None:
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


def parse_local_time(value) -> datetime | None:
    """解析数据集的本地时间字段（'YYYYMMDDHH'，兼容 8 位日期）；失败返回 None。

    数据集里同一时间戳被大量记录重复使用（探索集 87% 的 local_time 重复），
    因此对规范化后的字符串做有界缓存（datetime 不可变，可安全共享）。先用
    str()/strip() 归一化再查缓存，避免 1 与 1.0 之类的哈希等价键命中同一槽位。
    """
    return _parse_local_time_str(str(value or "").strip())


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


def linear_score(df: pd.DataFrame, features, mean, scale, coef, intercept) -> np.ndarray:
    """通用线性模型打分：z-score 标准化后的逻辑回归 -> P∈(0,1)。

    所有自定义特征子集的打分（如仅标题特征）复用这一实现，避免重复公式。
    """
    z = (df[list(features)].fillna(0).values - np.asarray(mean)) / np.asarray(scale)
    logit = z @ np.asarray(coef) + intercept
    return 1.0 / (1.0 + np.exp(-logit))


def detector_score(df: pd.DataFrame, model: dict, robust: bool = True) -> np.ndarray:
    """用保存的逻辑回归参数对 DataFrame 打分；robust=True 用稳健特征模型。"""
    if robust:
        return linear_score(df, model["robust_features"], model["robust_scaler_mean"],
                            model["robust_scaler_scale"], model["robust_coef"],
                            model["robust_intercept"])
    return linear_score(df, model["features"], model["scaler_mean"],
                        model["scaler_scale"], model["coef"], model["intercept"])


def cliffs_delta_from_u(u: float, n1: int, n2: int) -> float:
    return 2.0 * u / (n1 * n2) - 1.0


def cliffs_delta(a, b) -> tuple[float, float]:
    """返回 (Cliff's delta, Mann-Whitney U 双侧 p 值)。"""
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return cliffs_delta_from_u(u, len(a), len(b)), float(p)


def cliff_share_test(indicator, values) -> dict:
    """一组「命中」指示变量在连续值上的 Cliff's δ 与 Mann–Whitney p。

    用于检验某个二值属性（如「含分点行」「含 AI 专属标签」）的帖子在连续
    指标（如 P(AI) 得分）上是否系统性偏高；命中/未命中任一侧为空时返回
    中性结果 (δ=0, p=1)。原实现位于 symbol_analysis.py，此处上移供各分析复用。
    """
    indicator = np.asarray(indicator, dtype=bool)
    values = np.asarray(values)
    if indicator.any() and (~indicator).any():
        delta, p = cliffs_delta(values[indicator], values[~indicator])
    else:
        delta, p = 0.0, 1.0
    return {"delta": round(float(delta), 3), "p": float(p)}


def gini_coefficient(values) -> float:
    """基尼系数（0=完全均等，1=完全集中）；负值按 0 处理，空/全零返回 0。"""
    if not isinstance(values, np.ndarray):
        values = list(values)
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr[~np.isnan(arr)], 0.0, None)
    total = float(arr.sum())
    if arr.size == 0 or total <= 0:
        return 0.0
    arr.sort()
    n = arr.size
    weights = 2.0 * np.arange(1, n + 1) - n - 1
    return float(weights.dot(arr) / (n * total))


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


def cv_auc(df: pd.DataFrame, feats: list[str], cv=None,
           label_col: str = "dataset", positive: str = "aigc",
           seed: int = 42) -> dict:
    """特征子集在 human vs AIGC 上的 5 折分层交叉验证（逻辑回归）。

    与 `paper_detector.py` 完全同口径：StandardScaler + 平衡类权重
    LogisticRegression(max_iter=2000)，同一 `StratifiedKFold(shuffle=True,
    random_state=42)`（cv=None 时内部构造）。原实现在 title_style_analysis.py
    与本函数逐行一致，检测器消融类分析统一复用这里。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X = df[list(feats)].fillna(0).values
    y = (df[label_col] == positive).astype(int).values
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000,
                                            class_weight="balanced"))
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, pred)),
        "acc": float(accuracy_score(y, pred)),
        "n_features": len(feats),
    }


def threshold_metrics(y, proba, thresholds=(0.3, 0.5, 0.7, 0.9, 0.95),
                      with_ci: bool = True) -> list[dict]:
    """一组阈值下二分类器的精确率/召回率/F1/阳性占比。

    y=1 为正类（如 AIGC），proba 为正类概率。`flagged_neg_share` 是负类
    （人类）被标为正类的比例（假阳性率），`flagged_pos_share` 即召回率。
    """
    from sklearn.metrics import f1_score

    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n_pos if n_pos else 0.0
        row = {
            "threshold": float(t),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "flagged_neg_share": round(fp / n_neg, 4) if n_neg else 0.0,
            "flagged_pos_share": round(recall, 4),
        }
        if with_ci:
            row["precision_ci"] = [round(v, 4) for v in wilson_ci(tp, tp + fp)]
        rows.append(row)
    return rows


def prior_corrected_proba(proba, n_neg: int, n_pos: int) -> np.ndarray:
    """把 `class_weight="balanced"` 训练出的概率校正回原始类先验。

    平衡权重等价于在 logit 上平移 log(n_neg/n_pos)（见 King & Zeng, 2001），
    校正后 P̂ 可直接当作自然分布下的概率使用（用于校准/阈值分析）。
    """
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1 - 1e-12)
    shift = np.log(float(n_neg) / float(n_pos)) if n_pos > 0 and n_neg > 0 else 0.0
    logit = np.log(p / (1.0 - p)) - shift
    return 1.0 / (1.0 + np.exp(-logit))


def reliability_bins(proba, y, bins: int = 10) -> dict:
    """把预测概率分箱，返回可靠性图所需的逐箱统计。

    返回 dict：bin_lo/bin_hi（等宽箱边界）、mean_pred、mean_true、count。
    空箱的 mean_pred/mean_true 为 NaN（由调用方忽略）。
    """
    p = np.asarray(proba, dtype=float)
    labels = np.asarray(y, dtype=float)
    if p.size == 0 or p.size != labels.size:
        raise ValueError("proba and y must be non-empty and have equal length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    mean_pred = np.full(bins, np.nan)
    mean_true = np.full(bins, np.nan)
    count = np.zeros(bins, dtype=int)
    for b in range(bins):
        m = idx == b
        count[b] = int(m.sum())
        if count[b]:
            mean_pred[b] = float(p[m].mean())
            mean_true[b] = float(labels[m].mean())
    return {"bin_lo": edges[:-1].tolist(), "bin_hi": edges[1:].tolist(),
            "mean_pred": mean_pred.tolist(), "mean_true": mean_true.tolist(),
            "count": count.tolist()}


def expected_calibration_error(proba, y, bins: int = 10) -> float:
    """Expected Calibration Error（等宽分箱，|平均预测-实际频率| 的样本加权）。"""
    r = reliability_bins(proba, y, bins)
    count = np.asarray(r["count"], dtype=float)
    n = count.sum()
    if n <= 0:
        return 0.0
    pred = np.asarray(r["mean_pred"], dtype=float)
    true = np.asarray(r["mean_true"], dtype=float)
    valid = count > 0
    return float((count[valid] / n * np.abs(pred[valid] - true[valid])).sum())


def bootstrap_median_ci(values, n_boot: int = 1000, seed: int = 42,
                        alpha: float = 0.05) -> dict:
    """对一维样本做 bootstrap，返回中位数及其 (1-alpha) 置信区间。

    分块重采样避免构造 (n_boot, n) 的大矩阵；NaN 自动剔除。
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"median": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    n = arr.size
    meds = np.empty(n_boot)
    chunk = max(1, 2_000_000 // max(n, 1))
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = rng.integers(0, n, (size, n))
        meds[done:done + size] = np.median(arr[idx], axis=1)
        done += size
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"median": float(np.median(arr)), "ci_low": float(lo),
            "ci_high": float(hi), "n": int(n)}


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def _stream_bytes_equal(stream, data: bytes, chunk: int = 65536) -> bool:
    """把二进制流与 bytes 分块比较，避免把整个解压内容读进内存。"""
    pos = 0
    n = len(data)
    while pos < n:
        block = stream.read(min(chunk, n - pos))
        if not block or block != data[pos:pos + len(block)]:
            return False
        pos += len(block)
    return not stream.read(1)


def write_csv_gz_if_changed(df: pd.DataFrame, path: str) -> bool:
    """把 DataFrame 写为 gzip CSV；内容与现有文件一致时一个字节都不动。

    研究数据文件（*.csv.gz）由 CI 逐字节校验保护：重跑分析脚本时若派生内容
    没有变化，就不应该产生无意义的字节churn。这里先把新内容序列化到内存
    （gzip mtime 固定为 0，输出确定性），再分块流式解压现有文件逐字节比较
    （不整体读入内存）；只有内容真正变化时才经临时文件原子替换。返回是否
    实际写入了文件。
    """
    import gzip

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    if os.path.exists(path):
        try:
            with gzip.open(path, "rb") as f:
                if _stream_bytes_equal(f, csv_bytes):
                    return False
        except OSError:
            pass
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
            gz.write(csv_bytes)
    os.replace(tmp, path)
    return True
