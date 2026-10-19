# -*- coding: utf-8 -*-
"""深潜分析之十五：语气情态层（谨慎表达 / 断言强调 / 程度强化 / 社区呼语）。

既有分析覆盖了词汇表、句子节奏、框架模板、标题、符号与互动结构，但把
「AI 味」落到**话语语气**层面（说话人有多确定、多克制的措辞，以及是否
直接呼唤社区受众）的系统分析还缺失。本脚本用四个先验定义的词表类别
（不依赖任何监督信号）回答三组问题：

1. 跨子集使用率：三类语料如何使用这些语气标记？生成集有没有模仿社区口癖？
2. 探索集内的 P(AI) 梯度：哪些语气标记在野外真正区分「像人 / 像 AI」，
   哪些只是生成集内部的风格指纹？梯度是否稳健（去重 / 长度分层）？
3. 互动回报：在控制 P(AI)、正文长度与标签数后，语气标记与点赞、收藏的
   偏效应方向与大小（月份×领域固定效应 + 整群 bootstrap CI）。

口径说明：
- 正文清洗复用 analysis_utils.clean_body（与 paper_features.py 一致）；
- 人类/探索集按 note_id 去重（保留元数据最完整一条，规则同 deep_dive_analysis），
  AIGC 无 note_id，逐记录处理；
- 探索集 p_ai 按行序对齐 paper_exploring_scores.csv.gz，脚本内用 content_len
  逐条校验（不一致时直接报错退出）；
- 语气密度统一按「正文每百字」计；正文不足 MIN_BODY_LEN 字的帖子
  （纯标签/空正文等）不参与统计，数量单独报告；
- 互动回归与 engagement_drivers_analysis.py 完全同协议（特征先 z-score
  再单元内去均值、单元 ≥100 条、整群 bootstrap 300 次），额外加入四个
  语气密度特征。

词表（先验定义，不做数据驱动的筛选）：
- 谨慎表达 hedge：可能 也许 或许 大概 大约 好像 似乎 应该 不一定 算是
  稍微 有点 说不定 个人觉得
- 断言强调 boost：一定（排除「不一定」）绝对 必须 肯定 千万 保证 务必
  毫无疑问
- 程度强化 intens：超级 简直 真的 非常 特别 完全 彻底
- 社区呼语 vocative：姐妹们 宝子们 家人们 集美
- 类别计数为「正文中出现次数（非重叠匹配）」，等价于按词长降序的类别正则
  交替匹配；`一定` 只计入不以「不」开头的位置（即扣除「不一定」）。

输出：
  insights/stance_lexicon.png   跨子集类别密度 + 逐词 AIGC−人类使用差
  insights/stance_gradient.png  探索集 P(AI) 梯度 / 逐词单元内 ρ / 互动偏效应 / 家族指纹
  stance_results.json           全部数值结果
  stance_results.md             结果摘要
"""
from __future__ import annotations

from array import array
from collections import defaultdict
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from analysis_utils import (BASE, INSIGHTS, PALETTE, body_cache, init_plot_style,
                            save_fig, save_json, cliffs_delta, cell_rank_cache,
                            cell_groups, clean_body, content_of, decile_table,
                            dedup_indices, domain_standardized_series,
                            engagement_of, length_bin_codes,
                            parse_local_time, run_parallel, within_cell_spearman)
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

MIN_BODY_LEN = 50        # 语气密度在更短正文（纯标签/空正文）上无意义
MIN_CELL_N = 100         # 单元内统计的最小单元样本量（与论文口径一致）
MONTH_FROM = "202301"    # 互动分析起始月份（与 paper_engagement 一致）
N_BOOT = 300             # 整群 bootstrap 次数
BOOT_SEED = 42
MIN_QUARTER_N = 300
MIN_FAMILY_N = 100


CATEGORY_WORDS = {
    "hedge": ("谨慎表达", ["可能", "也许", "或许", "大概", "大约", "好像", "似乎",
                          "应该", "不一定", "算是", "稍微", "有点", "说不定",
                          "个人觉得"]),
    "boost": ("断言强调", ["一定", "绝对", "必须", "肯定", "千万", "保证",
                          "务必", "毫无疑问"]),
    "intens": ("程度强化", ["超级", "简直", "真的", "非常", "特别", "完全", "彻底"]),
    "vocative": ("社区呼语", ["姐妹们", "宝子们", "家人们", "集美"]),
}
CATEGORY_LABELS = {k: v[0] for k, v in CATEGORY_WORDS.items()}
CATEGORY_COLORS = {"hedge": "#4C72B0", "boost": "#C44E52",
                   "intens": "#DD8452", "vocative": "#8172B3"}
CATEGORY_ORDER = ["hedge", "boost", "intens", "vocative"]
SUBSET_ORDER = ["human", "aigc", "exploring"]
ALL_WORDS = [w for k in CATEGORY_ORDER for w in CATEGORY_WORDS[k][1]]
WORD_CATEGORY = {w: k for k in CATEGORY_ORDER for w in CATEGORY_WORDS[k][1]}

# 逐词展平（模块级预计算）：(词, 类别, 需扣除的重叠词或 None)；
# 热循环内不再做逐词字符串比较（"必须"/"一定"）分派。
_FLAT_STANCE_WORDS = [
    (w, cat, {"必须": "务必须", "一定": "不一定"}.get(w))
    for cat in CATEGORY_ORDER for w in CATEGORY_WORDS[cat][1]
]

FAMILY_NAME = {"anthropic": "Claude", "deepseek": "DeepSeek", "gemini": "Gemini",
               "glm": "GLM", "gpt": "GPT", "qwen": "Qwen"}
FAMILIES = ["anthropic", "deepseek", "gemini", "glm", "gpt", "qwen"]


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def word_counts(text: str) -> dict:
    """对一段正文统计四个类别 + 每个词的命中次数。

    返回 {"hedge": {"可能": 2, ...}, ..., "_words": {词: 次数}}；计数口径
    与 paper_features.py 的 body.count(word) 一致（非重叠匹配），唯一
    例外是 boost 中的「一定」不算「不一定」中的出现。

    实现为逐词的 `str.count`（C 层非重叠计数），与原「按词长降序的类别
    正则交替匹配」逐位一致：词表内不存在同类别前缀抢匹配（如「留言区」
    之于「留言」）的组合，唯一的类别内相邻重叠是「务必」尾字起头的
    「必须」（如「任务必须」——正则先匹配「务必」后吃掉「必须」的首字），
    因此对「必须」扣除「务必须」的出现次数；跨类别的「一定」扣除
    「不一定」（与 `(?<!不)一定` 等价）。逐词计数比整段正则快约 2–3 倍。
    """
    out: dict = {cat: {} for cat in CATEGORY_ORDER}
    words: dict = {}
    for w, cat, overlap in _FLAT_STANCE_WORDS:
        n = text.count(w)
        if overlap is not None:
            n -= text.count(overlap)
        if n:
            out[cat][w] = n
            words[w] = words.get(w, 0) + n
    out["_words"] = words
    return out


def density_per_100(count: int, body_len: int) -> float:
    """次数 / 正文字数 * 100；空正文返回 0。"""
    return count / body_len * 100 if body_len > 0 else 0.0


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
_WORD_INDEX = {w: i for i, w in enumerate(ALL_WORDS)}


def scan_corpus(path: str) -> dict:
    """单次流式扫描一个语料，返回逐记录数组。

    每条记录保存：正文长度、note_id、元数据完整度（用于去重取舍）、
    月份、领域、模型家族/模型（AIGC），以及四个类别的命中次数与逐词次数。

    逐词计数写入按 `ALL_WORDS` 顺序拉平的 `array('i')`，最后一次性
    reshape/切片：避免每帖 33 次 `dict.get` + `list.append`（全量三语料
    约 500 万次 Python 调用），且 int32 平铺存储比 33 个 Python int 列表
    更省内存。`content.strip()` 只用于「正文是否有非空白内容」的元数据
    评分，与原先 `"".join(content.split())` 的布尔值完全一致。
    """
    body_len, note_id, meta, month, domain = [], [], [], [], []
    family, model = [], []
    word_index = _WORD_INDEX
    n_words = len(ALL_WORDS)
    cat_counts = {k: [] for k in CATEGORY_WORDS}
    flat_words = array("i")
    likes = []
    n_total = 0
    # 重复正文（探索集约 45%）直接复用上一次的逐词计数，只读不改。
    cached = body_cache(word_counts)
    for rec in iter_cached(path):
        n_total += 1
        content = content_of(rec)
        body = clean_body(content)
        L = len(body)
        counts = cached(body)
        body_len.append(L)
        note_id.append(rec.get("note_id"))
        dom = rec.get("domain")
        domain.append(dom if dom not in (None, "") else "未知")
        likes_v, colls, comms = engagement_of(rec)
        likes.append(likes_v)
        has_meta = bool(rec.get("user_id")) and dom not in (None, "", "未知")
        dt = parse_local_time(rec.get("local_time"))
        m = f"{dt.year:04d}{dt.month:02d}" if dt else None
        month.append(m)
        score = (int(has_meta) + int(bool(likes_v)) + int(bool(colls))
                 + int(bool(comms)) + int(dt is not None)
                 + int(bool(content.strip())))
        meta.append(score)
        family.append(rec.get("model_family") or "")
        model.append(rec.get("model") or "")
        for k in CATEGORY_WORDS:
            cat_counts[k].append(sum(counts[k].values()))
        row = [0] * n_words
        for w, n in counts["_words"].items():
            row[word_index[w]] = n
        flat_words.extend(row)
    word_matrix = np.frombuffer(flat_words, dtype=np.int32).reshape(-1, n_words)
    return {
        "path": path, "n_total": n_total,
        "body_len": np.asarray(body_len, dtype=np.int32),
        "note_id": note_id, "meta": np.asarray(meta, dtype=np.int32),
        "month": month, "domain": domain, "likes": np.asarray(likes, dtype=float),
        "family": family, "model": model,
        "cat_counts": {k: np.asarray(v, dtype=np.int32)
                       for k, v in cat_counts.items()},
        "word_counts": {w: word_matrix[:, i] for i, w in enumerate(ALL_WORDS)},
    }


def dedup_keep(scan: dict) -> np.ndarray:
    """按 note_id 去重（保留元数据最完整一条），返回排序后的保留索引。"""
    keep = dedup_indices(scan["note_id"], scan["meta"])
    return np.asarray(sorted(keep), dtype=int)


def rates_block(scan: dict, keep: np.ndarray, min_len: int = MIN_BODY_LEN) -> dict:
    """一个子集的类别/逐词使用率（仅正文 >= min_len 的记录）。"""
    sel = keep[scan["body_len"][keep] >= min_len]
    n = int(len(sel))
    chars = int(scan["body_len"][sel].sum())
    cat = {}
    for k in CATEGORY_ORDER:
        c = scan["cat_counts"][k][sel].astype(float)
        cat[k] = {
            "n": n, "chars": chars,
            "density100": float(c.sum() / chars * 100) if chars else 0.0,
            "share": float((c > 0).mean()) if n else 0.0,
        }
    words = {}
    for w, arr in scan["word_counts"].items():
        c = arr[sel].astype(float)
        words[w] = {
            "n": n,
            "per10k_chars": float(c.sum() / chars * 10000) if chars else 0.0,
            "share": float((c > 0).mean()) if n else 0.0,
        }
    return {"n": n, "chars": chars, "categories": cat, "words": words}


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_lexicon(subset_rates: dict, word_table: dict) -> None:
    """左：三个子集的类别密度；右：逐词 AIGC−人类使用差（探索集为圆点）。"""
    init_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.6),
                             gridspec_kw={"width_ratios": [1.0, 1.45]})

    # ---- (a) 类别 × 子集密度 ----
    ax = axes[0]
    y = np.arange(len(CATEGORY_ORDER))
    h = 0.24
    for j, subset in enumerate(SUBSET_ORDER):
        vals = [subset_rates[subset]["categories"][k]["density100"]
                for k in CATEGORY_ORDER]
        pos = y + (j - 1) * h
        ax.barh(pos, vals, height=h, color=PALETTE[subset],
                label={"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
                       "exploring": "探索集(后LLM)"}[subset])
        for yy, v in zip(pos, vals):
            ax.annotate(f"{v:.3f}", xy=(v, yy), xytext=(3, -3),
                        textcoords="offset points", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([CATEGORY_LABELS[k] for k in CATEGORY_ORDER])
    ax.set_xlabel("语气标记密度（次/百字，正文 ≥ %d 字）" % MIN_BODY_LEN)
    ax.set_title("(a) 三类语料的语气标记使用密度")
    ax.legend(loc="lower right")

    # ---- (b) 逐词 AIGC−人类 使用差 ----
    ax = axes[1]
    rows = []
    for w, rec in word_table.items():
        rows.append((w, rec["category"],
                     rec["aigc"]["per10k_chars"] - rec["human"]["per10k_chars"],
                     rec["exploring"]["per10k_chars"]))
    # 先按类别分组、组内按过量使用降序，类别块连续时交替底色才有意义
    rows.sort(key=lambda r: (CATEGORY_ORDER.index(r[1]), -r[2]))
    ypos = np.arange(len(rows))
    diffs = np.asarray([r[2] for r in rows])
    colors = [CATEGORY_COLORS[r[1]] for r in rows]
    for cat_i, k in enumerate(CATEGORY_ORDER):
        idx = [i for i, r in enumerate(rows) if r[1] == k]
        if cat_i % 2 == 1 and idx:
            ax.axhspan(min(idx) - 0.5, max(idx) + 0.5, color="#EEEEEE", zorder=0)
    ax.barh(ypos, diffs, color=colors, height=0.72, alpha=0.9, zorder=2)
    ax.scatter([r[3] for r in rows], ypos, s=22, color="black", zorder=4,
               edgecolor="white", linewidth=0.5, label="探索集使用率")
    ax.axvline(0, color="black", lw=0.8, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    for i, r in enumerate(rows):
        v = r[2]
        # 词名贴着零线放（正差在零线左侧、负差在右侧），减少视线移动
        ax.annotate(r[0], xy=(0, ypos[i]), xytext=(-7 if v >= 0 else 7, 0),
                    textcoords="offset points", ha="right" if v >= 0 else "left",
                    va="center", fontsize=8.5)
        ax.annotate(f"{v:+.2f}", xy=(v, ypos[i]),
                    xytext=(6 if v >= 0 else -6, 4),
                    textcoords="offset points", fontsize=7.2,
                    ha="left" if v >= 0 else "right")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("AIGC − 人类 使用率差（次/万字符；越右=生成集越爱用）")
    ax.set_title("(b) 生成集对每个语气词的过量使用（黑点=探索集使用率）")
    handles = [plt.Line2D([], [], color=CATEGORY_COLORS[k], lw=6,
                          label=CATEGORY_LABELS[k]) for k in CATEGORY_ORDER]
    handles.append(plt.Line2D([], [], color="black", marker="o", ls="",
                              ms=4, label="探索集使用率"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    save_fig(fig, INSIGHTS + "stance_lexicon.png")


def plot_gradient(grad: dict, word_rho: dict, fe_likes: dict,
                  fe_collections: dict, family_table: dict) -> None:
    """四联图：P(AI) 十分位梯度 / 逐词单元内 ρ / 互动偏效应 / 家族指纹。"""
    init_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12.6))

    # ---- (a) P(AI) 十分位密度曲线 ----
    ax = axes[0, 0]
    for k in CATEGORY_ORDER:
        dec = grad["categories"][k]["deciles"]
        x = [d["decile"] for d in dec]
        yv = [d["mean"] for d in dec]
        lo = [d["ci_low"] for d in dec]
        hi = [d["ci_high"] for d in dec]
        ax.plot(x, yv, "o-", color=CATEGORY_COLORS[k], lw=1.8, ms=5,
                label=f"{CATEGORY_LABELS[k]}（ρ={grad['categories'][k]['weighted_rho']:+.3f}）")
        ax.fill_between(x, lo, hi, color=CATEGORY_COLORS[k], alpha=0.16)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("检测器 P(AI) 十分位（1=最像人，10=最像 AI）")
    ax.set_ylabel("密度（次/百字）")
    ax.set_title("(a) 探索集：语气标记随 P(AI) 的梯度（含 95% CI）")
    ax.legend(fontsize=9)

    # ---- (b) 逐词单元内 ρ ----
    ax = axes[0, 1]
    items = sorted(word_rho.items(), key=lambda kv: (CATEGORY_ORDER.index(
        kv[1]["category"]), kv[1]["weighted_rho"]))
    ypos = np.arange(len(items))
    vals = [kv[1]["weighted_rho"] for kv in items]
    colors = [CATEGORY_COLORS[kv[1]["category"]] for kv in items]
    ax.barh(ypos, vals, color=colors, alpha=0.9, height=0.72)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([kv[0] for kv in items], fontsize=8.5)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.3f}", xy=(v, ypos[i]),
                    xytext=(5 if v >= 0 else -5, -3),
                    textcoords="offset points", fontsize=7,
                    ha="left" if v >= 0 else "right")
    for k in CATEGORY_ORDER:
        rho = grad["categories"][k]["weighted_rho"]
        ax.axvline(rho, color=CATEGORY_COLORS[k], lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("月份×领域单元内加权 Spearman ρ（词密度 vs P(AI)）")
    ax.set_title("(b) 逐词语气标记的野外 P(AI) 梯度（虚线=类别整体 ρ）")
    handles = [plt.Line2D([], [], color=CATEGORY_COLORS[k], lw=6,
                          label=CATEGORY_LABELS[k]) for k in CATEGORY_ORDER]
    ax.legend(handles=handles, loc="lower left", fontsize=8)

    # ---- (c) 互动偏效应森林图 ----
    ax = axes[1, 0]
    names = list(fe_likes["coef"].keys())
    order = list(reversed(names))
    ypos = np.arange(len(order))
    for target, marker, offset, color in (
            ("likes", "o", -0.14, "#4C72B0"),
            ("collections", "^", 0.14, "#DD8452")):
        fe = fe_likes if target == "likes" else fe_collections
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
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    labels = {"hedge": "谨慎表达", "boost": "断言强调", "intens": "程度强化",
              "vocative": "社区呼语", "p_ai": "P(AI)",
              "content_len": "正文长度", "n_hashtags": "标签数"}
    ax.set_yticklabels([labels.get(n, n) for n in order])
    ax.set_xlabel("单元内 FE β（每 1 SD → log10(1+互动)）")
    ax.set_title(f"(c) 语气特征的互动偏效应（n={fe_likes['n_obs']:,}，"
                 f"{fe_likes['n_cells']} 单元，整群 bootstrap）")
    ax.legend(loc="lower right")

    # ---- (d) 家族指纹（按类别列内 z-score 着色的热力图） ----
    ax = axes[1, 1]
    fams = [f for f in FAMILIES if family_table.get(f, {}).get("n", 0) >= MIN_FAMILY_N]
    mat = np.asarray([[family_table[f]["density100"][k] for k in CATEGORY_ORDER]
                      for f in fams], dtype=float)
    z = (mat - mat.mean(axis=0)) / np.where(mat.std(axis=0) > 0,
                                            mat.std(axis=0), 1.0)
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
    ax.grid(False)
    for i in range(len(fams)):
        for j in range(len(CATEGORY_ORDER)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=9.5,
                    color="white" if abs(z[i, j]) > 1.15 else "black")
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels([CATEGORY_LABELS[k] for k in CATEGORY_ORDER], fontsize=9)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f"{FAMILY_NAME[f]}（n={family_table[f]['n']}）"
                        for f in fams], fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("家族差异（列内 z-score）", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("(d) AIGC 训练集的家族语气指纹（格内为密度，次/百字）")
    save_fig(fig, INSIGHTS + "stance_gradient.png")


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
    mismatch = int((scores["content_len"].to_numpy()
                    != scans["exploring"]["body_len"]).sum())
    if mismatch:
        raise SystemExit(f"探索集 p_ai 行序未对齐：content_len mismatch={mismatch}")

    # ---- 去重 + 子集使用率 ----
    keeps = {name: dedup_keep(sc) for name, sc in scans.items()}
    subset_rates = {name: rates_block(scans[name], keeps[name])
                    for name in SUBSET_ORDER}
    for name in SUBSET_ORDER:
        sc = scans[name]
        short = int((sc["body_len"][keeps[name]] < MIN_BODY_LEN).sum())
        print(f"  {name}: 去重后 {len(keeps[name]):,} 条，正文 <{MIN_BODY_LEN} 字 "
              f"排除 {short:,} 条")

    word_table = {}
    for w in ALL_WORDS:
        word_table[w] = {
            "category": WORD_CATEGORY[w],
            "human": subset_rates["human"]["words"][w],
            "aigc": subset_rates["aigc"]["words"][w],
            "exploring": subset_rates["exploring"]["words"][w],
        }

    # ---- 探索集 P(AI) 梯度（去重 + 长度门槛）----
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
    print(f"梯度样本：{len(sel):,} 条（去重 + 正文 ≥{MIN_BODY_LEN} 字），"
          f"单元 {len(np.unique(cells))} 个")

    groups = cell_groups(cells, MIN_CELL_N)
    # p_ai 是所有类别/逐词的同一对侧向量：预算一次单元内平均秩供全部调用
    # 复用（原先每个类别与逐词都会重算一次，共 32 次重复排序）；上下十分位
    # 阈值同样只与 p_ai 有关，循环外只算一次。
    p_ai_ranks = cell_rank_cache(p_ai, groups)
    groups_strat = cell_groups(strat, MIN_CELL_N)
    p_ai_ranks_strat = cell_rank_cache(p_ai, groups_strat)
    p_hi, p_lo = float(np.quantile(p_ai, 0.9)), float(np.quantile(p_ai, 0.1))
    grad_cat = {}
    for k in CATEGORY_ORDER:
        dens = expl["cat_counts"][k][sel].astype(float) / body * 100
        dec = decile_table(dens, p_ai)
        wc = within_cell_spearman(dens, p_ai, cells, min_cell_n=MIN_CELL_N,
                                  groups=groups, y_ranks=p_ai_ranks)
        st = within_cell_spearman(dens, p_ai, strat, min_cell_n=MIN_CELL_N,
                                  groups=groups_strat, y_ranks=p_ai_ranks_strat)
        lo = dec[0]["p_range"][0] if dec else float("nan")
        hi = dec[-1]["p_range"][1] if dec else float("nan")
        hi_p = p_ai >= p_hi
        lo_p = p_ai <= p_lo
        delta, dp = cliffs_delta(dens[hi_p], dens[lo_p])
        grad_cat[k] = {
            "deciles": dec,
            "weighted_rho": round(wc["weighted_rho"], 4),
            "neg_cell_share": round(wc["neg_cell_share"], 4),
            "n_cells": wc["n_cells"],
            "stratified_rho": round(st["weighted_rho"], 4),
            "stratified_cells": st["n_cells"],
            "cliff_delta_top_vs_bottom": round(delta, 4),
            "cliff_p": float(dp),
            "decile_p_range": [lo, hi],
        }
        print(f"  {k}: wcell ρ={wc['weighted_rho']:+.4f} "
              f"(长度分层 {st['weighted_rho']:+.4f})，δ={delta:+.4f}")

    word_rho = {}
    for w in ALL_WORDS:
        arr = expl["word_counts"][w]
        dens = arr[sel].astype(float) / body * 100
        if dens.std() == 0:
            continue
        wc = within_cell_spearman(dens, p_ai, cells, min_cell_n=MIN_CELL_N,
                                  groups=groups, y_ranks=p_ai_ranks)
        word_rho[w] = {"category": WORD_CATEGORY[w],
                       "weighted_rho": round(wc["weighted_rho"], 4),
                       "neg_cell_share": round(wc["neg_cell_share"], 4),
                       "n_cells": wc["n_cells"]}

    # ---- 互动关联（与 engagement_drivers 同协议）----
    model = pd.read_csv(FEATURES_PATH,
                        usecols=["month", "domain", "content_len", "n_hashtags",
                                 "likes", "collections"])
    if len(model) != n_scanned:
        raise SystemExit("features 与 scores 行数不一致")
    mm2 = int((model["content_len"].to_numpy()
               != scans["exploring"]["body_len"]).sum())
    if mm2:
        raise SystemExit(f"features content_len 与扫描不一致：{mm2}")
    likes_mismatch = int((model["likes"].to_numpy()
                          != scans["exploring"]["likes"]).sum())
    stance = np.column_stack([
        scans["exploring"]["cat_counts"][k].astype(float)
        / np.maximum(scans["exploring"]["body_len"], 1) * 100
        for k in CATEGORY_ORDER])
    target_likes = np.log10(1 + np.clip(model["likes"].to_numpy(dtype=float), 0, None))
    y_coll = np.log10(1 + np.clip(model["collections"].to_numpy(dtype=float), 0, None))
    month_all = model["month"].astype(str).to_numpy()
    dom_all = model["domain"].astype(str).to_numpy()
    keep_eng = (month_all >= MONTH_FROM) & (scans["exploring"]["body_len"] >= MIN_BODY_LEN)
    eng_cells_raw = np.asarray([f"{m}|{d}" for m, d in zip(month_all, dom_all)])
    X = np.column_stack([stance, scores["p_ai"].to_numpy(dtype=float),
                         scans["exploring"]["body_len"].astype(float),
                         model["n_hashtags"].to_numpy(dtype=float)])
    feature_names = ["hedge", "boost", "intens", "vocative", "p_ai",
                     "content_len", "n_hashtags"]
    eng = {}
    for target, y in (("likes", target_likes), ("collections", y_coll)):
        Xk = X[keep_eng]
        yk = y[keep_eng]
        ck = eng_cells_raw[keep_eng]
        prep = prepare_fe(Xk, ck, MIN_CELL_N)
        fit = fe_ols(Xk, yk, ck, prepared=prep)
        boot = cluster_bootstrap_fe_ols(fit["X_demeaned"], fit["y_demeaned"],
                                        fit["cell_codes"], n_boot=N_BOOT,
                                        seed=BOOT_SEED)
        ci = np.percentile(boot, [2.5, 97.5], axis=0)
        eng[target] = {
            "features": feature_names,
            "coef": {n: round(float(v), 4) for n, v in zip(feature_names, fit["coef"])},
            "ci_low": {n: round(float(v), 4) for n, v in zip(feature_names, ci[0])},
            "ci_high": {n: round(float(v), 4) for n, v in zip(feature_names, ci[1])},
            "n_obs": fit["n_obs"], "n_cells": fit["n_cells"],
        }
        print(f"  互动 FE（{target}）: n={fit['n_obs']:,}, "
              f"hedge β={fit['coef'][0]:+.4f} [{ci[0][0]:+.4f}, {ci[1][0]:+.4f}]")
    uni_eng = {}
    groups_eng = cell_groups(eng_cells_raw[keep_eng], MIN_CELL_N)
    y_ranks = cell_rank_cache(target_likes[keep_eng], groups_eng)
    yc_ranks = cell_rank_cache(y_coll[keep_eng], groups_eng)
    for i, k in enumerate(CATEGORY_ORDER):
        uni_eng[k] = {
            "likes": within_cell_spearman(X[keep_eng, i], target_likes[keep_eng],
                                          eng_cells_raw[keep_eng],
                                          min_cell_n=MIN_CELL_N,
                                          groups=groups_eng, y_ranks=y_ranks),
            "collections": within_cell_spearman(X[keep_eng, i], y_coll[keep_eng],
                                                eng_cells_raw[keep_eng],
                                                min_cell_n=MIN_CELL_N,
                                                groups=groups_eng,
                                                y_ranks=yc_ranks),
        }

    # ---- AIGC 家族 / 单模型指纹 ----
    aigc = scans["aigc"]
    ok = aigc["body_len"] >= MIN_BODY_LEN
    fam_agg: dict = defaultdict(lambda: {"n": 0, "chars": 0, "counts": defaultdict(int)})
    mod_agg: dict = defaultdict(lambda: {"n": 0, "chars": 0, "counts": defaultdict(int)})
    for i in np.flatnonzero(ok):
        for agg, key in ((fam_agg, aigc["family"][i]), (mod_agg, aigc["model"][i])):
            slot = agg[key]
            slot["n"] += 1
            slot["chars"] += int(aigc["body_len"][i])
            for k in CATEGORY_ORDER:
                slot["counts"][k] += int(aigc["cat_counts"][k][i])

    def agg_to_table(agg):
        out = {}
        for key, slot in agg.items():
            if not key:
                continue
            out[key] = {
                "n": slot["n"],
                "density100": {
                    k: round(slot["counts"][k] / slot["chars"] * 100, 4)
                    if slot["chars"] else 0.0
                    for k in CATEGORY_ORDER},
            }
        return out

    family_table = agg_to_table(fam_agg)
    model_table = agg_to_table(mod_agg)

    # ---- 季度趋势（去重；原始 + 固定领域权重）----
    q_sel = keeps["exploring"][expl["body_len"][keeps["exploring"]] >= MIN_BODY_LEN]
    q_df = pd.DataFrame({
        "quarter": [f"{m[:4]}Q{(int(m[4:6]) - 1) // 3 + 1}" if m else "未知"
                    for m in (expl["month"][i] for i in q_sel)],
        "domain": [expl["domain"][i] for i in q_sel],
        "body_len": expl["body_len"][q_sel].astype(float),
    })
    for k in CATEGORY_ORDER:
        q_df[k] = expl["cat_counts"][k][q_sel].astype(float) / q_df["body_len"] * 100
    q_df = q_df[q_df["quarter"] != "未知"]
    q_counts = q_df.groupby("quarter").size()
    valid_q = sorted(q_counts[q_counts >= MIN_QUARTER_N].index)
    q_df = q_df[q_df["quarter"].isin(valid_q)]
    quarterly = {"quarters": valid_q, "n": {q: int(q_counts[q]) for q in valid_q},
                 "raw": {}, "domain_std": {}, "trend": {}, "share_any": {}}
    for k in CATEGORY_ORDER:
        raw = q_df.groupby("quarter")[k].mean().reindex(valid_q)
        std = domain_standardized_series(q_df, k, "quarter").reindex(valid_q)
        rho, p = spearmanr(range(len(valid_q)), raw.to_numpy())
        rho_s, p_s = spearmanr(range(len(valid_q)), std.to_numpy())
        quarterly["raw"][k] = [round(float(v), 4) for v in raw]
        quarterly["domain_std"][k] = [round(float(v), 4) for v in std]
        quarterly["trend"][k] = {"rho": round(float(rho), 4), "p": float(p),
                                 "rho_domain_std": round(float(rho_s), 4),
                                 "p_domain_std": float(p_s)}
        share = q_df.assign(hit=q_df[k] > 0).groupby("quarter")["hit"].mean().reindex(valid_q)
        quarterly["share_any"][k] = [round(float(v), 4) for v in share]

    # ---- 结果 ----
    out = {
        "meta": {
            "min_body_len": MIN_BODY_LEN, "min_cell_n": MIN_CELL_N,
            "month_from": MONTH_FROM, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "min_quarter_n": MIN_QUARTER_N,
            "exploring_content_len_mismatch": mismatch,
            "exploring_likes_mismatch": likes_mismatch,
            "note": "人类/探索集按 note_id 去重（保留元数据最完整一条）；"
                    "AIGC 无 note_id 逐记录；p_ai 按行序对齐 paper_exploring_scores.csv.gz",
        },
        "lexicon": {k: {"label": CATEGORY_LABELS[k], "words": CATEGORY_WORDS[k][1]}
                    for k in CATEGORY_ORDER},
        "subset_rates": {
            name: {
                "n": subset_rates[name]["n"],
                "chars": subset_rates[name]["chars"],
                "categories": subset_rates[name]["categories"],
            } for name in SUBSET_ORDER},
        "word_rates": word_table,
        "exploring_gradient": {
            "n": int(len(sel)),
            "short_excluded": int((expl["body_len"][keep] < MIN_BODY_LEN).sum()),
            "n_cells": int(len(np.unique(cells))),
            "categories": grad_cat,
            "words": word_rho,
        },
        "engagement": {
            "month_from": MONTH_FROM, "min_body_len": MIN_BODY_LEN,
            "univariate": {
                k: {t: {"weighted_rho": round(v[t]["weighted_rho"], 4),
                        "neg_cell_share": round(v[t]["neg_cell_share"], 4),
                        "n_cells": v[t]["n_cells"]}
                    for t in ("likes", "collections")}
                for k, v in uni_eng.items()},
            "fe": eng,
        },
        "family": {"families": family_table, "models": model_table},
        "quarterly": quarterly,
    }
    save_json(out, BASE + "stance_results.json")
    print("已写出 stance_results.json")
    write_markdown(out)
    plot_lexicon(subset_rates, word_table)
    plot_gradient({"categories": grad_cat}, word_rho, eng["likes"],
                  eng["collections"], family_table)
    print("已写出 stance_results.md 与 insights/stance_*.png")


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def write_markdown(out: dict) -> None:
    meta = out["meta"]
    rates = out["subset_rates"]
    grad = out["exploring_gradient"]
    eng = out["engagement"]
    q = out["quarterly"]
    labels = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)",
              "exploring": "探索集(后LLM)"}
    lines = [
        "# 语气情态分析结果（Stance）",
        "",
        f"> 由 `stance_analysis.py` 全量运行生成；正文清洗与 `paper_features.py` 一致，"
        f"人类/探索集按 note_id 去重；正文 ≥{meta['min_body_len']} 字；"
        f"探索集 p_ai 按行序对齐 `paper_exploring_scores.csv.gz`"
        f"（content_len 校验 mismatch={meta['exploring_content_len_mismatch']}）。",
        "",
        "## 1 跨子集语气标记使用密度（次/百字）",
        "",
        "| 子集 | 有效帖数 | " + " | ".join(CATEGORY_LABELS[k] for k in CATEGORY_ORDER) + " |",
        "|---|---:|" + "---:|" * len(CATEGORY_ORDER),
    ]
    for name in SUBSET_ORDER:
        cats = rates[name]["categories"]
        lines.append(
            f"| {labels[name]} | {rates[name]['n']:,} | "
            + " | ".join(f"{cats[k]['density100']:.3f}" for k in CATEGORY_ORDER)
            + " |")
    lines += [
        "",
        "> 注：三个子集的正文字数分布不同，密度按「总命中次数/总字符数」合并计算；",
        "> 「谨慎表达」与论文特征 `hum_lex100` 有词表重叠（应该/非常），"
        "「程度强化」与 `AI_WORDS` 有重叠（超级/简直），逐词结果见 `stance_results.json`。",
        "",
        "## 2 探索集内的 P(AI) 梯度（去重，正文 ≥%d 字）" % meta["min_body_len"],
        "",
        "| 类别 | 单元内加权 ρ | 负单元占比 | 长度分层 ρ | 首尾十分位 δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in CATEGORY_ORDER:
        g = grad["categories"][k]
        lines.append(
            f"| {CATEGORY_LABELS[k]} | {g['weighted_rho']:+.4f} | "
            f"{g['neg_cell_share']:.1%} | {g['stratified_rho']:+.4f} | "
            f"{g['cliff_delta_top_vs_bottom']:+.3f} |")
    lines += [
        "",
        f"（样本 {grad['n']:,} 条 / {grad['n_cells']} 个「月份×领域」单元；"
        f"长度分层 = 月份×领域×长度分箱单元。）",
        "",
        "**逐词单元内 ρ（Top/Bottom 8）：**",
        "",
    ]
    items = sorted(grad["words"].items(), key=lambda kv: kv[1]["weighted_rho"])
    show = items[:8] + items[-8:]
    lines.append("| 词 | 类别 | 单元内 ρ | 负单元占比 |")
    lines.append("|---|---|---:|---:|")
    for w, r in show:
        lines.append(f"| {w} | {CATEGORY_LABELS[r['category']]} | "
                     f"{r['weighted_rho']:+.4f} | {r['neg_cell_share']:.0%} |")
    lines += [
        "",
        "## 3 互动偏效应（月份×领域 FE + 整群 bootstrap %d 次）" % meta["n_boot"],
        "",
        "| 特征 | 点赞 β [95% CI] | 收藏 β [95% CI] |",
        "|---|---:|---:|",
    ]
    for name in eng["fe"]["likes"]["features"]:
        fl, fc = eng["fe"]["likes"], eng["fe"]["collections"]
        lines.append(
            f"| {name} | {fl['coef'][name]:+.4f} "
            f"[{fl['ci_low'][name]:+.4f}, {fl['ci_high'][name]:+.4f}] | "
            f"{fc['coef'][name]:+.4f} "
            f"[{fc['ci_low'][name]:+.4f}, {fc['ci_high'][name]:+.4f}] |")
    lines += [
        "",
        f"（n={eng['fe']['likes']['n_obs']:,}，{eng['fe']['likes']['n_cells']} 个单元；"
        f"特征已 z-score，β 为每 1 SD 的偏效应；{meta['month_from']} 起、正文 ≥"
        f"{meta['min_body_len']} 字。）",
        "",
        "## 4 AIGC 家族语气指纹（次/百字）",
        "",
        "| 家族 | n | " + " | ".join(CATEGORY_LABELS[k] for k in CATEGORY_ORDER) + " |",
        "|---|---:|" + "---:|" * len(CATEGORY_ORDER),
    ]
    for fam in FAMILIES:
        slot = out["family"]["families"].get(fam)
        if not slot:
            continue
        lines.append(
            f"| {FAMILY_NAME.get(fam, fam)} | {slot['n']} | "
            + " | ".join(f"{slot['density100'][k]:.3f}" for k in CATEGORY_ORDER)
            + " |")
    lines += [
        "",
        "## 5 季度趋势（探索集去重）",
        "",
        "| 季度 | n | " + " | ".join(CATEGORY_LABELS[k] for k in CATEGORY_ORDER) + " |",
        "|---|---:|" + "---:|" * len(CATEGORY_ORDER),
    ]
    for i, qq in enumerate(q["quarters"]):
        lines.append(
            f"| {qq} | {q['n'][qq]:,} | "
            + " | ".join(f"{q['raw'][k][i]:.3f}" for k in CATEGORY_ORDER)
            + " |")
    lines.append("")
    for k in CATEGORY_ORDER:
        t = q["trend"][k]
        lines.append(f"- {CATEGORY_LABELS[k]}：季度趋势 ρ={t['rho']:+.3f}（p={t['p']:.3f}），"
                     f"固定领域权重后 ρ={t['rho_domain_std']:+.3f}（p={t['p_domain_std']:.3f}）。")
    lines += [
        "",
        "图表见 `insights/stance_lexicon.png`、`insights/stance_gradient.png`。",
    ]
    with open(BASE + "stance_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
