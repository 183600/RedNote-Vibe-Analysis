# -*- coding: utf-8 -*-
"""深潜分析之十六：具体性 / 实指表达（concreteness）。

既有分析覆盖了词汇表、语气情态、框架模板、标题、符号与互动结构，但把
「AI 味」落到**信息具体程度**层面的系统分析还缺失：帖子给出的是可核验的
具体信息（数字、带单位的量值、地点），还是模糊的笼统表述？本脚本用四个
先验定义的标记类别（不依赖任何监督信号）回答四组问题：

1. 跨子集对比：三类语料谁更「实指」？发生率（帖子层）与密度（字符层）、
   领域标准化后的差异，以及长度控制（分箱 + 长度匹配差异）；
2. 野外梯度：探索集内 P(AI) 越高的帖子是否越模糊？单元内（月份×领域）
   秩相关是否稳健（长度分层）？
3. 互动回报：在控制 P(AI)、正文长度与标签数后，具体性标记与点赞、收藏的
   偏效应方向与大小（月份×领域固定效应 + 整群 bootstrap CI）；
4. 生成器指纹与时间演化：17 个 LLM 的家族指纹差异；探索集季度趋势
   （原始 + 固定领域权重）。

口径说明：
- 正文清洗复用 analysis_utils.clean_body（与 paper_features.py 一致）；
- 人类/探索集按 note_id 去重（保留元数据最完整一条，规则同 deep_dive_analysis），
  AIGC 无 note_id，逐记录处理；
- 探索集 p_ai 按行序对齐 paper_exploring_scores.csv.gz，脚本内用 content_len
  逐条校验（不一致时直接报错退出）；
- 发生率（share）按帖子层统计「至少命中一次」的比例；密度按「总命中次数 /
  总字符数 × 100」合并计算；正文不足 MIN_BODY_LEN 字的帖子（纯标签/空正文等）
  不参与统计，数量单独报告；
- 互动回归与 engagement_drivers_analysis.py 完全同协议（特征先 z-score
  再单元内去均值、单元 ≥100 条、整群 bootstrap 300 次），额外加入四个
  具体性密度特征。

标记（先验定义，不做数据驱动的筛选）：
- 具体数字 num：阿拉伯数字（未带单位的量值），如 3、150、2024；
- 带单位量值 unit：数字 + 金额/百分比/度量/时间单位，如 50元、3.5%、5kg、
  30分钟、3天、2024年；子类 money/percent/measure/time 单独计数；
- 模糊量词 vague：很多 一些 各种 许多 不少 大量 部分 多数 ……（反向标记）；
- 地点实指 place：固定地名表（省级行政区 + 主要城市）的命中，如 北京、上海、
  成都、三亚。

输出：
  insights/concreteness_lexicon.png   发生率 / 密度 / 长度分箱发生率
  insights/concreteness_gradient.png  P(AI) 梯度 / 单元内 ρ / 互动偏效应 /
                                      季度趋势 / 家族指纹 / 单标记 AUC
  concreteness_results.json           全部数值结果
  concreteness_results.md             结果摘要
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from analysis_utils import (BASE, DATASET_LABELS, INSIGHTS, LENGTH_BINS, PALETTE,
                            body_cache, bootstrap_auc, cell_groups,
                            cell_rank_cache, clean_body, cliffs_delta, content_of,
                            cv_auc, decile_table, dedup_indices,
                            domain_standardized_series, engagement_of,
                            init_plot_style, length_bin_codes,
                            parse_local_time, save_fig, save_json,
                            standardize_weights, wilson_ci,
                            run_parallel, within_cell_spearman)
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

MIN_BODY_LEN = 50        # 具体性标记在更短正文（纯标签/空正文）上无意义
MIN_CELL_N = 100         # 单元内统计的最小单元样本量（与论文口径一致）
MONTH_FROM = "202301"    # 互动分析起始月份（与 paper_engagement 一致）
N_BOOT = 300             # 整群 bootstrap 次数
AUC_BOOT = 300           # 单标记 AUC bootstrap 次数
BOOT_SEED = 42
MIN_QUARTER_N = 300
MIN_FAMILY_N = 100

CATEGORY_ORDER = ["num", "unit", "vague", "place"]
CATEGORY_LABELS = {"num": "具体数字", "unit": "带单位量值",
                   "vague": "模糊量词", "place": "地点实指"}
CATEGORY_COLORS = {"num": "#4C72B0", "unit": "#DD8452",
                   "vague": "#C44E52", "place": "#55A868"}
SUBSET_ORDER = ["human", "aigc", "exploring"]
SUBTYPE_ORDER = ["money", "percent", "measure", "time"]
SUBTYPE_LABELS = {"money": "金额", "percent": "百分比",
                  "measure": "度量", "time": "时间"}

# --------------------------------------------------------------------------
# 先验标记（正则按「最具体优先」排序：金额 > 百分比 > 度量 > 时间 > 裸数字）
# --------------------------------------------------------------------------
MASTER_RE = re.compile(r"""
 (?P<money>[¥￥$]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:万元|万块|块钱|元|块))
|(?P<percent>\d+(?:\.\d+)?\s*[%％]|百分之[零一二两三四五六七八九十百千万]+)
|(?P<measure>\d+(?:\.\d+)?\s*(?:mAh|kg|KG|Kg|公斤|斤|克|毫克|mg|毫升|ml|ML|升|厘米|cm|CM|毫米|mm|千米|公里|km|KM|米|寸|尺|摄氏度|大卡|千卡|卡路里|卡|毫安|瓦|GB|MB|TB|gb|mb|tb))
|(?P<time>\d+(?:\.\d+)?\s*(?:分钟|小时|天|周|星期|个月|月|年|岁|秒|点半?)|(?:\d{1,2}[:：]\d{2}))
 |(?P<num>\d+(?:\.\d+)?)
 """, re.X)
# 快速预筛选：MASTER_RE 的任一分支命中文本必含 数字/¥/￥/$/百，先在
# 整个文本上做一次廉价 search，无提示字符时跳过昂贵的逐匹配循环。
_MASTER_HINT_RE = re.compile(r"[\d¥￥$百]")
_UNIT_STRIP_RE = re.compile(r"[\d.\s]")

# MASTER_RE 保留为规范定义（测试与文档引用）；classify() 使用下面这套
# 等价但更快的实现：以提示字符定位候选起点（数字 / ¥ ￥ $ / 百），再按
# 「金额 > 百分比 > 度量 > 时间 > 裸数字」的优先级手工判定后缀，避免
# 多分支正则在长文本上的回溯开销。已用全量语料与模糊串做差分验证。
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_CLOCK_RE = re.compile(r"\d{1,2}[:：]\d{2}")
_PERCENT_CN_RE = re.compile(r"百分之[零一二两三四五六七八九十百千万]+")
_MONEY_SYMBOL_RE = re.compile(r"[¥￥$]\s*")
_WS_RE = re.compile(r"\s*")

# 各分支单位表（与 MASTER_RE 逐字一致）；按首字符索引、同首字符最长优先，
# 与原交替分支内「更具体优先」的书写顺序等价（跨分支不存在互为前缀的单位）。
_UNIT_TABLE = {
    "money": ("万元", "万块", "块钱", "元", "块"),
    "percent": ("%", "％"),
    "measure": ("mAh", "kg", "KG", "Kg", "公斤", "斤", "克", "毫克", "mg",
                "毫升", "ml", "ML", "升", "厘米", "cm", "CM", "毫米", "mm",
                "千米", "公里", "km", "KM", "米", "寸", "尺", "摄氏度", "大卡",
                "千卡", "卡路里", "卡", "毫安", "瓦", "GB", "MB", "TB", "gb",
                "mb", "tb"),
    "time": ("分钟", "小时", "天", "周", "星期", "个月", "月", "年", "岁",
             "秒", "点半", "点"),
}
_SUFFIX_BY_FIRST: dict = {}
for _unit_cat, _unit_list in _UNIT_TABLE.items():
    for _unit_str in _unit_list:
        _SUFFIX_BY_FIRST.setdefault(_unit_str[0], []).append((_unit_str, _unit_cat))
for _unit_cands in _SUFFIX_BY_FIRST.values():
    _unit_cands.sort(key=lambda item: -len(item[0]))

VAGUE_WORDS = ["很多", "一些", "有些", "各种", "许多", "不少", "好多", "大量",
               "无数", "些许", "若干", "部分", "多数", "少数", "大多", "挺多",
               "蛮多", "一堆", "一批", "一系列", "各种各样", "或多或少"]
VAGUE_RE = re.compile("|".join(VAGUE_WORDS))

# 固定地名表：34 个省级行政区 + 主要城市（先验清单，不做数据驱动筛选）。
PROVINCES = ["北京", "上海", "天津", "重庆", "河北", "山西", "辽宁", "吉林",
             "黑龙江", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
             "湖北", "湖南", "广东", "海南", "四川", "贵州", "云南", "陕西",
             "甘肃", "青海", "台湾", "内蒙古", "广西", "西藏", "宁夏", "新疆",
             "香港", "澳门"]
CITIES = ["广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "苏州", "长沙",
          "郑州", "青岛", "宁波", "东莞", "无锡", "厦门", "福州", "合肥", "大连",
          "沈阳", "哈尔滨", "济南", "昆明", "南宁", "贵阳", "太原", "石家庄",
          "兰州", "西宁", "银川", "乌鲁木齐", "呼和浩特", "南昌", "温州", "佛山",
          "泉州", "徐州", "烟台", "常州", "珠海", "中山", "惠州", "嘉兴", "南通",
          "绍兴", "台州", "金华", "保定", "洛阳", "临沂", "潍坊", "扬州", "盐城",
          "泰州", "镇江", "湖州", "芜湖", "桂林", "大理", "丽江", "张家界",
          "敦煌", "三亚", "海口", "拉萨", "秦皇岛", "威海", "日照", "湛江",
          "汕头", "潮州", "开封", "襄阳", "宜昌", "岳阳", "株洲", "湘潭",
          "衡阳", "绵阳", "乐山", "宜宾", "遵义", "宝鸡", "咸阳", "天水",
          "包头", "鄂尔多斯", "大庆", "齐齐哈尔", "吉林市", "鞍山", "抚顺",
          "丹东", "锦州", "营口", "承德", "唐山", "廊坊", "沧州", "邢台",
          "邯郸", "淄博", "济宁", "泰安", "聊城", "菏泽", "枣庄", "德州",
          "滨州", "淮安", "连云港", "宿迁", "衢州", "丽水", "舟山", "莆田",
          "三明", "龙岩", "南平", "赣州", "九江", "景德镇", "上饶", "宜春",
          "吉安", "黄石", "十堰", "荆州", "荆门", "黄冈", "孝感", "咸宁",
          "阳江", "茂名", "肇庆", "江门", "清远", "韶关", "梅州", "河源",
          "汕尾", "云浮", "揭阳", "西双版纳", "香格里拉", "九寨沟", "峨眉山"]
# 长度降序 + 字典序打破并列：`key=len` 的稳定排序在同长度词之间依赖 set
# 迭代顺序（受 PYTHONHASHSEED 影响），会导致产物内容不变但序列化顺序随机。
_PLACE_NAMES = sorted(set(PROVINCES) | set(CITIES), key=lambda s: (-len(s), s))
PLACE_RE = re.compile("|".join(map(re.escape, _PLACE_NAMES)))

CATEGORY_NOTE = {
    "num": "阿拉伯数字（未带单位）",
    "unit": "数字 + 金额/百分比/度量/时间单位",
    "vague": "模糊量词（先验词表）",
    "place": "固定地名表（省级行政区 + 主要城市）",
}

FAMILY_NAME = {"anthropic": "Claude", "deepseek": "DeepSeek", "gemini": "Gemini",
               "glm": "GLM", "gpt": "GPT", "qwen": "Qwen"}
FAMILIES = ["anthropic", "deepseek", "gemini", "glm", "gpt", "qwen"]


# --------------------------------------------------------------------------
# 纯函数（可单元测试）
# --------------------------------------------------------------------------
def unit_token_of(raw: str) -> str:
    """从命中的量值文本中剥离数字与空白，返回单位记号（纯符号返回符号本身）。"""
    token = _UNIT_STRIP_RE.sub("", raw)
    return token or "符号"


def classify(text: str) -> tuple:
    """统计一段文本的四类标记。

    返回 (counts, subtypes, unit_tokens, place_tokens)：
    - counts: {num/unit/vague/place: 次数}，unit = 四个子类之和；
    - subtypes: {money/percent/measure/time: 次数}；
    - unit_tokens: 单位记号 Counter（如 元/天/kg）；
    - place_tokens: 命中地名 Counter。
    裸数字（num）与带单位量值（unit）互斥：金额/百分比/度量/时间优先匹配。

    实现与 MASTER_RE 逐分支等价：人民币符号 + 数字为金额；数字（可含小数）
    + 空白 + 单位按 金额/百分比/度量/时间 判定；「百分之 + 中文数字」计为
    百分比；`d{1,2}[:：]d{2}` 计为时间；其余数字为裸数字。命中区间与
    MASTER_RE.finditer 一致（下一个候选起点从匹配末尾继续扫描）。
    """
    counts = {k: 0 for k in CATEGORY_ORDER}
    subtypes = {k: 0 for k in SUBTYPE_ORDER}
    unit_tokens: Counter = Counter()
    place_tokens: Counter = Counter()
    unit = money = percent = measure = time = 0
    next_start = 0
    for hint in _MASTER_HINT_RE.finditer(text):
        pos = hint.start()
        if pos < next_start:
            continue
        ch = text[pos]
        if ch in "¥￥$":
            sym = _MONEY_SYMBOL_RE.match(text, pos)
            number = _NUMBER_RE.match(text, sym.end()) if sym else None
            if number is None:            # 符号后无数字：不构成金额
                next_start = pos + 1
                continue
            end = number.end()
            unit += 1
            money += 1
            unit_tokens[unit_token_of(text[pos:end])] += 1
            next_start = end
            continue
        if ch != "百":
            clock = _CLOCK_RE.match(text, pos)
            if clock is not None:         # 时间分支的第二种写法
                unit += 1
                time += 1
                unit_tokens[unit_token_of(clock.group(0))] += 1
                next_start = clock.end()
                continue
            number = _NUMBER_RE.match(text, pos)
            if number is None:
                next_start = pos + 1
                continue
            end = number.end()
            unit_pos = _WS_RE.match(text, end).end()
            cands = _SUFFIX_BY_FIRST.get(text[unit_pos:unit_pos + 1])
            if cands is not None:
                for suffix, cat in cands:
                    if text.startswith(suffix, unit_pos):
                        unit += 1
                        if cat == "money":
                            money += 1
                        elif cat == "percent":
                            percent += 1
                        elif cat == "measure":
                            measure += 1
                        else:
                            time += 1
                        unit_tokens[unit_token_of(suffix)] += 1
                        next_start = unit_pos + len(suffix)
                        break
                else:
                    counts["num"] += 1
                    next_start = end
                continue
            counts["num"] += 1
            next_start = end
            continue
        pct = _PERCENT_CN_RE.match(text, pos)   # 百分比分支的中文写法
        if pct is not None:
            unit += 1
            percent += 1
            unit_tokens[unit_token_of(pct.group(0))] += 1
            next_start = pct.end()
        else:
            next_start = pos + 1
    counts["unit"] = unit
    subtypes["money"] = money
    subtypes["percent"] = percent
    subtypes["measure"] = measure
    subtypes["time"] = time
    counts["vague"] = len(VAGUE_RE.findall(text))
    places = PLACE_RE.findall(text)
    counts["place"] = len(places)
    place_tokens.update(places)
    return counts, subtypes, unit_tokens, place_tokens


def density_per_100(count: int, body_len: int) -> float:
    """次数 / 正文字数 * 100；空正文返回 0。"""
    return count / body_len * 100 if body_len > 0 else 0.0


def concrete_count(counts: dict) -> int:
    """复合「具体性」命中数 = 数字 + 带单位量值 + 地点（模糊量词为反向标记）。"""
    return counts["num"] + counts["unit"] + counts["place"]


def length_matched_gap(hit_a, n_a, hit_b, n_b, weights) -> dict:
    """按固定长度权重加权的人−机发生率差异及解析 SE。

    `hit_a/n_a`、`hit_b/n_b` 为逐分箱命中数与样本量；`weights` 为分箱权重
    （本脚本用人类样本的长度分布）。缺失分箱（任一侧 n=0）不参与并按剩余
    权重重新归一化。返回 {gap, se, ci_low, ci_high, used_bins}。
    """
    w = np.asarray(weights, dtype=float).copy()
    p_a = np.asarray(hit_a, dtype=float) / np.maximum(np.asarray(n_a, dtype=float), 1)
    p_b = np.asarray(hit_b, dtype=float) / np.maximum(np.asarray(n_b, dtype=float), 1)
    ok = (np.asarray(n_a) > 0) & (np.asarray(n_b) > 0)
    if not ok.any():
        return {"gap": float("nan"), "se": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "used_bins": 0}
    w[~ok] = 0.0
    w = np.asarray([v for v in standardize_weights(dict(enumerate(w))).values()])
    gap = float(np.sum(w * (p_a - p_b)))
    var = float(np.sum(w ** 2 * (p_a * (1 - p_a) / np.maximum(n_a, 1)
                                 + p_b * (1 - p_b) / np.maximum(n_b, 1))))
    se = float(np.sqrt(var))
    return {"gap": gap, "se": se, "ci_low": gap - 1.96 * se,
            "ci_high": gap + 1.96 * se, "used_bins": int(ok.sum())}


def quarter_of(month: str | None) -> str | None:
    """YYYYMM -> YYYYQn；非法输入返回 None。"""
    if not month or len(month) < 6 or not month[:6].isdigit():
        return None
    return f"{month[:4]}Q{(int(month[4:6]) - 1) // 3 + 1}"


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
def scan_corpus(path: str) -> dict:
    """单次流式扫描一个语料，返回逐记录数组 + 词表级计数。"""
    body_len, note_id, meta, month, domain = [], [], [], [], []
    family, model = [], []
    likes, collections = [], []
    cat_counts = {k: [] for k in CATEGORY_ORDER}
    subtype_counts = {k: [] for k in SUBTYPE_ORDER}
    unit_tokens: Counter = Counter()
    place_tokens: Counter = Counter()
    n_total = 0
    # 重复正文（探索集约 45%）直接复用上一次的分类结果；缓存值只读：
    # 下面 unit_tokens.update(units) / place_tokens.update(places) 均不改动
    # 被缓存的 Counter 本体。
    cached = body_cache(classify)
    for rec in iter_cached(path):
        n_total += 1
        content = content_of(rec)
        body = clean_body(content)
        counts, subtypes, units, places = cached(body)
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
        # 正文仅用于「是否有非空白内容」的评分：strip 的布尔值与原先
        # `"".join(content.split())` 完全一致，并省掉逐帖的 split/join 拷贝。
        # 打分项与 analysis_utils.dedup_indices 文档的共享规则一致（含评论数），
        # 与其他全量扫描脚本的去重保留记录完全相同。
        score = (int(has_meta) + int(bool(likes_v)) + int(bool(colls))
                 + int(bool(comms)) + int(dt is not None)
                 + int(bool(content.strip())))
        meta.append(score)
        family.append(rec.get("model_family") or "")
        model.append(rec.get("model") or "")
        for k in CATEGORY_ORDER:
            cat_counts[k].append(counts[k])
        for k in SUBTYPE_ORDER:
            subtype_counts[k].append(subtypes[k])
        unit_tokens.update(units)
        place_tokens.update(places)
    return {
        "path": path, "n_total": n_total,
        "body_len": np.asarray(body_len, dtype=np.int32),
        "note_id": note_id, "meta": np.asarray(meta, dtype=np.int32),
        "month": month, "domain": domain,
        "likes": np.asarray(likes, dtype=float),
        "collections": np.asarray(collections, dtype=float),
        "family": family, "model": model,
        "cat_counts": {k: np.asarray(v, dtype=np.int32)
                       for k, v in cat_counts.items()},
        "subtype_counts": {k: np.asarray(v, dtype=np.int32)
                           for k, v in subtype_counts.items()},
        "unit_tokens": unit_tokens, "place_tokens": place_tokens,
    }


def dedup_keep(scan: dict) -> np.ndarray:
    """按 note_id 去重（保留元数据最完整一条），返回排序后的保留索引。"""
    keep = dedup_indices(scan["note_id"], scan["meta"])
    return np.asarray(sorted(keep), dtype=int)


def rates_block(scan: dict, keep: np.ndarray, min_len: int = MIN_BODY_LEN) -> dict:
    """一个子集的四类标记发生率（帖子层）与密度（字符层）。"""
    sel = keep[scan["body_len"][keep] >= min_len]
    n = int(len(sel))
    chars = int(scan["body_len"][sel].sum())
    cat = {}
    for k in CATEGORY_ORDER:
        c = scan["cat_counts"][k][sel].astype(float)
        k_hit = int((c > 0).sum())
        lo, hi = wilson_ci(k_hit, n) if n else (0.0, 0.0)
        cat[k] = {
            "n": n, "chars": chars,
            "share": float(k_hit / n) if n else 0.0,
            "share_ci_low": float(lo), "share_ci_high": float(hi),
            "density100": float(c.sum() / chars * 100) if chars else 0.0,
        }
    conc = (scan["cat_counts"]["num"][sel].astype(float)
            + scan["cat_counts"]["unit"][sel].astype(float)
            + scan["cat_counts"]["place"][sel].astype(float))
    k_hit = int((conc > 0).sum())
    lo, hi = wilson_ci(k_hit, n) if n else (0.0, 0.0)
    cat["concrete"] = {
        "n": n, "chars": chars,
        "share": float(k_hit / n) if n else 0.0,
        "share_ci_low": float(lo), "share_ci_high": float(hi),
        "density100": float(conc.sum() / chars * 100) if chars else 0.0,
    }
    return {"n": n, "chars": chars, "categories": cat}


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_lexicon(subset_rates: dict, length_bins: dict) -> None:
    """三联图：发生率 / 密度 / 长度分箱发生率（长度控制）。"""
    init_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.4),
                             gridspec_kw={"width_ratios": [1.05, 1.05, 1.2]})
    cats = CATEGORY_ORDER + ["concrete"]
    labels = dict(CATEGORY_LABELS)
    labels["concrete"] = "任一具体性标记"

    # ---- (a) 发生率 ----
    ax = axes[0]
    y = np.arange(len(cats))
    h = 0.24
    for j, subset in enumerate(SUBSET_ORDER):
        vals = [subset_rates[subset]["categories"][k]["share"] * 100 for k in cats]
        pos = y + (j - 1) * h
        ax.barh(pos, vals, height=h, color=PALETTE[subset],
                label=DATASET_LABELS[subset].replace("(", "（").replace(")", "）"))
        for yy, v in zip(pos, vals):
            ax.annotate(f"{v:.1f}", xy=(v, yy), xytext=(3, -3),
                        textcoords="offset points", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in cats])
    ax.set_xlabel("至少命中一次的帖子占比（%）")
    ax.set_title("(a) 具体性标记发生率（帖子层）")
    ax.legend(loc="lower right")

    # ---- (b) 密度 ----
    ax = axes[1]
    for j, subset in enumerate(SUBSET_ORDER):
        vals = [subset_rates[subset]["categories"][k]["density100"] for k in cats]
        pos = y + (j - 1) * h
        ax.barh(pos, vals, height=h, color=PALETTE[subset])
        for yy, v in zip(pos, vals):
            ax.annotate(f"{v:.3f}", xy=(v, yy), xytext=(3, -3),
                        textcoords="offset points", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("密度（次/百字）")
    ax.set_title("(b) 具体性标记密度（字符层）")

    # ---- (c) 长度分箱发生率（复合具体性标记） ----
    ax = axes[2]
    for subset in SUBSET_ORDER:
        rec = length_bins[subset]
        x = np.arange(len(rec["labels"]))
        # 空分箱（n=0）画成 NaN 断开，避免把「无样本」当成 0%
        ok = np.asarray([b["n"] > 0 for b in rec["bins"]])
        yv = np.where(ok, [b["share"] for b in rec["bins"]], np.nan) * 100
        lo = np.where(ok, [b["ci_low"] for b in rec["bins"]], np.nan) * 100
        hi = np.where(ok, [b["ci_high"] for b in rec["bins"]], np.nan) * 100
        ax.plot(x, yv, "o-", color=PALETTE[subset], lw=1.9, ms=5,
                label=DATASET_LABELS[subset].replace("(", "（").replace(")", "）"))
        ax.fill_between(x, lo, hi, color=PALETTE[subset], alpha=0.15)
    ax.set_xticks(range(len(length_bins["human"]["labels"])))
    ax.set_xticklabels(length_bins["human"]["labels"], fontsize=9)
    ax.set_xlabel("正文长度分箱")
    ax.set_ylabel("任一具体性标记的帖子占比（%）")
    ax.set_title("(c) 长度控制：分箱内的具体性发生率")
    ax.legend(fontsize=9)

    fig.tight_layout()
    save_fig(fig, INSIGHTS + "concreteness_lexicon.png")


def plot_gradient(grad: dict, eng: dict, quarterly: dict,
                  family_table: dict, auc: dict) -> None:
    """六联图：梯度 / 单元内 ρ / 互动偏效应 / 季度趋势 / 家族指纹 / AUC。"""
    init_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 12.8))

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
    ax.set_title("(a) 探索集：具体性标记随 P(AI) 的梯度（含 95% CI）")
    ax.legend(fontsize=9)

    # ---- (b) 单元内 ρ（原始 + 长度分层） ----
    ax = axes[0, 1]
    ypos = np.arange(len(CATEGORY_ORDER))
    h = 0.32
    raw = [grad["categories"][k]["weighted_rho"] for k in CATEGORY_ORDER]
    strat = [grad["categories"][k]["stratified_rho"] for k in CATEGORY_ORDER]
    ax.barh(ypos - h / 2, raw, height=h, color=[CATEGORY_COLORS[k]
                                                for k in CATEGORY_ORDER],
            alpha=0.95, label="月份×领域单元内 ρ")
    ax.barh(ypos + h / 2, strat, height=h,
            color=[CATEGORY_COLORS[k] for k in CATEGORY_ORDER], alpha=0.5,
            label="再加长度分层的单元内 ρ")
    for i, (r, s) in enumerate(zip(raw, strat)):
        ax.annotate(f"{r:+.3f}", xy=(r, ypos[i] - h / 2),
                    xytext=(5 if r >= 0 else -5, -3), textcoords="offset points",
                    fontsize=8, ha="left" if r >= 0 else "right")
        ax.annotate(f"{s:+.3f}", xy=(s, ypos[i] + h / 2),
                    xytext=(5 if s >= 0 else -5, -3), textcoords="offset points",
                    fontsize=8, ha="left" if s >= 0 else "right")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([CATEGORY_LABELS[k] for k in CATEGORY_ORDER])
    ax.set_xlabel("加权 Spearman ρ（密度 vs P(AI)）")
    ax.set_title("(b) 野外 P(AI) 梯度的稳健性（长度分层对照）")
    ax.legend(fontsize=8.5, loc="lower left")

    # ---- (c) 互动偏效应森林图 ----
    ax = axes[0, 2]
    names = list(eng["likes"]["coef"].keys())
    order = list(reversed(names))
    ypos = np.arange(len(order))
    for target, marker, offset, color in (
            ("likes", "o", -0.14, "#4C72B0"),
            ("collections", "^", 0.14, "#DD8452")):
        fe = eng[target]
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
    labels = dict(CATEGORY_LABELS)
    labels.update({"p_ai": "P(AI)", "content_len": "正文长度",
                   "n_hashtags": "标签数"})
    ax.set_yticklabels([labels.get(n, n) for n in order])
    ax.set_xlabel("单元内 FE β（每 1 SD → log10(1+互动)）")
    ax.set_title(f"(c) 具体性特征的互动偏效应（n={eng['likes']['n_obs']:,}，"
                 f"{eng['likes']['n_cells']} 单元，整群 bootstrap）")
    ax.legend(loc="lower right")

    # ---- (d) 季度趋势（固定领域权重） ----
    ax = axes[1, 0]
    qs = quarterly["quarters"]
    x = np.arange(len(qs))
    for k in CATEGORY_ORDER:
        vals = quarterly["domain_std"][k]
        t = quarterly["trend"][k]
        ax.plot(x, vals, "o-", color=CATEGORY_COLORS[k], lw=1.8, ms=4.5,
                label=f"{CATEGORY_LABELS[k]}（ρ={t['rho_domain_std']:+.2f}）")
    ax.set_xticks(x)
    ax.set_xticklabels(qs, rotation=35, fontsize=8.5, ha="right")
    ax.set_ylabel("密度（次/百字，固定领域权重）")
    ax.set_title("(d) 探索集季度趋势（领域标准化）")
    ax.legend(fontsize=8.5)

    # ---- (e) 家族指纹热力图 ----
    ax = axes[1, 1]
    fams = [f for f in FAMILIES if family_table.get(f, {}).get("n", 0) >= MIN_FAMILY_N]
    fams.sort(key=lambda f: -family_table[f]["share"]["concrete"])
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
    ax.set_title("(e) AIGC 训练集的家族具体性指纹（格内为密度，次/百字）")

    # ---- (f) 单标记 AUC（人类 vs AIGC） ----
    ax = axes[1, 2]
    keys = CATEGORY_ORDER + ["concrete", "cv_lr"]
    labels2 = dict(CATEGORY_LABELS)
    labels2["concrete"] = "任一具体性标记（计数）"
    labels2["cv_lr"] = "四标记 LR（5 折 CV）"
    mark_keys = CATEGORY_ORDER + ["concrete"]
    aucs = [auc["markers"][k]["auc"] for k in mark_keys] + [auc["cv_lr"]["auc"]]
    # cv_auc() 只返回点估计，没有 bootstrap CI：LR 行用 NaN 占位而不是把点
    # 估计当作区间端点（否则会画出一条零宽度的“95% CI”线段）。
    los = [auc["markers"][k]["ci_low"] for k in mark_keys] + [float("nan")]
    his = [auc["markers"][k]["ci_high"] for k in mark_keys] + [float("nan")]
    colors = [CATEGORY_COLORS[k] for k in CATEGORY_ORDER] + ["#8172B3", "#937860"]
    ypos = np.arange(len(keys))
    for i, (v, lo, hi, c) in enumerate(zip(aucs, los, his, colors)):
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [ypos[i], ypos[i]], color=c, lw=2.0,
                    solid_capstyle="butt")
        ax.scatter([v], [ypos[i]], s=52, color=c, zorder=3)
        ax.annotate(f"{v:.3f}", xy=(v, ypos[i]), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8.5)
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels2[k] for k in keys])
    ax.set_xlabel("AUC（得分越高=该标记越偏向人类；虚线=随机）")
    ax.set_xlim(0.45, max(h for h in his if np.isfinite(h)) + 0.03)
    ax.set_title("(f) 标记单独区分人类 / AIGC 的能力（单标记含 95% CI）")

    fig.tight_layout()
    save_fig(fig, INSIGHTS + "concreteness_gradient.png")


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

    # ---- 去重 + 子集发生率 ----
    keeps = {name: dedup_keep(sc) for name, sc in scans.items()}
    subset_rates = {name: rates_block(scans[name], keeps[name])
                    for name in SUBSET_ORDER}
    for name in SUBSET_ORDER:
        sc = scans[name]
        short = int((sc["body_len"][keeps[name]] < MIN_BODY_LEN).sum())
        print(f"  {name}: 去重后 {len(keeps[name]):,} 条，正文 <{MIN_BODY_LEN} 字 "
              f"排除 {short:,} 条")

    # ---- 领域标准化发生率（固定为人类领域权重） ----
    human_kept = keeps["human"][scans["human"]["body_len"][keeps["human"]]
                                >= MIN_BODY_LEN]
    human_dom_share = pd.Series(
        [scans["human"]["domain"][i] for i in human_kept]).value_counts(
        normalize=True).to_dict()
    domain_std = {}
    for name in SUBSET_ORDER:
        sc = scans[name]
        sel = keeps[name][sc["body_len"][keeps[name]] >= MIN_BODY_LEN]
        df = pd.DataFrame({"domain": [sc["domain"][i] for i in sel]})
        for k in CATEGORY_ORDER:
            df[k] = sc["cat_counts"][k][sel] > 0
        df["concrete"] = (sc["cat_counts"]["num"][sel]
                          + sc["cat_counts"]["unit"][sel]
                          + sc["cat_counts"]["place"][sel]) > 0
        block = {}
        for k in CATEGORY_ORDER + ["concrete"]:
            means = df.groupby("domain")[k].mean().to_dict()
            w = standardize_weights({d: human_dom_share.get(d, 0.0) for d in means})
            block[k] = float(sum(means[d] * w[d] for d in means))
        domain_std[name] = block
        print(f"  {name}: 领域标准化发生率 concrete="
              f"{block['concrete']:.3f}（原始 {subset_rates[name]['categories']['concrete']['share']:.3f}）")

    # ---- 长度分箱发生率（长度控制） ----
    length_bins = {}
    for name in SUBSET_ORDER:
        sc = scans[name]
        sel = keeps[name][sc["body_len"][keeps[name]] >= MIN_BODY_LEN]
        codes = length_bin_codes(sc["body_len"][sel])
        hit = (sc["cat_counts"]["num"][sel]
               + sc["cat_counts"]["unit"][sel]
               + sc["cat_counts"]["place"][sel]) > 0
        bins = []
        for b in range(len(LENGTH_BINS)):
            m = codes == b
            n = int(m.sum())
            k_hit = int(hit[m].sum())
            lo, hi = wilson_ci(k_hit, n) if n else (0.0, 0.0)
            bins.append({"bin": b, "n": n, "share": k_hit / n if n else 0.0,
                         "ci_low": float(lo), "ci_high": float(hi)})
        length_bins[name] = {"labels": [lab for _, _, lab in LENGTH_BINS],
                             "bins": bins}
    w_human = np.asarray([b["n"] for b in length_bins["human"]["bins"]],
                         dtype=float)
    gap_matched = {}
    for name in ("aigc", "exploring"):
        gap_matched[name] = length_matched_gap(
            [b["share"] * b["n"] for b in length_bins["human"]["bins"]],
            [b["n"] for b in length_bins["human"]["bins"]],
            [b["share"] * b["n"] for b in length_bins[name]["bins"]],
            [b["n"] for b in length_bins[name]["bins"]],
            w_human)
        print(f"  长度匹配差异 human−{name}: {gap_matched[name]['gap']:+.3f} "
              f"[{gap_matched[name]['ci_low']:+.3f}, {gap_matched[name]['ci_high']:+.3f}]")

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
    # p_ai 是所有类别的同一对侧向量：预算一次单元内平均秩供全部调用复用
    # （原先每个类别都会重算片上秩，且长度分层单元也每次重新分组）；
    # 上下十分位阈值同样只与 p_ai 有关，循环外只算一次。
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

    # ---- 单标记 AUC（human vs AIGC，标签 human=1） ----
    h_sel = keeps["human"][scans["human"]["body_len"][keeps["human"]]
                           >= MIN_BODY_LEN]
    a_sel = scans["aigc"]["body_len"] >= MIN_BODY_LEN
    a_sel = np.flatnonzero(a_sel)
    y = np.r_[np.ones(len(h_sel), dtype=int), np.zeros(len(a_sel), dtype=int)]
    auc_markers = {}
    for k in CATEGORY_ORDER + ["concrete"]:
        if k == "concrete":
            h_counts = (scans["human"]["cat_counts"]["num"][h_sel]
                        + scans["human"]["cat_counts"]["unit"][h_sel]
                        + scans["human"]["cat_counts"]["place"][h_sel])
            a_counts = (scans["aigc"]["cat_counts"]["num"][a_sel]
                        + scans["aigc"]["cat_counts"]["unit"][a_sel]
                        + scans["aigc"]["cat_counts"]["place"][a_sel])
        else:
            h_counts = scans["human"]["cat_counts"][k][h_sel]
            a_counts = scans["aigc"]["cat_counts"][k][a_sel]
        score = np.r_[h_counts, a_counts].astype(float)
        auc_markers[k] = bootstrap_auc(y, score, n_boot=AUC_BOOT, seed=BOOT_SEED)
        print(f"  AUC[{k}] = {auc_markers[k]['auc']:.4f} "
              f"[{auc_markers[k]['ci_low']:.4f}, {auc_markers[k]['ci_high']:.4f}]")
    # 四标记 5 折 CV（逻辑回归）
    cv_df = pd.DataFrame({
        "dataset": ["human"] * len(h_sel) + ["aigc"] * len(a_sel)})
    for k in CATEGORY_ORDER:
        cv_df[k] = np.r_[scans["human"]["cat_counts"][k][h_sel],
                         scans["aigc"]["cat_counts"][k][a_sel]].astype(float)
    cv_lr = cv_auc(cv_df, CATEGORY_ORDER, seed=BOOT_SEED)
    print(f"  四标记 LR 5 折 CV AUC = {cv_lr['auc']:.4f}")

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
    # 互动指标一律采用 paper_features_*.csv.gz 的口径（与论文互动分析一致；
    # 个别记录 liked_count=0 但 likes 非空，features 的 `or` 回退取后者），
    # 与流式扫描的 engagement_of 口径可能存在差异，仅记录不报错（同 stance）。
    likes_mismatch = int((model["likes"].to_numpy()
                          != scans["exploring"]["likes"]).sum())
    marker_mat = np.column_stack([
        scans["exploring"]["cat_counts"][k].astype(float)
        / np.maximum(scans["exploring"]["body_len"], 1) * 100
        for k in CATEGORY_ORDER])
    target_likes = np.log10(1 + np.clip(model["likes"].to_numpy(dtype=float), 0, None))
    y_coll = np.log10(1 + np.clip(model["collections"].to_numpy(dtype=float), 0, None))
    month_all = model["month"].astype(str).to_numpy()
    dom_all = model["domain"].astype(str).to_numpy()
    keep_eng = (month_all >= MONTH_FROM) & (scans["exploring"]["body_len"] >= MIN_BODY_LEN)
    eng_cells_raw = np.asarray([f"{m}|{d}" for m, d in zip(month_all, dom_all)])
    X = np.column_stack([marker_mat, scores["p_ai"].to_numpy(dtype=float),
                         scans["exploring"]["body_len"].astype(float),
                         model["n_hashtags"].to_numpy(dtype=float)])
    feature_names = CATEGORY_ORDER + ["p_ai", "content_len", "n_hashtags"]
    eng = {}
    for target, yv in (("likes", target_likes), ("collections", y_coll)):
        Xk = X[keep_eng]
        yk = yv[keep_eng]
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
              f"num β={fit['coef'][0]:+.4f} [{ci[0][0]:+.4f}, {ci[1][0]:+.4f}]")
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
    fam_agg: dict = defaultdict(lambda: {"n": 0, "chars": 0,
                                         "counts": defaultdict(int),
                                         "concrete_hits": 0})
    mod_agg: dict = defaultdict(lambda: {"n": 0, "chars": 0,
                                         "counts": defaultdict(int),
                                         "concrete_hits": 0})
    for i in np.flatnonzero(ok):
        conc = concrete_count({k: int(aigc["cat_counts"][k][i])
                               for k in CATEGORY_ORDER})
        for agg, key in ((fam_agg, aigc["family"][i]), (mod_agg, aigc["model"][i])):
            slot = agg[key]
            slot["n"] += 1
            slot["chars"] += int(aigc["body_len"][i])
            slot["concrete_hits"] += int(conc > 0)
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
                "share": {"concrete": round(slot["concrete_hits"] / slot["n"], 4)
                          if slot["n"] else 0.0},
            }
        return out

    family_table = agg_to_table(fam_agg)
    model_table = agg_to_table(mod_agg)

    # ---- 季度趋势（去重；原始 + 固定领域权重）----
    q_sel = keeps["exploring"][expl["body_len"][keeps["exploring"]] >= MIN_BODY_LEN]
    q_df = pd.DataFrame({
        "quarter": [quarter_of(expl["month"][i]) or "未知" for i in q_sel],
        "domain": [expl["domain"][i] for i in q_sel],
        "body_len": expl["body_len"][q_sel].astype(float),
    })
    for k in CATEGORY_ORDER:
        q_df[k] = expl["cat_counts"][k][q_sel].astype(float) / q_df["body_len"] * 100
    conc_q = (expl["cat_counts"]["num"][q_sel].astype(float)
              + expl["cat_counts"]["unit"][q_sel].astype(float)
              + expl["cat_counts"]["place"][q_sel].astype(float))
    q_df["concrete"] = conc_q / q_df["body_len"] * 100
    q_df = q_df[q_df["quarter"] != "未知"]
    q_counts = q_df.groupby("quarter").size()
    valid_q = sorted(q_counts[q_counts >= MIN_QUARTER_N].index)
    q_df = q_df[q_df["quarter"].isin(valid_q)]
    quarterly = {"quarters": valid_q,
                 "n": {qq: int(q_counts[qq]) for qq in valid_q},
                 "raw": {}, "domain_std": {}, "trend": {},
                 "share_any": {}}
    for k in CATEGORY_ORDER + ["concrete"]:
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
            "month_from": MONTH_FROM, "n_boot": N_BOOT, "auc_boot": AUC_BOOT,
            "boot_seed": BOOT_SEED, "min_quarter_n": MIN_QUARTER_N,
            "exploring_content_len_mismatch": mismatch,
            "exploring_likes_mismatch": likes_mismatch,
            "note": "人类/探索集按 note_id 去重（保留元数据最完整一条）；"
                    "AIGC 无 note_id 逐记录；p_ai 按行序对齐 "
                    "paper_exploring_scores.csv.gz；单标记 AUC 标签 human=1；"
                    "top_units/top_places 为全语料原始记录聚合（未去重、不限长度）",
        },
        "lexicon": {
            "categories": {k: {"label": CATEGORY_LABELS[k], "definition": CATEGORY_NOTE[k]}
                           for k in CATEGORY_ORDER},
            "subtypes": {k: SUBTYPE_LABELS[k] for k in SUBTYPE_ORDER},
            "vague_words": VAGUE_WORDS,
            "place_names": _PLACE_NAMES,
        },
        "subset_rates": {
            name: {
                "n": subset_rates[name]["n"],
                "chars": subset_rates[name]["chars"],
                "categories": subset_rates[name]["categories"],
                "subtypes": {
                    k: int(scans[name]["subtype_counts"][k][
                        keeps[name][scans[name]["body_len"][keeps[name]]
                                   >= MIN_BODY_LEN]].sum())
                    for k in SUBTYPE_ORDER},
                "top_units": dict(scans[name]["unit_tokens"].most_common(15)),
                "top_places": dict(scans[name]["place_tokens"].most_common(15)),
            } for name in SUBSET_ORDER},
        "domain_standardized_share": domain_std,
        "human_domain_weights": {k: round(v, 4) for k, v in human_dom_share.items()},
        "length_control": {
            "bins": [lab for _, _, lab in
                     __import__("analysis_utils").LENGTH_BINS],
            "by_subset": length_bins,
            "matched_gap": gap_matched,
        },
        "exploring_gradient": {
            "n": int(len(sel)),
            "short_excluded": int((expl["body_len"][keep] < MIN_BODY_LEN).sum()),
            "n_cells": int(len(np.unique(cells))),
            "categories": grad_cat,
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
        "auc": {"markers": auc_markers, "cv_lr": cv_lr},
        "quarterly": quarterly,
    }
    save_json(out, BASE + "concreteness_results.json")
    print("已写出 concreteness_results.json")
    write_markdown(out)
    plot_lexicon(subset_rates, length_bins)
    plot_gradient({"categories": grad_cat}, eng, quarterly, family_table, out["auc"])
    print("已写出 concreteness_results.md 与 insights/concreteness_*.png")


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def write_markdown(out: dict) -> None:
    meta = out["meta"]
    rates = out["subset_rates"]
    grad = out["exploring_gradient"]
    eng = out["engagement"]
    q = out["quarterly"]
    lines = [
        "# 具体性分析结果（Concreteness）",
        "",
        f"> 由 `concreteness_analysis.py` 全量运行生成；正文清洗与 `paper_features.py` "
        f"一致，人类/探索集按 note_id 去重；正文 ≥{meta['min_body_len']} 字；"
        f"探索集 p_ai 按行序对齐 `paper_exploring_scores.csv.gz`"
        f"（content_len 校验 mismatch={meta['exploring_content_len_mismatch']}）。",
        "",
        "## 1 跨子集具体性标记（帖子层发生率与字符层密度）",
        "",
        "| 子集 | 有效帖数 | 具体数字 | 带单位量值 | 模糊量词 | 地点实指 | 任一具体性标记 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SUBSET_ORDER:
        cats = rates[name]["categories"]
        lines.append(
            f"| {DATASET_LABELS[name]} | {rates[name]['n']:,} | "
            + " | ".join(f"{cats[k]['share']:.1%}" for k in CATEGORY_ORDER)
            + f" | {cats['concrete']['share']:.1%} |")
    lines += [
        "",
        "密度（次/百字）：",
        "",
        "| 子集 | 具体数字 | 带单位量值 | 模糊量词 | 地点实指 | 任一具体性标记 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in SUBSET_ORDER:
        cats = rates[name]["categories"]
        lines.append(
            f"| {DATASET_LABELS[name]} | "
            + " | ".join(f"{cats[k]['density100']:.3f}" for k in CATEGORY_ORDER)
            + f" | {cats['concrete']['density100']:.3f} |")
    lines += [
        "",
        "领域标准化发生率（固定为人类领域权重）：",
        "",
        "| 子集 | 具体数字 | 带单位量值 | 模糊量词 | 地点实指 | 任一具体性标记 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in SUBSET_ORDER:
        d = out["domain_standardized_share"][name]
        lines.append(f"| {DATASET_LABELS[name]} | "
                     + " | ".join(f"{d[k]:.1%}" for k in CATEGORY_ORDER)
                     + f" | {d['concrete']:.1%} |")
    lines += [
        "",
        "带单位量值的子类构成（命中次数）：",
        "",
        "| 子集 | 金额 | 百分比 | 度量 | 时间 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in SUBSET_ORDER:
        sub = rates[name]["subtypes"]
        lines.append(f"| {DATASET_LABELS[name]} | "
                     + " | ".join(f"{sub[k]:,}" for k in SUBTYPE_ORDER) + " |")
    lines += [
        "",
        "## 2 长度控制（任一具体性标记的发生率，单个分箱内）",
        "",
        "| 长度分箱 | " + " | ".join(
            DATASET_LABELS[n] for n in SUBSET_ORDER) + " |",
        "|---|" + "---:|" * len(SUBSET_ORDER),
    ]
    for b in range(len(out["length_control"]["bins"])):
        cells = []
        for name in SUBSET_ORDER:
            rec = out["length_control"]["by_subset"][name]["bins"][b]
            cells.append("—（n=0）" if rec["n"] == 0
                         else f"{rec['share']:.1%}（n={rec['n']:,}）")
        lines.append(f"| {out['length_control']['bins'][b]} | " + " | ".join(cells) + " |")
    lines.append("")
    for name, tag in (("aigc", "AIGC"), ("exploring", "探索集")):
        g = out["length_control"]["matched_gap"][name]
        lines.append(f"- 长度匹配差异（人类−{tag}，按人类长度分布加权）："
                     f"{g['gap'] * 100:+.1f} 个百分点"
                     f" [{g['ci_low'] * 100:+.1f}, {g['ci_high'] * 100:+.1f}]"
                     f"（{g['used_bins']} 个分箱参与，n=0 的分箱不参与）。")
    lines += [
        "",
        "## 3 探索集内的 P(AI) 梯度（去重，正文 ≥%d 字）" % meta["min_body_len"],
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
        "## 4 互动偏效应（月份×领域 FE + 整群 bootstrap %d 次）" % meta["n_boot"],
        "",
        "| 特征 | 点赞 β [95% CI] | 收藏 β [95% CI] |",
        "|---|---:|---:|",
    ]
    for name in eng["fe"]["likes"]["features"]:
        fl, fc = eng["fe"]["likes"], eng["fe"]["collections"]
        label = CATEGORY_LABELS.get(name, {"p_ai": "P(AI)", "content_len": "正文长度",
                                           "n_hashtags": "标签数"}.get(name, name))
        lines.append(
            f"| {label} | {fl['coef'][name]:+.4f} "
            f"[{fl['ci_low'][name]:+.4f}, {fl['ci_high'][name]:+.4f}] | "
            f"{fc['coef'][name]:+.4f} "
            f"[{fc['ci_low'][name]:+.4f}, {fc['ci_high'][name]:+.4f}] |")
    lines += [
        "",
        f"（n={eng['fe']['likes']['n_obs']:,}，{eng['fe']['likes']['n_cells']} 个单元；"
        f"特征已 z-score，β 为每 1 SD 的偏效应；{meta['month_from']} 起、正文 ≥"
        f"{meta['min_body_len']} 字。）",
        "",
        "## 5 AIGC 家族具体性指纹（密度=次/百字；具体性发生率=含任一具体性标记的帖子占比）",
        "",
        "| 家族 | n | 具体数字 | 带单位量值 | 模糊量词 | 地点实指 | 具体性发生率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMILIES:
        slot = out["family"]["families"].get(fam)
        if not slot:
            continue
        lines.append(
            f"| {FAMILY_NAME.get(fam, fam)} | {slot['n']:,} | "
            + " | ".join(f"{slot['density100'][k]:.3f}" for k in CATEGORY_ORDER)
            + f" | {slot['share']['concrete']:.1%} |")
    auc = out["auc"]
    lines += [
        "",
        "## 6 单标记区分度（人类 vs AIGC，标签 human=1）",
        "",
        "| 标记 | AUC [95% CI] |",
        "|---|---:|",
    ]
    for k in CATEGORY_ORDER + ["concrete"]:
        a = auc["markers"][k]
        label = CATEGORY_LABELS.get(k, "任一具体性标记")
        lines.append(f"| {label} | {a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}] |")
    lines += [
        f"| 四标记逻辑回归（5 折 CV） | {auc['cv_lr']['auc']:.3f} |",
        "",
        "## 7 季度趋势（探索集去重）",
        "",
        "| 季度 | n | 具体数字 | 带单位量值 | 模糊量词 | 地点实指 | 任一具体性标记 |",
        "|---|---:|" + "---:|" * (len(CATEGORY_ORDER) + 1),
    ]
    for i, qq in enumerate(q["quarters"]):
        lines.append(
            f"| {qq} | {q['n'][qq]:,} | "
            + " | ".join(f"{q['domain_std'][k][i]:.3f}"
                         for k in CATEGORY_ORDER)
            + f" | {q['domain_std']['concrete'][i]:.3f} |")
    lines.append("")
    for k in CATEGORY_ORDER:
        t = q["trend"][k]
        lines.append(f"- {CATEGORY_LABELS[k]}：季度趋势 ρ={t['rho']:+.3f}（p={t['p']:.3f}），"
                     f"固定领域权重后 ρ={t['rho_domain_std']:+.3f}"
                     f"（p={t['p_domain_std']:.3f}）。")
    lines += [
        "",
        "图表见 `insights/concreteness_lexicon.png`、"
        "`insights/concreteness_gradient.png`。",
    ]
    with open(BASE + "concreteness_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
