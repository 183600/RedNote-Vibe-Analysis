# -*- coding: utf-8 -*-
"""第一步：数据驱动地发现区分 AIGC 与人类的词汇（informative Dirichlet log-odds）。

输出 paper_lexicon_results.json：AIGC 偏好词与人类偏好词各 Top-60，
供人工筛选出"风格词表"（排除纯话题词），并写入论文表格。
"""
import json
import re
import math
from collections import Counter

import jieba

from analysis_utils import BASE, load_posts, make_pool, close_pool

# 与 paper_features 同口径：`#标签[话题]#` 的闭合 `#` 一并删除，避免残留在正文。
HUATI = re.compile(r"\[话题\]#?")
HASHTAG = re.compile(r"#[^#\s]{1,30}")
XHS_EMO = re.compile(r"\[[^\[\]]{1,8}R\]")
HAN_WORD = re.compile(r"[一-鿿]")
PINYIN_WORD = re.compile(r"[a-zA-Z]{2,10}")


def clean(text):
    """去掉爬取痕迹与话题标签，保留正文风格。"""
    text = HUATI.sub("", text)
    text = HASHTAG.sub(" ", text)
    return text


def count_clean_words(cleaned):
    """对已经 `clean` 过的文本分词计数（与 count_words 相同的过滤口径）。

    供已经自行清洗过文本的调用方复用，避免同一段文本被清洗/分词两遍。
    """
    cnt = Counter()
    for w in jieba.cut(cleaned):
        w = w.strip()
        if not w:
            continue
        # 只保留含中文的词或常见拼音化网络词（预编译正则，避免逐词查找缓存）
        if HAN_WORD.search(w) or PINYIN_WORD.fullmatch(w):
            cnt[w] += 1
    return cnt


def count_words(texts):
    cnt = Counter()
    for t in texts:
        cnt.update(count_clean_words(clean(t)))
    return cnt


def _count_chunk(texts):
    """worker：批量清洗+分词并合并为一个 Counter（计数与顺序无关）。"""
    cnt = Counter()
    for t in texts:
        cnt.update(count_clean_words(clean(t)))
    return cnt


def count_words_parallel(texts, pool=None, chunk: int = 400):
    """与 count_words 结果完全相同；pool 非空时按块并行。"""
    if pool is None:
        return count_words(texts)
    cnt = Counter()
    for c in pool.map(_count_chunk,
                      [texts[i:i + chunk] for i in range(0, len(texts), chunk)]):
        cnt.update(c)
    return cnt


def log_odds(cnt_a, cnt_b, min_count=30):
    """Monroe et al. (2008) informative Dirichlet prior log-odds, z-score."""
    n_a = sum(cnt_a.values())
    n_b = sum(cnt_b.values())
    n_p = n_a + n_b
    alpha0 = 500.0  # 先验强度
    results = {}
    for w in set(cnt_a) | set(cnt_b):
        c = cnt_a[w] + cnt_b[w]
        if c < min_count:
            continue
        pw = c / n_p
        aw = alpha0 * pw
        la = math.log((cnt_a[w] + aw) / (n_a + alpha0 - cnt_a[w] - aw))
        lb = math.log((cnt_b[w] + aw) / (n_b + alpha0 - cnt_b[w] - aw))
        delta = la - lb
        var = 1.0 / (cnt_a[w] + aw) + 1.0 / (cnt_b[w] + aw)
        results[w] = delta / math.sqrt(var)
    return results


def rate(cnt, n, w):
    return cnt[w] / n * 1e4  # 每万词


def main():
    print("加载文本 ...")
    human = load_posts(BASE + "training_set_human_fixed.jsonl", sample=20000)
    aigc = load_posts(BASE + "training_set_aigc.jsonl")
    print(f"human sample={len(human)}, aigc={len(aigc)}")

    print("分词统计 ...")
    pool = make_pool(preload=jieba.initialize)
    try:
        cnt_h = count_words_parallel(human, pool)
        cnt_a = count_words_parallel(aigc, pool)
    finally:
        close_pool(pool)
    n_h = sum(cnt_h.values())
    n_a = sum(cnt_a.values())
    print(f"tokens: human={n_h}, aigc={n_a}")

    z = log_odds(cnt_a, cnt_h, min_count=30)
    top_aigc = sorted(z.items(), key=lambda x: -x[1])[:60]
    top_human = sorted(z.items(), key=lambda x: x[1])[:60]

    out = {
        "aigc_words": [
            {
                "word": w,
                "z": round(s, 2),
                "aigc_per10k": round(rate(cnt_a, n_a, w), 2),
                "human_per10k": round(rate(cnt_h, n_h, w), 2),
            }
            for w, s in top_aigc
        ],
        "human_words": [
            {
                "word": w,
                "z": round(s, 2),
                "aigc_per10k": round(rate(cnt_a, n_a, w), 2),
                "human_per10k": round(rate(cnt_h, n_h, w), 2),
            }
            for w, s in top_human
        ],
    }
    with open(BASE + "paper_lexicon_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n=== AIGC 偏好词 Top-40 ===")
    for item in out["aigc_words"][:40]:
        print(f"{item['word']}\tz={item['z']}\tA={item['aigc_per10k']}\tH={item['human_per10k']}")
    print("\n=== 人类偏好词 Top-40 ===")
    for item in out["human_words"][:40]:
        print(f"{item['word']}\tz={item['z']}\tA={item['aigc_per10k']}\tH={item['human_per10k']}")


if __name__ == "__main__":
    main()
