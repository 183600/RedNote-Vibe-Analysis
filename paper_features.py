# -*- coding: utf-8 -*-
"""第二步：对三个数据集统一抽取可解释风格特征 -> paper_features_*.csv.gz

设计要点：
1. 清洗爬取痕迹 [话题]，把话题标签 #xxx 从正文剥离单独计数，
   避免标签灌水污染字符级风格特征，也避免跨数据集爬取格式差异造成偏差。
2. 全部特征为可解释的字符级/词表级统计，不依赖分词，速度快。
3. AI 词表与人类词表来自 log-odds 分析（paper_lexicon.py），仅保留风格词、
   排除话题词（如"穿搭""夏日"），且全部为不易歧义的 2 字以上词。
"""
import json
import re
import gzip
import csv
import os
from functools import partial

from analysis_utils import close_pool, make_pool

BASE = os.path.dirname(os.path.abspath(__file__)) + "/"

HUATI = re.compile(r"\[话题\]")
HASHTAG = re.compile(r"#[^#\s]{1,30}")
XHS_EMO = re.compile(r"\[[^\[\]]{1,8}R\]")
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF"
    "\U00002B00-\U00002BFF\U0001F900-\U0001F9FF✨❤️✅⭐]"
)
LIST_LINE = re.compile(r"^\s*(\d+[\.、\)）]|[①-⑩]|[✅🔥📌💡⭐️\-·•])")

# 来自 log-odds 分析的风格词表（排除话题词，仅保留跨领域风格词）
AI_WORDS = [
    "简直", "瞬间", "超级", "绝对", "宝藏", "神器", "必备", "搞定", "轻松",
    "完美", "居然", "姐妹们", "宝子们", "家人们", "谁懂", "绝绝子", "拿捏",
    "天花板", "闭眼入", "亲测", "码住", "yyds", "快冲", "冲鸭", "记得",
]
HUMAN_WORDS = [
    "但是", "因为", "所以", "如果", "然后", "之后", "比较", "应该", "需要",
    "对于", "能够", "时候", "很多", "没有", "非常", "可以",
]
PARTICLES = list("的了是我你就都很有不")

RE_DIGIT = re.compile(r"\d")

FIELDS = [
    # 元数据
    "dataset", "domain", "month", "model_family", "model",
    "likes", "collections", "comments",
    # 长度与结构
    "content_len", "n_lines", "avg_line_len", "n_hashtags",
    "colon_line_ratio", "list_line_ratio",
    # 标点（每百字）
    "excl100", "ques100", "ellip100", "tilde100",
    "comma100", "pause100", "period100",
    # 表情
    "emoji100", "emoji_distinct", "xhsemo100",
    # 虚词（每百字）
    *[f"p_{c}" for c in PARTICLES],
    # 词表（每百字）
    "ai_lex100", "hum_lex100",
    # 多样性
    "uniq_char_ratio", "uniq_bigram_ratio",
    # 标题
    "title_len", "title_excl", "title_ques", "title_emoji",
    "title_pipe", "title_num", "title_bracket",
]


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def extract(title, content):
    feats = {}
    content = HUATI.sub("", content or "")
    title = HUATI.sub("", title or "")
    hashtags = HASHTAG.findall(content)
    body = HASHTAG.sub(" ", content)
    body = body.strip()
    n = len(body)
    n_safe = max(n, 1)

    lines = [l for l in body.split("\n") if l.strip()]
    n_lines = max(len(lines), 1)
    feats["content_len"] = n
    feats["n_lines"] = len(lines)
    feats["avg_line_len"] = round(n / n_lines, 2)
    feats["n_hashtags"] = len(hashtags)
    feats["colon_line_ratio"] = round(
        sum(1 for l in lines if ("：" in l[:14] or ":" in l[:14])) / n_lines, 4)
    feats["list_line_ratio"] = round(
        sum(1 for l in lines if LIST_LINE.match(l)) / n_lines, 4)

    def per100(cnt, _round=round, _n=n_safe):
        return _round(cnt / _n * 100, 4)

    feats["excl100"] = per100(sum(map(body.count, "！!")))
    feats["ques100"] = per100(sum(map(body.count, "？?")))
    feats["ellip100"] = per100(sum(map(body.count, ("…", "。。。", "..."))))
    feats["tilde100"] = per100(sum(map(body.count, "~～")))
    feats["comma100"] = per100(body.count("，"))
    feats["pause100"] = per100(body.count("、"))
    feats["period100"] = per100(body.count("。"))

    emojis = EMOJI.findall(body)
    feats["emoji100"] = per100(len(emojis))
    feats["emoji_distinct"] = len(set(emojis))
    feats["xhsemo100"] = per100(len(XHS_EMO.findall(body)))

    for c, cnt in zip(PARTICLES, map(body.count, PARTICLES)):
        feats[f"p_{c}"] = per100(cnt)

    feats["ai_lex100"] = per100(sum(map(body.count, AI_WORDS)))
    feats["hum_lex100"] = per100(sum(map(body.count, HUMAN_WORDS)))

    feats["uniq_char_ratio"] = round(len(set(body)) / n_safe, 4)
    n_bigrams = max(n - 1, 0)
    uniq_bigrams = len({body[i:i + 2] for i in range(n_bigrams)})
    feats["uniq_bigram_ratio"] = round(uniq_bigrams / max(n_bigrams, 1), 4)

    feats["title_len"] = len(title)
    feats["title_excl"] = int("！" in title or "!" in title)
    feats["title_ques"] = int("？" in title or "?" in title)
    feats["title_emoji"] = int(bool(EMOJI.search(title)))
    feats["title_pipe"] = int("｜" in title or "|" in title)
    feats["title_num"] = int(bool(RE_DIGIT.search(title)))
    feats["title_bracket"] = int("【" in title or "】" in title)
    return feats


def row_of(d: dict, dataset: str) -> dict:
    """由一条记录构造输出行（纯函数，可在子进程中并行执行）。"""
    content = d.get("note_content") or d.get("desc") or ""
    title = d.get("note_title") or ""
    row = extract(title, content)
    lt = str(d.get("local_time") or "")
    row.update({
        "dataset": dataset,
        "domain": d.get("domain") or "",
        "month": lt[:6] if len(lt) >= 6 else "",
        "model_family": d.get("model_family") or "",
        "model": d.get("model") or "",
        "likes": to_int(d.get("liked_count") or d.get("likes")),
        "collections": to_int(d.get("collected_count") or d.get("collections")),
        "comments": to_int(d.get("comments_count") or d.get("comments")),
    })
    return row


def row_from_line(line: str, dataset: str) -> dict:
    """解析一行 JSONL 并抽取输出行（并行任务的入口，可 pickle）。"""
    return row_of(json.loads(line), dataset)


def process(path, dataset, out_writer, pool=None):
    """流式抽取特征并写出；传 pool 时并行抽取，imap 保序因此输出逐行一致。"""
    n = 0
    if pool is None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line or line.isspace():
                    continue
                row = row_from_line(line, dataset)
                out_writer.writerow([row[k] for k in FIELDS])
                n += 1
        return n
    with open(path, encoding="utf-8") as f:
        lines = (line for line in f if line and not line.isspace())
        for row in pool.imap(partial(row_from_line, dataset=dataset), lines,
                             chunksize=64):
            out_writer.writerow([row[k] for k in FIELDS])
            n += 1
    return n


def main():
    jobs = [
        ("training_set_human_fixed.jsonl", "human"),
        ("training_set_aigc.jsonl", "aigc"),
        ("exploring_set_fixed.jsonl", "exploring"),
    ]
    pool = make_pool()
    try:
        for fname, ds in jobs:
            out = BASE + f"paper_features_{ds}.csv.gz"
            with gzip.open(out, "wt", encoding="utf-8", newline="") as g:
                w = csv.writer(g)
                w.writerow(FIELDS)
                n = process(BASE + fname, ds, w, pool=pool)
            print(f"{ds}: {n} rows -> {out}")
    finally:
        close_pool(pool)


if __name__ == "__main__":
    main()
