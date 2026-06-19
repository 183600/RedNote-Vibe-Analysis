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

BASE = "/home/qwe12345678/RedNote-Vibe-Analysis/"

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

    def per100(cnt):
        return round(cnt / n_safe * 100, 4)

    feats["excl100"] = per100(body.count("！") + body.count("!"))
    feats["ques100"] = per100(body.count("？") + body.count("?"))
    feats["ellip100"] = per100(body.count("…") + body.count("。。。") + body.count("..."))
    feats["tilde100"] = per100(body.count("~") + body.count("～"))
    feats["comma100"] = per100(body.count("，"))
    feats["pause100"] = per100(body.count("、"))
    feats["period100"] = per100(body.count("。"))

    emojis = EMOJI.findall(body)
    feats["emoji100"] = per100(len(emojis))
    feats["emoji_distinct"] = len(set(emojis))
    feats["xhsemo100"] = per100(len(XHS_EMO.findall(body)))

    for c in PARTICLES:
        feats[f"p_{c}"] = per100(body.count(c))

    feats["ai_lex100"] = per100(sum(body.count(w) for w in AI_WORDS))
    feats["hum_lex100"] = per100(sum(body.count(w) for w in HUMAN_WORDS))

    feats["uniq_char_ratio"] = round(len(set(body)) / n_safe, 4)
    bigrams = [body[i:i + 2] for i in range(0, max(n - 1, 0))]
    feats["uniq_bigram_ratio"] = round(len(set(bigrams)) / max(len(bigrams), 1), 4)

    feats["title_len"] = len(title)
    feats["title_excl"] = int("！" in title or "!" in title)
    feats["title_ques"] = int("？" in title or "?" in title)
    feats["title_emoji"] = int(bool(EMOJI.search(title)))
    feats["title_pipe"] = int("｜" in title or "|" in title)
    feats["title_num"] = int(bool(re.search(r"\d", title)))
    feats["title_bracket"] = int("【" in title or "】" in title)
    return feats


def process(path, dataset, out_writer):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
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
            out_writer.writerow([row.get(k, "") for k in FIELDS])
            n += 1
    return n


def main():
    jobs = [
        ("training_set_human_fixed.jsonl", "human"),
        ("training_set_aigc.jsonl", "aigc"),
        ("exploring_set_fixed.jsonl", "exploring"),
    ]
    for fname, ds in jobs:
        out = BASE + f"paper_features_{ds}.csv.gz"
        with gzip.open(out, "wt", encoding="utf-8", newline="") as g:
            w = csv.writer(g)
            w.writerow(FIELDS)
            n = process(BASE + fname, ds, w)
        print(f"{ds}: {n} rows -> {out}")


if __name__ == "__main__":
    main()
