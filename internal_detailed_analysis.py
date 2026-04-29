# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
深入分析：Human vs AIGC vs Exploring 内部差异和相关性
避免与已有分析重复，增加新的分析维度
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
import math


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_chinese(text):
    if not text:
        return []
    return re.findall(r"[\u4e00-\u9fff]+", text)


def get_text(item):
    """Get text field from different data formats"""
    if "note_content" in item:
        return item.get("note_content", "")
    return item.get("desc", "")


print("Loading data...")
human_data = load_jsonl("training_set_human.jsonl")
aigc_data = load_jsonl("training_set_aigc.jsonl")
exploring_data = load_jsonl("exploring_set.jsonl")

print(
    f"Human: {len(human_data)}, AIGC: {len(aigc_data)}, Exploring: {len(exploring_data)}"
)

positive_words = [
    "喜欢",
    "爱",
    "开心",
    "快乐",
    "幸福",
    "美好",
    "推荐",
    "赞",
    "棒",
    "太好了",
    "好看",
    "满意",
    "惊喜",
    "感动",
    "治愈",
    "温暖",
    "完美",
    "感谢",
    "加油",
    "优秀",
    "成功",
    "进步",
    "成长",
]
negative_words = [
    "焦虑",
    "难过",
    "痛苦",
    "累",
    "烦",
    "失望",
    "伤",
    "哭",
    "难",
    "怕",
    "担心",
    "恐惧",
    "抑郁",
    "崩溃",
    "糟糕",
    "无力",
    "压抑",
    "自责",
    "内耗",
    "后悔",
]

# ==================== 1. 内部领域分布 ====================
print("\n=== 1. Internal Domain Distribution ===")


def analyze_domain(data, name):
    domains = Counter()
    for item in data:
        d = item.get("domain", "Unknown")
        domains[d] += 1
    total = sum(domains.values())
    print(f"\n{name}:")
    for domain, count in domains.most_common(12):
        print(f"  {domain}: {count} ({count / total * 100:.1f}%)")


analyze_domain(human_data, "Human")
analyze_domain(aigc_data, "AIGC")
analyze_domain(exploring_data, "Exploring")

# ==================== 2. 情感分析 ====================
print("\n=== 2. Sentiment Analysis ===")


def sentiment(data, name, use_title=False):
    pos = neg = 0
    for item in data:
        if use_title:
            text = item.get("note_title", "")
        else:
            text = get_text(item) + " " + item.get("note_title", "")
        pos += sum(1 for w in positive_words if w in text)
        neg += sum(1 for w in negative_words if w in text)
    total = len(data)
    label = "Title" if use_title else "Content"
    print(
        f"\n{name} ({label}): Positive={pos}, Negative={neg}, Ratio={pos / neg if neg > 0 else pos:.2f}"
    )


print("\n=== 2. Sentiment Analysis (Content) ===")
sentiment(human_data, "Human", False)
sentiment(aigc_data, "AIGC", False)
sentiment(exploring_data, "Exploring", False)

print("\n=== 2b. Sentiment Analysis (Title) ===")
sentiment(human_data, "Human", True)
sentiment(aigc_data, "AIGC", True)
sentiment(exploring_data, "Exploring", True)

# ==================== 3. Engagement ====================
print("\n=== 3. Engagement Stats ===")


def engagement(data, name):
    likes = [item.get("liked_count", 0) for item in data]
    collects = [item.get("collected_count", 0) for item in data]
    comments = [item.get("comments_count", 0) for item in data]
    print(
        f"\n{name}: Likes={sum(likes) / len(likes):.1f}, Collects={sum(collects) / len(collects):.1f}, Comments={sum(comments) / len(comments):.1f}"
    )


engagement(human_data, "Human")
engagement(aigc_data, "AIGC")
engagement(exploring_data, "Exploring")

# ==================== 4. Structure ====================
print("\n=== 4. Content Structure ===")


def structure(data, name, use_title=False):
    para = lists = bullet = emoji = hashtag = 0
    emoji_pat = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]"
    )
    for item in data:
        if use_title:
            text = item.get("note_title", "")
        else:
            text = get_text(item) + " " + item.get("note_title", "")
        paras = [p for p in text.split("\n") if p.strip()]
        para += len(paras)
        lists += len(re.findall(r"^\d+[.、]", text, re.MULTILINE))
        bullet += len(re.findall(r"^[、•·]", text, re.MULTILINE))
        emoji += len(emoji_pat.findall(text))
        hashtag += len(re.findall(r"#[\u4e00-\u9fff]+", text))
    n = len(data)
    label = "Title" if use_title else "Content"
    print(
        f"\n{name} ({label}): Para={para / n:.1f}, List={lists / n:.1f}, Emoji={emoji / n:.1f}, Tag={hashtag / n:.1f}"
    )


print("\n=== 4. Content Structure (Content) ===")
structure(human_data, "Human", False)
structure(aigc_data, "AIGC", False)
structure(exploring_data, "Exploring", False)

print("\n=== 4b. Content Structure (Title) ===")
structure(human_data, "Human", True)
structure(aigc_data, "AIGC", True)
structure(exploring_data, "Exploring", True)

# ==================== 5. Complexity ====================
print("\n=== 5. Text Complexity ===")


def complexity(data, name, use_title=False):
    chars_list = unique_ratios = sent_lens = []
    for item in data:
        if use_title:
            text = item.get("note_title", "")
        else:
            text = get_text(item)
        chars = extract_chinese(text)
        if chars:
            total = sum(len(c) for c in chars)
            unique = len(set("".join(chars)))
            chars_list.append(total)
            unique_ratios.append(unique / total if total > 0 else 0)
        sents = re.split(r"[。！？\n]", text)
        for s in sents:
            if s.strip():
                sent_lens.append(len(s.strip()))

    avg_chars = sum(chars_list) / len(chars_list) if chars_list else 0
    avg_ratio = sum(unique_ratios) / len(unique_ratios) if unique_ratios else 0
    avg_sent = sum(sent_lens) / len(sent_lens) if sent_lens else 0
    label = "Title" if use_title else "Content"
    print(
        f"\n{name} ({label}): AvgChars={avg_chars:.0f}, UniqueRatio={avg_ratio:.3f}, AvgSentLen={avg_sent:.0f}"
    )


print("\n=== 5. Text Complexity (Content) ===")
complexity(human_data, "Human", False)
complexity(aigc_data, "AIGC", False)
complexity(exploring_data, "Exploring", False)

print("\n=== 5b. Text Complexity (Title) ===")
complexity(human_data, "Human", True)
complexity(aigc_data, "AIGC", True)
complexity(exploring_data, "Exploring", True)

# ==================== 6. Title Style ====================
print("\n=== 6. Title Style ===")


def title_style(data, name):
    q = ex = num = emo = empty = 0
    emoji_pat = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]"
    )
    for item in data:
        title = item.get("note_title", "")
        if not title:
            empty += 1
            continue
        if "?" in title or "？" in title:
            q += 1
        if "!" in title or "！" in title:
            ex += 1
        if re.search(r"\d", title):
            num += 1
        if emoji_pat.search(title):
            emo += 1
    n = len(data)
    print(
        f"\n{name}: Question={q / n * 100:.1f}%, Exclaim={ex / n * 100:.1f}%, Number={num / n * 100:.1f}%, Emoji={emo / n * 100:.1f}%, Empty={empty / n * 100:.1f}%"
    )


title_style(human_data, "Human")
title_style(aigc_data, "AIGC")
title_style(exploring_data, "Exploring")

# ==================== 7. TTR ====================
print("\n=== 7. Vocabulary Diversity (TTR) ===")


def vocab_ttr(data, name):
    all_words = []
    for item in data:
        desc = get_text(item)
        words = extract_chinese(desc)
        all_words.extend(words)
    unique = len(set(all_words))
    total = len(all_words)
    ttr = unique / total if total > 0 else 0
    print(f"\n{name}: Total={total}, Unique={unique}, TTR={ttr:.4f}")


vocab_ttr(human_data, "Human")
vocab_ttr(aigc_data, "AIGC")
vocab_ttr(exploring_data, "Exploring")

# ==================== 8. Metadata ====================
print("\n=== 8. Metadata Fields ===")


def metadata(data, name):
    user = sum(1 for item in data if "user_id" in item)
    tm = sum(1 for item in data if "time" in item)
    lt = sum(1 for item in data if "local_time" in item)
    like = sum(1 for item in data if "liked_count" in item)
    collect = sum(1 for item in data if "collected_count" in item)
    comment = sum(1 for item in data if "comments_count" in item)
    n = len(data)
    print(
        f"\n{name}: user_id={user / n * 100:.1f}%, time={tm / n * 100:.1f}%, local_time={lt / n * 100:.1f}%, likes={like / n * 100:.1f}%"
    )


metadata(human_data, "Human")
metadata(aigc_data, "AIGC")
metadata(exploring_data, "Exploring")

# ==================== 9. Domain-Style Correlation ====================
print("\n=== 9. Domain-Style Correlation ===")


def domain_style(data, name):
    domain_sent = defaultdict(lambda: [0, 0])
    for item in data:
        domain = item.get("domain", "Unknown")
        text = get_text(item)
        pos = sum(1 for w in positive_words if w in text)
        neg = sum(1 for w in negative_words if w in text)
        domain_sent[domain][0] += pos
        domain_sent[domain][1] += neg

    print(f"\n{name}:")
    for d, (p, n) in list(domain_sent.items())[:5]:
        ratio = p / n if n > 0 else p
        print(f"  {d}: pos={p}, neg={n}, ratio={ratio:.2f}")


domain_style(human_data, "Human")
domain_style(aigc_data, "AIGC")
domain_style(exploring_data, "Exploring")

# ==================== 10. Consistency ====================
print("\n=== 10. Data Consistency ===")


def consistency(data, name):
    empty_desc = empty_title = short = 0
    titles = [item.get("note_title", "") for item in data]
    for item in data:
        text = get_text(item)
        title = item.get("note_title", "")
        if not text.strip():
            empty_desc += 1
        if not title:
            empty_title += 1
        if text and len(text.strip()) < 20:
            short += 1
    dup = sum(1 for t, c in Counter(titles).items() if c > 1)
    n = len(data)
    print(
        f"\n{name}: EmptyDesc={empty_desc / n * 100:.1f}%, EmptyTitle={empty_title / n * 100:.1f}%, Short<20={short / n * 100:.1f}%, DuplicateTitle={dup}"
    )


consistency(human_data, "Human")
consistency(aigc_data, "AIGC")
consistency(exploring_data, "Exploring")

# ==================== 11. Time Distribution ====================
print("\n=== 11. Time Distribution ===")


def time_dist(data, name):
    months = Counter()
    for item in data:
        lt = item.get("local_time", "")
        if lt and len(lt) >= 6:
            months[lt[:6]] += 1
    print(f"\n{name}:")
    for m, c in months.most_common(8):
        print(f"  {m}: {c}")


time_dist(human_data, "Human")
time_dist(aigc_data, "AIGC")
time_dist(exploring_data, "Exploring")

# ==================== 12. Character patterns ====================
print("\n=== 12. Character Patterns ===")


def char_patterns(data, name, use_title=False):
    punct = {"!": 0, "?": 0, "...": 0}
    for item in data:
        if use_title:
            text = item.get("note_title", "")
        else:
            text = get_text(item)
        punct["!"] += text.count("!") + text.count("！")
        punct["?"] += text.count("?") + text.count("？")
        punct["..."] += text.count("...")
    n = len(data)
    label = "Title" if use_title else "Content"
    print(
        f"\n{name} ({label}): Exclaim={punct['!'] / n:.2f}, Question={punct['?'] / n:.2f}, Ellipsis={punct['...'] / n:.2f}"
    )


print("\n=== 12. Character Patterns (Content) ===")
char_patterns(human_data, "Human", False)
char_patterns(aigc_data, "AIGC", False)
char_patterns(exploring_data, "Exploring", False)

print("\n=== 12b. Character Patterns (Title) ===")
char_patterns(human_data, "Human", True)
char_patterns(aigc_data, "AIGC", True)
char_patterns(exploring_data, "Exploring", True)

# ==================== 13. Domain vs Length ====================
print("\n=== 13. Domain vs Content Length ===")


def domain_len(data, name):
    domain_lens = defaultdict(list)
    for item in data:
        d = item.get("domain", "Unknown")
        text = get_text(item)
        domain_lens[d].append(len(text))

    print(f"\n{name}:")
    for d, lens in sorted(
        domain_lens.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True
    )[:5]:
        avg = sum(lens) / len(lens)
        print(f"  {d}: avg_len={avg:.0f}")


domain_len(human_data, "Human")
domain_len(aigc_data, "AIGC")
domain_len(exploring_data, "Exploring")

print("\n=== Analysis Complete ===")
