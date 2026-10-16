# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import re
import random
import statistics
from collections import Counter

PATTERN_KEYS = ("!", "?", "...", "#", "@", "numbers", "links",
                "first_person", "list")

RE_HASHTAG = re.compile(r"#\S+")
RE_MENTION = re.compile(r"@\S+")
RE_NUMBERS = re.compile(r"\d+")
RE_LINKS = re.compile(r"https?://|www\.")
RE_LIST = re.compile(r"\n\d+")


def iter_records(filepath):
    """流式读取 JSONL，跳过无法解析的行（与原 load_jsonl 的容错口径一致）。"""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def pattern_counts(text):
    return {
        "!": text.count("!") + text.count("！"),
        "?": text.count("?") + text.count("？"),
        "...": (text.count("...") + text.count("…") + text.count("。。。")),
        "#": len(RE_HASHTAG.findall(text)),
        "@": len(RE_MENTION.findall(text)),
        "numbers": len(RE_NUMBERS.findall(text)),
        "links": len(RE_LINKS.findall(text)),
        "first_person": 1 if ("我" in text or "我们" in text) else 0,
        "list": 1 if RE_LIST.search(text) else 0,
    }


def content_of(item):
    return item.get("desc", "") or item.get("note_content", "")


def sample_indices(total, sample_size, seed=42):
    return random.Random(seed).sample(range(total), min(sample_size, total))


def collect(filepath, sample_size=3000):
    """单遍流式统计（标题/领域）+ 按索引抽样统计正文，避免全量载入记录。"""
    n = 0
    title_lengths = []
    title_patterns = {k: 0 for k in PATTERN_KEYS}
    domains = Counter()

    for item in iter_records(filepath):
        n += 1
        title = item.get("note_title", "")
        if title:
            title = str(title)
            title_lengths.append(len(title))
            for k, v in pattern_counts(title).items():
                title_patterns[k] += v
        if "domain" in item:
            domains[item.get("domain", "")] += 1

    wanted = set(sample_indices(n, sample_size))
    lengths = []
    content_patterns = {k: 0 for k in PATTERN_KEYS}
    for i, item in enumerate(iter_records(filepath)):
        if i not in wanted:
            continue
        text = content_of(item)
        if not text:
            continue
        text = str(text)
        lengths.append(len(text))
        for k, v in pattern_counts(text).items():
            content_patterns[k] += v
        if len(lengths) == len(wanted):
            break

    sample_n = min(sample_size, n)
    content = {
        "length": {
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
        },
        "patterns": {k: round(v / sample_n, 2)
                     for k, v in content_patterns.items()},
    }
    title = {
        "length": {
            "mean": round(sum(title_lengths) / len(title_lengths), 1)
            if title_lengths else 0,
            "median": statistics.median(title_lengths) if title_lengths else 0,
        },
        "patterns": {k: round(v / n, 2) for k, v in title_patterns.items()},
    }
    domain = {"unique": len(domains), "top3": dict(domains.most_common(3))} \
        if domains else {}
    return content, title, domain, n


def main():
    files = [
        ("training_set_human.jsonl", "Human"),
        ("training_set_aigc.jsonl", "AIGC"),
        ("exploring_set.jsonl", "Exploring"),
    ]

    datasets = {}
    for filepath, name in files:
        stats = collect(filepath)
        datasets[name] = stats
        print("Loaded " + name + ": " + str(stats[3]) + " records")

    print("\n" + "=" * 70)
    print("INTERNAL DIFFERENCES ANALYSIS - WITHIN EACH CATEGORY")
    print("=" * 70)

    print("\n" + "-" * 50)
    print("1. CONTENT LENGTH COMPARISON")
    print("-" * 50)
    for name, (content, _, _, _) in datasets.items():
        r = content
        print(
            name
            + ": mean="
            + str(r["length"]["mean"])
            + ", median="
            + str(r["length"]["median"])
        )

    print("\n" + "-" * 50)
    print("2. TITLE LENGTH & PATTERNS COMPARISON")
    print("-" * 50)
    for name, (_, title, _, _) in datasets.items():
        r = title
        print(
            "\n"
            + name
            + " (Title): mean="
            + str(r["length"]["mean"])
            + ", median="
            + str(r["length"]["median"])
        )
        p = r["patterns"]
        print("  Exclamation marks: " + str(p["!"]))
        print("  Question marks: " + str(p["?"]))
        print("  Ellipses: " + str(p["..."]))
        print("  Hashtags (#): " + str(p["#"]))
        print("  @mentions: " + str(p["@"]))
        print("  Numbers: " + str(p["numbers"]))
        print("  Links: " + str(p["links"]))
        print("  First person (I/we): " + str(round(p["first_person"] * 100, 2)) + "%")
        print("  List format: " + str(round(p["list"] * 100, 2)) + "%")

    print("\n" + "-" * 50)
    print("3. FORMATTING PATTERNS (avg per post)")
    print("-" * 50)
    for name, (content, _, _, _) in datasets.items():
        r = content
        p = r["patterns"]
        print("\n" + name + ":")
        print("  Exclamation marks: " + str(p["!"]))
        print("  Question marks: " + str(p["?"]))
        print("  Ellipses: " + str(p["..."]))
        print("  Hashtags (#): " + str(p["#"]))
        print("  @mentions: " + str(p["@"]))
        print("  Numbers: " + str(p["numbers"]))
        print("  Links: " + str(p["links"]))
        print("  First person (I/we): " + str(round(p["first_person"] * 100, 2)) + "%")
        print("  List format: " + str(round(p["list"] * 100, 2)) + "%")

    print("\n" + "-" * 50)
    print("4. DOMAIN DISTRIBUTION")
    print("-" * 50)
    for name, (_, _, domain, _) in datasets.items():
        d = domain
        print("\n" + name + ":")
        print("  Unique domains: " + str(d.get("unique", "N/A")))
        print("  Top 3: " + str(d.get("top3", {})))

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
