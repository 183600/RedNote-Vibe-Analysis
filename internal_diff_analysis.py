# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import json
import re
import codecs
from collections import Counter
import random
import io


def load_jsonl(filepath):
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                texts.append(data)
            except:
                continue
    return texts


def analyze_content(texts, sample_size=3000):
    if len(texts) < sample_size:
        sample_size = len(texts)
    else:
        sample_size = min(sample_size, len(texts))
    indices = random.sample(range(len(texts)), sample_size)
    sample = [texts[i] for i in indices]

    lengths = []
    patterns = {
        "!": 0,
        "?": 0,
        "...": 0,
        "#": 0,
        "@": 0,
        "numbers": 0,
        "links": 0,
        "first_person": 0,
        "list": 0,
    }

    for item in sample:
        text = item.get("desc", "") or item.get("note_content", "")
        if not text:
            continue
        text = str(text)
        lengths.append(len(text))
        patterns["!"] += text.count("!")
        patterns["?"] += text.count("?")
        patterns["..."] += text.count("...")
        patterns["#"] += len(re.findall(r"#\S+", text))
        patterns["@"] += len(re.findall(r"@\S+", text))
        patterns["numbers"] += len(re.findall(r"\d+", text))
        patterns["links"] += len(re.findall(r"http|www", text))
        if "我" in text or "我们" in text:
            patterns["first_person"] += 1
        if re.search(r"\n\d+", text):
            patterns["list"] += 1

    return {
        "length": {
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        },
        "patterns": {k: round(v / len(sample), 2) for k, v in patterns.items()},
    }


def analyze_title(texts):
    lengths = []
    patterns = {
        "!": 0,
        "?": 0,
        "...": 0,
        "#": 0,
        "@": 0,
        "numbers": 0,
        "links": 0,
        "first_person": 0,
        "list": 0,
    }

    for item in texts:
        title = item.get("note_title", "")
        if not title:
            continue
        title = str(title)
        lengths.append(len(title))
        patterns["!"] += title.count("!")
        patterns["?"] += title.count("?")
        patterns["..."] += title.count("...")
        patterns["#"] += len(re.findall(r"#\S+", title))
        patterns["@"] += len(re.findall(r"@\S+", title))
        patterns["numbers"] += len(re.findall(r"\d+", title))
        patterns["links"] += len(re.findall(r"http|www", title))
        if "我" in title or "我们" in title:
            patterns["first_person"] += 1
        if re.search(r"\n\d+", title):
            patterns["list"] += 1

    return {
        "length": {
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        },
        "patterns": {k: round(v / len(texts), 2) for k, v in patterns.items()},
    }


def analyze_domain(texts):
    domains = [item.get("domain", "") for item in texts if "domain" in item]
    if not domains:
        return {}
    freq = Counter(domains)
    return {"unique": len(freq), "top3": dict(freq.most_common(3))}


def main():
    files = [
        ("training_set_human.jsonl", "Human"),
        ("training_set_aigc.jsonl", "AIGC"),
        ("exploring_set.jsonl", "Exploring"),
    ]

    datasets = {}
    for filepath, name in files:
        datasets[name] = load_jsonl(filepath)
        print("Loaded " + name + ": " + str(len(datasets[name])) + " records")

    print("\n" + "=" * 70)
    print("INTERNAL DIFFERENCES ANALYSIS - WITHIN EACH CATEGORY")
    print("=" * 70)

    print("\n" + "-" * 50)
    print("1. CONTENT LENGTH COMPARISON")
    print("-" * 50)
    for name, texts in datasets.items():
        r = analyze_content(texts)
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
    for name, texts in datasets.items():
        r = analyze_title(texts)
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
        print("  First person (I/we): " + str(p["first_person"]) + "%")
        print("  List format: " + str(p["list"]) + "%")

    print("\n" + "-" * 50)
    print("3. FORMATTING PATTERNS (avg per post)")
    print("-" * 50)
    for name, texts in datasets.items():
        r = analyze_content(texts)
        p = r["patterns"]
        print("\n" + name + ":")
        print("  Exclamation marks: " + str(p["!"]))
        print("  Question marks: " + str(p["?"]))
        print("  Ellipses: " + str(p["..."]))
        print("  Hashtags (#): " + str(p["#"]))
        print("  @mentions: " + str(p["@"]))
        print("  Numbers: " + str(p["numbers"]))
        print("  Links: " + str(p["links"]))
        print("  First person (I/we): " + str(p["first_person"]) + "%")
        print("  List format: " + str(p["list"]) + "%")

    print("\n" + "-" * 50)
    print("4. DOMAIN DISTRIBUTION")
    print("-" * 50)
    for name, texts in datasets.items():
        d = analyze_domain(texts)
        print("\n" + name + ":")
        print("  Unique domains: " + str(d.get("unique", "N/A")))
        print("  Top 3: " + str(d.get("top3", {})))

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
