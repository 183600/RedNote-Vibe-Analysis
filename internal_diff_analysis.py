# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import json
import re
import codecs
from collections import Counter
import random
import io

reload(sys)
sys.setdefaultencoding("utf-8")


def load_jsonl(filepath):
    texts = []
    with io.open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                texts.append(data)
            except:
                continue
    return texts


def analyze_content(texts, sample_size=3000):
    if len(texts) < sample_size:
        sample = texts
    else:
        sample = random.sample(texts, sample_size)

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
        try:
            text = unicode(text)
        except:
            continue
        lengths.append(len(text))
        patterns["!"] += text.count("!")
        patterns["?"] += text.count("?")
        patterns["..."] += text.count("...")
        patterns["#"] += len(re.findall("#\S+", text))
        patterns["@"] += len(re.findall("@\S+", text))
        patterns["numbers"] += len(re.findall("\d+", text))
        patterns["links"] += len(re.findall("http|www", text))
        if "\u6211" in text or "\u6211\u4eec" in text:
            patterns["first_person"] += 1
        if re.search("\n\d+", text):
            patterns["list"] += 1

    return {
        "length": {
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
        },
        "patterns": {k: round(v / len(sample), 2) for k, v in patterns.items()},
    }


def analyze_title(texts):
    lengths = []
    for item in texts:
        title = item.get("note_title", "")
        if title:
            try:
                title = unicode(title)
                lengths.append(len(title))
            except:
                pass
    if not lengths:
        return {}
    lengths.sort()
    return {
        "mean": round(sum(lengths) / len(lengths), 1),
        "median": lengths[len(lengths) // 2],
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
    print("2. TITLE LENGTH COMPARISON")
    print("-" * 50)
    for name, texts in datasets.items():
        r = analyze_title(texts)
        print(
            name
            + ": mean="
            + str(r.get("mean", "N/A"))
            + ", median="
            + str(r.get("median", "N/A"))
        )

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
