# -*- coding: utf-8 -*-
import heapq
import json
import re
import numpy as np
from analysis_utils import init_plot_style
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter, defaultdict
import os

init_plot_style()

STOPWORDS = set(
    [
        "de",
        "shi",
        "zai",
        "le",
        "he",
        "yu",
        "huo",
        "ba",
        "bei",
        "gei",
        "wo",
        "ni",
        "ta",
        "zhe",
        "na",
        "you",
        "ye",
        "jiu",
        "du",
        "er",
        "ji",
        "zhe",
        "yige",
        "shenme",
        "zenme",
        "yi",
        "bu",
        "hen",
        "yao",
        "hui",
        "keyi",
        "neng",
        "dao",
        "shuo",
        "hai",
        "cong",
        "yi",
        "wei",
        "shang",
        "xia",
        "lai",
        "qu",
        "ru",
        "rang",
        "danshi",
        "suoyi",
        "yinywei",
        "ruguo",
        "erqie",
        "bing",
        "qie",
        "zhi",
        "zhishi",
        "you",
        "zai",
        "yi",
        "yijing",
        "geng",
        "zui",
        "tai",
        "zhen",
        "hao",
        "de",
        "guo",
        "kan",
        "xiang",
        "zuo",
        "yong",
        "ge",
        "liang",
        "zhi",
        "qi",
        "zhong",
        "hou",
        "qian",
        "li",
        "dan",
        "que",
        "shi",
        "jiang",
        "dui",
        "ke",
        "话题",
        "day",
    ]
)


RE_DIGITS = re.compile(r"\d+")
# 连续的非 \w 字符（标点/空白）作为分隔符直接切词，等价于原先「标点→空格再 split」。
RE_NONWORD = re.compile(r"[^\w]+")
# 小红书表情占位符（如 [笑哭R]）不是词，先删掉再分词（口径同 word_freq_analysis）。
RE_XHS_EMO = re.compile(r"\[[^\[\]]{1,8}R\]")
RE_HASHTAG = re.compile(r"#(\w+)")


def tokenize(text):
    if not text:
        return []
    text = str(text)
    text = RE_XHS_EMO.sub(" ", text)
    text = RE_DIGITS.sub("", text)
    words = RE_NONWORD.split(text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def stream_stats(filepath):
    freq_desc = Counter()
    freq_title = Counter()
    lengths_desc = []
    lengths_title = []
    domains = Counter()
    hashtags_desc = Counter()
    hashtags_title = Counter()
    total = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue
            total += 1

            text = item.get("desc", "") or item.get("note_content", "")
            freq_desc.update(tokenize(text))
            if text:
                lengths_desc.append(len(text))
                hashtags_desc.update(RE_HASHTAG.findall(text))

            title = item.get("note_title", "")
            freq_title.update(tokenize(title))
            if title:
                lengths_title.append(len(title))
                hashtags_title.update(RE_HASHTAG.findall(title))

            domain = item.get("domain", "unknown")
            if domain:
                domains[domain] += 1

    return {
        "total": total,
        "freq_desc": freq_desc,
        "freq_title": freq_title,
        "lengths_desc": lengths_desc,
        "lengths_title": lengths_title,
        "domains": domains,
        "hashtags_desc": hashtags_desc,
        "hashtags_title": hashtags_title,
    }


def compute_word_freq(texts, use_title=False):
    all_words = []
    for item in texts:
        if use_title:
            text = item.get("note_title", "")
        else:
            text = item.get("desc", "") or item.get("note_content", "")
        all_words.extend(tokenize(text))
    return Counter(all_words)


def plot_top_words_comparison(freqs, files, output_prefix):
    top_n = 30
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for idx, (f, freq) in enumerate(zip(files, freqs)):
        top_words = freq.most_common(top_n)
        words = [w for w, c in top_words]
        counts = [c for w, c in top_words]

        axes[idx].barh(range(len(words)), counts)
        axes[idx].set_yticks(range(len(words)))
        axes[idx].set_yticklabels(words, fontsize=8)
        axes[idx].invert_yaxis()
        axes[idx].set_title(f.replace(".jsonl", "") + f" (Top {top_n})")

    plt.tight_layout()
    plt.savefig(output_prefix + "_top_words.png", dpi=150)
    plt.close()
    print("Saved: " + output_prefix + "_top_words.png")


def analyze_unique_words(freqs, files, word_sets=None):
    if word_sets is None:
        word_sets = [set(freq.keys()) for freq in freqs]
    results = {}

    for i, (f1, freq1) in enumerate(zip(files, freqs)):
        keys1 = word_sets[i]
        for j, (f2, freq2) in enumerate(zip(files[i + 1 :], freqs[i + 1 :]),
                                          start=i + 1):
            keys2 = word_sets[j]

            unique1_top = heapq.nsmallest(
                20,
                ((w, freq1[w]) for w in keys1 if w not in keys2),
                key=lambda x: (-x[1], x[0]),
            )
            unique2_top = heapq.nsmallest(
                20,
                ((w, freq2[w]) for w in keys2 if w not in keys1),
                key=lambda x: (-x[1], x[0]),
            )

            results[f"{f1}_unique_in_{f2}"] = unique1_top
            results[f"{f2}_unique_in_{f1}"] = unique2_top

    return results


def analyze_content_length(texts):
    lengths = []
    for item in texts:
        text = item.get("desc", "") or item.get("note_content", "")
        if text:
            lengths.append(len(text))
    return lengths


def analyze_title_length(texts):
    lengths = []
    for item in texts:
        text = item.get("note_title", "")
        if text:
            lengths.append(len(text))
    return lengths


def plot_length_distribution(all_lengths, files, output_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (f, lengths) in enumerate(zip(files, all_lengths)):
        if lengths:
            axes[idx].hist(lengths, bins=50, alpha=0.7, edgecolor="black")
            axes[idx].set_title(f.replace(".jsonl", "") + " Content Length")
            axes[idx].set_xlabel("Length (chars)")
            axes[idx].set_ylabel("Count")
            axes[idx].axvline(np.mean(lengths), color="r", linestyle="--", label="Mean")
            axes[idx].axvline(
                np.median(lengths), color="g", linestyle="--", label="Median"
            )
            axes[idx].legend()

    plt.tight_layout()
    plt.savefig(output_prefix + "_length_distribution.png", dpi=150)
    plt.close()
    print("Saved: " + output_prefix + "_length_distribution.png")


def analyze_domain(texts):
    domains = Counter()
    for item in texts:
        domain = item.get("domain", "unknown")
        if domain:
            domains[domain] += 1
    return domains


def plot_domain_distribution(domain_freqs, files, output_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for idx, (f, domains) in enumerate(zip(files, domain_freqs)):
        top_domains = dict(domains.most_common(15))
        axes[idx].barh(list(top_domains.keys()), list(top_domains.values()))
        axes[idx].invert_yaxis()
        axes[idx].set_title(f.replace(".jsonl", "") + " Domain Distribution")

    plt.tight_layout()
    plt.savefig(output_prefix + "_domain_distribution.png", dpi=150)
    plt.close()
    print("Saved: " + output_prefix + "_domain_distribution.png")


def analyze_hashtag(texts, use_title=False):
    hashtags = Counter()
    for item in texts:
        if use_title:
            text = item.get("note_title", "")
        else:
            text = item.get("desc", "") or item.get("note_content", "")
        if text:
            matches = re.findall(r"#(\w+)", text)
            for tag in matches:
                hashtags[tag] += 1
    return hashtags


def main():
    files = [
        "training_set_human.jsonl",
        "training_set_aigc.jsonl",
        "exploring_set.jsonl",
    ]

    print("=" * 60)
    print("Additional Analysis Results")
    print("=" * 60)

    stats = {}
    for f in files:
        stats[f] = stream_stats(f)
        print(f"Loaded {f}: {stats[f]['total']} records")

    print("\n--- Top Words (Content/Desc) ---")
    freqs = [stats[f]["freq_desc"] for f in files]
    for f, freq in zip(files, freqs):
        top10 = freq.most_common(10)
        print(f"\n{f}:")
        for w, c in top10:
            print(f"  {w}: {c}")

    print("\n--- Top Words (Title) ---")
    freqs_title = [stats[f]["freq_title"] for f in files]
    for f, freq in zip(files, freqs_title):
        top10 = freq.most_common(10)
        print(f"\n{f}:")
        for w, c in top10:
            print(f"  {w}: {c}")

    print("\n--- Unique Words Analysis ---")
    unique_results = analyze_unique_words(freqs, files)
    for key, words in unique_results.items():
        print(f"\n{key}:")
        for w, c in words[:10]:
            print(f"  {w}: {c}")

    print("\n--- Content Length Statistics ---")
    for f in files:
        lengths = stats[f]["lengths_desc"]
        print(f"\n{f}:")
        print(f"  Mean: {np.mean(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")
        print(f"  Std: {np.std(lengths):.1f}")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")

    print("\n--- Title Length Statistics ---")
    for f in files:
        lengths = stats[f]["lengths_title"]
        print(f"\n{f}:")
        print(f"  Mean: {np.mean(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")

    print("\n--- Domain Distribution (Top 10) ---")
    domain_freqs = [stats[f]["domains"] for f in files]
    for f, domains in zip(files, domain_freqs):
        print(f"\n{f}:")
        for domain, count in domains.most_common(10):
            print(f"  {domain}: {count}")

    print("\n--- Hashtag Analysis (Content) ---")
    hashtag_freqs = [stats[f]["hashtags_desc"] for f in files]
    for f, hashtags in zip(files, hashtag_freqs):
        print(f"\n{f}:")
        for tag, count in hashtags.most_common(15):
            print(f"  #{tag}: {count}")

    print("\n--- Hashtag Analysis (Title) ---")
    hashtag_freqs_title = [stats[f]["hashtags_title"] for f in files]
    for f, hashtags in zip(files, hashtag_freqs_title):
        print(f"\n{f}:")
        for tag, count in hashtags.most_common(15):
            print(f"  #{tag}: {count}")

    plot_top_words_comparison(freqs, files, "desc")
    plot_top_words_comparison(freqs_title, files, "title")

    all_lengths = [stats[f]["lengths_desc"] for f in files]
    plot_length_distribution(all_lengths, files, "content")

    plot_domain_distribution(domain_freqs, files, "domain")

    print("\n" + "=" * 60)
    print("Analysis Complete. Saved plots:")
    print("- desc_top_words.png")
    print("- title_top_words.png")
    print("- content_length_distribution.png")
    print("- domain_distribution.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
