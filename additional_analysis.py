# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter, defaultdict
import os

font_dir = "/usr/share/fonts/noto-cjk"
for f in os.listdir(font_dir):
    if "NotoSansCJK" in f and f.endswith(".ttc"):
        font_path = os.path.join(font_dir, f)
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.sans-serif"] = [prop.get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False

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
    ]
)


def tokenize(text):
    if not text:
        return []
    text = str(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    words = text.lower().split()
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


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


def analyze_unique_words(freqs, files):
    results = {}

    for i, (f1, freq1) in enumerate(zip(files, freqs)):
        for f2, freq2 in zip(files[i + 1 :], freqs[i + 1 :]):
            unique1 = set(freq1.keys()) - set(freq2.keys())
            unique2 = set(freq2.keys()) - set(freq1.keys())

            unique1_top = sorted([(w, freq1[w]) for w in unique1], key=lambda x: -x[1])[
                :20
            ]
            unique2_top = sorted([(w, freq2[w]) for w in unique2], key=lambda x: -x[1])[
                :20
            ]

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

    texts_data = {}
    for f in files:
        texts_data[f] = load_jsonl(f)
        print(f"Loaded {f}: {len(texts_data[f])} records")

    print("\n--- Top Words (Content/Desc) ---")
    freqs = [compute_word_freq(texts_data[f], False) for f in files]
    for f, freq in zip(files, freqs):
        top10 = freq.most_common(10)
        print(f"\n{f}:")
        for w, c in top10:
            print(f"  {w}: {c}")

    print("\n--- Top Words (Title) ---")
    freqs_title = [compute_word_freq(texts_data[f], True) for f in files]
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
        lengths = analyze_content_length(texts_data[f])
        print(f"\n{f}:")
        print(f"  Mean: {np.mean(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")
        print(f"  Std: {np.std(lengths):.1f}")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")

    print("\n--- Title Length Statistics ---")
    for f in files:
        lengths = analyze_title_length(texts_data[f])
        print(f"\n{f}:")
        print(f"  Mean: {np.mean(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")

    print("\n--- Domain Distribution (Top 10) ---")
    domain_freqs = [analyze_domain(texts_data[f]) for f in files]
    for f, domains in zip(files, domain_freqs):
        print(f"\n{f}:")
        for domain, count in domains.most_common(10):
            print(f"  {domain}: {count}")

    print("\n--- Hashtag Analysis (Content) ---")
    hashtag_freqs = [analyze_hashtag(texts_data[f], False) for f in files]
    for f, hashtags in zip(files, hashtag_freqs):
        print(f"\n{f}:")
        for tag, count in hashtags.most_common(15):
            print(f"  #{tag}: {count}")

    print("\n--- Hashtag Analysis (Title) ---")
    hashtag_freqs_title = [analyze_hashtag(texts_data[f], True) for f in files]
    for f, hashtags in zip(files, hashtag_freqs_title):
        print(f"\n{f}:")
        for tag, count in hashtags.most_common(15):
            print(f"  #{tag}: {count}")

    plot_top_words_comparison(freqs, files, "desc")
    plot_top_words_comparison(freqs_title, files, "title")

    all_lengths = [analyze_content_length(texts_data[f]) for f in files]
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
