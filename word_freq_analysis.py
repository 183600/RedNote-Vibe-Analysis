# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter
from scipy import stats

import os

font_dir = "/usr/share/fonts/noto-cjk"
noto_files = sorted(
    [f for f in os.listdir(font_dir) if "NotoSansCJK" in f and f.endswith(".ttc")]
)
if noto_files:
    font_path = os.path.join(font_dir, noto_files[0])
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
else:
    for fallback in [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
    ]:
        if any(fallback in f.get_name() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fallback
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


def compute_correlation(freq1, freq2, top_n=100):
    words1 = set(freq1.keys())
    words2 = set(freq2.keys())
    common = words1 & words2

    if len(common) < 10:
        all_words = list(words1 | words2)[:top_n]
    else:
        combined = dict((w, freq1.get(w, 0) + freq2.get(w, 0)) for w in common)
        all_words = sorted(combined.keys(), key=lambda w: combined[w], reverse=True)[
            :top_n
        ]

    vec1 = np.array([freq1.get(w, 0) for w in all_words], dtype=float)
    vec2 = np.array([freq2.get(w, 0) for w in all_words], dtype=float)

    norm1 = np.linalg.norm(vec1) + 1e-10
    norm2 = np.linalg.norm(vec2) + 1e-10
    vec1 = vec1 / norm1
    vec2 = vec2 / norm2

    corr = np.dot(vec1, vec2)
    return corr, all_words, vec1, vec2


def plot_scatter(vec1, vec2, corr_val, f1_name, f2_name, label):
    plt.figure(figsize=(10, 8))
    plt.scatter(vec1, vec2, alpha=0.6, edgecolors="none", s=50)

    max_val = max(max(vec1), max(vec2)) * 1.1
    plt.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y=x")

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(vec1, vec2)
        x_line = np.linspace(0, max(vec1), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, "g-", alpha=0.7, label="Regression")
    except Exception:
        pass

    plt.xlabel("Word Frequency (File 1)")
    plt.ylabel("Word Frequency (File 2)")
    title_str = (
        label + ": " + f1_name + " vs " + f2_name + ", corr=" + str(round(corr_val, 4))
    )
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    filename = (
        "scatter_"
        + label
        + "_"
        + f1_name.replace(".jsonl", "")
        + "_vs_"
        + f2_name.replace(".jsonl", "")
        + ".png"
    )
    plt.savefig(filename, dpi=150)
    plt.close()
    print("Saved: " + filename)


def main():
    files = [
        "training_set_human.jsonl",
        "training_set_aigc.jsonl",
        "exploring_set.jsonl",
    ]

    results = {}

    for use_title in [False, True]:
        label = "title_only" if use_title else "desc_only"

        freqs = {}
        for f in files:
            data = load_jsonl(f)
            freqs[f] = compute_word_freq(data, use_title)
            print(
                "Loaded "
                + f
                + ": "
                + str(len(data))
                + " records, "
                + label
                + " words: "
                + str(len(freqs[f]))
            )

        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                f1 = files[i]
                f2 = files[j]
                corr, words, vec1, vec2 = compute_correlation(freqs[f1], freqs[f2])

                key = label + "_" + f1 + "_vs_" + f2
                results[key] = corr

                f1_name = f1.replace(".jsonl", "")
                f2_name = f2.replace(".jsonl", "")
                plot_scatter(vec1, vec2, corr, f1_name, f2_name, label)

    print("\nCorrelation Matrix (DESC only, ignoring title):")
    print("=" * 60)
    header = "                "
    for f in files:
        header += f.replace(".jsonl", "")[:15].ljust(15) + " "
    print(header)

    for f1 in files:
        row = f1.replace(".jsonl", "")[:15].ljust(15) + " "
        for f2 in files:
            if f1 == f2:
                row += "1.0000".ljust(15) + " "
            else:
                key = "desc_only_" + f1 + "_vs_" + f2
                key_rev = "desc_only_" + f2 + "_vs_" + f1
                if key in results:
                    row += str(round(results[key], 4)).ljust(15) + " "
                elif key_rev in results:
                    row += str(round(results[key_rev], 4)).ljust(15) + " "
                else:
                    row += "N/A".ljust(15) + " "
        print(row)

    print("\nCorrelation Matrix (TITLE only):")
    print("=" * 60)
    print(header)

    for f1 in files:
        row = f1.replace(".jsonl", "")[:15].ljust(15) + " "
        for f2 in files:
            if f1 == f2:
                row += "1.0000".ljust(15) + " "
            else:
                key = "title_only_" + f1 + "_vs_" + f2
                key_rev = "title_only_" + f2 + "_vs_" + f1
                if key in results:
                    row += str(round(results[key], 4)).ljust(15) + " "
                elif key_rev in results:
                    row += str(round(results[key_rev], 4)).ljust(15) + " "
                else:
                    row += "N/A".ljust(15) + " "
        print(row)


if __name__ == "__main__":
    main()
