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
import warnings
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

warnings.filterwarnings("ignore")

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
        "yijing",
        "geng",
        "zui",
        "tai",
        "zhen",
        "hao",
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


def clean_tokenize(text):
    if not text:
        return []
    text = str(text)
    text = re.sub(r"#\w+\[话题\]", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    return [
        w
        for w in text.lower().split()
        if len(w) > 1 and w not in STOPWORDS and len(w) > 1
    ]


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
        all_words.extend(clean_tokenize(text))
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


def plot_scatter(vec1, vec2, corr_val, f1_name, f2_name, label, output_dir):
    f1_save = f1_name.replace(".jsonl", "")
    f2_save = f2_name.replace(".jsonl", "")

    plt.figure(figsize=(10, 8))
    plt.scatter(vec1, vec2, alpha=0.6, edgecolors="none", s=50)

    max_val = max(max(vec1), max(vec2)) * 1.1
    plt.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y=x")

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(vec1, vec2)
        x_line = np.linspace(0, max(vec1), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, "g-", alpha=0.7, label="Regression")
    except:
        pass

    if use_title:
        plt.xlabel("Word Frequency (Title File 1)")
        plt.ylabel("Word Frequency (Title File 2)")
    else:
        plt.xlabel("Word Frequency (Content File 1)")
        plt.ylabel("Word Frequency (File 2)")

    title_str = (
        label + ": " + f1_save + " vs " + f2_save + ", corr=" + str(round(corr_val, 4))
    )
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    filename = (
        output_dir + "/scatter_" + label + "_" + f1_save + "_vs_" + f2_save + ".png"
    )
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename


def plot_top_words(freqs, files, use_title, output_dir):
    top_n = 30
    label = "title" if use_title else "content"

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for idx, (f, freq) in enumerate(zip(files, freqs)):
        top_words = freq.most_common(top_n)
        words = [w for w, c in top_words]
        counts = [c for w, c in top_words]

        color = "#1f77b4" if idx == 0 else "#ff7f0e" if idx == 1 else "#2ca02c"
        axes[idx].barh(range(len(words)), counts, color=color)
        axes[idx].set_yticks(range(len(words)))
        axes[idx].set_yticklabels(words, fontsize=8)
        axes[idx].invert_yaxis()
        name = f.replace(".jsonl", "")
        axes[idx].set_title(name + f" Top {top_n} ({label})")

    plt.tight_layout()
    filename = output_dir + f"/{label}_top_words.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename


def plot_length_distribution(texts_data, files, use_title, output_dir):
    label = "title" if use_title else "content"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, f in enumerate(files):
        lengths = []
        for item in texts_data[f]:
            if use_title:
                text = item.get("note_title", "")
            else:
                text = item.get("desc", "") or item.get("note_content", "")
            if text:
                lengths.append(len(text))

        if lengths:
            axes[idx].hist(
                lengths, bins=50, alpha=0.7, edgecolor="black", color="steelblue"
            )
            axes[idx].set_title(f.replace(".jsonl", "") + f" {label} length")
            axes[idx].set_xlabel("Length (chars)")
            axes[idx].set_ylabel("Count")
            axes[idx].axvline(np.mean(lengths), color="r", linestyle="--", label="Mean")
            axes[idx].axvline(
                np.median(lengths), color="g", linestyle="--", label="Median"
            )
            axes[idx].legend()

    plt.tight_layout()
    filename = output_dir + f"/{label}_length_distribution.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename


def plot_correlation_heatmaps(corr_matrices, output_dir):
    files = ["training_set_human", "training_set_aigc", "exploring_set"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (title, matrix) in enumerate(corr_matrices.items()):
        im = axes[idx].imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=1)
        axes[idx].set_xticks(range(3))
        axes[idx].set_yticks(range(3))
        axes[idx].set_xticklabels(files, rotation=45)
        axes[idx].set_yticklabels(files)
        axes[idx].set_title(title)

        for i in range(3):
            for j in range(3):
                text = axes[idx].text(
                    j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="black"
                )

        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    filename = output_dir + "/correlation_heatmaps.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename


def main():
    files = [
        "training_set_human.jsonl",
        "training_set_aigc.jsonl",
        "exploring_set.jsonl",
    ]
    file_names = ["training_set_human", "training_set_aigc", "exploring_set"]

    print("Loading data...")
    texts_data = {}
    for f in files:
        texts_data[f] = load_jsonl(f)
        print(f"  {f}: {len(texts_data[f])} records")

    results = {"content": {}, "title": {}}

    for use_title in [False, True]:
        label = "title" if use_title else "content"
        print(f"\n=== {label.upper()} Analysis ===")

        freqs = [compute_word_freq(texts_data[f], use_title) for f in files]

        print(f"\nTop 10 words ({label}):")
        for f, freq in zip(file_names, freqs):
            top10 = ", ".join([w for w, c in freq.most_common(10)])
            print(f"  {f}: {top10}")

        print(f"\n{label.capitalize()} length stats:")
        for f in files:
            lengths = []
            for item in texts_data[f]:
                if use_title:
                    text = item.get("note_title", "")
                else:
                    text = item.get("desc", "") or item.get("note_content", "")
                if text:
                    lengths.append(len(text))
            print(
                f"  {f}: mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}"
            )

        corr_matrix = np.zeros((3, 3))
        for i in range(3):
            corr_matrix[i, i] = 1.0
            for j in range(i + 1, 3):
                corr, _, vec1, vec2 = compute_correlation(freqs[i], freqs[j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                f1 = file_names[i]
                f2 = file_names[j]
                results[label][f"{f1}_vs_{f2}"] = corr
                print(f"\nCorrelation ({label}): {f1} vs {f2} = {corr:.4f}")

        print(f"\nCorrelation Matrix ({label}):")
        print(corr_matrix)

    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- content_top_words.png")
    print("- title_top_words.png")
    print("- content_length_distribution.png")
    print("- title_length_distribution.png")
    print("- 6 scatter plots (3 content + 3 title)")
    print("- correlation_heatmaps.png")


if __name__ == "__main__":
    main()
