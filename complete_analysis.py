# -*- coding: utf-8 -*-
import heapq
import json
import re
from functools import partial

import numpy as np
from analysis_utils import init_plot_style, run_parallel
from derived_cache import iter_cached
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter
from scipy import stats
import warnings
import os

init_plot_style()

warnings.filterwarnings("ignore")

RE_TOPIC_TAG = re.compile(r"#[^#\s]+\s*\[话题\]#?")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_XHS_EMO = re.compile(r"\[[^\[\]]{1,8}R\]")
RE_DIGITS = re.compile(r"\d+")
# 连续的非 \w 字符（标点/空白）作为分隔符直接切词，等价于原先「标点→空格再 split」。
RE_NONWORD = re.compile(r"[^\w]+")

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
    if "[话题]" in text:
        text = RE_TOPIC_TAG.sub(" ", text)
    if "#" in text:
        text = RE_HASHTAG.sub(r"\1", text)
    if "[" in text:
        text = RE_XHS_EMO.sub(" ", text)
    text = text.lower()
    text = RE_DIGITS.sub("", text)
    return [w for w in RE_NONWORD.split(text) if len(w) > 1 and w not in STOPWORDS]


def stream_stats(filepath):
    freq_desc = Counter()
    freq_title = Counter()
    lengths_desc = []
    lengths_title = []
    total = 0
    # 派生缓存读取投影字段；缓存缺失时自动回退 iter_jsonl。
    for item in iter_cached(filepath):
        total += 1
        text = item.get("desc", "") or item.get("note_content", "")
        freq_desc.update(clean_tokenize(text))
        if text:
            lengths_desc.append(len(text))
        title = item.get("note_title", "")
        freq_title.update(clean_tokenize(title))
        if title:
            lengths_title.append(len(title))
    return {
        "total": total,
        "freq_desc": freq_desc,
        "freq_title": freq_title,
        "lengths_desc": lengths_desc,
        "lengths_title": lengths_title,
    }


def compute_correlation(freq1, freq2, top_n=100):
    words1 = set(freq1.keys())
    words2 = set(freq2.keys())
    common = words1 & words2

    candidates = (words1 | words2) if len(common) < 10 else common
    # 只取 Top-N，避免对全词表做完整排序；直接对 (-合并频次, 词) 元组做
    # nsmallest（键唯一，结果与原先 key=(-combined[w], w) 的后排序切片一致），
    # 省掉中间 combined 字典与堆比较时的 lambda 调用。
    all_words = [w for _, w in heapq.nsmallest(
        top_n,
        ((-(freq1.get(w, 0) + freq2.get(w, 0)), w) for w in candidates),
    )]

    vec1 = np.array([freq1.get(w, 0) for w in all_words], dtype=float)
    vec2 = np.array([freq2.get(w, 0) for w in all_words], dtype=float)

    corr = float(np.corrcoef(vec1, vec2)[0, 1])
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

    if label == "title":
        plt.xlabel("Word Frequency (Title File 1)")
        plt.ylabel("Word Frequency (Title File 2)")
    else:
        plt.xlabel("Word Frequency (Content File 1)")
        plt.ylabel("Word Frequency (Content File 2)")

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


def plot_length_distribution(lengths_by_file, files, use_title, output_dir):
    label = "title" if use_title else "content"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, f in enumerate(files):
        lengths = lengths_by_file[f]

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
    # 三个数据集互不依赖，进程池并行统计；结果与串行执行完全一致。
    stats = dict(zip(files, run_parallel(
        [partial(stream_stats, f) for f in files])))
    for f in files:
        print(f"  {f}: {stats[f]['total']} records")

    results = {"content": {}, "title": {}}
    corr_matrices = {}
    generated = []
    os.makedirs("output", exist_ok=True)

    for use_title in [False, True]:
        label = "title" if use_title else "content"
        print(f"\n=== {label.upper()} Analysis ===")

        freqs = [
            stats[f]["freq_title" if use_title else "freq_desc"] for f in files
        ]

        print(f"\nTop 10 words ({label}):")
        for f, freq in zip(file_names, freqs):
            top10 = ", ".join([w for w, c in freq.most_common(10)])
            print(f"  {f}: {top10}")

        generated.append(plot_top_words(freqs, files, use_title, "output"))

        print(f"\n{label.capitalize()} length stats:")
        lengths_by_file = {
            f: stats[f]["lengths_title" if use_title else "lengths_desc"]
            for f in files
        }
        for f in files:
            lengths = lengths_by_file[f]
            print(
                f"  {f}: mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}"
            )

        generated.append(
            plot_length_distribution(lengths_by_file, files, use_title, "output")
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
                generated.append(
                    plot_scatter(vec1, vec2, corr, files[i], files[j], label,
                                 "output")
                )
                print(f"\nCorrelation ({label}): {f1} vs {f2} = {corr:.4f}")

        print(f"\nCorrelation Matrix ({label}):")
        print(corr_matrix)
        corr_matrices[label] = corr_matrix

    generated.append(plot_correlation_heatmaps(corr_matrices, "output"))

    print("\n=== Analysis Complete ===")
    print("Generated files:")
    for path in generated:
        print("- " + path)


if __name__ == "__main__":
    main()
