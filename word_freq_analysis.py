# -*- coding: utf-8 -*-
import heapq
import json
import re
from functools import partial

import numpy as np
from analysis_utils import init_plot_style, clean_body, run_parallel
from derived_cache import iter_cached
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter
from scipy import stats

import os

init_plot_style()

RE_XHS_EMO = re.compile(r"\[[^\[\]]{1,8}R\]")
# 热循环内使用的正则统一预编译（等价于原来的 re.sub 字面量）。
RE_NONWORD = re.compile(r"[^\w]+")
RE_DIGITS = re.compile(r"\d+")

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
    text = clean_body(str(text))
    if "[" in text:
        text = RE_XHS_EMO.sub(" ", text)
    text = RE_DIGITS.sub("", text)
    # 直接把非 \w 串当作分隔符：与「先 re.sub 成空格再 split」得到相同的
    # 词序列（相邻分隔符合并、空片段被 len>1 过滤）。
    words = RE_NONWORD.split(text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def stream_word_freq(filepath):
    freq_desc = Counter()
    freq_title = Counter()
    total = 0
    # 走派生缓存（JSONL 投影字段 pickle）：只用到正文/标题，与直接解析
    # JSONL 逐字段一致；缓存缺失时 iter_cached 自动回退 iter_jsonl。
    for item in iter_cached(filepath):
        total += 1
        freq_desc.update(tokenize(item.get("desc", "") or item.get("note_content", "")))
        freq_title.update(tokenize(item.get("note_title", "")))
    return freq_desc, freq_title, total


def compute_correlation(freq1, freq2, top_n=100):
    words1 = set(freq1.keys())
    words2 = set(freq2.keys())
    common = words1 & words2

    candidates = (words1 | words2) if len(common) < 10 else common
    # 直接对 (-合并频次, 词) 元组做 nsmallest（键唯一，与原先
    # key=(-combined[w], w) 的排序结果一致），省掉中间 combined 字典与
    # 堆比较时的 lambda 调用。
    all_words = [w for _, w in heapq.nsmallest(
        top_n,
        ((-(freq1.get(w, 0) + freq2.get(w, 0)), w) for w in candidates),
    )]

    vec1 = np.array([freq1.get(w, 0) for w in all_words], dtype=float)
    vec2 = np.array([freq2.get(w, 0) for w in all_words], dtype=float)

    corr = float(np.corrcoef(vec1, vec2)[0, 1])
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

    freqs_by_label = {"desc_only": {}, "title_only": {}}
    totals = {}
    # 三个数据集互不依赖，进程池并行统计；结果与串行执行完全一致。
    for f, (freq_desc, freq_title, total) in zip(
            files, run_parallel([partial(stream_word_freq, f) for f in files])):
        freqs_by_label["desc_only"][f] = freq_desc
        freqs_by_label["title_only"][f] = freq_title
        totals[f] = total

    for use_title in [False, True]:
        label = "title_only" if use_title else "desc_only"

        freqs = freqs_by_label[label]
        for f in files:
            print(
                "Loaded "
                + f
                + ": "
                + str(totals[f])
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
