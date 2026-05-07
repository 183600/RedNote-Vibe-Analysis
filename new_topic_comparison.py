# -*- coding: utf-8 -*-
"""
New Topic Comparison Analysis: Exploring vs AIGC vs Human
Different from previous analyses - focuses on new comparison dimensions
"""

import json
import re
import os
import warnings
import numpy as np
from collections import Counter, defaultdict
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")

DOMAINS = [
    "健康",
    "穿搭",
    "美食",
    "职场",
    "宠物",
    "学习",
    "运动",
    "情感",
    "旅行",
    "心理",
]

STOPWORDS = set(
    [
        "的",
        "是",
        "在",
        "了",
        "和",
        "与",
        "或",
        "把",
        "被",
        "给",
        "我",
        "你",
        "他",
        "她",
        "它",
        "这",
        "那",
        "有",
        "也",
        "就",
        "都",
        "而",
        "及",
        "着",
        "但",
        "却",
        "因为",
        "所以",
        "如果",
        "而且",
        "并",
        "且",
        "只是",
        "已经",
        "更",
        "最",
        "真",
        "好",
        "过",
        "看",
        "想",
        "说",
        "还",
        "会",
        "能",
        "要",
        "到",
        "来",
        "去",
        "上",
        "下",
        "里",
        "中",
        "后",
        "前",
        "间",
        "内",
        "外",
        "为",
        "之",
        "其",
        "所",
        "以",
        "然",
        "当",
        "然后",
        "这个",
        "那个",
        "什么",
        "怎么",
        "一",
        "不",
        "很",
        "可以",
        "出来",
        "起来",
        "话题",
        "day",
        "一个",
        "自己",
        "时候",
        "没有",
        "就是",
    ]
)

POSITIVE_WORDS = set(
    [
        "喜欢",
        "爱",
        "开心",
        "快乐",
        "幸福",
        "美好",
        "推荐",
        "赞",
        "棒",
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
)

NEGATIVE_WORDS = set(
    [
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
)

EMOJI_PATTERN = re.compile(
    r"[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff\U00002702-\U000027b0\U000024c2-\U0001f251]+",
    flags=re.UNICODE,
)


def setup_fonts():
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


def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                continue
    return data


def get_text(item):
    return item.get("note_content", "") or item.get("desc", "") or ""


def clean_tokenize(text):
    if not text:
        return []
    text = str(text)
    text = re.sub(r"#\w+\[话题\]", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    return [w for w in text.split() if len(w) > 1 and w not in STOPWORDS]


def analyze_engagement_patterns(data_dict):
    print("\n" + "=" * 80)
    print("1. ENGAGEMENT PATTERN ANALYSIS: Like/Collect/Comment Ratios")
    print("=" * 80)

    results = {}
    for ds_name, data in data_dict.items():
        results[ds_name] = {}
        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items:
                continue

            likes = [item.get("liked_count", 0) for item in domain_items]
            collects = [item.get("collected_count", 0) for item in domain_items]
            comments = [item.get("comments_count", 0) for item in domain_items]

            total_likes = sum(likes)
            total_collects = sum(collects)
            total_comments = sum(comments)
            total_eng = total_likes + total_collects + total_comments

            results[ds_name][domain] = {
                "total_engagement": total_eng,
                "like_ratio": total_likes / total_eng if total_eng > 0 else 0,
                "collect_ratio": total_collects / total_eng if total_eng > 0 else 0,
                "comment_ratio": total_comments / total_eng if total_eng > 0 else 0,
                "collect_to_like": total_collects / total_likes
                if total_likes > 0
                else 0,
                "comment_to_like": total_comments / total_likes
                if total_likes > 0
                else 0,
                "avg_likes": np.mean(likes),
                "avg_collects": np.mean(collects),
                "avg_comments": np.mean(comments),
            }

    for domain in DOMAINS:
        print("")
        print("--- " + domain + " ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                r = results[ds][domain]
                print(
                    "  "
                    + ds
                    + ": Like="
                    + str(round(r["like_ratio"] * 100, 1))
                    + "%, Collect="
                    + str(round(r["collect_ratio"] * 100, 1))
                    + "%, Comment="
                    + str(round(r["comment_ratio"] * 100, 1))
                    + "%, C/L="
                    + str(round(r["collect_to_like"], 3))
                )

    return results


def plot_engagement_patterns(results, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    common_domains = [
        d for d in DOMAINS if all(d in results.get(ds, {}) for ds in datasets)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    x = np.arange(len(common_domains))
    width = 0.25

    for i, ds in enumerate(datasets):
        vals = [
            results[ds].get(d, {}).get("collect_to_like", 0) for d in common_domains
        ]
        axes[0, 0].bar(
            x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85
        )
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(common_domains, rotation=45)
    axes[0, 0].set_title("Collect-to-Like Ratio by Domain")
    axes[0, 0].set_ylabel("Ratio")
    axes[0, 0].legend()

    for i, ds in enumerate(datasets):
        vals = [
            results[ds].get(d, {}).get("comment_to_like", 0) for d in common_domains
        ]
        axes[0, 1].bar(
            x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85
        )
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(common_domains, rotation=45)
    axes[0, 1].set_title("Comment-to-Like Ratio by Domain")
    axes[0, 1].set_ylabel("Ratio")
    axes[0, 1].legend()

    for i, ds in enumerate(datasets):
        vals = [results[ds].get(d, {}).get("avg_likes", 0) for d in common_domains]
        axes[1, 0].bar(
            x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85
        )
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(common_domains, rotation=45)
    axes[1, 0].set_title("Average Likes by Domain")
    axes[1, 0].set_ylabel("Likes")
    axes[1, 0].legend()

    for i, ds in enumerate(datasets):
        vals = [results[ds].get(d, {}).get("avg_comments", 0) for d in common_domains]
        axes[1, 1].bar(
            x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85
        )
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(common_domains, rotation=45)
    axes[1, 1].set_title("Average Comments by Domain")
    axes[1, 1].set_ylabel("Comments")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "engagement_patterns.png"), dpi=150)
    plt.close()
    print("Saved: " + os.path.join(output_dir, "engagement_patterns.png"))


def analyze_title_content_ratio(data_dict):
    print("\n" + "=" * 80)
    print("2. TITLE-CONTENT RATIO ANALYSIS")
    print("=" * 80)

    results = {}
    for ds_name, data in data_dict.items():
        results[ds_name] = {}
        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items:
                continue

            title_lens = [
                len(item.get("note_title", "") or "") for item in domain_items
            ]
            content_lens = [len(get_text(item)) for item in domain_items]

            ratios = []
            for tl, cl in zip(title_lens, content_lens):
                if cl > 0:
                    ratios.append(tl / cl)

            results[ds_name][domain] = {
                "avg_title_len": np.mean(title_lens),
                "avg_content_len": np.mean(content_lens),
                "avg_ratio": np.mean(ratios),
                "median_ratio": np.median(ratios),
                "long_title_count": sum(1 for t in title_lens if t > 20),
            }

    for domain in DOMAINS:
        print("")
        print("--- " + domain + " ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                r = results[ds][domain]
                print(
                    "  "
                    + ds
                    + ": Title="
                    + str(round(r["avg_title_len"], 1))
                    + ", Content="
                    + str(round(r["avg_content_len"], 1))
                    + ", Ratio="
                    + str(round(r["avg_ratio"], 4))
                )

    return results


def plot_title_content_ratio(results, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    common_domains = [
        d for d in DOMAINS if all(d in results.get(ds, {}) for ds in datasets)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(common_domains))
    width = 0.25

    for i, ds in enumerate(datasets):
        vals = [results[ds].get(d, {}).get("avg_ratio", 0) for d in common_domains]
        axes[0].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(common_domains, rotation=45)
    axes[0].set_title("Title-to-Content Ratio by Domain")
    axes[0].set_ylabel("Ratio")
    axes[0].legend()

    for i, ds in enumerate(datasets):
        vals = [results[ds].get(d, {}).get("avg_title_len", 0) for d in common_domains]
        axes[1].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(common_domains, rotation=45)
    axes[1].set_title("Average Title Length by Domain")
    axes[1].set_ylabel("Characters")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "title_content_ratio.png"), dpi=150)
    plt.close()
    print("Saved: " + os.path.join(output_dir, "title_content_ratio.png"))


def analyze_unique_vocabulary(data_dict):
    print("\n" + "=" * 80)
    print("3. UNIQUE VOCABULARY ANALYSIS: Dataset-Specific Words per Domain")
    print("=" * 80)

    word_freqs = {}
    for ds_name, data in data_dict.items():
        word_freqs[ds_name] = {}
        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items:
                continue

            counter = Counter()
            for item in domain_items:
                counter.update(clean_tokenize(get_text(item)))
            word_freqs[ds_name][domain] = counter

    results = {}
    for domain in DOMAINS:
        results[domain] = {}
        all_datasets = [
            ds
            for ds in ["Human", "AIGC", "Exploring"]
            if ds in word_freqs and domain in word_freqs[ds]
        ]

        for ds in all_datasets:
            other_datasets = [d for d in all_datasets if d != ds]
            other_words = set()
            for od in other_datasets:
                other_words.update(
                    w for w, c in word_freqs[od][domain].most_common(100)
                )

            ds_words = dict(word_freqs[ds][domain].most_common(50))
            unique_words = {w: c for w, c in ds_words.items() if w not in other_words}

            results[domain][ds] = {
                "unique_words": list(unique_words.keys())[:15],
                "unique_count": len(unique_words),
            }

    for domain in DOMAINS:
        print("")
        print("--- " + domain + " ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results.get(domain, {}):
                unique = results[domain][ds]["unique_words"]
                print("  " + ds + " unique words: " + ", ".join(unique[:10]))

    return results


def plot_unique_vocabulary(results, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    common_domains = [
        d for d in DOMAINS if all(ds in results.get(d, {}) for ds in datasets)
    ]

    unique_counts = {
        ds: [results[d].get(ds, {}).get("unique_count", 0) for d in common_domains]
        for ds in datasets
    }

    x = np.arange(len(common_domains))
    width = 0.25

    plt.figure(figsize=(14, 6))
    for i, ds in enumerate(datasets):
        plt.bar(
            x + i * width,
            unique_counts[ds],
            width,
            label=ds,
            color=colors[i],
            alpha=0.85,
        )
    plt.xticks(x + width, common_domains, rotation=45)
    plt.title("Unique Vocabulary Count per Domain")
    plt.ylabel("Count (unique in top 50)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unique_vocabulary.png"), dpi=150)
    plt.close()
    print("Saved: " + os.path.join(output_dir, "unique_vocabulary.png"))


def analyze_sentence_features(data_dict):
    print("\n" + "=" * 80)
    print("4. SENTENCE-LEVEL FEATURE ANALYSIS")
    print("=" * 80)

    def count_sentences(text):
        sentences = re.split(r"[。！？.!?]", str(text))
        return len([s for s in sentences if s.strip()])

    results = {}
    for ds_name, data in data_dict.items():
        results[ds_name] = {}
        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items:
                continue

            sentence_counts = [count_sentences(get_text(item)) for item in domain_items]
            content_lens = [len(get_text(item)) for item in domain_items]

            avg_sent_len = []
            for s, l in zip(sentence_counts, content_lens):
                if s > 0:
                    avg_sent_len.append(l / s)

            results[ds_name][domain] = {
                "avg_sentences": np.mean(sentence_counts),
                "median_sentences": np.median(sentence_counts),
                "avg_sent_length": np.mean(avg_sent_len),
                "short_post_pct": sum(1 for s in sentence_counts if s <= 2)
                / len(sentence_counts)
                * 100,
                "long_post_pct": sum(1 for s in sentence_counts if s >= 5)
                / len(sentence_counts)
                * 100,
            }

    for domain in DOMAINS:
        print("")
        print("--- " + domain + " ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                r = results[ds][domain]
                print(
                    "  "
                    + ds
                    + ": AvgSent="
                    + str(round(r["avg_sentences"], 1))
                    + ", SentLen="
                    + str(round(r["avg_sent_length"], 1))
                    + ", Short%="
                    + str(round(r["short_post_pct"], 1))
                    + "%"
                )

    return results


def plot_sentence_features(results, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    common_domains = [
        d for d in DOMAINS if all(d in results.get(ds, {}) for ds in datasets)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    x = np.arange(len(common_domains))
    width = 0.25

    for i, ds in enumerate(datasets):
        vals = [results[ds].get(d, {}).get("avg_sentences", 0) for d in common_domains]
        axes[0, 0].bar(
            x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85
        )
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(common_domains, rotation=45)
    axes[0, 0].set_title("Average Sentence Count by Domain")
    axes[0, 0].set_ylabel("Sentences")
    axes[0, 0].legend()

    for i, ds in enumerate(datasets):
        vals = [
            results[ds].get(d, {}).get("avg_sent_length", 0) for d in common_domains
        ]
        axes[0, 1].bar(
            x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85
        )
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(common_domains, rotation=45)
    axes[0, 1].set_title("Average Characters per Sentence")
    axes[0, 1].set_ylabel("Characters")
    axes[0, 1].legend()

    for i, ds in enumerate(datasets):
        vals = [results[ds].get(d, {}).get("short_post_pct", 0) for d in common_domains]
        axes[1, 0].bar(
            x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85
        )
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(common_domains, rotation=45)
    axes[1, 0].set_title("Short Posts (1-2 sentences) Percentage")
    axes[1, 0].set_ylabel("Percentage")
    axes[1, 0].legend()

    for i, ds in enumerate(datasets):
        vals = [results[ds].get(d, {}).get("long_post_pct", 0) for d in common_domains]
        axes[1, 1].bar(
            x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85
        )
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(common_domains, rotation=45)
    axes[1, 1].set_title("Long Posts (5+ sentences) Percentage")
    axes[1, 1].set_ylabel("Percentage")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentence_features.png"), dpi=150)
    plt.close()
    print("Saved: " + os.path.join(output_dir, "sentence_features.png"))


def analyze_keyword_cooccurrence(data_dict):
    print("\n" + "=" * 80)
    print("5. KEYWORD CO-OCCURRENCE ANALYSIS")
    print("=" * 80)

    cooccurrence = {}
    for ds_name, data in data_dict.items():
        cooccurrence[ds_name] = {}
        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items or len(domain_items) < 10:
                continue

            word_counter = Counter()
            for item in domain_items:
                word_counter.update(clean_tokenize(get_text(item)))

            top_words = [w for w, c in word_counter.most_common(30)]

            co_word_counts = defaultdict(Counter)
            for item in domain_items:
                words = set(clean_tokenize(get_text(item)))
                for w in words:
                    if w in top_words:
                        for w2 in words:
                            if w2 != w and w2 in top_words:
                                co_word_counts[w][w2] += 1

            strong_pairs = []
            for w1, counter in co_word_counts.items():
                for w2, count in counter.most_common(3):
                    if count >= 5:
                        strong_pairs.append((w1, w2, count))

            cooccurrence[ds_name][domain] = {
                "top_words": top_words[:10],
                "strong_pairs": strong_pairs[:10],
            }

    for domain in DOMAINS:
        print("")
        print("--- " + domain + " ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in cooccurrence and domain in cooccurrence[ds]:
                pairs = cooccurrence[ds][domain]["strong_pairs"]
                top = cooccurrence[ds][domain]["top_words"]
                print("  " + ds + " top words: " + ", ".join(top[:8]))
                pair_strs = [
                    p[0] + "-" + p[1] + "(" + str(p[2]) + ")" for p in pairs[:5]
                ]
                print("  " + ds + " strong pairs: " + ", ".join(pair_strs))

    return cooccurrence


def analyze_content_consistency(data_dict):
    print("\n" + "=" * 80)
    print("6. CONTENT CONSISTENCY ANALYSIS: Within-Domain Variance")
    print("=" * 80)

    results = {}
    for ds_name, data in data_dict.items():
        results[ds_name] = {}
        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items or len(domain_items) < 5:
                continue

            lengths = [len(get_text(item)) for item in domain_items]
            title_lens = [
                len(item.get("note_title", "") or "") for item in domain_items
            ]

            results[ds_name][domain] = {
                "length_cv": np.std(lengths) / np.mean(lengths)
                if np.mean(lengths) > 0
                else 0,
                "title_cv": np.std(title_lens) / np.mean(title_lens)
                if np.mean(title_lens) > 0
                else 0,
                "length_std": np.std(lengths),
                "sample_size": len(domain_items),
            }

    for domain in DOMAINS:
        print("")
        print("--- " + domain + " ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                r = results[ds][domain]
                print(
                    "  "
                    + ds
                    + ": LengthCV="
                    + str(round(r["length_cv"], 3))
                    + ", LengthStd="
                    + str(round(r["length_std"], 1))
                    + ", N="
                    + str(r["sample_size"])
                )

    return results


def plot_content_consistency(results, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    common_domains = [
        d for d in DOMAINS if all(d in results.get(ds, {}) for ds in datasets)
    ]

    length_cvs = {
        ds: [results[ds].get(d, {}).get("length_cv", 0) for d in common_domains]
        for ds in datasets
    }

    x = np.arange(len(common_domains))
    width = 0.25

    plt.figure(figsize=(14, 6))
    for i, ds in enumerate(datasets):
        plt.bar(
            x + i * width, length_cvs[ds], width, label=ds, color=colors[i], alpha=0.85
        )
    plt.xticks(x + width, common_domains, rotation=45)
    plt.title("Content Length Coefficient of Variation (Lower = More Consistent)")
    plt.ylabel("CV (std/mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "content_consistency.png"), dpi=150)
    plt.close()
    print("Saved: " + os.path.join(output_dir, "content_consistency.png"))


def analyze_emotion_intensity(data_dict):
    print("\n" + "=" * 80)
    print("7. EMOTION INTENSITY ANALYSIS: Positive vs Negative Word Strength")
    print("=" * 80)

    INTENSITY_WORDS = {
        "positive_strong": [
            "超爱",
            "太喜欢",
            "超级喜欢",
            "完美",
            "绝绝子",
            "太棒了",
            "爱死了",
            "惊艳",
            "震撼",
            "极致",
        ],
        "positive_mild": [
            "喜欢",
            "好",
            "不错",
            "可以",
            "还行",
            "满意",
            "推荐",
            "棒",
            "赞",
            "开心",
        ],
        "negative_strong": [
            "崩溃",
            "绝望",
            "彻底",
            "完全",
            "太难了",
            "受够了",
            "糟透了",
            "极度",
            "相当",
            "难受",
        ],
        "negative_mild": [
            "有点",
            "可能",
            "担心",
            "焦虑",
            "难过",
            "累",
            "烦",
            "失望",
            "怕",
            "不舒服",
        ],
    }

    results = {}
    for ds_name, data in data_dict.items():
        results[ds_name] = {}
        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items:
                continue

            text_all = " ".join(get_text(item) for item in domain_items)

            pos_strong = sum(
                text_all.count(w) for w in INTENSITY_WORDS["positive_strong"]
            )
            pos_mild = sum(text_all.count(w) for w in INTENSITY_WORDS["positive_mild"])
            neg_strong = sum(
                text_all.count(w) for w in INTENSITY_WORDS["negative_strong"]
            )
            neg_mild = sum(text_all.count(w) for w in INTENSITY_WORDS["negative_mild"])

            total_words = len(clean_tokenize(text_all))

            results[ds_name][domain] = {
                "pos_strong_per_k": pos_strong / total_words * 1000
                if total_words > 0
                else 0,
                "pos_mild_per_k": pos_mild / total_words * 1000
                if total_words > 0
                else 0,
                "neg_strong_per_k": neg_strong / total_words * 1000
                if total_words > 0
                else 0,
                "neg_mild_per_k": neg_mild / total_words * 1000
                if total_words > 0
                else 0,
                "pos_intensity": pos_strong / (pos_mild + 1),
                "neg_intensity": neg_strong / (neg_mild + 1),
            }

    for domain in DOMAINS:
        print("")
        print("--- " + domain + " ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                r = results[ds][domain]
                print(
                    "  "
                    + ds
                    + ": PosInt="
                    + str(round(r["pos_intensity"], 2))
                    + ", NegInt="
                    + str(round(r["neg_intensity"], 2))
                    + ", PosStrong="
                    + str(round(r["pos_strong_per_k"], 2))
                    + "/k"
                )

    return results


def plot_emotion_intensity(results, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    common_domains = [
        d for d in DOMAINS if all(d in results.get(ds, {}) for ds in datasets)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(common_domains))
    width = 0.25

    for i, ds in enumerate(datasets):
        vals = [results[ds].get(d, {}).get("pos_intensity", 0) for d in common_domains]
        axes[0].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(common_domains, rotation=45)
    axes[0].set_title("Positive Emotion Intensity (Strong/Mild)")
    axes[0].set_ylabel("Intensity Ratio")
    axes[0].legend()

    for i, ds in enumerate(datasets):
        vals = [results[ds].get(d, {}).get("neg_intensity", 0) for d in common_domains]
        axes[1].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(common_domains, rotation=45)
    axes[1].set_title("Negative Emotion Intensity (Strong/Mild)")
    axes[1].set_ylabel("Intensity Ratio")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "emotion_intensity.png"), dpi=150)
    plt.close()
    print("Saved: " + os.path.join(output_dir, "emotion_intensity.png"))


def analyze_time_distribution(data_dict):
    print("\n" + "=" * 80)
    print("8. TEMPORAL PATTERN ANALYSIS: Posting Time Distribution")
    print("=" * 80)

    results = {}
    for ds_name, data in data_dict.items():
        results[ds_name] = {}
        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items:
                continue

            hours = []
            for item in domain_items:
                time_str = item.get("local_time", "") or item.get("time", "")
                if time_str:
                    match = re.search(r"(\d{2}):(\d{2})", str(time_str))
                    if match:
                        hours.append(int(match.group(1)))

            if not hours:
                continue

            morning = sum(1 for h in hours if 6 <= h < 12)
            afternoon = sum(1 for h in hours if 12 <= h < 18)
            evening = sum(1 for h in hours if 18 <= h < 24)
            night = sum(1 for h in hours if 0 <= h < 6)

            results[ds_name][domain] = {
                "morning_pct": morning / len(hours) * 100,
                "afternoon_pct": afternoon / len(hours) * 100,
                "evening_pct": evening / len(hours) * 100,
                "night_pct": night / len(hours) * 100,
                "peak_hour": Counter(hours).most_common(1)[0][0] if hours else 0,
            }

    for domain in DOMAINS:
        has_data = False
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                has_data = True
                break
        if not has_data:
            continue
        print("")
        print("--- " + domain + " ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                r = results[ds][domain]
                print(
                    "  "
                    + ds
                    + ": Morning="
                    + str(round(r["morning_pct"], 1))
                    + "%, Afternoon="
                    + str(round(r["afternoon_pct"], 1))
                    + "%, Evening="
                    + str(round(r["evening_pct"], 1))
                    + "%, Peak="
                    + str(r["peak_hour"])
                    + ":00"
                )

    return results


def analyze_hashtag_usage(data_dict):
    print("\n" + "=" * 80)
    print("9. HASHTAG USAGE ANALYSIS")
    print("=" * 80)

    def extract_hashtags(text):
        if not text:
            return []
        hashtags = re.findall(r"#(\w+)", str(text))
        return [h for h in hashtags if "话题" not in h]

    results = {}
    for ds_name, data in data_dict.items():
        results[ds_name] = {}
        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items:
                continue

            hashtag_counter = Counter()
            posts_with_hashtag = 0

            for item in domain_items:
                text = get_text(item)
                tags = extract_hashtags(text)
                if tags:
                    posts_with_hashtag += 1
                    hashtag_counter.update(tags)

            results[ds_name][domain] = {
                "hashtag_rate": posts_with_hashtag / len(domain_items) * 100,
                "avg_hashtags_per_post": sum(hashtag_counter.values())
                / len(domain_items),
                "unique_hashtags": len(hashtag_counter),
                "top_hashtags": hashtag_counter.most_common(10),
            }

    for domain in DOMAINS:
        print("")
        print("--- " + domain + " ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                r = results[ds][domain]
                tags = ", ".join([t[0] for t in r["top_hashtags"][:5]])
                print(
                    "  "
                    + ds
                    + ": Rate="
                    + str(round(r["hashtag_rate"], 1))
                    + "%, Avg="
                    + str(round(r["avg_hashtags_per_post"], 2))
                    + ", Tags="
                    + tags
                )

    return results


def plot_hashtag_usage(results, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    common_domains = [
        d for d in DOMAINS if all(d in results.get(ds, {}) for ds in datasets)
    ]

    hashtag_rates = {
        ds: [results[ds].get(d, {}).get("hashtag_rate", 0) for d in common_domains]
        for ds in datasets
    }

    x = np.arange(len(common_domains))
    width = 0.25

    plt.figure(figsize=(14, 6))
    for i, ds in enumerate(datasets):
        plt.bar(
            x + i * width,
            hashtag_rates[ds],
            width,
            label=ds,
            color=colors[i],
            alpha=0.85,
        )
    plt.xticks(x + width, common_domains, rotation=45)
    plt.title("Hashtag Usage Rate by Domain")
    plt.ylabel("Percentage of Posts with Hashtags")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hashtag_usage.png"), dpi=150)
    plt.close()
    print("Saved: " + os.path.join(output_dir, "hashtag_usage.png"))


def analyze_punctuation_diversity(data_dict):
    print("\n" + "=" * 80)
    print("10. PUNCTUATION DIVERSITY ANALYSIS")
    print("=" * 80)

    results = {}
    for ds_name, data in data_dict.items():
        results[ds_name] = {}
        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items:
                continue

            text_all = "".join(get_text(item) for item in domain_items)

            comma_count = text_all.count(",") + text_all.count("，")
            period_count = text_all.count(".") + text_all.count("。")
            exclaim_count = text_all.count("!") + text_all.count("！")
            question_count = text_all.count("?") + text_all.count("？")
            colon_count = text_all.count(":") + text_all.count("：")
            semicolon_count = text_all.count(";") + text_all.count("；")

            total_posts = len(domain_items)

            results[ds_name][domain] = {
                "comma_per_post": comma_count / total_posts,
                "period_per_post": period_count / total_posts,
                "exclaim_per_post": exclaim_count / total_posts,
                "question_per_post": question_count / total_posts,
                "colon_per_post": colon_count / total_posts,
                "semicolon_per_post": semicolon_count / total_posts,
            }

    for domain in DOMAINS:
        print("")
        print("--- " + domain + " ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                r = results[ds][domain]
                print(
                    "  "
                    + ds
                    + ": Comma="
                    + str(round(r["comma_per_post"], 2))
                    + ", Exclaim="
                    + str(round(r["exclaim_per_post"], 2))
                    + ", Question="
                    + str(round(r["question_per_post"], 2))
                )

    return results


def plot_punctuation_diversity(results, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    common_domains = [
        d for d in DOMAINS if all(d in results.get(ds, {}) for ds in datasets)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(common_domains))
    width = 0.25

    for i, ds in enumerate(datasets):
        vals = [results[ds].get(d, {}).get("comma_per_post", 0) for d in common_domains]
        axes[0].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(common_domains, rotation=45)
    axes[0].set_title("Comma Usage per Post")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    for i, ds in enumerate(datasets):
        vals = [
            results[ds].get(d, {}).get("exclaim_per_post", 0) for d in common_domains
        ]
        axes[1].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(common_domains, rotation=45)
    axes[1].set_title("Exclamation per Post")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    for i, ds in enumerate(datasets):
        vals = [
            results[ds].get(d, {}).get("question_per_post", 0) for d in common_domains
        ]
        axes[2].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[2].set_xticks(x + width)
    axes[2].set_xticklabels(common_domains, rotation=45)
    axes[2].set_title("Question Mark per Post")
    axes[2].set_ylabel("Count")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "punctuation_diversity.png"), dpi=150)
    plt.close()
    print("Saved: " + os.path.join(output_dir, "punctuation_diversity.png"))


def main():
    setup_fonts()
    output_dir = "/home/qwe12345678/RedNote-Vibe-Analysis/output"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    human_data = load_jsonl("training_set_human.jsonl")
    aigc_data = load_jsonl("training_set_aigc.jsonl")
    exploring_data = load_jsonl("exploring_set.jsonl")

    print("  Human: " + str(len(human_data)) + " records")
    print("  AIGC: " + str(len(aigc_data)) + " records")
    print("  Exploring: " + str(len(exploring_data)) + " records")

    data_dict = {"Human": human_data, "AIGC": aigc_data, "Exploring": exploring_data}

    print("\n" + "=" * 80)
    print("NEW TOPIC COMPARISON ANALYSIS: Exploring vs AIGC vs Human")
    print("=" * 80)

    eng_results = analyze_engagement_patterns(data_dict)
    plot_engagement_patterns(eng_results, output_dir)

    title_results = analyze_title_content_ratio(data_dict)
    plot_title_content_ratio(title_results, output_dir)

    unique_vocab_results = analyze_unique_vocabulary(data_dict)
    plot_unique_vocabulary(unique_vocab_results, output_dir)

    sentence_results = analyze_sentence_features(data_dict)
    plot_sentence_features(sentence_results, output_dir)

    analyze_keyword_cooccurrence(data_dict)

    consistency_results = analyze_content_consistency(data_dict)
    plot_content_consistency(consistency_results, output_dir)

    emotion_results = analyze_emotion_intensity(data_dict)
    plot_emotion_intensity(emotion_results, output_dir)

    time_results = analyze_time_distribution(data_dict)

    hashtag_results = analyze_hashtag_usage(data_dict)
    plot_hashtag_usage(hashtag_results, output_dir)

    punct_results = analyze_punctuation_diversity(data_dict)
    plot_punctuation_diversity(punct_results, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("")
    print("Generated visualizations:")
    print("  - " + os.path.join(output_dir, "engagement_patterns.png"))
    print("  - " + os.path.join(output_dir, "title_content_ratio.png"))
    print("  - " + os.path.join(output_dir, "unique_vocabulary.png"))
    print("  - " + os.path.join(output_dir, "sentence_features.png"))
    print("  - " + os.path.join(output_dir, "content_consistency.png"))
    print("  - " + os.path.join(output_dir, "emotion_intensity.png"))
    print("  - " + os.path.join(output_dir, "hashtag_usage.png"))
    print("  - " + os.path.join(output_dir, "punctuation_diversity.png"))


if __name__ == "__main__":
    main()
