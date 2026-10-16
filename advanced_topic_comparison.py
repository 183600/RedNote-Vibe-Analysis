# -*- coding: utf-8 -*-
"""
Advanced Topic Comparison Analysis: Exploring vs AIGC vs Human
Compares differences across 11 domains between all three datasets
Generates comprehensive visualizations and statistical comparisons
"""

import json
import re
import os
import warnings
import numpy as np
from collections import Counter, defaultdict
from scipy import stats

from analysis_utils import init_plot_style, engagement_of
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
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002702-\U000027b0"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "]+",
    flags=re.UNICODE,
)

RE_TOPIC_TAG = re.compile(r"#\w+\[话题\]")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_DIGITS = re.compile(r"\d+")
# 连续的非 \w 字符（标点/空白）作为分隔符直接切词，等价于原先「标点→空格再 split」。
RE_NONWORD = re.compile(r"[^\w]+")
RE_HAS_DIGIT = re.compile(r"\d")


def setup_fonts():
    init_plot_style()


# 下游分析只读取这些字段；加载时按白名单投影，避免整条记录（含 note_id /
# user_id / time 等未使用字段）长期驻留内存。容错口径与原实现一致。
_ITEM_KEYS = ("note_title", "note_content", "desc", "domain",
              "liked_count", "collected_count", "comments_count",
              "likes", "collections", "comments")


def iter_jsonl_items(filepath):
    """流式读取并按下游用到的字段白名单投影；容错口径与旧 load_jsonl 一致。"""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.isspace():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            yield {k: item[k] for k in _ITEM_KEYS if k in item}


def load_jsonl(filepath):
    return list(iter_jsonl_items(filepath))


def group_by_domain(filepath):
    """单次流式扫描 → ({domain: [items]}, 总记录数)，域内记录顺序不变。"""
    grouped = {d: [] for d in DOMAINS}
    total = 0
    for item in iter_jsonl_items(filepath):
        total += 1
        d = item.get("domain")
        if isinstance(d, str) and d in grouped:
            grouped[d].append(item)
    return grouped, total


def get_text(item):
    if "note_content" in item:
        return item.get("note_content", "") or ""
    return item.get("desc", "") or ""


def clean_tokenize(text):
    if not text:
        return []
    text = str(text)
    text = RE_TOPIC_TAG.sub(" ", text)
    text = RE_HASHTAG.sub(r"\1", text)
    text = RE_DIGITS.sub("", text)
    return [w for w in RE_NONWORD.split(text) if len(w) > 1 and w not in STOPWORDS]


def count_emojis(text):
    if not text:
        return 0
    return len(EMOJI_PATTERN.findall(str(text)))


def compute_sentiment(text):
    if not text:
        return 0, 0
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    return pos, neg


def compute_engagement(item):
    return sum(engagement_of(item))


def extract_domain_items(data, domain):
    return [item for item in data if item.get("domain") == domain]


def analyze_domain_stats(items):
    if not items:
        return None

    texts = [get_text(item) for item in items]

    lengths = [len(t) for t in texts]
    title_lengths = [len(item.get("note_title", "") or "") for item in items]
    emoji_counts = [count_emojis(t) for t in texts]
    engagements = [compute_engagement(item) for item in items]

    pos_total, neg_total = 0, 0
    for t in texts:
        p, n = compute_sentiment(t)
        pos_total += p
        neg_total += n

    word_freq = Counter()
    for t in texts:
        word_freq.update(clean_tokenize(t))

    # 逐帖计算「独特字/总字」再取均值：语料级 TTR 会随语料规模系统性下降，
    # 而三个数据集的域内帖数相差一个数量级，直接比较语料级 TTR 会引入规模混杂。
    char_ratios = [len(set(t)) / len(t) for t in texts if t]
    ttr = float(np.mean(char_ratios)) if char_ratios else 0.0

    exclaim = sum(t.count("!") + t.count("！") for t in texts)
    question = sum(t.count("?") + t.count("？") for t in texts)

    first_person = sum(1 for t in texts if "我" in t or "我们" in t)

    has_numbers = sum(1 for t in texts if RE_HAS_DIGIT.search(t))

    paragraphs = sum(len([p for p in t.split("\n") if p.strip()]) for t in texts)

    if neg_total > 0:
        sentiment_ratio = pos_total / neg_total
    else:
        sentiment_ratio = float("inf") if pos_total > 0 else 0.0

    return {
        "count": len(items),
        # 原始数组随统计结果一并返回，供统计检验复用，避免重复扫描文本
        "lengths": lengths,
        "emojis": emoji_counts,
        "engagements": engagements,
        "avg_length": np.mean(lengths),
        "median_length": np.median(lengths),
        "std_length": np.std(lengths),
        "avg_title_length": np.mean(title_lengths),
        "avg_emoji": np.mean(emoji_counts),
        "median_emoji": np.median(emoji_counts),
        "avg_engagement": np.mean(engagements),
        "median_engagement": np.median(engagements),
        "pos_count": pos_total,
        "neg_count": neg_total,
        "sentiment_ratio": sentiment_ratio,
        "top_words": word_freq.most_common(20),
        "ttr": ttr,
        "exclaim_per_post": exclaim / len(items),
        "question_per_post": question / len(items),
        "first_person_pct": first_person / len(items) * 100,
        "has_numbers_pct": has_numbers / len(items) * 100,
        "avg_paragraphs": paragraphs / len(items),
    }


def compute_word_similarity(words1, words2, top_n=50):
    set1 = set(w for w, c in words1.most_common(top_n))
    set2 = set(w for w, c in words2.most_common(top_n))
    intersection = set1 & set2
    union = set1 | set2
    jaccard = len(intersection) / len(union) if union else 0
    overlap_count = len(intersection)
    return jaccard, overlap_count


def plot_three_way_domain_comparison(all_stats, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    common_domains = sorted(
        set.intersection(*[set(s.keys()) for s in all_stats.values()])
    )

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    metrics = [
        ("avg_length", "Average Content Length", "chars"),
        ("avg_title_length", "Average Title Length", "chars"),
        ("avg_emoji", "Average Emoji Count", "count"),
        ("avg_engagement", "Average Engagement", "count"),
        ("sentiment_ratio", "Sentiment Ratio (Pos/Neg)", "ratio"),
        ("ttr", "Vocabulary Diversity (TTR)", "ratio"),
        ("exclaim_per_post", "Exclamation Marks per Post", "count"),
        ("question_per_post", "Question Marks per Post", "count"),
        ("first_person_pct", "First Person Usage", "%"),
    ]

    for idx, (metric, title, unit) in enumerate(metrics):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        x = np.arange(len(common_domains))
        width = 0.25

        for i, ds_name in enumerate(datasets):
            values = [all_stats[ds_name][d][metric] for d in common_domains]
            ax.bar(
                x + i * width, values, width, label=ds_name, color=colors[i], alpha=0.85
            )

        ax.set_xticks(x + width)
        ax.set_xticklabels(common_domains, rotation=45, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit, fontsize=10)
        if idx == 0:
            ax.legend(fontsize=9)

    for idx in range(len(metrics), 9):
        row, col = idx // 3, idx % 3
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "topic_comparison_three_way.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_dir}/topic_comparison_three_way.png")


def plot_domain_word_similarity(all_stats, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    common_domains = sorted(
        set.intersection(*[set(s.keys()) for s in all_stats.values()])
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [("Human", "AIGC"), ("Human", "Exploring"), ("AIGC", "Exploring")]

    for idx, (ds1, ds2) in enumerate(pairs):
        ax = axes[idx]
        similarities = []
        for domain in common_domains:
            words1 = Counter(dict(all_stats[ds1][domain]["top_words"]))
            words2 = Counter(dict(all_stats[ds2][domain]["top_words"]))
            jaccard, _ = compute_word_similarity(words1, words2)
            similarities.append(jaccard)

        bars = ax.bar(common_domains, similarities, color=colors_hack(idx), alpha=0.85)
        ax.set_xticklabels(common_domains, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"Word Overlap: {ds1} vs {ds2}", fontsize=11)
        ax.set_ylabel("Jaccard Similarity", fontsize=10)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, similarities):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "topic_word_similarity.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_dir}/topic_word_similarity.png")


def colors_hack(idx):
    return ["#2196F3", "#FF5722", "#4CAF50"][idx]


def plot_domain_distribution_comparison(all_stats, totals, output_dir):
    domains = DOMAINS
    h_counts = [all_stats["Human"].get(d, {}).get("count", 0) for d in domains]
    a_counts = [all_stats["AIGC"].get(d, {}).get("count", 0) for d in domains]
    e_counts = [all_stats["Exploring"].get(d, {}).get("count", 0) for d in domains]

    # 分母用「有领域标注的记录数」而非全部记录：探索集有 ~19% 记录缺 domain，
    # 用人均全部记录作分母会把所有领域占比整体压小，破坏跨数据集可比性。
    h_base = sum(h_counts) or 1
    a_base = sum(a_counts) or 1
    e_base = sum(e_counts) or 1
    h_pct = [c / h_base * 100 for c in h_counts]
    a_pct = [c / a_base * 100 for c in a_counts]
    e_pct = [c / e_base * 100 for c in e_counts]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(domains))
    width = 0.25

    axes[0].bar(x - width, h_pct, width, label="Human", color="#2196F3", alpha=0.85)
    axes[0].bar(x, a_pct, width, label="AIGC", color="#FF5722", alpha=0.85)
    axes[0].bar(x + width, e_pct, width, label="Exploring", color="#4CAF50", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(domains, rotation=45, ha="right")
    axes[0].set_title("Domain Distribution (%) Across Datasets")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].legend()

    # 复用 analyze_domain_stats 已算好的平均互动，避免逐域重复遍历记录
    h_eng_by_domain = [all_stats["Human"].get(d, {}).get("avg_engagement", 0)
                       for d in domains]
    a_eng_by_domain = [all_stats["AIGC"].get(d, {}).get("avg_engagement", 0)
                       for d in domains]
    e_eng_by_domain = [all_stats["Exploring"].get(d, {}).get("avg_engagement", 0)
                       for d in domains]

    axes[1].bar(
        x - width, h_eng_by_domain, width, label="Human", color="#2196F3", alpha=0.85
    )
    axes[1].bar(x, a_eng_by_domain, width, label="AIGC", color="#FF5722", alpha=0.85)
    axes[1].bar(
        x + width,
        e_eng_by_domain,
        width,
        label="Exploring",
        color="#4CAF50",
        alpha=0.85,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(domains, rotation=45, ha="right")
    axes[1].set_title("Average Engagement by Domain Across Datasets")
    axes[1].set_ylabel("Avg Engagement (likes+collects+comments)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "topic_distribution_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_dir}/topic_distribution_comparison.png")


def plot_domain_sentiment_comparison(all_stats, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    common_domains = sorted(
        set.intersection(*[set(s.keys()) for s in all_stats.values()])
    )
    colors_map = {"Human": "#2196F3", "AIGC": "#FF5722", "Exploring": "#4CAF50"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ds_name in datasets:
        pos_vals = [
            all_stats[ds_name][d]["pos_count"] / all_stats[ds_name][d]["count"]
            for d in common_domains
        ]
        neg_vals = [
            all_stats[ds_name][d]["neg_count"] / all_stats[ds_name][d]["count"]
            for d in common_domains
        ]
        x = np.arange(len(common_domains))
        width = 0.25
        offset = ["Human", "AIGC", "Exploring"].index(ds_name)

        axes[0].bar(
            x + offset * width,
            pos_vals,
            width,
            label=f"{ds_name} Pos",
            color=colors_map[ds_name],
            alpha=0.85,
        )
        axes[1].bar(
            x + offset * width,
            neg_vals,
            width,
            label=f"{ds_name} Neg",
            color=colors_map[ds_name],
            alpha=0.5,
        )

    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(common_domains, rotation=45, ha="right")
    axes[0].set_title("Positive Word Frequency by Domain")
    axes[0].set_ylabel("Avg Positive Words per Post")
    axes[0].legend(fontsize=8)

    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(common_domains, rotation=45, ha="right")
    axes[1].set_title("Negative Word Frequency by Domain")
    axes[1].set_ylabel("Avg Negative Words per Post")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "topic_sentiment_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_dir}/topic_sentiment_comparison.png")


def plot_domain_structure_comparison(all_stats, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    common_domains = sorted(
        set.intersection(*[set(s.keys()) for s in all_stats.values()])
    )
    colors_map = {"Human": "#2196F3", "AIGC": "#FF5722", "Exploring": "#4CAF50"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_labels = [
        ("first_person_pct", "First Person Usage (%)", 0, 0),
        ("has_numbers_pct", "Posts with Numbers (%)", 0, 1),
        ("avg_paragraphs", "Average Paragraphs", 1, 0),
        ("question_per_post", "Question Marks per Post", 1, 1),
    ]

    for metric, title, row, col in metrics_labels:
        ax = axes[row, col]
        x = np.arange(len(common_domains))
        width = 0.25
        for i, ds_name in enumerate(datasets):
            values = [all_stats[ds_name][d][metric] for d in common_domains]
            ax.bar(
                x + i * width,
                values,
                width,
                label=ds_name,
                color=colors_map[ds_name],
                alpha=0.85,
            )
        ax.set_xticks(x + width)
        ax.set_xticklabels(common_domains, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel(title.split("(")[0].strip() if "(" in title else title)
        if row == 0 and col == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "topic_structure_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_dir}/topic_structure_comparison.png")


def print_detailed_comparison(all_stats):
    datasets = ["Human", "AIGC", "Exploring"]
    common_domains = sorted(
        set.intersection(*[set(s.keys()) for s in all_stats.values()])
    )

    print("\n" + "=" * 100)
    print("DETAILED TOPIC COMPARISON: HUMAN vs AIGC vs EXPLORING")
    print("=" * 100)

    for domain in common_domains:
        print(f"\n{'=' * 60}")
        print(f"DOMAIN: {domain}")
        print(f"{'=' * 60}")

        print(f"{'Metric':<30} {'Human':>12} {'AIGC':>12} {'Exploring':>12}")
        print("-" * 70)

        metrics_display = [
            ("count", "Sample Count", "{:.0f}"),
            ("avg_length", "Avg Content Length", "{:.1f}"),
            ("median_length", "Median Content Length", "{:.1f}"),
            ("std_length", "Length Std Dev", "{:.1f}"),
            ("avg_title_length", "Avg Title Length", "{:.1f}"),
            ("avg_emoji", "Avg Emoji Count", "{:.2f}"),
            ("avg_engagement", "Avg Engagement", "{:.1f}"),
            ("sentiment_ratio", "Sentiment Ratio", "{:.2f}"),
            ("ttr", "Vocabulary Diversity", "{:.4f}"),
            ("exclaim_per_post", "Exclamation/Post", "{:.2f}"),
            ("question_per_post", "Question/Post", "{:.2f}"),
            ("first_person_pct", "First Person %", "{:.1f}%"),
            ("has_numbers_pct", "Has Numbers %", "{:.1f}%"),
            ("avg_paragraphs", "Avg Paragraphs", "{:.2f}"),
        ]

        for metric, label, fmt in metrics_display:
            vals = []
            for ds in datasets:
                v = all_stats[ds][domain][metric]
                if "%" in fmt:
                    vals.append(f"{v:.1f}%")
                else:
                    vals.append(
                        f"{v:>{12}.1f}"
                        if "Ratio" not in label and "Diversity" not in label
                        else f"{v:>12.4f}"
                        if "Diversity" in label
                        else f"{v:>12.2f}"
                    )
            print(f"{label:<30} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

        print(f"\n  Top Words Comparison:")
        for ds in datasets:
            top_words = [w for w, c in all_stats[ds][domain]["top_words"][:10]]
            print(f"    {ds:<12}: {', '.join(top_words)}")

        print(f"\n  Word Overlap (Top 20):")
        for i in range(3):
            for j in range(i + 1, 3):
                ds1, ds2 = datasets[i], datasets[j]
                words1 = Counter(dict(all_stats[ds1][domain]["top_words"]))
                words2 = Counter(dict(all_stats[ds2][domain]["top_words"]))
                jaccard, overlap = compute_word_similarity(words1, words2, top_n=20)
                print(
                    f"    {ds1} vs {ds2}: Jaccard={jaccard:.3f}, Overlap={overlap}/20"
                )


def print_cross_topic_analysis(all_stats):
    datasets = ["Human", "AIGC", "Exploring"]
    common_domains = sorted(
        set.intersection(*[set(s.keys()) for s in all_stats.values()])
    )

    print("\n" + "=" * 100)
    print("CROSS-TOPIC ANALYSIS: Between-Dataset, Within-Topic Variations")
    print("=" * 100)

    print("\n--- Length Differences Across Datasets (by Topic) ---")
    print(
        f"{'Domain':<10} {'Human->AIGC':>15} {'Human->Exploring':>18} {'AIGC->Exploring':>18}"
    )
    print("-" * 65)
    for domain in common_domains:
        h_len = all_stats["Human"][domain]["avg_length"]
        a_len = all_stats["AIGC"][domain]["avg_length"]
        e_len = all_stats["Exploring"][domain]["avg_length"]
        diff_ha = a_len - h_len
        diff_he = e_len - h_len
        diff_ae = e_len - a_len
        print(f"{domain:<10} {diff_ha:>+15.1f} {diff_he:>+18.1f} {diff_ae:>+18.1f}")

    print("\n--- Emoji Usage Differences (by Topic) ---")
    print(
        f"{'Domain':<10} {'Human->AIGC':>15} {'Human->Exploring':>18} {'AIGC->Exploring':>18}"
    )
    print("-" * 65)
    for domain in common_domains:
        h_emoji = all_stats["Human"][domain]["avg_emoji"]
        a_emoji = all_stats["AIGC"][domain]["avg_emoji"]
        e_emoji = all_stats["Exploring"][domain]["avg_emoji"]
        print(
            f"{domain:<10} {a_emoji - h_emoji:>+15.2f} {e_emoji - h_emoji:>+18.2f} {e_emoji - a_emoji:>+18.2f}"
        )

    print("\n--- Sentiment Ratio Differences (by Topic) ---")
    print(
        f"{'Domain':<10} {'Human':>10} {'AIGC':>10} {'Exploring':>12} {'Max Diff':>10}"
    )
    print("-" * 56)
    for domain in common_domains:
        h_sent = all_stats["Human"][domain]["sentiment_ratio"]
        a_sent = all_stats["AIGC"][domain]["sentiment_ratio"]
        e_sent = all_stats["Exploring"][domain]["sentiment_ratio"]
        max_diff = max(abs(h_sent - a_sent), abs(h_sent - e_sent), abs(a_sent - e_sent))
        print(
            f"{domain:<10} {h_sent:>10.2f} {a_sent:>10.2f} {e_sent:>12.2f} {max_diff:>10.2f}"
        )

    print("\n--- Vocabulary Diversity Differences (by Topic) ---")
    print(
        f"{'Domain':<10} {'Human':>10} {'AIGC':>10} {'Exploring':>12} {'Most Diverse':>15}"
    )
    print("-" * 60)
    for domain in common_domains:
        h_ttr = all_stats["Human"][domain]["ttr"]
        a_ttr = all_stats["AIGC"][domain]["ttr"]
        e_ttr = all_stats["Exploring"][domain]["ttr"]
        most = max(
            [("Human", h_ttr), ("AIGC", a_ttr), ("Exploring", e_ttr)],
            key=lambda x: x[1],
        )
        print(
            f"{domain:<10} {h_ttr:>10.4f} {a_ttr:>10.4f} {e_ttr:>12.4f} {most[0]:>15}"
        )

    print("\n--- Top Domain by Dataset for Each Metric ---")
    metrics_interesting = [
        ("avg_length", "Longest Content"),
        ("avg_title_length", "Longest Titles"),
        ("avg_emoji", "Most Emojis"),
        ("avg_engagement", "Highest Engagement"),
        ("sentiment_ratio", "Most Positive"),
        ("first_person_pct", "Most Personal"),
        ("question_per_post", "Most Questions"),
    ]

    for metric, label in metrics_interesting:
        print(f"\n  {label}:")
        for ds in datasets:
            best_domain = max(common_domains, key=lambda d: all_stats[ds][d][metric])
            best_val = all_stats[ds][best_domain][metric]
            print(f"    {ds:<12}: {best_domain} ({best_val:.2f})")

    print("\n--- Domain Specialization Analysis ---")
    print("Which topics have the most distinctive vocabulary per dataset?")
    print()
    for domain in common_domains:
        word_sets = {}
        for ds in datasets:
            word_sets[ds] = set(w for w, c in all_stats[ds][domain]["top_words"][:30])

        unique_to_h = word_sets["Human"] - word_sets["AIGC"] - word_sets["Exploring"]
        unique_to_a = word_sets["AIGC"] - word_sets["Human"] - word_sets["Exploring"]
        unique_to_e = word_sets["Exploring"] - word_sets["Human"] - word_sets["AIGC"]

        print(f"  {domain}:")
        print(
            f"    Human-only top words:    {', '.join(list(unique_to_h)[:8]) if unique_to_h else 'none'}"
        )
        print(
            f"    AIGC-only top words:     {', '.join(list(unique_to_a)[:8]) if unique_to_a else 'none'}"
        )
        print(
            f"    Exploring-only top words: {', '.join(list(unique_to_e)[:8]) if unique_to_e else 'none'}"
        )


def print_statistical_tests(all_stats):
    datasets = ["Human", "AIGC", "Exploring"]
    common_domains = sorted(
        set.intersection(*[set(s.keys()) for s in all_stats.values()])
    )

    print("\n" + "=" * 100)
    print("STATISTICAL SIGNIFICANCE TESTS (Within-Topic, Between-Dataset)")
    print("=" * 100)

    pairs = [("Human", "AIGC"), ("Human", "Exploring"), ("AIGC", "Exploring")]

    for domain in common_domains:
        print(f"\n--- Domain: {domain} ---")
        print(
            f"{'Comparison':<25} {'Length (p)':>15} {'Emoji (p)':>15} {'Engagement (p)':>18}"
        )
        print("-" * 75)

        for ds1, ds2 in pairs:
            # 复用分析阶段已提取的长度/表情/互动数组，不再重复处理文本
            s1 = all_stats[ds1].get(domain)
            s2 = all_stats[ds2].get(domain)
            lens1 = s1["lengths"] if s1 else []
            lens2 = s2["lengths"] if s2 else []

            if len(lens1) < 5 or len(lens2) < 5:
                print(f"{ds1} vs {ds2:<17} {'N/A':>15} {'N/A':>15} {'N/A':>18}")
                continue

            _, p_len = stats.mannwhitneyu(lens1, lens2, alternative="two-sided")
            _, p_emoji = stats.mannwhitneyu(s1["emojis"], s2["emojis"],
                                            alternative="two-sided")
            _, p_eng = stats.mannwhitneyu(s1["engagements"], s2["engagements"],
                                          alternative="two-sided")

            sig_len = (
                "***"
                if p_len < 0.001
                else "**"
                if p_len < 0.01
                else "*"
                if p_len < 0.05
                else "ns"
            )
            sig_emoji = (
                "***"
                if p_emoji < 0.001
                else "**"
                if p_emoji < 0.01
                else "*"
                if p_emoji < 0.05
                else "ns"
            )
            sig_eng = (
                "***"
                if p_eng < 0.001
                else "**"
                if p_eng < 0.01
                else "*"
                if p_eng < 0.05
                else "ns"
            )

            print(
                f"{ds1} vs {ds2:<17} {p_len:>10.4f}{sig_len:>4} {p_emoji:>10.4f}{sig_emoji:>4} {p_eng:>13.4f}{sig_eng:>4}"
            )


def main():
    setup_fonts()
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    grouped = {}
    totals = {}
    for ds_name, filepath in [
        ("Human", "training_set_human.jsonl"),
        ("AIGC", "training_set_aigc.jsonl"),
        ("Exploring", "exploring_set.jsonl"),
    ]:
        # 逐数据集流式读取并一次性按域分桶，避免整份 load 与逐域重复扫描
        grouped[ds_name], totals[ds_name] = group_by_domain(filepath)
        print(f"  {ds_name}: {totals[ds_name]} records")

    print("\nAnalyzing domain statistics for each dataset...")
    all_stats = {"Human": {}, "AIGC": {}, "Exploring": {}}

    for domain in DOMAINS:
        for ds_name in ["Human", "AIGC", "Exploring"]:
            items = grouped[ds_name][domain]
            if items:
                stats = analyze_domain_stats(items)
                if stats:
                    all_stats[ds_name][domain] = stats
                    print(f"  {ds_name} - {domain}: {stats['count']} items processed")

    # 统计结果已包含后续所需的全部数组，释放原始记录（大文本）再绘图
    del grouped

    print("\nGenerating visualizations...")
    plot_three_way_domain_comparison(all_stats, output_dir)
    plot_domain_word_similarity(all_stats, output_dir)
    plot_domain_distribution_comparison(all_stats, totals, output_dir)
    plot_domain_sentiment_comparison(all_stats, output_dir)
    plot_domain_structure_comparison(all_stats, output_dir)

    print_detailed_comparison(all_stats)
    print_cross_topic_analysis(all_stats)
    print_statistical_tests(all_stats)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print("\nGenerated visualizations:")
    print(f"  - {output_dir}/topic_comparison_three_way.png")
    print(f"  - {output_dir}/topic_word_similarity.png")
    print(f"  - {output_dir}/topic_distribution_comparison.png")
    print(f"  - {output_dir}/topic_sentiment_comparison.png")
    print(f"  - {output_dir}/topic_structure_comparison.png")


if __name__ == "__main__":
    main()
