# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter, defaultdict
import warnings
import os


def extract_face_emojis(text):
    """Extract emoji count excluding [话题] topic tags"""
    all_emojis = re.findall(r"\[.*?\]", text)
    return len([e for e in all_emojis if e != "[话题]"])


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

warnings.filterwarnings("ignore")

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
        "的",
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


def clean_tokenize(text):
    if not text:
        return []
    text = str(text)
    text = re.sub(r"#\w+\[话题\]", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    return [w for w in text.split() if len(w) > 1 and w not in STOPWORDS]


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


def analyze_domain_differences(texts_data, output_dir):
    domains = [
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

    results = {}
    for domain in domains:
        domain_texts = [t for t in texts_data if t.get("domain") == domain]
        if not domain_texts:
            continue

        lengths = []
        word_freq = Counter()
        emoji_counts = []

        for item in domain_texts:
            desc = item.get("desc", "") or item.get("note_content", "")
            if desc:
                lengths.append(len(desc))
                words = clean_tokenize(desc)
                word_freq.update(words)

                emojis = extract_face_emojis(desc)
                emoji_counts.append(emojis)

        results[domain] = {
            "count": len(domain_texts),
            "avg_length": np.mean(lengths) if lengths else 0,
            "median_length": np.median(lengths) if lengths else 0,
            "top_words": word_freq.most_common(30),
            "avg_emojis": np.mean(emoji_counts) if emoji_counts else 0,
            "liked": [item.get("liked_count", 0) for item in domain_texts],
            "collected": [item.get("collected_count", 0) for item in domain_texts],
            "comments": [item.get("comments_count", 0) for item in domain_texts],
        }

    print("\n=== Domain Differences Analysis ===")
    for domain, data in results.items():
        eng = data["liked"] + data["collected"] + data["comments"]
        print(f"\n{domain}:")
        print(
            f"  Count: {data['count']}, Avg length: {data['avg_length']:.1f}, Avg emojis: {data['avg_emojis']:.2f}"
        )
        print(f"  Top 10 words: {[w for w, c in data['top_words'][:10]]}")
        print(f"  Avg engagement: {np.mean(eng):.2f}")

    plot_domain_comparison(results, output_dir)
    return results


def plot_domain_comparison(results, output_dir):
    domains = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    counts = [results[d]["count"] for d in domains]
    axes[0, 0].bar(domains, counts, color="steelblue")
    axes[0, 0].set_title("Sample Count by Domain")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis="x", rotation=45)

    avg_lens = [results[d]["avg_length"] for d in domains]
    axes[0, 1].bar(domains, avg_lens, color="coral")
    axes[0, 1].set_title("Average Text Length by Domain")
    axes[0, 1].set_ylabel("Length (chars)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    avg_emojis = [results[d]["avg_emojis"] for d in domains]
    axes[1, 0].bar(domains, avg_emojis, color="mediumseagreen")
    axes[1, 0].set_title("Average Emoji Count by Domain")
    axes[1, 0].set_ylabel("Emoji count")
    axes[1, 0].tick_params(axis="x", rotation=45)

    eng_means = [
        np.mean(results[d]["liked"] + results[d]["collected"] + results[d]["comments"])
        for d in domains
    ]
    axes[1, 1].bar(domains, eng_means, color="orchid")
    axes[1, 1].set_title("Average Engagement by Domain")
    axes[1, 1].set_ylabel("Engagement")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir + "/domain_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/domain_comparison.png")


def compare_aigc_vs_human_by_domain(human_data, aigc_data, output_dir):
    domains = [
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

    comparison = {}
    for domain in domains:
        human_texts = [t for t in human_data if t.get("domain") == domain]
        aigc_texts = [t for t in aigc_data if t.get("domain") == domain]

        if not human_texts or not aigc_texts:
            continue

        h_lengths = [
            len(t.get("desc", "") or t.get("note_content", "")) for t in human_texts
        ]
        a_lengths = [
            len(t.get("desc", "") or t.get("note_content", "")) for t in aigc_texts
        ]

        h_emoji = [
            extract_face_emojis(t.get("desc", "") or t.get("note_content", ""))
            for t in human_texts
        ]
        a_emoji = [
            extract_face_emojis(t.get("desc", "") or t.get("note_content", ""))
            for t in aigc_texts
        ]

        h_eng = [
            t.get("liked_count", 0)
            + t.get("collected_count", 0)
            + t.get("comments_count", 0)
            for t in human_texts
        ]
        a_eng = [
            t.get("liked_count", 0)
            + t.get("collected_count", 0)
            + t.get("comments_count", 0)
            for t in aigc_texts
        ]

        h_words = Counter()
        for t in human_texts:
            desc = t.get("desc", "") or t.get("note_content", "")
            h_words.update(clean_tokenize(desc))

        a_words = Counter()
        for t in aigc_texts:
            desc = t.get("desc", "") or t.get("note_content", "")
            a_words.update(clean_tokenize(desc))

        comparison[domain] = {
            "human_count": len(human_texts),
            "aigc_count": len(aigc_texts),
            "human_avg_length": np.mean(h_lengths),
            "aigc_avg_length": np.mean(a_lengths),
            "human_avg_emoji": np.mean(h_emoji),
            "aigc_avg_emoji": np.mean(a_emoji),
            "human_avg_engagement": np.mean(h_eng),
            "aigc_avg_engagement": np.mean(a_eng),
            "human_top_words": h_words.most_common(20),
            "aigc_top_words": a_words.most_common(20),
        }

    print("\n=== AIGC vs Human by Domain ===")
    for domain, data in comparison.items():
        print(f"\n{domain}:")
        print(
            f"  Human: {data['human_count']} samples, avg_length={data['human_avg_length']:.1f}, emoji={data['human_avg_emoji']:.2f}, eng={data['human_avg_engagement']:.1f}"
        )
        print(
            f"  AIGC:  {data['aigc_count']} samples, avg_length={data['aigc_avg_length']:.1f}, emoji={data['aigc_avg_emoji']:.2f}, eng={data['aigc_avg_engagement']:.1f}"
        )

    plot_aigc_human_comparison(comparison, output_dir)
    return comparison


def plot_aigc_human_comparison(comparison, output_dir):
    domains = list(comparison.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    h_counts = [comparison[d]["human_count"] for d in domains]
    a_counts = [comparison[d]["aigc_count"] for d in domains]
    x = np.arange(len(domains))
    width = 0.35
    axes[0, 0].bar(x - width / 2, h_counts, width, label="Human", color="steelblue")
    axes[0, 0].bar(x + width / 2, a_counts, width, label="AIGC", color="coral")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(domains, rotation=45)
    axes[0, 0].set_title("Sample Count: Human vs AIGC")
    axes[0, 0].legend()

    h_lens = [comparison[d]["human_avg_length"] for d in domains]
    a_lens = [comparison[d]["aigc_avg_length"] for d in domains]
    axes[0, 1].bar(x - width / 2, h_lens, width, label="Human", color="steelblue")
    axes[0, 1].bar(x + width / 2, a_lens, width, label="AIGC", color="coral")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(domains, rotation=45)
    axes[0, 1].set_title("Average Text Length: Human vs AIGC")
    axes[0, 1].legend()

    h_emoji = [comparison[d]["human_avg_emoji"] for d in domains]
    a_emoji = [comparison[d]["aigc_avg_emoji"] for d in domains]
    axes[1, 0].bar(x - width / 2, h_emoji, width, label="Human", color="steelblue")
    axes[1, 0].bar(x + width / 2, a_emoji, width, label="AIGC", color="coral")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(domains, rotation=45)
    axes[1, 0].set_title("Average Emoji Count: Human vs AIGC")
    axes[1, 0].legend()

    h_eng = [comparison[d]["human_avg_engagement"] for d in domains]
    a_eng = [comparison[d]["aigc_avg_engagement"] for d in domains]
    axes[1, 1].bar(x - width / 2, h_eng, width, label="Human", color="steelblue")
    axes[1, 1].bar(x + width / 2, a_eng, width, label="AIGC", color="coral")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(domains, rotation=45)
    axes[1, 1].set_title("Average Engagement: Human vs AIGC")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir + "/aigc_human_domain_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/aigc_human_domain_comparison.png")


def analyze_domain_word_overlap(human_data, aigc_data, output_dir):
    domains = [
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

    print("\n=== Domain Word Overlap Analysis ===")
    overlap_results = {}

    for domain in domains:
        human_texts = [t for t in human_data if t.get("domain") == domain]
        aigc_texts = [t for t in aigc_data if t.get("domain") == domain]

        if not human_texts or not aigc_texts:
            continue

        h_words = Counter()
        for t in human_texts:
            desc = t.get("desc", "") or t.get("note_content", "")
            h_words.update(clean_tokenize(desc))

        a_words = Counter()
        for t in aigc_texts:
            desc = t.get("desc", "") or t.get("note_content", "")
            a_words.update(clean_tokenize(desc))

        common_words = set(h_words.keys()) & set(a_words.keys())

        top_h = set([w for w, c in h_words.most_common(50)])
        top_a = set([w for w, c in a_words.most_common(50)])
        top_overlap = top_h & top_a

        overlap_results[domain] = {
            "common_count": len(common_words),
            "top50_overlap": len(top_overlap),
            "top50_jaccard": len(top_overlap) / len(top_h | top_a)
            if top_h | top_a
            else 0,
        }

        print(f"\n{domain}:")
        print(f"  Common words: {len(common_words)}")
        print(
            f"  Top 50 overlap: {len(top_overlap)} / 50 ({len(top_overlap) / 50 * 100:.1f}%)"
        )
        print(f"  Jaccard similarity: {overlap_results[domain]['top50_jaccard']:.3f}")

    plot_word_overlap(overlap_results, output_dir)
    return overlap_results


def plot_word_overlap(overlap_results, output_dir):
    domains = list(overlap_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    common_counts = [overlap_results[d]["common_count"] for d in domains]
    axes[0].bar(domains, common_counts, color="steelblue")
    axes[0].set_title("Common Words Between Human & AIGC by Domain")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    jaccards = [overlap_results[d]["top50_jaccard"] for d in domains]
    axes[1].bar(domains, jaccards, color="coral")
    axes[1].set_title("Top 50 Words Jaccard Similarity by Domain")
    axes[1].set_ylabel("Jaccard Index")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir + "/domain_word_overlap.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/domain_word_overlap.png")


def compute_domain_sentiment(texts_data):
    positive_words = set(
        [
            "好",
            "喜欢",
            "爱",
            "开心",
            "快乐",
            "幸福",
            "美",
            "棒",
            "赞",
            "优秀",
            "完美",
            "可爱",
            "漂亮",
            "精彩",
            "感谢",
            "感动",
            "满足",
            "舒服",
            "满意",
        ]
    )
    negative_words = set(
        [
            "差",
            "不好",
            "讨厌",
            "难过",
            "痛苦",
            "累",
            "失望",
            "糟糕",
            "烦",
            "恶心",
            "无奈",
            "后悔",
            "生气",
            "悲伤",
            "焦虑",
            "担心",
            "害怕",
            "寂寞",
        ]
    )

    domains = [
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
    sentiment_results = {}

    for domain in domains:
        domain_texts = [t for t in texts_data if t.get("domain") == domain]
        if not domain_texts:
            continue

        pos_count = 0
        neg_count = 0
        total_words = 0

        for item in domain_texts:
            desc = item.get("desc", "") or item.get("note_content", "")
            words = clean_tokenize(desc)
            total_words += len(words)

            for w in words:
                if w in positive_words:
                    pos_count += 1
                elif w in negative_words:
                    neg_count += 1

        sentiment_results[domain] = {
            "positive_ratio": pos_count / total_words if total_words > 0 else 0,
            "negative_ratio": neg_count / total_words if total_words > 0 else 0,
            "sentiment_score": (pos_count - neg_count) / total_words
            if total_words > 0
            else 0,
        }

    print("\n=== Sentiment by Domain ===")
    for domain, data in sentiment_results.items():
        print(
            f"{domain}: positive={data['positive_ratio']:.4f}, negative={data['negative_ratio']:.4f}, score={data['sentiment_score']:.4f}"
        )

    return sentiment_results


def compare_domains_between_datasets(human_data, aigc_data, output_dir):
    h_domains = Counter(t.get("domain") for t in human_data)
    a_domains = Counter(t.get("domain") for t in aigc_data)

    all_domains = set(h_domains.keys()) | set(a_domains.keys())

    print("\n=== Domain Distribution Comparison ===")
    for domain in sorted(all_domains):
        h_pct = h_domains.get(domain, 0) / len(human_data) * 100
        a_pct = a_domains.get(domain, 0) / len(aigc_data) * 100
        diff = h_pct - a_pct
        print(f"{domain}: Human={h_pct:.1f}%, AIGC={a_pct:.1f}%, Diff={diff:+.1f}%")

    plot_domain_distribution(h_domains, a_domains, output_dir)


def plot_domain_distribution(h_domains, a_domains, output_dir):
    all_domains = sorted(set(h_domains.keys()) | set(a_domains.keys()))

    h_pcts = [h_domains.get(d, 0) for d in all_domains]
    a_pcts = [a_domains.get(d, 0) for d in all_domains]

    h_total = sum(h_pcts)
    a_total = sum(a_pcts)
    h_pcts = [x / h_total * 100 if h_total > 0 else 0 for x in h_pcts]
    a_pcts = [x / a_total * 100 if a_total > 0 else 0 for x in a_pcts]

    x = np.arange(len(all_domains))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, h_pcts, width, label="Human", color="steelblue")
    plt.bar(x + width / 2, a_pcts, width, label="AIGC", color="coral")
    plt.xticks(x, all_domains, rotation=45)
    plt.title("Domain Distribution: Human vs AIGC")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir + "/domain_distribution_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/domain_distribution_comparison.png")


def main():
    output_dir = "/home/qwe12345678/RedNote-Vibe-Analysis/output"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    human_data = load_jsonl("training_set_human_fixed.jsonl")
    aigc_data = load_jsonl("exploring_set_fixed.jsonl")

    print(f"Human: {len(human_data)} records")
    print(f"AIGC: {len(aigc_data)} records")

    analyze_domain_differences(human_data, output_dir)
    compare_aigc_vs_human_by_domain(human_data, aigc_data, output_dir)
    analyze_domain_word_overlap(human_data, aigc_data, output_dir)
    compute_domain_sentiment(human_data)
    compute_domain_sentiment(aigc_data)
    compare_domains_between_datasets(human_data, aigc_data, output_dir)

    print("\n=== Topic Comparison Analysis Complete ===")
    print("Generated files:")
    print("- domain_comparison.png")
    print("- aigc_human_domain_comparison.png")
    print("- domain_word_overlap.png")
    print("- domain_distribution_comparison.png")


if __name__ == "__main__":
    main()
