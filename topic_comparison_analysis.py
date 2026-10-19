# -*- coding: utf-8 -*-
import re
from array import array

import numpy as np
from analysis_utils import init_plot_style
# 两个语料均经共享派生缓存读取（只用 PROJECT_KEYS 内字段）；缓存缺失时
# 自动逐位回退到逐行 JSON 解析，统计口径不变。
from derived_cache import iter_cached
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter, defaultdict
import warnings
import os

RE_TOPIC_TAG = re.compile(r"#\w+\[话题\]")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_XHS_EMO = re.compile(r"\[[^\[\]]{1,8}R\]")
RE_DIGITS = re.compile(r"\d+")
# 直接匹配长度 >1 的 \w 串，等价于原先「非 \w 切分后过滤长度 >1」。
RE_WORD2 = re.compile(r"\w{2,}")
# 方括号表情：负向前瞻剔除 [话题]，等价于原「匹配全部方括号再排除」。
RE_FACE_EMOJI = re.compile(r"\[(?!话题\])[^\]\n]*\]")


def extract_face_emojis(text):
    """Extract emoji count excluding [话题] topic tags"""
    return len(RE_FACE_EMOJI.findall(text))


init_plot_style()

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


POSITIVE_WORDS = set(
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
NEGATIVE_WORDS = set(
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

DOMAIN_LIST = [
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

# 中文描述无空格，clean_tokenize 产出的是连续片段（如 "今天好开心啊"），
# 直接用 `w in POSITIVE_WORDS` 会因整段不等于词表词而几乎永不命中。
# 与 advanced_topic_comparison.py 的子串匹配口径一致：在片段内做词表匹配，
# 正负词合并为单个模式、长词优先（如 "不好" 先于 "好"），避免把否定词
# 误判为正向，也避免同一位置重复计数。
SENTIMENT_PATTERN = re.compile(
    "(?P<pos>"
    + "|".join(sorted((re.escape(w) for w in POSITIVE_WORDS), key=len, reverse=True))
    + ")|(?P<neg>"
    + "|".join(sorted((re.escape(w) for w in NEGATIVE_WORDS), key=len, reverse=True))
    + ")"
)

# 词表全部用字：片段不含其中任何字符时不可能命中任何情感词，直接跳过正则。
_SENTIMENT_CHARS = frozenset("".join(POSITIVE_WORDS | NEGATIVE_WORDS))


def count_sentiment_words(token):
    """统计单个片段内正/负向词的出现次数。"""
    if _SENTIMENT_CHARS.isdisjoint(token):
        return 0, 0
    pos = neg = 0
    for m in SENTIMENT_PATTERN.finditer(token):
        if m.lastgroup == "pos":
            pos += 1
        else:
            neg += 1
    return pos, neg


def clean_tokenize(text):
    if not text:
        return []
    text = str(text)
    text = RE_TOPIC_TAG.sub(" ", text)
    text = RE_HASHTAG.sub(r"\1", text)
    text = RE_XHS_EMO.sub(" ", text)
    text = RE_DIGITS.sub("", text)
    return [w for w in RE_WORD2.findall(text) if w not in STOPWORDS]


def load_aggregates(filepath):
    # 逐记录指标只用于均值/中位数，用紧凑的 array('d')（8 字节/值）替代
    # Python 列表（~36 字节/值），内存从「记录数 × 5 × 36B」降为 1/4 以下；
    # 聚合结果（np.mean/np.median）与列表口径完全一致。
    domains = {}
    for domain in DOMAIN_LIST:
        domains[domain] = {
            "count": 0,
            "lengths": array("d"),
            "emoji": array("d"),
            "liked": array("d"),
            "collected": array("d"),
            "comments": array("d"),
            "words": Counter(),
            "pos": 0,
            "neg": 0,
            "total_words": 0,
        }
    domain_counter = Counter()
    total = 0
    has_engagement = False
    for item in iter_cached(filepath):
        total += 1
        if any(k in item for k in ("liked_count", "likes", "collected_count",
                                   "collections", "comments_count", "comments")):
            has_engagement = True
        domain = item.get("domain")
        domain_counter[domain] += 1
        if domain in domains:
            d = domains[domain]
            d["count"] += 1
            desc = item.get("desc", "") or item.get("note_content", "")
            d["lengths"].append(len(desc))
            d["emoji"].append(extract_face_emojis(desc))
            d["liked"].append(item.get("liked_count", 0))
            d["collected"].append(item.get("collected_count", 0))
            d["comments"].append(item.get("comments_count", 0))
            words = clean_tokenize(desc)
            d["words"].update(words)
            d["total_words"] += len(words)
            for w in words:
                pos_n, neg_n = count_sentiment_words(w)
                d["pos"] += pos_n
                d["neg"] += neg_n
    return {"total": total, "domain_counter": domain_counter, "domains": domains,
            "has_engagement": has_engagement}


def analyze_domain_differences(human_agg, output_dir):
    domains = DOMAIN_LIST

    results = {}
    for domain in domains:
        d = human_agg["domains"][domain]
        if d["count"] == 0:
            continue

        lengths = [l for l in d["lengths"] if l > 0]
        emoji_counts = [e for e, l in zip(d["emoji"], d["lengths"]) if l > 0]
        eng = [x + y + z for x, y, z in zip(d["liked"], d["collected"], d["comments"])]

        results[domain] = {
            "count": d["count"],
            "avg_length": np.mean(lengths) if lengths else 0,
            "median_length": np.median(lengths) if lengths else 0,
            "top_words": d["words"].most_common(30),
            "avg_emojis": np.mean(emoji_counts) if emoji_counts else 0,
            "avg_engagement": np.mean(eng) if eng else 0,
            "liked": d["liked"],
            "collected": d["collected"],
            "comments": d["comments"],
        }

    print("\n=== Domain Differences Analysis ===")
    for domain, data in results.items():
        print(f"\n{domain}:")
        print(
            f"  Count: {data['count']}, Avg length: {data['avg_length']:.1f}, Avg emojis: {data['avg_emojis']:.2f}"
        )
        print(f"  Top 10 words: {[w for w, c in data['top_words'][:10]]}")
        print(f"  Avg engagement: {data['avg_engagement']:.2f}")

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

    eng_means = [results[d]["avg_engagement"] for d in domains]
    axes[1, 1].bar(domains, eng_means, color="orchid")
    axes[1, 1].set_title("Average Engagement by Domain")
    axes[1, 1].set_ylabel("Engagement")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir + "/domain_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/domain_comparison.png")


def compare_aigc_vs_human_by_domain(human_agg, aigc_agg, output_dir):
    domains = DOMAIN_LIST

    comparison = {}
    for domain in domains:
        h = human_agg["domains"][domain]
        a = aigc_agg["domains"][domain]

        if h["count"] == 0 or a["count"] == 0:
            continue

        h_eng = [x + y + z for x, y, z in zip(h["liked"], h["collected"], h["comments"])]
        a_eng = [x + y + z for x, y, z in zip(a["liked"], a["collected"], a["comments"])]
        # AIGC 语料没有互动字段：缺失不等于互动量为 0，记为 N/A（NaN）
        h_has = human_agg.get("has_engagement", True)
        a_has = aigc_agg.get("has_engagement", True)

        comparison[domain] = {
            "human_count": h["count"],
            "aigc_count": a["count"],
            "human_avg_length": np.mean(h["lengths"]),
            "aigc_avg_length": np.mean(a["lengths"]),
            "human_avg_emoji": np.mean(h["emoji"]),
            "aigc_avg_emoji": np.mean(a["emoji"]),
            "human_avg_engagement": np.mean(h_eng) if h_has else float("nan"),
            "aigc_avg_engagement": np.mean(a_eng) if a_has else float("nan"),
            "human_has_engagement": h_has,
            "aigc_has_engagement": a_has,
            "human_top_words": h["words"].most_common(20),
            "aigc_top_words": a["words"].most_common(20),
        }

    print("\n=== AIGC vs Human by Domain ===")
    for domain, data in comparison.items():
        print(f"\n{domain}:")
        h_eng_str = (f"{data['human_avg_engagement']:.1f}"
                     if data["human_has_engagement"] else "N/A")
        a_eng_str = (f"{data['aigc_avg_engagement']:.1f}"
                     if data["aigc_has_engagement"] else "N/A")
        print(
            f"  Human: {data['human_count']} samples, avg_length={data['human_avg_length']:.1f}, emoji={data['human_avg_emoji']:.2f}, eng={h_eng_str}"
        )
        print(
            f"  AIGC:  {data['aigc_count']} samples, avg_length={data['aigc_avg_length']:.1f}, emoji={data['aigc_avg_emoji']:.2f}, eng={a_eng_str}"
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


def analyze_domain_word_overlap(human_agg, aigc_agg, output_dir):
    domains = DOMAIN_LIST

    print("\n=== Domain Word Overlap Analysis ===")
    overlap_results = {}

    for domain in domains:
        h = human_agg["domains"][domain]
        a = aigc_agg["domains"][domain]

        if h["count"] == 0 or a["count"] == 0:
            continue

        h_words = h["words"]
        a_words = a["words"]

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


def compute_domain_sentiment(agg):
    domains = DOMAIN_LIST
    sentiment_results = {}

    for domain in domains:
        d = agg["domains"][domain]
        if d["count"] == 0:
            continue

        pos_count = d["pos"]
        neg_count = d["neg"]
        total_words = d["total_words"]

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


def compare_domains_between_datasets(human_agg, aigc_agg, output_dir):
    h_domains = human_agg["domain_counter"]
    a_domains = aigc_agg["domain_counter"]

    all_domains = set(h_domains.keys()) | set(a_domains.keys())

    print("\n=== Domain Distribution Comparison ===")
    for domain in sorted(all_domains):
        h_pct = h_domains.get(domain, 0) / human_agg["total"] * 100
        a_pct = a_domains.get(domain, 0) / aigc_agg["total"] * 100
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
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    human_agg = load_aggregates("training_set_human_fixed.jsonl")
    print(f"Human: {human_agg['total']} records")
    aigc_agg = load_aggregates("training_set_aigc.jsonl")
    print(f"AIGC: {aigc_agg['total']} records")

    analyze_domain_differences(human_agg, output_dir)
    compare_aigc_vs_human_by_domain(human_agg, aigc_agg, output_dir)
    analyze_domain_word_overlap(human_agg, aigc_agg, output_dir)
    compute_domain_sentiment(human_agg)
    compute_domain_sentiment(aigc_agg)
    compare_domains_between_datasets(human_agg, aigc_agg, output_dir)

    print("\n=== Topic Comparison Analysis Complete ===")
    print("Generated files:")
    print("- domain_comparison.png")
    print("- aigc_human_domain_comparison.png")
    print("- domain_word_overlap.png")
    print("- domain_distribution_comparison.png")


if __name__ == "__main__":
    main()
