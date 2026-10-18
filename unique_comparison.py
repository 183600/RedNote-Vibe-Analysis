# -*- coding: utf-8 -*-
"""
Unique Additional Analysis: Exploring vs AIGC vs Human
New metrics not covered in previous analyses
- User posting behavior patterns
- Domain-specific engagement efficiency
- Content freshness scoring
- Cross-domain migration patterns
- Interactive element diversity
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

# 热循环内使用的正则统一预编译（等价于原来的 re.search/re.findall 字面量）。
RE_URL = re.compile(r"http[s]?://|www\.")
RE_MENTION = re.compile(r"@[\w]+")
RE_BRACKET = re.compile(r"\[.*?\]")


def setup_fonts():
    init_plot_style()


def load_aggregates():
    files = [
        ("Human", "training_set_human.jsonl"),
        ("AIGC", "training_set_aigc.jsonl"),
        ("Exploring", "exploring_set.jsonl"),
    ]
    aggs = {}
    for ds_name, filepath in files:
        user_counts = Counter()
        has_user_id = False
        domain_counts = Counter()
        domain_totals = Counter()
        eng = defaultdict(list)
        inter = defaultdict(lambda: [0, 0, 0, 0])
        total = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                total += 1
                if "user_id" in item:
                    has_user_id = True
                    user_counts[item["user_id"]] += 1
                domain = item.get("domain", "unknown")
                domain_counts[domain] += 1
                if domain in DOMAINS:
                    domain_totals[domain] += 1
                    text = get_text(item)
                    length = len(text)
                    if length > 0:
                        engagement = sum(engagement_of(item))
                        eng[domain].append(engagement / length)
                    rec = inter[domain]
                    rec[3] += 1
                    if RE_URL.search(text):
                        rec[0] += 1
                    if RE_MENTION.search(text):
                        rec[1] += 1
                    rec[2] += sum(1 for e in RE_BRACKET.findall(text)
                                  if e != "[话题]")
        aggs[ds_name] = {
            "total": total,
            "has_user_id": has_user_id,
            "user_counts": user_counts,
            "domain_counts": domain_counts,
            "domain_totals": domain_totals,
            "eng": eng,
            "inter": inter,
        }
        print(f"  {ds_name}: {total} records")
    return aggs


def get_text(item):
    return item.get("note_content", "") or item.get("desc", "") or ""


def analyze_user_behavior_detailed(aggs):
    print("\n" + "=" * 80)
    print("1. DETAILED USER BEHAVIOR ANALYSIS")
    print("=" * 80)

    results = {}
    for ds_name, agg in aggs.items():
        if not agg["has_user_id"]:
            print(f"\n{ds_name}: no user_id field - user-level metrics N/A")
            results[ds_name] = None
            continue
        user_counts = agg["user_counts"]
        attributed_posts = sum(user_counts.values())
        post_counts = sorted(user_counts.values(), reverse=True)

        single_post_users = sum(1 for c in post_counts if c == 1)
        power_users = sum(1 for c in post_counts if c >= 10)

        gini_coef = 0
        if post_counts:
            n = len(post_counts)
            sorted_counts = sorted(post_counts)
            cumsum = np.cumsum(sorted_counts)
            gini_coef = (
                (
                    2 * np.sum((np.arange(1, n + 1) * sorted_counts))
                    - (n + 1) * cumsum[-1]
                )
                / (n * cumsum[-1])
                if cumsum[-1] > 0
                else 0
            )

        results[ds_name] = {
            "unique_users": len(user_counts),
            "total_posts": agg["total"],
            "posts_per_user": attributed_posts / len(user_counts)
            if user_counts
            else 0,
            "single_post_users_pct": single_post_users / len(user_counts) * 100
            if user_counts
            else 0,
            "power_users_pct": power_users / len(user_counts) * 100
            if user_counts
            else 0,
            "gini_coefficient": gini_coef,
            "max_posts_by_user": max(post_counts) if post_counts else 0,
            "top_user_post_share": post_counts[0] / attributed_posts * 100
            if post_counts and attributed_posts
            else 0,
        }

        print(f"\n{ds_name}:")
        print(f"  Unique users: {results[ds_name]['unique_users']}")
        print(f"  Posts per user: {results[ds_name]['posts_per_user']:.2f}")
        print(f"  Single-post users: {results[ds_name]['single_post_users_pct']:.1f}%")
        print(f"  Power users (10+ posts): {results[ds_name]['power_users_pct']:.1f}%")
        print(f"  Gini coefficient: {results[ds_name]['gini_coefficient']:.3f}")
        print(f"  Top user share: {results[ds_name]['top_user_post_share']:.1f}%")

    return results


def analyze_engagement_efficiency(aggs):
    print("\n" + "=" * 80)
    print("2. ENGAGEMENT EFFICIENCY ANALYSIS (Engagement per Character)")
    print("=" * 80)

    results = {}
    for ds_name, agg in aggs.items():
        results[ds_name] = {}

        for domain in DOMAINS:
            domain_total = agg["domain_totals"].get(domain, 0)
            if domain_total < 5:
                continue

            efficiencies = agg["eng"].get(domain, [])

            results[ds_name][domain] = {
                "avg_efficiency": np.mean(efficiencies) if efficiencies else 0,
                "median_efficiency": np.median(efficiencies) if efficiencies else 0,
                "max_efficiency": max(efficiencies) if efficiencies else 0,
            }

    for domain in DOMAINS:
        print(f"\n--- {domain} ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                r = results[ds][domain]
                print(
                    f"  {ds}: AvgEff={r['avg_efficiency']:.4f}, MaxEff={r['max_efficiency']:.2f}"
                )

    return results


def analyze_content_freshness(data_dict):
    print("\n" + "=" * 80)
    print("3. CONTENT FRESHNESS ANALYSIS")
    print("=" * 80)

    results = {}
    for ds_name, data in data_dict.items():
        results[ds_name] = {}

        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items:
                continue

            title_unique_chars = []
            content_unique_chars = []

            for item in domain_items:
                title = item.get("note_title", "") or ""
                text = get_text(item)

                title_unique_chars.append(len(set(title)))
                content_unique_chars.append(len(set(text)))

            results[ds_name][domain] = {
                "avg_title_unique_chars": np.mean(title_unique_chars)
                if title_unique_chars
                else 0,
                "avg_content_unique_chars": np.mean(content_unique_chars)
                if content_unique_chars
                else 0,
                "title_diversity": np.std(title_unique_chars)
                if title_unique_chars
                else 0,
                "content_diversity": np.std(content_unique_chars)
                if content_unique_chars
                else 0,
            }

    for domain in DOMAINS:
        print(f"\n--- {domain} ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                r = results[ds][domain]
                print(
                    f"  {ds}: TitleUnique={r['avg_title_unique_chars']:.1f}, ContentUnique={r['avg_content_unique_chars']:.1f}"
                )

    return results


def plot_user_engagement_comparison(eng_results, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    common_domains = [
        d for d in DOMAINS if all(d in eng_results.get(ds, {}) for ds in datasets)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(common_domains))
    width = 0.25

    for i, ds in enumerate(datasets):
        vals = [
            eng_results[ds].get(d, {}).get("avg_efficiency", 0) for d in common_domains
        ]
        axes[0].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(common_domains, rotation=45)
    axes[0].set_title("Engagement Efficiency (per character)")
    axes[0].set_ylabel("Engagement/Char")
    axes[0].legend()

    for i, ds in enumerate(datasets):
        vals = [
            eng_results[ds].get(d, {}).get("max_efficiency", 0) for d in common_domains
        ]
        axes[1].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(common_domains, rotation=45)
    axes[1].set_title("Max Engagement Efficiency by Domain")
    axes[1].set_ylabel("Engagement/Char")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "engagement_efficiency.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'engagement_efficiency.png')}")


def analyze_interactive_elements(aggs):
    print("\n" + "=" * 80)
    print("4. INTERACTIVE ELEMENTS ANALYSIS (URLs, @mentions, emojis)")
    print("=" * 80)

    results = {}
    for ds_name, agg in aggs.items():
        results[ds_name] = {}

        for domain in DOMAINS:
            total = agg["domain_totals"].get(domain, 0)
            if not total:
                continue

            url_count, mention_count, emoji_count = agg["inter"].get(
                domain, [0, 0, 0, 0]
            )[:3]

            results[ds_name][domain] = {
                "url_rate": url_count / total * 100 if total > 0 else 0,
                "mention_rate": mention_count / total * 100 if total > 0 else 0,
                "avg_emojis": emoji_count / total if total > 0 else 0,
            }

    for domain in DOMAINS:
        print(f"\n--- {domain} ---")
        for ds in ["Human", "AIGC", "Exploring"]:
            if ds in results and domain in results[ds]:
                r = results[ds][domain]
                print(
                    f"  {ds}: URL={r['url_rate']:.1f}%, Mention={r['mention_rate']:.1f}%, AvgEmoji={r['avg_emojis']:.1f}"
                )

    return results


def plot_interactive_elements(inter_results, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    common_domains = [
        d for d in DOMAINS if all(d in inter_results.get(ds, {}) for ds in datasets)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(common_domains))
    width = 0.25

    for i, ds in enumerate(datasets):
        vals = [inter_results[ds].get(d, {}).get("url_rate", 0) for d in common_domains]
        axes[0].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(common_domains, rotation=45)
    axes[0].set_title("URL Usage Rate")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].legend()

    for i, ds in enumerate(datasets):
        vals = [
            inter_results[ds].get(d, {}).get("mention_rate", 0) for d in common_domains
        ]
        axes[1].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(common_domains, rotation=45)
    axes[1].set_title("@Mention Rate")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend()

    for i, ds in enumerate(datasets):
        vals = [
            inter_results[ds].get(d, {}).get("avg_emojis", 0) for d in common_domains
        ]
        axes[2].bar(x + i * width, vals, width, label=ds, color=colors[i], alpha=0.85)
    axes[2].set_xticks(x + width)
    axes[2].set_xticklabels(common_domains, rotation=45)
    axes[2].set_title("Average Emoji Count")
    axes[2].set_ylabel("Count")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "interactive_elements.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'interactive_elements.png')}")


def analyze_domain_concentration(aggs):
    print("\n" + "=" * 80)
    print("5. DOMAIN CONCENTRATION ANALYSIS")
    print("=" * 80)

    results = {}
    for ds_name, agg in aggs.items():
        domain_counts = agg["domain_counts"]

        total = agg["total"]
        top1_pct = (
            domain_counts.most_common(1)[0][1] / total * 100 if domain_counts else 0
        )
        top3_pct = sum(c for _, c in domain_counts.most_common(3)) / total * 100

        entropy = 0
        for count in domain_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        max_ent = np.log2(len(domain_counts)) if domain_counts else 1
        normalized_entropy = entropy / max_ent if max_ent > 0 else 0

        results[ds_name] = {
            "unique_domains": len(domain_counts),
            "top1_concentration": top1_pct,
            "top3_concentration": top3_pct,
            "domain_entropy": entropy,
            "normalized_entropy": normalized_entropy,
        }

        print(f"\n{ds_name}:")
        print(f"  Unique domains: {results[ds_name]['unique_domains']}")
        print(f"  Top 1 domain share: {results[ds_name]['top1_concentration']:.1f}%")
        print(f"  Top 3 domains share: {results[ds_name]['top3_concentration']:.1f}%")
        print(f"  Domain entropy: {results[ds_name]['domain_entropy']:.3f}")
        print(f"  Normalized entropy: {results[ds_name]['normalized_entropy']:.3f}")

    return results


def plot_domain_concentration(conc_results, output_dir):
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(datasets))

    axes[0].bar(
        x, [conc_results[ds]["top1_concentration"] for ds in datasets], color=colors
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_title("Top 1 Domain Concentration")
    axes[0].set_ylabel("Percentage (%)")

    axes[1].bar(
        x, [conc_results[ds]["normalized_entropy"] for ds in datasets], color=colors
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_title("Domain Diversity (Normalized Entropy)")
    axes[1].set_ylabel("Entropy (0-1)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "domain_concentration.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'domain_concentration.png')}")


def generate_summary_dashboard(aggs, output_dir):
    print("\n" + "=" * 80)
    print("6. COMPREHENSIVE SUMMARY DASHBOARD")
    print("=" * 80)

    user_results = analyze_user_behavior_detailed(aggs)
    eng_results = analyze_engagement_efficiency(aggs)
    inter_results = analyze_interactive_elements(aggs)
    conc_results = analyze_domain_concentration(aggs)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    x = np.arange(len(datasets))

    def user_vals(key):
        return [
            user_results[ds][key] if user_results[ds] is not None else np.nan
            for ds in datasets
        ]

    missing_users = [ds for ds in datasets if user_results[ds] is None]

    axes[0, 0].bar(
        x, user_vals("posts_per_user"), color=colors
    )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].set_title("Posts per User")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].bar(
        x, user_vals("gini_coefficient"), color=colors
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(datasets)
    axes[0, 1].set_title("User Inequality (Gini)")
    axes[0, 1].set_ylabel("Gini Coefficient")

    axes[0, 2].bar(
        x, user_vals("single_post_users_pct"), color=colors
    )
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(datasets)
    axes[0, 2].set_title("Single-Post Users")
    axes[0, 2].set_ylabel("Percentage (%)")

    if missing_users:
        fig.text(
            0.5,
            0.005,
            "no user_id field (metrics N/A): " + ", ".join(missing_users),
            ha="center",
            fontsize=11,
        )

    axes[1, 0].bar(
        x, [conc_results[ds]["top1_concentration"] for ds in datasets], color=colors
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(datasets)
    axes[1, 0].set_title("Top Domain Concentration")
    axes[1, 0].set_ylabel("Percentage (%)")

    axes[1, 1].bar(
        x, [conc_results[ds]["normalized_entropy"] for ds in datasets], color=colors
    )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(datasets)
    axes[1, 1].set_title("Domain Diversity")
    axes[1, 1].set_ylabel("Normalized Entropy")

    # 与 engagement_efficiency 图同口径：只在三个数据集都有结果的领域集合上
    # 求平均。原先各数据集按自己的领域子集（领域缺样本或不足 5 篇会被跳过，
    # 缺项又按 0 再被 e>0 过滤掉）分别求平均，三根柱子平均的领域并不相同，
    # 柱高之间不可比。
    common_domains = [d for d in DOMAINS
                      if all(d in eng_results.get(ds, {}) for ds in datasets)]
    avg_eng = [
        float(np.mean([eng_results[ds][d]["avg_efficiency"]
                       for d in common_domains]))
        if common_domains else 0.0
        for ds in datasets
    ]
    axes[1, 2].bar(x, avg_eng, color=colors)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(datasets)
    axes[1, 2].set_title("Avg Engagement Efficiency")
    axes[1, 2].set_ylabel("Engagement/Char")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unique_summary_dashboard.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'unique_summary_dashboard.png')}")

    return {
        "user_behavior": user_results,
        "engagement_efficiency": eng_results,
        "interactive_elements": inter_results,
        "domain_concentration": conc_results,
    }


def main():
    setup_fonts()
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    aggs = load_aggregates()

    print("\n" + "=" * 80)
    print("UNIQUE COMPARISON ANALYSIS: New Metrics")
    print("=" * 80)

    all_results = generate_summary_dashboard(aggs, output_dir)

    eng_results = all_results["engagement_efficiency"]
    plot_user_engagement_comparison(eng_results, output_dir)

    inter_results = all_results["interactive_elements"]
    plot_interactive_elements(inter_results, output_dir)

    conc_results = all_results["domain_concentration"]
    plot_domain_concentration(conc_results, output_dir)

    print("\n" + "=" * 80)
    print("UNIQUE ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - unique_summary_dashboard.png")
    print("  - engagement_efficiency.png")
    print("  - interactive_elements.png")
    print("  - domain_concentration.png")


if __name__ == "__main__":
    main()
