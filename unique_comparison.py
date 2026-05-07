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


def analyze_user_behavior_detailed(data_dict):
    print("\n" + "=" * 80)
    print("1. DETAILED USER BEHAVIOR ANALYSIS")
    print("=" * 80)

    results = {}
    for ds_name, data in data_dict.items():
        user_counts = Counter(item.get("user_id", "unknown") for item in data)
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
            "total_posts": len(data),
            "posts_per_user": len(data) / len(user_counts) if user_counts else 0,
            "single_post_users_pct": single_post_users / len(user_counts) * 100
            if user_counts
            else 0,
            "power_users_pct": power_users / len(user_counts) * 100
            if user_counts
            else 0,
            "gini_coefficient": gini_coef,
            "max_posts_by_user": max(post_counts) if post_counts else 0,
            "top_user_post_share": post_counts[0] / len(data) * 100
            if post_counts
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


def analyze_engagement_efficiency(data_dict):
    print("\n" + "=" * 80)
    print("2. ENGAGEMENT EFFICIENCY ANALYSIS (Engagement per Character)")
    print("=" * 80)

    results = {}
    for ds_name, data in data_dict.items():
        results[ds_name] = {}

        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items or len(domain_items) < 5:
                continue

            efficiencies = []
            for item in domain_items:
                text = get_text(item)
                length = len(text)
                if length > 0:
                    engagement = (
                        item.get("liked_count", 0)
                        + item.get("collected_count", 0)
                        + item.get("comments_count", 0)
                    )
                    efficiencies.append(engagement / length)

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


def analyze_interactive_elements(data_dict):
    print("\n" + "=" * 80)
    print("4. INTERACTIVE ELEMENTS ANALYSIS (URLs, @mentions, emojis)")
    print("=" * 80)

    results = {}
    for ds_name, data in data_dict.items():
        results[ds_name] = {}

        for domain in DOMAINS:
            domain_items = [item for item in data if item.get("domain") == domain]
            if not domain_items:
                continue

            url_count = 0
            mention_count = 0
            emoji_count = 0

            for item in domain_items:
                text = get_text(item)

                if re.search(r"http[s]?://|www\.", text):
                    url_count += 1
                if re.search(r"@[\w]+", text):
                    mention_count += 1
                emoji_count += len(re.findall(r"\[.*?\]", text))

            total = len(domain_items)
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


def analyze_domain_concentration(data_dict):
    print("\n" + "=" * 80)
    print("5. DOMAIN CONCENTRATION ANALYSIS")
    print("=" * 80)

    results = {}
    for ds_name, data in data_dict.items():
        domain_counts = Counter(item.get("domain", "unknown") for item in data)

        total = len(data)
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


def generate_summary_dashboard(data_dict, output_dir):
    print("\n" + "=" * 80)
    print("6. COMPREHENSIVE SUMMARY DASHBOARD")
    print("=" * 80)

    user_results = analyze_user_behavior_detailed(data_dict)
    eng_results = analyze_engagement_efficiency(data_dict)
    inter_results = analyze_interactive_elements(data_dict)
    conc_results = analyze_domain_concentration(data_dict)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    datasets = ["Human", "AIGC", "Exploring"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    x = np.arange(len(datasets))

    axes[0, 0].bar(
        x, [user_results[ds]["posts_per_user"] for ds in datasets], color=colors
    )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].set_title("Posts per User")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].bar(
        x, [user_results[ds]["gini_coefficient"] for ds in datasets], color=colors
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(datasets)
    axes[0, 1].set_title("User Inequality (Gini)")
    axes[0, 1].set_ylabel("Gini Coefficient")

    axes[0, 2].bar(
        x, [user_results[ds]["single_post_users_pct"] for ds in datasets], color=colors
    )
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(datasets)
    axes[0, 2].set_title("Single-Post Users")
    axes[0, 2].set_ylabel("Percentage (%)")

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

    avg_eng = []
    for ds in datasets:
        domain_engs = [
            eng_results[ds].get(d, {}).get("avg_efficiency", 0) for d in DOMAINS
        ]
        avg_eng.append(np.mean([e for e in domain_engs if e > 0]))
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
    output_dir = "/home/qwe12345678/RedNote-Vibe-Analysis/output"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    human_data = load_jsonl("training_set_human.jsonl")
    aigc_data = load_jsonl("training_set_aigc.jsonl")
    exploring_data = load_jsonl("exploring_set.jsonl")

    print(f"  Human: {len(human_data)} records")
    print(f"  AIGC: {len(aigc_data)} records")
    print(f"  Exploring: {len(exploring_data)} records")

    data_dict = {"Human": human_data, "AIGC": aigc_data, "Exploring": exploring_data}

    print("\n" + "=" * 80)
    print("UNIQUE COMPARISON ANALYSIS: New Metrics")
    print("=" * 80)

    all_results = generate_summary_dashboard(data_dict, output_dir)

    eng_results = analyze_engagement_efficiency(data_dict)
    plot_user_engagement_comparison(eng_results, output_dir)

    inter_results = analyze_interactive_elements(data_dict)
    plot_interactive_elements(inter_results, output_dir)

    conc_results = analyze_domain_concentration(data_dict)
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
