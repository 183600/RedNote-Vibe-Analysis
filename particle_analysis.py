# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter, defaultdict
from scipy import stats

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

COMMON_PARTICLES = [
    "的",
    "了",
    "是",
    "在",
    "有",
    "我",
    "你",
    "他",
    "她",
    "它",
    "们",
    "这",
    "那",
    "就",
    "也",
    "还",
    "都",
    "而",
    "与",
    "和",
    "把",
    "被",
    "让",
    "给",
    "对",
    "从",
    "到",
    "为",
    "以",
    "及",
    "或",
    "但",
    "却",
    "又",
    "可",
    "要",
    "会",
    "能",
    "可以",
    "已经",
    "没有",
    "因为",
    "所以",
    "如果",
    "虽然",
    "但是",
    "而且",
    "然后",
    "还是",
    "只是",
    "不过",
    "可能",
    "应该",
    "这样",
    "那样",
    "怎么",
    "什么",
    "为什么",
    "一个",
    "这个",
    "那个",
    "自己",
    "现在",
    "以前",
    "以后",
    "一直",
    "一下",
    "一点",
    "非常",
    "特别",
    "真的",
    "好像",
    "感觉",
    "觉得",
    "知道",
    "看到",
    "听到",
    "想到",
    "希望",
    "喜欢",
    "需要",
    "开始",
    "结束",
    "继续",
    "改变",
    "形成",
    "成为",
    "得到",
    "包括",
    "作为",
    "比如",
    "例如",
    "关于",
    "通过",
    "根据",
    "由于",
    "因此",
    "所以",
    "然后",
    "接着",
    "于是",
    "最后",
    "总之",
    "反正",
    "总共",
    "一共",
    "至少",
    "最多",
    "最少",
    "大概",
    "大约",
    "左右",
    "以上",
    "以下",
    "之内",
    "之外",
    "当中",
    "中间",
    "内部",
    "外部",
    "前面",
    "后面",
    "上面",
    "下面",
    "旁边",
    "附近",
    "这里",
    "那里",
    "哪里",
    "有些",
    "有的",
    "有些",
    "某些",
    "每个",
    "各个",
    "有人",
    "有人",
    "没人",
    "人人",
    "别人",
    "大家",
    "咱们",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "有人",
    "何时",
    "何地",
    "何事",
    "何人",
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


def count_particle(text, particle):
    if not text:
        return 0
    return text.count(particle)


def count_particles_in_text(text, particles):
    if not text:
        return {}
    result = {}
    for p in particles:
        result[p] = text.count(p)
    return result


def compute_particle_ratio(text, particle):
    if not text or len(text) == 0:
        return 0
    count = text.count(particle)
    return count / len(text)


def compute_all_particle_ratios(text, particles):
    if not text or len(text) == 0:
        return {p: 0 for p in particles}
    result = {}
    for p in particles:
        result[p] = text.count(p) / len(text)
    return result


def load_particle_data(files, particles):
    data = {}
    for f in files:
        dataset = load_jsonl(f)
        data[f] = dataset
        print("Loaded {}: {} records".format(f, len(dataset)))

    particle_stats = {}
    for f in files:
        dataset = data[f]
        stats_list = []
        for item in dataset:
            title = item.get("note_title", "")
            desc = item.get("desc", "") or item.get("note_content", "")
            text = title + " " + (desc if desc else "")

            ratios = compute_all_particle_ratios(text, particles)
            stats_list.append(
                {
                    "title": title,
                    "desc": desc,
                    "text": text,
                    "ratios": ratios,
                    "domain": item.get("domain", ""),
                    "local_time": item.get("local_time", ""),
                    "liked_count": item.get("liked_count", 0),
                    "collected_count": item.get("collected_count", 0),
                }
            )
        particle_stats[f] = stats_list
    return particle_stats


def analyze_particle_distribution(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Particle Distribution Analysis")
    print("=" * 80)

    results = {}
    for f in files:
        stats_list = particle_stats[f]
        all_ratios = defaultdict(list)
        for item in stats_list:
            for p in top_particles:
                all_ratios[p].append(item["ratios"].get(p, 0))

        results[f] = {}
        for p in top_particles:
            ratios = all_ratios[p]
            results[f][p] = {
                "mean": np.mean(ratios) if ratios else 0,
                "std": np.std(ratios) if ratios else 0,
                "median": np.median(ratios) if ratios else 0,
                "total": sum(ratios) if ratios else 0,
            }

    print("\nMean Ratio of Particles:")
    print("-" * 80)
    header = "Particle".ljust(15)
    for f in files:
        header += f.replace(".jsonl", "")[:12].ljust(12)
    print(header)

    for p in top_particles[:30]:
        row = p.ljust(15)
        for f in files:
            row += str(round(results[f][p]["mean"], 6)).ljust(12)
        print(row)

    return results


def analyze_particle_by_domain(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Particle Distribution by Domain")
    print("=" * 80)

    domain_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for f in files:
        stats_list = particle_stats[f]
        for item in stats_list:
            domain = item.get("domain", "unknown")
            for p in top_particles:
                domain_results[f][domain][p].append(item["ratios"].get(p, 0))

    for f in files:
        print("\n{}:".format(f))
        for domain in sorted(domain_results[f].keys()):
            print("  {}:".format(domain))
            for p in top_particles[:10]:
                ratios = domain_results[f][domain][p]
                if ratios:
                    mean_val = np.mean(ratios)
                    print("    {}: {} (n={})".format(p, mean_val, len(ratios)))


def analyze_particle_by_time(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Particle Distribution over Time")
    print("=" * 80)

    time_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for f in files:
        stats_list = particle_stats[f]
        for item in stats_list:
            local_time = item.get("local_time", "")
            if len(local_time) >= 6:
                month = local_time[:6]
            else:
                month = "unknown"
            for p in top_particles:
                time_results[f][month][p].append(item["ratios"].get(p, 0))

    for f in files:
        print("\n{}:".format(f))
        for month in sorted(time_results[f].keys())[:12]:
            for p in ["的", "了", "是", "在"]:
                ratios = time_results[f][month][p]
                if ratios:
                    mean_val = np.mean(ratios)
                    print("  {} - {}: {}".format(month, p, mean_val))


def analyze_particle_correlation(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Particle Correlation Analysis between Datasets")
    print("=" * 80)

    for p in top_particles[:20]:
        correlations = []
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                f1 = files[i]
                f2 = files[j]
                stats1 = particle_stats[f1]
                stats2 = particle_stats[f2]

                ratios1 = [item["ratios"].get(p, 0) for item in stats1]
                ratios2 = [item["ratios"].get(p, 0) for item in stats2]

                min_len = min(len(ratios1), len(ratios2))
                if min_len > 10:
                    corr, _ = stats.pearsonr(ratios1[:min_len], ratios2[:min_len])
                    correlations.append((f1, f2, corr))

        print("\nParticle '{}':".format(p))
        for f1, f2, corr in correlations:
            print("  {} vs {}: {}".format(f1, f2, corr))


def plot_particle_comparison(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Generating Particle Comparison Plots")
    print("=" * 80)

    particles_to_plot = ["的", "了", "是", "在", "有", "我", "你", "他", "这", "那"]
    particles_to_plot = [p for p in particles_to_plot if p in top_particles]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Common Particle Distribution Comparison", fontsize=14)

    color_map = {
        "training_set_human.jsonl": "blue",
        "training_set_aigc.jsonl": "red",
        "exploring_set.jsonl": "green",
    }
    label_map = {
        "training_set_human.jsonl": "Human",
        "training_set_aigc.jsonl": "AIGC",
        "exploring_set.jsonl": "Exploring",
    }

    for idx, p in enumerate(particles_to_plot[:4]):
        ax = axes[idx // 2, idx % 2]
        for f in files:
            stats_list = particle_stats[f]
            ratios = [item["ratios"].get(p, 0) for item in stats_list]
            if ratios:
                ax.hist(
                    ratios, bins=30, alpha=0.5, label=label_map[f], color=color_map[f]
                )
        ax.set_xlabel("Ratio of '{}'".format(p))
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of '{}'".format(p))
        ax.legend()

    plt.tight_layout()
    plt.savefig("particle_distribution_comparison.png", dpi=150)
    plt.close()
    print("Saved: particle_distribution_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Mean Particle Ratios by Dataset", fontsize=14)

    for idx, p in enumerate(particles_to_plot[:4]):
        ax = axes[idx // 2, idx % 2]
        means = []
        labels = []
        for f in files:
            stats_list = particle_stats[f]
            ratios = [item["ratios"].get(p, 0) for item in stats_list]
            means.append(np.mean(ratios) if ratios else 0)
            labels.append(label_map[f])

        ax.bar(labels, means, color=[color_map[f] for f in files])
        ax.set_ylabel("Mean Ratio")
        ax.set_title("Mean Ratio of '{}'".format(p))

    plt.tight_layout()
    plt.savefig("particle_mean_comparison.png", dpi=150)
    plt.close()
    print("Saved: particle_mean_comparison.png")


def analyze_content_length_with_particles(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Content Length vs Particle Usage")
    print("=" * 80)

    for f in files:
        stats_list = particle_stats[f]
        lengths = []
        particle_counts = {p: [] for p in top_particles[:10]}

        for item in stats_list:
            text = item.get("text", "")
            text_len = len(text)
            if text_len > 0:
                lengths.append(text_len)
                for p in top_particles[:10]:
                    particle_counts[p].append(item["ratios"].get(p, 0) * text_len)

        print("\n{}:".format(f))
        print("  Average text length: {}".format(np.mean(lengths)))
        for p in top_particles[:10]:
            if lengths:
                avg_count = np.mean(particle_counts[p])
                print("  Average '{}' count per text: {}".format(p, avg_count))


def analyze_particle_with_engagement(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Particle Usage vs Engagement Metrics")
    print("=" * 80)

    for f in files:
        stats_list = particle_stats[f]
        for p in ["的", "了", "是", "在"]:
            highs = []
            lows = []
            for item in stats_list:
                ratio = item["ratios"].get(p, 0)
                liked = item.get("liked_count", 0)
                if liked > 10:
                    highs.append(ratio)
                elif liked == 0:
                    lows.append(ratio)

            if highs and lows:
                high_mean = np.mean(highs)
                low_mean = np.mean(lows)
                print(
                    "{} - '{}': popular={} (n={}), zero_like={} (n={})".format(
                        f, p, high_mean, len(highs), low_mean, len(lows)
                    )
                )


def compute_detailed_correlation(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Detailed Particle Frequency Correlation")
    print("=" * 80)

    particles_to_use = ["的", "了", "是", "在", "有", "我", "你", "这", "那", "就"]
    particles_to_use = [p for p in particles_to_use if p in top_particles]

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            f1 = files[i]
            f2 = files[j]
            stats1 = particle_stats[f1]
            stats2 = particle_stats[f2]

            print("\n{} vs {}:".format(f1, f2))

            for p in particles_to_use:
                ratios1 = [item["ratios"].get(p, 0) for item in stats1]
                ratios2 = [item["ratios"].get(p, 0) for item in stats2]

                min_len = min(len(ratios1), len(ratios2))
                if min_len > 10:
                    corr, pval = stats.pearsonr(ratios1[:min_len], ratios2[:min_len])
                    print("  {}: corr={}, p-value={}".format(p, corr, pval))


def plot_particle_heatmap(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Generating Particle Heatmap")
    print("=" * 80)

    key_particles = [
        "的",
        "了",
        "是",
        "在",
        "有",
        "我",
        "你",
        "他",
        "她",
        "它",
        "们",
        "这",
        "那",
        "就",
        "也",
        "还",
        "都",
        "而",
        "与",
        "和",
        "把",
        "被",
        "让",
        "给",
        "对",
        "从",
        "到",
        "为",
        "以",
        "及",
        "但",
        "却",
        "又",
        "可",
        "要",
        "会",
        "能",
        "可以",
        "已经",
        "没有",
        "因为",
        "所以",
        "如果",
        "虽然",
        "但是",
        "而且",
        "然后",
        "还是",
        "这样",
        "那样",
        "怎么",
        "什么",
        "为什么",
        "一个",
        "这个",
        "那个",
        "自己",
        "现在",
        "以前",
        "以后",
        "一直",
        "一下",
        "一点",
        "非常",
        "特别",
        "真的",
        "好像",
        "感觉",
        "觉得",
        "知道",
        "看到",
        "听到",
        "希望",
        "喜欢",
        "需要",
        "开始",
        "结束",
        "继续",
        "改变",
        "形成",
        "成为",
        "得到",
        "包括",
        "作为",
        "比如",
        "例如",
        "关于",
        "通过",
    ]
    key_particles = [p for p in key_particles if p in top_particles][:60]

    fig, axes = plt.subplots(1, 3, figsize=(20, 12))
    fig.suptitle("Particle Mean Ratios by Dataset", fontsize=14)

    label_map = {
        "training_set_human.jsonl": "Human",
        "training_set_aigc.jsonl": "AIGC",
        "exploring_set.jsonl": "Exploring",
    }

    for idx, f in enumerate(files):
        ax = axes[idx]
        stats_list = particle_stats[f]
        means = []
        for item in stats_list:
            for p in key_particles:
                means.append(item["ratios"].get(p, 0))

        np_means = np.array(means).reshape(len(stats_list), len(key_particles))
        mean_vals = np.mean(np_means, axis=0)

        ax.barh(range(len(key_particles)), mean_vals)
        ax.set_yticks(range(len(key_particles)))
        ax.set_yticklabels(key_particles, fontsize=8)
        ax.set_xlabel("Mean Ratio")
        ax.set_title(label_map[f])

    plt.tight_layout()
    plt.savefig("particle_all_comparison.png", dpi=150)
    plt.close()
    print("Saved: particle_all_comparison.png")

    fig, ax = plt.subplots(figsize=(16, 10))
    matrix = []
    labels = []
    for f in files:
        stats_list = particle_stats[f]
        mean_vals = []
        for p in key_particles:
            ratios = [item["ratios"].get(p, 0) for item in stats_list]
            mean_vals.append(np.mean(ratios))
        matrix.append(mean_vals)
        labels.append(label_map[f])

    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(key_particles)))
    ax.set_xticklabels(key_particles, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(files)))
    ax.set_yticklabels(labels)
    ax.set_title("Particle Mean Ratios Heatmap")
    plt.colorbar(im, ax=ax, label="Mean Ratio")

    plt.tight_layout()
    plt.savefig("particle_heatmap.png", dpi=150)
    plt.close()
    print("Saved: particle_heatmap.png")


def analyze_particle_pairs(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Particle Co-occurrence Analysis")
    print("=" * 80)

    target_pairs = [
        ("的", "了"),
        ("的", "是"),
        ("的", "在"),
        ("了", "是"),
        ("我", "你"),
        ("这", "那"),
        ("的", "我"),
        ("的", "你"),
        ("有", "没有"),
        ("可以", "能"),
        ("是", "在"),
        ("了", "在"),
    ]

    for f in files:
        print("\n{}:".format(f))
        stats_list = particle_stats[f]

        for p1, p2 in target_pairs:
            counts1 = []
            counts2 = []
            for item in stats_list:
                text = item.get("text", "")
                text_len = len(text) if text else 1
                if text_len > 0:
                    r1 = item["ratios"].get(p1, 0) * text_len
                    r2 = item["ratios"].get(p2, 0) * text_len
                    counts1.append(r1)
                    counts2.append(r2)

            if counts1 and counts2 and len(counts1) > 10:
                corr, pval = stats.pearsonr(counts1, counts2)
                print(
                    "  '{}' vs '{}': corr={:.4f}, p={:.2e}".format(p1, p2, corr, pval)
                )


def main():
    files = [
        "training_set_human.jsonl",
        "training_set_aigc.jsonl",
        "exploring_set.jsonl",
    ]

    particle_stats = load_particle_data(files, COMMON_PARTICLES)

    top_particles = COMMON_PARTICLES

    analyze_particle_distribution(particle_stats, files, top_particles)

    analyze_particle_by_domain(particle_stats, files, top_particles)

    analyze_particle_by_time(particle_stats, files, top_particles)

    analyze_particle_correlation(particle_stats, files, top_particles)

    plot_particle_comparison(particle_stats, files, top_particles)

    analyze_content_length_with_particles(particle_stats, files, top_particles)

    analyze_particle_with_engagement(particle_stats, files, top_particles)

    compute_detailed_correlation(particle_stats, files, top_particles)

    plot_particle_heatmap(particle_stats, files, top_particles)

    analyze_particle_pairs(particle_stats, files, top_particles)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
