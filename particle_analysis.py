# -*- coding: utf-8 -*-
import array
import json
import re
from functools import partial

import numpy as np
from analysis_utils import init_plot_style, engagement_of, run_parallel
from derived_cache import iter_cached
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter, defaultdict
from scipy import stats

import os

init_plot_style()

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
                data = json.loads(line)
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


def _group_indices(labels):
    """把记录按标签分组为索引列表（保持组内原记录顺序）。"""
    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)
    return groups


def _load_particle_file(f, stored_particles, particle_index):
    """流式读取单个数据集并统计各粒子的占比（供 load_particle_data 并行调用）。"""
    k = len(stored_particles)
    ratios_flat = array.array("d")
    text_lens = array.array("q")
    domain_cache = {}
    month_cache = {}
    domains = []
    months = []
    liked_vals = []
    text_cache = {}
    total = 0
    # 派生缓存（投影字段 pickle）比逐行 json.loads 快；缺失时自动回退。
    for item in iter_cached(f):
        total += 1
        title = item.get("note_title", "")
        desc = item.get("desc", "") or item.get("note_content", "")
        text = title + " " + (desc if desc else "")
        text_len = len(text)

        if not text or text_len == 0:
            ratios_flat.extend([0.0] * k)
        else:
            # 缓存存「首次出现的行号」而不是整段 ratio 数组：重复文本
            # 命中时从 ratios_flat 拷贝该行，省掉每个唯一文本一份
            # float64 副本（探索集约 6 万个唯一文本 → 省约 30MB）。
            cached_row = text_cache.get(text)
            if cached_row is None:
                text_cache[text] = len(text_lens)
                ratios_flat.extend(
                    array.array(
                        "d",
                        [text.count(p) / text_len
                         for p in stored_particles],
                    )
                )
            else:
                start = cached_row * k
                ratios_flat.extend(ratios_flat[start:start + k])

        text_lens.append(text_len)
        dom = item.get("domain", "")
        domains.append(domain_cache.setdefault(dom, dom))
        local_time = item.get("local_time", "")
        month = local_time[:6] if len(local_time) >= 6 else "unknown"
        months.append(month_cache.setdefault(month, month))
        liked_vals.append(engagement_of(item)[0])
    ratios = np.frombuffer(ratios_flat, dtype=np.float64).reshape(total, k)
    return {
        "n": total,
        "ratios": ratios,
        "zeros": np.zeros(total),
        "particle_index": particle_index,
        "stored_particles": stored_particles,
        "text_len": np.frombuffer(text_lens, dtype=np.int64),
        "domain": domains,
        "domain_index": _group_indices(domains),
        "month_index": _group_indices(months),
        "liked": np.asarray(liked_vals, dtype=np.int64),
    }


def load_particle_data(files, particles):
    # 去重后仅保留下游真正访问的粒子：分布/相关分析取前 30/20/10 个，
    # 热力图取 key_particles[:60]，其最大索引落在去重序列的第 65 个之前。
    # 存前 65 个即覆盖全部访问点（此前需存全量，为修复热力图后半段为 0 的
    # slicing bug；65 已覆盖其索引上限）。
    #
    # 存储改为按记录连续排布的 float64 数组（array('d') + frombuffer 零拷贝），
    # 不再为每条记录构造 dict，内存从逐记录 dict 的数倍降为 ~8 字节/粒子。
    stored_particles = list(dict.fromkeys(particles))[:65]
    particle_index = {p: i for i, p in enumerate(stored_particles)}

    # 三个数据集互不依赖，进程池并行统计（返回 numpy 数组与列表，结果与
    # 串行执行完全一致）。
    results = run_parallel([
        partial(_load_particle_file, f, stored_particles, particle_index)
        for f in files
    ])
    for f, stats_data in zip(files, results):
        print("Loaded {}: {} records".format(f, stats_data["n"]))
    return dict(zip(files, results))


def _ratios_for(stats, particle):
    """取某粒子在各记录上的 ratio 列；未存储的粒子与旧口径一致按 0 处理。"""
    idx = stats["particle_index"].get(particle)
    if idx is None:
        return stats["zeros"]
    return stats["ratios"][:, idx]


def analyze_particle_distribution(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Particle Distribution Analysis")
    print("=" * 80)

    results = {}
    for f in files:
        stats_data = particle_stats[f]
        results[f] = {}
        for p in top_particles[:30]:
            ratios = _ratios_for(stats_data, p)
            results[f][p] = {
                "mean": np.mean(ratios) if ratios.size else 0,
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

    for f in files:
        stats_data = particle_stats[f]
        domain_index = stats_data["domain_index"]
        print("\n{}:".format(f))
        for domain in sorted(domain_index.keys()):
            idxs = domain_index[domain]
            print("  {}:".format(domain))
            for p in top_particles[:10]:
                ratios = _ratios_for(stats_data, p)[idxs]
                if ratios.size:
                    mean_val = np.mean(ratios)
                    print("    {}: {} (n={})".format(p, mean_val, len(ratios)))


def analyze_particle_by_time(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Particle Distribution over Time")
    print("=" * 80)

    for f in files:
        stats_data = particle_stats[f]
        month_index = stats_data["month_index"]
        print("\n{}:".format(f))
        for month in sorted(month_index.keys())[:12]:
            idxs = month_index[month]
            for p in ["的", "了", "是", "在"]:
                ratios = _ratios_for(stats_data, p)[idxs]
                if ratios.size:
                    mean_val = np.mean(ratios)
                    print("  {} - {}: {}".format(month, p, mean_val))


def _domain_mean_ratios(stats_data, particle):
    """按 domain 聚合该粒子的平均 ratio（用于跨语料的可比统计）。"""
    ratios = _ratios_for(stats_data, particle)
    out = {}
    for dom, idxs in stats_data["domain_index"].items():
        if dom:
            out[dom] = float(np.mean(ratios[idxs]))
    return out


def _domain_correlation(stats1, stats2, particle):
    """在共同 domain 上比较两语料的粒子平均 ratio；返回 (corr, p, n) 或 None。

    两语料的记录顺序互不相关，逐条配对没有意义；按 domain 聚合后比较，
    衡量的是「不同话题下粒子用法是否在两个语料中同向」。
    """
    m1 = _domain_mean_ratios(stats1, particle)
    m2 = _domain_mean_ratios(stats2, particle)
    common = sorted(set(m1) & set(m2))
    if len(common) < 3:
        return None
    v1 = np.asarray([m1[d] for d in common], dtype=float)
    v2 = np.asarray([m2[d] for d in common], dtype=float)
    if np.std(v1) <= 0 or np.std(v2) <= 0:
        return None
    corr, pval = stats.pearsonr(v1, v2)
    return corr, pval, len(common)


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
                res = _domain_correlation(particle_stats[f1],
                                          particle_stats[f2], p)
                if res is not None:
                    corr, _, n_dom = res
                    correlations.append((f1, f2, corr, n_dom))

        print("\nParticle '{}':".format(p))
        for f1, f2, corr, n_dom in correlations:
            print("  {} vs {}: {} ({} common domains)".format(
                f1, f2, corr, n_dom))


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
            ratios = _ratios_for(particle_stats[f], p)
            if ratios.size:
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
            ratios = _ratios_for(particle_stats[f], p)
            means.append(np.mean(ratios) if ratios.size else 0)
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
        stats_data = particle_stats[f]
        text_len = stats_data["text_len"]
        mask = text_len > 0
        lengths = text_len[mask]

        print("\n{}:".format(f))
        print("  Average text length: {}".format(np.mean(lengths)))
        for p in top_particles[:10]:
            if lengths.size:
                ratios = _ratios_for(stats_data, p)[mask]
                avg_count = np.mean(ratios * lengths)
                print("  Average '{}' count per text: {}".format(p, avg_count))


def analyze_particle_with_engagement(particle_stats, files, top_particles):
    print("\n" + "=" * 80)
    print("Particle Usage vs Engagement Metrics")
    print("=" * 80)

    for f in files:
        stats_data = particle_stats[f]
        liked = stats_data["liked"]
        high_mask = liked > 10
        low_mask = liked == 0
        for p in ["的", "了", "是", "在"]:
            ratios = _ratios_for(stats_data, p)
            highs = ratios[high_mask]
            lows = ratios[low_mask]

            if highs.size and lows.size:
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
                res = _domain_correlation(stats1, stats2, p)
                if res is not None:
                    corr, pval, n_dom = res
                    print("  {}: corr={}, p-value={} ({} common domains)".format(
                        p, corr, pval, n_dom))


def _mean_ratios(stats_data, key_particles, chunk_size=20000):
    """分块累加每个 particle 的平均 ratio，避免一次性构造大矩阵。"""
    k = len(key_particles)
    n = stats_data["n"]
    if n == 0:
        return np.zeros(k)
    total = np.zeros(k)
    col_idx = [stats_data["particle_index"].get(p) for p in key_particles]
    ratios = stats_data["ratios"]
    for start in range(0, n, chunk_size):
        block = ratios[start:start + chunk_size]
        arr = np.empty((block.shape[0], k), dtype=float)
        for j, ix in enumerate(col_idx):
            if ix is None:
                arr[:, j] = 0.0
            else:
                arr[:, j] = block[:, ix]
        total += arr.sum(axis=0)
    return total / n


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

    mean_by_file = {f: _mean_ratios(particle_stats[f], key_particles) for f in files}
    for idx, f in enumerate(files):
        ax = axes[idx]
        mean_vals = mean_by_file[f]

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
        matrix.append(mean_by_file[f])
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
        stats_data = particle_stats[f]
        text_len = stats_data["text_len"]
        safe_len = np.where(text_len == 0, 1, text_len)

        for p1, p2 in target_pairs:
            counts1 = _ratios_for(stats_data, p1) * safe_len
            counts2 = _ratios_for(stats_data, p2) * safe_len

            if counts1.size and counts2.size and len(counts1) > 10:
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
