# -*- coding: utf-8 -*-
import json
import random
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from collections import Counter, defaultdict
from scipy import stats
import warnings
import os
import pandas as pd
from pandas import Timestamp, DatetimeIndex
from datetime import datetime, timezone

from analysis_utils import init_plot_style, load_records_with_total

init_plot_style()

warnings.filterwarnings("ignore")

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

EMOJI_FACE = re.compile("[\U0001f600-\U0001f64f]+", flags=re.UNICODE)

POSITIVE_WORDS = set(
    [
        "喜欢",
        "爱",
        "好",
        "棒",
        "赞",
        "美",
        "帅",
        "酷",
        "开心",
        "快乐",
        "幸福",
        "完美",
        "优秀",
        "厉害",
        "强大",
        "可爱",
        "漂亮",
        "好看",
        "精彩",
        "感动",
        "温暖",
        "甜蜜",
        "浪漫",
        "满足",
        "骄傲",
        "激动",
        "兴奋",
        "推荐",
        "超赞",
        "太强",
        "爱了",
        "绝了",
        "完美",
        "太好了",
        "好喜欢",
    ]
)

NEGATIVE_WORDS = set(
    [
        "讨厌",
        "恨",
        "差",
        "烂",
        "丑",
        "难过",
        "伤心",
        "痛苦",
        "生气",
        "失望",
        "后悔",
        "可怕",
        "恐怖",
        "恶心",
        "无聊",
        "累",
        "烦",
        "崩溃",
        "绝望",
        "悲剧",
        "坑",
        "骗",
        "假",
        "垃圾",
        "吐槽",
        "踩雷",
        "避雷",
    ]
)

NEUTRAL_WORDS = set(
    [
        "今天",
        "昨天",
        "明天",
        "分享",
        "记录",
        "看看",
        "感觉",
        "觉得",
        "其实",
        "可能",
        "应该",
        "知道",
        "看看",
        "就是",
        "然后",
    ]
)


SAMPLE_LIMIT = 5000


def load_jsonl(filepath, limit=SAMPLE_LIMIT, seed=42):
    return load_records_with_total(filepath, sample=limit, seed=seed)


def extract_emojis(text):
    if not text:
        return []
    emojis = EMOJI_PATTERN.findall(str(text))
    result = []
    for e in emojis:
        result.extend(list(e))
    return result


def extract_emoji_face(text):
    if not text:
        return "", 0
    faces = EMOJI_FACE.findall(str(text))
    if faces:
        return faces[0], len(faces)
    return "", 0


def extract_punctuation(text):
    if not text:
        return {}
    # ASCII 与对应全角标点都要统计：中文语料里 ！？：远超其半角形式，
    # 只统计半角会让热图低估主要句末标点（与各脚本统一的「半角+全角」口径一致）。
    punctuation = re.findall(r"([!?。.,，:：！？\"\'\-\(\)【】{}《》])", str(text))
    return Counter(punctuation)


def sentiment_analysis(text):
    if not text:
        return 0, 0, 0
    text = str(text).lower()
    pos_count = sum(1 for word in POSITIVE_WORDS if word in text)
    neg_count = sum(1 for word in NEGATIVE_WORDS if word in text)
    return pos_count, neg_count, pos_count - neg_count


def parse_timestamp(ts):
    if not ts:
        return None
    try:
        if isinstance(ts, (int, float)):
            # 兼容以整型存储的 "YYYYMMDDHH"（10 位）本地时间戳：
            # 与字符串分支同样先按该格式解析，避免被误当作 epoch 秒。
            if isinstance(ts, int) and len(str(ts)) == 10 and str(ts)[:2] in ("19", "20"):
                dt = datetime.strptime(str(ts), "%Y%m%d%H").replace(tzinfo=timezone.utc)
                return dt.timestamp()
            value = float(ts) / 1000.0 if ts > 1e10 else float(ts)
            return value + 8 * 3600
        ts_str = str(ts).strip()
        if len(ts_str) == 10 and ts_str.isdigit() and ts_str[:2] in ("19", "20"):
            dt = datetime.strptime(ts_str, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            return dt.timestamp()
        value = float(ts_str)
        value = value / 1000.0 if value > 1e10 else value
        return value + 8 * 3600
    except:
        return None


def get_time_features(item):
    ts = (
        item.get("local_time")
        or item.get("timestamp")
        or item.get("create_time")
        or item.get("time")
    )
    if ts:
        timestamp = parse_timestamp(ts)
        if timestamp:
            return timestamp
    return None


def compute_chi_square(counter1, counter2, top_n=10):
    """Chi-square test of homogeneity over the most frequent emoji types."""
    try:
        top = [e for e, _ in (counter1 + counter2).most_common(top_n)]
        if not top:
            return 0, 1
        observed = np.array(
            [[counter1.get(e, 0) for e in top], [counter2.get(e, 0) for e in top]]
        )
        chi2, p_value = stats.chi2_contingency(observed)[:2]
        return chi2, p_value
    except:
        return 0, 1


def compute_t_test(data1, data2):
    try:
        t_stat, p_value = stats.ttest_ind(data1, data2)
        return t_stat, p_value
    except:
        return 0, 1


def plot_emoji_distribution(emoji_counts, files, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (f, counts) in enumerate(zip(files, emoji_counts)):
        top_emojis = counts.most_common(20)
        if not top_emojis:
            continue
        emojis = [e for e, c in top_emojis]
        counts_list = [c for e, c in top_emojis]

        color = plt.cm.Set3(idx / 3)
        axes[idx].barh(range(len(emojis)), counts_list, color=color)
        axes[idx].set_yticks(range(len(emojis)))
        axes[idx].set_yticklabels(emojis, fontsize=8)
        axes[idx].invert_yaxis()
        axes[idx].set_title(f.replace(".jsonl", "") + " Emoji Distribution")
        axes[idx].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(output_dir + "/emoji_distribution.png", dpi=150)
    plt.close()


def _add_bar_group(ax, lefts, heights, width, color, label):
    """用单个 PolyCollection 批量绘制一组等宽柱体（几何与 `ax.bar` 相同）。

    5000 个样本 × 2 组的逐根 Rectangle 构建是绘图的主要耗时来源（数万个
    artist 的 Python 开销）；PolyCollection 在 C 层循环绘制完全相同的多边形，
    像素输出与 `ax.bar` 逐字节一致。sticky y=0 与 autolim 保证坐标范围一致。
    """
    verts = [
        ((left, 0.0), (left, h), (left + width, h), (left + width, 0.0))
        for left, h in zip(lefts, heights) if h
    ]
    coll = PolyCollection(verts, facecolors=color, edgecolors="none", label=label)
    coll.sticky_edges.y.append(0)
    ax.add_collection(coll, autolim=bool(verts))
    return coll


def plot_sentiment_comparison(sentiments, files, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (f, sent_data) in enumerate(zip(files, sentiments)):
        pos, neg, net = zip(*sent_data)

        x = np.arange(len(pos))
        width = 0.35

        _add_bar_group(axes[idx], x - width / 2, pos, width, "#2ecc71", "Positive")
        _add_bar_group(axes[idx], x + width / 2, neg, width, "#e74c3c", "Negative")
        axes[idx].autoscale_view()
        axes[idx].set_title(f.replace(".jsonl", "") + " Sentiment Analysis")
        axes[idx].set_xlabel("Sample Index")
        axes[idx].set_ylabel("Word Count")
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(output_dir + "/sentiment_comparison.png", dpi=150)
    plt.close()


def plot_time_distribution(timestamps, files, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (f, ts_list) in enumerate(zip(files, timestamps)):
        valid_ts = [t for t in ts_list if t]
        if not valid_ts:
            continue

        ts_arr = np.array(valid_ts)
        if ts_arr.max() > 1e10:
            ts_arr = ts_arr / 1000
        dates = [pd.Timestamp(ts, unit="s") for ts in ts_arr]

        try:
            idx_dt = DatetimeIndex(dates)
            axes[idx].hist(idx_dt.hour, bins=24, alpha=0.7, color="steelblue")
            axes[idx].set_title(f.replace(".jsonl", "") + " Post Hour Distribution")
            axes[idx].set_xlabel("Hour of Day")
            axes[idx].set_ylabel("Count")
        except:
            pass

    plt.tight_layout()
    plt.savefig(output_dir + "/time_distribution.png", dpi=150)
    plt.close()


def plot_punctuation_heatmap(punc_counts, files, output_dir):
    all_punct = set()
    for counts in punc_counts:
        all_punct.update(counts.keys())

    top_punct = sorted(
        all_punct, key=lambda x: sum(c.get(x, 0) for c in punc_counts), reverse=True
    )[:15]

    matrix = np.zeros((3, len(top_punct)))
    for i, counts in enumerate(punc_counts):
        for j, p in enumerate(top_punct):
            matrix[i, j] = counts.get(p, 0)

    plt.figure(figsize=(14, 6))
    plt.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.xticks(range(len(top_punct)), top_punct)
    plt.yticks(range(3), [f.replace(".jsonl", "") for f in files])
    plt.title("Punctuation Usage Heatmap")
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(output_dir + "/punctuation_heatmap.png", dpi=150)
    plt.close()


def plot_sentiment_box(sentiments, files, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (f, sent_data) in enumerate(zip(files, sentiments)):
        net_scores = [s[2] for s in sent_data]

        axes[idx].boxplot([net_scores])
        # 兼容 matplotlib <3.9：旧版本 boxplot 无 tick_labels 参数
        axes[idx].set_xticklabels([f.replace(".jsonl", "")])
        axes[idx].set_title("Net Sentiment Score Distribution")
        axes[idx].set_ylabel("Score (Positive - Negative)")

    plt.tight_layout()
    plt.savefig(output_dir + "/sentiment_box.png", dpi=150)
    plt.close()


def main():
    files = [
        "training_set_human.jsonl",
        "training_set_aigc.jsonl",
        "exploring_set.jsonl",
    ]
    file_names = [f.replace(".jsonl", "") for f in files]
    output_dir = "."

    print("Loading data...")
    texts_data = {}
    for f in files:
        texts_data[f], total = load_jsonl(f)
        print(f"  {f}: {total} records")
        if total > 5000:
            print(f"  -> Using a reproducible random sample of 5000 records")

    print("\n=== 1. Emoji Analysis ===")
    emoji_counts = []
    emoji_face_counts = []
    punc_counts = []
    sentiments = []
    timestamps = []

    for f in files:
        all_emojis = []
        face_counts = []
        all_punct = defaultdict(int)
        pos_scores = []
        neg_scores = []
        net_scores = []
        ts_list = []

        for item in texts_data[f]:
            text = item.get("desc", "") or item.get("note_content", "")
            all_emojis.extend(extract_emojis(text))

            face, count = extract_emoji_face(text)
            if face:
                face_counts.append(count)

            for p, c in extract_punctuation(text).items():
                all_punct[p] += c

            pos, neg, net = sentiment_analysis(text)
            pos_scores.append(pos)
            neg_scores.append(neg)
            net_scores.append(net)

            ts = get_time_features(item)
            if ts:
                ts_list.append(ts)

        emoji_counts.append(Counter(all_emojis))
        emoji_face_counts.append(face_counts)
        punc_counts.append(dict(all_punct))
        sentiments.append(list(zip(pos_scores, neg_scores, net_scores)))
        timestamps.append(ts_list)

    for idx, f in enumerate(files):
        counts = emoji_counts[idx]
        print(f"\n{f} Emoji Stats:")
        print(f"  Total emojis: {sum(counts.values())}")
        print(f"  Unique emojis: {len(counts)}")
        if counts:
            top5 = counts.most_common(5)
            print(f"  Top 5: {top5}")

    plot_emoji_distribution(emoji_counts, files, output_dir)
    print("\nGenerated: emoji_distribution.png")

    print("\n=== 2. Punctuation Analysis ===")
    for f, all_punct in zip(files, punc_counts):
        total = sum(all_punct.values())
        print(f"\n{f} Punctuation Stats:")
        print(f"  Total punctuation marks: {total}")
        top5 = sorted(all_punct.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5: {top5}")

    plot_punctuation_heatmap(punc_counts, files, output_dir)
    print("\nGenerated: punctuation_heatmap.png")

    print("\n=== 3. Sentiment Analysis ===")
    for f, sent_data in zip(files, sentiments):
        pos_scores = [s[0] for s in sent_data]
        print(f"\n{f} Sentiment Stats:")
        print(f"  Mean positive: {np.mean(pos_scores):.2f}")
        print(f"  Mean negative: {np.mean([s[1] for s in sent_data]):.2f}")
        print(f"  Mean net: {np.mean([s[2] for s in sent_data]):.2f}")
        print(
            f"  Positive ratio: {sum(1 for x in pos_scores if x > 0) / len(pos_scores):.2%}"
        )

    plot_sentiment_comparison(sentiments, files, output_dir)
    plot_sentiment_box(sentiments, files, output_dir)
    print("\nGenerated: sentiment_comparison.png, sentiment_box.png")

    print("\n=== 4. Time Series Analysis ===")
    for f, ts_list in zip(files, timestamps):
        if ts_list:
            ts_arr = np.array(ts_list)
            if ts_arr.max() > 1e10:
                ts_arr = ts_arr / 1000
            print(f"\n{f} Time Stats:")
            print(f"  Samples with timestamp: {len(ts_list)}")
            print(f"  Earliest: {pd.Timestamp(ts_arr.min(), unit='s')}")
            print(f"  Latest: {pd.Timestamp(ts_arr.max(), unit='s')}")

    try:
        plot_time_distribution(timestamps, files, output_dir)
        print("\nGenerated: time_distribution.png")
    except Exception as e:
        print(f"\n(Skipping time distribution: {e})")

    print("\n=== 5. Statistical Tests ===")
    print("\nChi-Square Tests (Emoji Usage):")
    for i in range(3):
        for j in range(i + 1, 3):
            chi2, p_value = compute_chi_square(emoji_counts[i], emoji_counts[j])
            print(
                f"  {file_names[i]} vs {file_names[j]}: chi2={chi2:.2f}, p={p_value:.4f}"
            )

    print("\nT-Tests (Sentiment Scores):")
    for i in range(3):
        for j in range(i + 1, 3):
            net1 = [s[2] for s in sentiments[i]]
            net2 = [s[2] for s in sentiments[j]]
            t_stat, p_value = compute_t_test(net1, net2)
            print(
                f"  {file_names[i]} vs {file_names[j]}: t={t_stat:.2f}, p={p_value:.4f}"
            )

    print("\n=== Analysis Complete ===")
    print("Generated visualizations:")
    print("- emoji_distribution.png")
    print("- punctuation_heatmap.png")
    print("- sentiment_comparison.png")
    print("- sentiment_box.png")
    try:
        print("- time_distribution.png")
    except:
        pass


if __name__ == "__main__":
    main()
