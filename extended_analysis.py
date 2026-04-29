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
import warnings
import os
import pandas as pd
from pandas import Timestamp, DatetimeIndex

font_dir = "/usr/share/fonts/noto-cjk"
for f in os.listdir(font_dir):
    if "NotoSansCJK" in f and f.endswith(".ttc"):
        font_path = os.path.join(font_dir, f)
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.sans-serif"] = [prop.get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
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
    punctuation = re.findall(r"([!?。.,，:\"\'\-\(\)【】{}《》])", str(text))
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
            if ts > 1e10:
                return ts
            return ts * 1000
        return float(ts)
    except:
        return None


def get_time_features(item):
    ts = item.get("timestamp") or item.get("create_time") or item.get("time")
    if ts:
        timestamp = parse_timestamp(ts)
        if timestamp:
            return timestamp
    return None


def compute_chi_square(counts1, counts2, total1, total2):
    try:
        observed = np.array([counts1, counts2])
        expected = np.array([total1 * 0.5, total2 * 0.5])
        chi2, p_value = stats.chisquare(observed, expected)
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


def plot_sentiment_comparison(sentiments, files, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (f, sent_data) in enumerate(zip(files, sentiments)):
        pos, neg, net = zip(*sent_data)

        x = np.arange(len(pos))
        width = 0.35

        axes[idx].bar(x - width / 2, pos, width, label="Positive", color="#2ecc71")
        axes[idx].bar(x + width / 2, neg, width, label="Negative", color="#e74c3c")
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

        axes[idx].boxplot([net_scores], labels=[f.replace(".jsonl", "")])
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
        texts_data[f] = load_jsonl(f)
        print(f"  {f}: {len(texts_data[f])} records")
        if len(texts_data[f]) > 5000:
            texts_data[f] = texts_data[f][:5000]
            print(f"  -> Using first 5000 records for efficiency")

    print("\n=== 1. Emoji Analysis ===")
    emoji_counts = []
    emoji_face_counts = []

    for f in files:
        all_emojis = []
        face_counts = []
        for item in texts_data[f]:
            text = item.get("desc", "") or item.get("note_content", "")
            all_emojis.extend(extract_emojis(text))

            face, count = extract_emoji_face(text)
            if face:
                face_counts.append(count)

        emoji_counts.append(Counter(all_emojis))
        emoji_face_counts.append(face_counts)

        print(f"\n{f} Emoji Stats:")
        print(f"  Total emojis: {len(all_emojis)}")
        print(f"  Unique emojis: {len(emoji_counts[-1])}")
        if emoji_counts[-1]:
            top5 = emoji_counts[-1].most_common(5)
            print(f"  Top 5: {top5}")

    plot_emoji_distribution(emoji_counts, files, output_dir)
    print("\nGenerated: emoji_distribution.png")

    print("\n=== 2. Punctuation Analysis ===")
    punc_counts = []

    for f in files:
        all_punct = defaultdict(int)
        for item in texts_data[f]:
            text = item.get("desc", "") or item.get("note_content", "")
            punct_dict = extract_punctuation(text)
            for p, c in punct_dict.items():
                all_punct[p] += c

        punc_counts.append(dict(all_punct))
        total = sum(all_punct.values())
        print(f"\n{f} Punctuation Stats:")
        print(f"  Total punctuation marks: {total}")
        top5 = sorted(all_punct.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5: {top5}")

    plot_punctuation_heatmap(punc_counts, files, output_dir)
    print("\nGenerated: punctuation_heatmap.png")

    print("\n=== 3. Sentiment Analysis ===")
    sentiments = []

    for f in files:
        pos_scores = []
        neg_scores = []
        net_scores = []

        for item in texts_data[f]:
            text = item.get("desc", "") or item.get("note_content", "")
            pos, neg, net = sentiment_analysis(text)
            pos_scores.append(pos)
            neg_scores.append(neg)
            net_scores.append(net)

        sentiments.append(list(zip(pos_scores, neg_scores, net_scores)))

        print(f"\n{f} Sentiment Stats:")
        print(f"  Mean positive: {np.mean(pos_scores):.2f}")
        print(f"  Mean negative: {np.mean(neg_scores):.2f}")
        print(f"  Mean net: {np.mean(net_scores):.2f}")
        print(
            f"  Positive ratio: {sum(1 for x in pos_scores if x > 0) / len(pos_scores):.2%}"
        )

    plot_sentiment_comparison(sentiments, files, output_dir)
    plot_sentiment_box(sentiments, files, output_dir)
    print("\nGenerated: sentiment_comparison.png, sentiment_box.png")

    print("\n=== 4. Time Series Analysis ===")
    timestamps = []

    for f in files:
        ts_list = []
        for item in texts_data[f]:
            ts = get_time_features(item)
            if ts:
                ts_list.append(ts)
        timestamps.append(ts_list)

        if ts_list:
            valid_ts = [t for t in ts_list if t]
            if valid_ts:
                ts_arr = np.array(valid_ts)
                if ts_arr.max() > 1e10:
                    ts_arr = ts_arr / 1000
                print(f"\n{f} Time Stats:")
                print(f"  Samples with timestamp: {len(valid_ts)}")
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
            c1 = len(emoji_counts[i])
            c2 = len(emoji_counts[j])
            total1 = len(texts_data[files[i]])
            total2 = len(texts_data[files[j]])
            chi2, p_value = compute_chi_square([c1, c2], [total1, total2])
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
