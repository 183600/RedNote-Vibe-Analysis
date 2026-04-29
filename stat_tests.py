# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
from collections import Counter
import warnings
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

warnings.filterwarnings("ignore")


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


def sentiment_analysis(text):
    if not text:
        return 0, 0, 0
    text = str(text).lower()
    pos_count = sum(1 for word in POSITIVE_WORDS if word in text)
    neg_count = sum(1 for word in NEGATIVE_WORDS if word in text)
    return pos_count, neg_count, pos_count - neg_count


files = ["training_set_human.jsonl", "training_set_aigc.jsonl", "exploring_set.jsonl"]
file_names = [f.replace(".jsonl", "") for f in files]

print("Loading data...")
texts_data = {}
for f in files:
    data = load_jsonl(f)
    print(f"  {f}: {len(data)} records")
    texts_data[f] = data[:3000]

print("\n=== Statistical Tests ===")

sentiments = {}
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

    sentiments[f] = {
        "pos": pos_scores,
        "neg": neg_scores,
        "net": net_scores,
        "length": [
            len(item.get("desc", "") or item.get("note_content", ""))
            for item in texts_data[f]
        ],
    }

print("\n1. T-Tests (Net Sentiment):")
for i in range(3):
    for j in range(i + 1, 3):
        f1, f2 = files[i], files[j]
        net1 = sentiments[f1]["net"]
        net2 = sentiments[f2]["net"]
        t_stat, p_value = stats.ttest_ind(net1, net2)
        sig = (
            "***"
            if p_value < 0.001
            else "**"
            if p_value < 0.01
            else "*"
            if p_value < 0.05
            else ""
        )
        print(
            f"  {file_names[i]} vs {file_names[j]}: t={t_stat:.3f}, p={p_value:.4f} {sig}"
        )

print("\n2. T-Tests (Content Length):")
for i in range(3):
    for j in range(i + 1, 3):
        f1, f2 = files[i], files[j]
        len1 = sentiments[f1]["length"]
        len2 = sentiments[f2]["length"]
        t_stat, p_value = stats.ttest_ind(len1, len2)
        sig = (
            "***"
            if p_value < 0.001
            else "**"
            if p_value < 0.01
            else "*"
            if p_value < 0.05
            else ""
        )
        print(
            f"  {file_names[i]} vs {file_names[j]}: t={t_stat:.3f}, p={p_value:.4f} {sig}"
        )

print("\n3. Mann-Whitney U Tests (Positive Words):")
for i in range(3):
    for j in range(i + 1, 3):
        f1, f2 = files[i], files[j]
        pos1 = sentiments[f1]["pos"]
        pos2 = sentiments[f2]["pos"]
        u_stat, p_value = stats.mannwhitneyu(pos1, pos2, alternative="two-sided")
        sig = (
            "***"
            if p_value < 0.001
            else "**"
            if p_value < 0.01
            else "*"
            if p_value < 0.05
            else ""
        )
        print(
            f"  {file_names[i]} vs {file_names[j]}: U={u_stat:.1f}, p={p_value:.4f} {sig}"
        )

print("\n4. Chi-Square Tests (Sentiment Distribution):")
for i in range(3):
    for j in range(i + 1, 3):
        f1, f2 = files[i], files[j]
        pos1 = sum(1 for x in sentiments[f1]["pos"] if x > 0)
        pos2 = sum(1 for x in sentiments[f2]["pos"] if x > 0)
        neg1 = sum(1 for x in sentiments[f1]["neg"] if x > 0)
        neg2 = sum(1 for x in sentiments[f2]["neg"] if x > 0)

        observed = [[pos1, neg1], [pos2, neg2]]
        chi2, p_value = stats.chi2_contingency(observed)[:2]
        sig = (
            "***"
            if p_value < 0.001
            else "**"
            if p_value < 0.01
            else "*"
            if p_value < 0.05
            else ""
        )
        print(
            f"  {file_names[i]} vs {file_names[j]}: chi2={chi2:.3f}, p={p_value:.4f} {sig}"
        )

print("\n5. ANOVA (All Datasets):")
all_net = [sentiments[f]["net"] for f in files]
f_stat, p_value = stats.f_oneway(*all_net)
sig = (
    "***"
    if p_value < 0.001
    else "**"
    if p_value < 0.01
    else "*"
    if p_value < 0.05
    else ""
)
print(f"  Net Sentiment: F={f_stat:.3f}, p={p_value:.4f} {sig}")

all_len = [sentiments[f]["length"] for f in files]
f_stat, p_value = stats.f_oneway(*all_len)
sig = (
    "***"
    if p_value < 0.001
    else "**"
    if p_value < 0.01
    else "*"
    if p_value < 0.05
    else ""
)
print(f"  Content Length: F={f_stat:.3f}, p={p_value:.4f} {sig}")

print("\n6. Effect Sizes (Cohen's d):")


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx - 1) * np.std(x) ** 2 + (ny - 1) * np.std(y) ** 2) / dof
    )


for i in range(3):
    for j in range(i + 1, 3):
        f1, f2 = files[i], files[j]
        d = cohens_d(sentiments[f1]["net"], sentiments[f2]["net"])
        eff = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
        print(f"  {file_names[i]} vs {file_names[j]}: d={d:.3f} ({eff})")

print("\n=== Summary ===")
print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")

print("\n=== Complete ===")
