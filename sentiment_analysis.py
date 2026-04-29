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

print("Loading data...")
texts_data = {}
for f in files:
    data = load_jsonl(f)
    print(f"  {f}: {len(data)} records")
    texts_data[f] = data[:5000]

print("\n=== Sentiment Analysis ===")
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

    print(f"\n{f}:")
    print(f"  Mean positive: {np.mean(pos_scores):.2f}")
    print(f"  Mean negative: {np.mean(neg_scores):.2f}")
    print(f"  Mean net: {np.mean(net_scores):.2f}")
    print(
        f"  Positive ratio: {sum(1 for x in pos_scores if x > 0) / len(pos_scores):.2%}"
    )

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (f, sent_data) in enumerate(zip(files, sentiments)):
    pos, neg, net = zip(*sent_data)

    x = np.arange(len(pos))
    width = 0.35

    axes[idx].bar(
        x[:50] - width / 2, pos[:50], width, label="Positive", color="#2ecc71"
    )
    axes[idx].bar(
        x[:50] + width / 2, neg[:50], width, label="Negative", color="#e74c3c"
    )
    axes[idx].set_title(f.replace(".jsonl", ""))
    axes[idx].set_xlabel("Sample Index")
    axes[idx].set_ylabel("Word Count")
    axes[idx].legend()

plt.tight_layout()
plt.savefig("sentiment_comparison.png", dpi=150)
print("\nGenerated: sentiment_comparison.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (f, sent_data) in enumerate(zip(files, sentiments)):
    net_scores = [s[2] for s in sent_data]

    axes[idx].boxplot([net_scores], labels=[f.replace(".jsonl", "")])
    axes[idx].set_title("Net Sentiment Score Distribution")
    axes[idx].set_ylabel("Score (Positive - Negative)")

plt.tight_layout()
plt.savefig("sentiment_box.png", dpi=150)
print("Generated: sentiment_box.png")

print("\n=== Complete ===")
