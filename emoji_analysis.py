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

# Configure font for Chinese character support
noto_fonts = sorted([f for f in font_manager.findSystemFonts() if "NotoSansCJK" in f])
if noto_fonts:
    font_path = noto_fonts[0]
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


def extract_emojis(text):
    if not text:
        return []
    emojis = EMOJI_PATTERN.findall(str(text))
    result = []
    for e in emojis:
        result.extend(list(e))
    return result


files = ["training_set_human.jsonl", "training_set_aigc.jsonl", "exploring_set.jsonl"]

print("Loading data...")
texts_data = {}
for f in files:
    data = load_jsonl(f)
    print("  {}: {} records".format(f, len(data)))
    texts_data[f] = data[:5000]

print("\n=== Emoji Analysis ===")
emoji_counts = []

for f in files:
    all_emojis = []
    for item in texts_data[f]:
        text = item.get("desc", "") or item.get("note_content", "")
        all_emojis.extend(extract_emojis(text))

    emoji_counts.append(Counter(all_emojis))

    print("\n{}:".format(f))
    print("  Total emojis: {}".format(len(all_emojis)))
    print("  Unique emojis: {}".format(len(emoji_counts[-1])))
    if emoji_counts[-1]:
        top5 = emoji_counts[-1].most_common(5)
        print("  Top 5: {}".format(top5))

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
plt.savefig("emoji_distribution.png", dpi=150)
print("\nGenerated: emoji_distribution.png")

print("\n=== Complete ===")
