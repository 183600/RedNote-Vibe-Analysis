# -*- coding: utf-8 -*-
import json
import random
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter, defaultdict
import warnings
import os

from analysis_utils import init_plot_style, load_records_with_total
from paper_features import EMOJI

# Configure font for Chinese character support
init_plot_style()

warnings.filterwarnings("ignore")


SAMPLE_LIMIT = 5000

# 变体选择符（FE0E/FE0F）与零宽连接符（ZWJ）在 paper_features.EMOJI 里会被
# 单独匹配，造成同一 emoji 重复计数（口径与 symbol_analysis 一致，仅剔除它们）。
_EMOJI_SKIP = frozenset("\ufe0e\ufe0f\u200d")


def load_jsonl(filepath, limit=SAMPLE_LIMIT, seed=42):
    return load_records_with_total(filepath, sample=limit, seed=seed)


def extract_emojis(text):
    if not text:
        return []
    return [c for c in EMOJI.findall(str(text)) if c not in _EMOJI_SKIP]


def main():
    files = ["training_set_human.jsonl", "training_set_aigc.jsonl", "exploring_set.jsonl"]

    print("Loading data...")
    texts_data = {}
    for f in files:
        data, total = load_jsonl(f)
        print("  {}: {} records".format(f, total))
        texts_data[f] = data

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


if __name__ == "__main__":
    main()
