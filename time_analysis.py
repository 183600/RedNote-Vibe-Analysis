# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter
import warnings
import os
import pandas as pd

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
    ts = (
        item.get("timestamp")
        or item.get("create_time")
        or item.get("time")
        or item.get("local_time")
    )
    if ts:
        return parse_timestamp(ts)
    return None


files = ["training_set_human.jsonl", "training_set_aigc.jsonl", "exploring_set.jsonl"]

print("Loading data...")
texts_data = {}
for f in files:
    data = load_jsonl(f)
    print(f"  {f}: {len(data)} records")
    texts_data[f] = data[:5000]

print("\n=== Time Series Analysis ===")
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
            print(f"\n{f}:")
            print(f"  Samples with timestamp: {len(valid_ts)}")
            print(f"  Earliest: {pd.Timestamp(ts_arr.min(), unit='s')}")
            print(f"  Latest: {pd.Timestamp(ts_arr.max(), unit='s')}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (f, ts_list) in enumerate(zip(files, timestamps)):
    valid_ts = [t for t in ts_list if t]
    if not valid_ts:
        continue

    ts_arr = np.array(valid_ts)
    if ts_arr.max() > 1e10:
        ts_arr = ts_arr / 1000

    dates = pd.to_datetime(ts_arr, unit="s")
    hours = dates.hour

    axes[idx].hist(hours, bins=24, alpha=0.7, color="steelblue")
    axes[idx].set_title(f.replace(".jsonl", "") + " Post Hour Distribution")
    axes[idx].set_xlabel("Hour of Day")
    axes[idx].set_ylabel("Count")

plt.tight_layout()
plt.savefig("time_distribution.png", dpi=150)
print("\nGenerated: time_distribution.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (f, ts_list) in enumerate(zip(files, timestamps)):
    valid_ts = [t for t in ts_list if t]
    if not valid_ts:
        continue

    ts_arr = np.array(valid_ts)
    if ts_arr.max() > 1e10:
        ts_arr = ts_arr / 1000

    dates = pd.to_datetime(ts_arr, unit="s")

    axes[idx].hist(dates.month, bins=12, alpha=0.7, color="coral")
    axes[idx].set_title(f.replace(".jsonl", "") + " Post Month Distribution")
    axes[idx].set_xlabel("Month")
    axes[idx].set_ylabel("Count")

plt.tight_layout()
plt.savefig("month_distribution.png", dpi=150)
print("Generated: month_distribution.png")

print("\n=== Complete ===")
