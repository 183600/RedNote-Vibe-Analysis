# -*- coding: utf-8 -*-
import json
import random
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter
import warnings
import os
import pandas as pd
from datetime import datetime, timezone

from analysis_utils import init_plot_style, load_records_with_total

init_plot_style()

warnings.filterwarnings("ignore")


SAMPLE_LIMIT = 5000


def load_jsonl(filepath, limit=SAMPLE_LIMIT, seed=42):
    return load_records_with_total(filepath, sample=limit, seed=seed)


def parse_timestamp(ts):
    if not ts:
        return None
    try:
        if isinstance(ts, (int, float)):
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
        return parse_timestamp(ts)
    return None


def main():
    files = ["training_set_human.jsonl", "training_set_aigc.jsonl", "exploring_set.jsonl"]

    print("Loading data...")
    texts_data = {}
    for f in files:
        data, total = load_jsonl(f)
        print(f"  {f}: {total} records")
        texts_data[f] = data

    print("\n=== Time Series Analysis ===")
    file_dates = []

    for f in files:
        ts_list = []
        for item in texts_data[f]:
            ts = get_time_features(item)
            if ts:
                ts_list.append(ts)
        valid_ts = ts_list  # ts_list 仅在真值时才追加，无需再复制过滤
        dates = None
        if valid_ts:
            ts_arr = np.array(valid_ts)
            if ts_arr.max() > 1e10:
                ts_arr = ts_arr / 1000
            print(f"\n{f}:")
            print(f"  Samples with timestamp: {len(valid_ts)}")
            print(f"  Earliest: {pd.Timestamp(ts_arr.min(), unit='s')}")
            print(f"  Latest: {pd.Timestamp(ts_arr.max(), unit='s')}")
            dates = pd.to_datetime(ts_arr, unit="s")
        file_dates.append(dates)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (f, dates) in enumerate(zip(files, file_dates)):
        axes[idx].set_title(f.replace(".jsonl", "") + " Post Hour Distribution")
        axes[idx].set_xlabel("Hour of Day")
        axes[idx].set_ylabel("Count")
        if dates is None:
            axes[idx].text(0.5, 0.5, "无时间字段", ha="center", va="center",
                           transform=axes[idx].transAxes, fontsize=12)
            continue

        hours = np.asarray(dates.hour)
        if np.unique(hours).size <= 1:
            axes[idx].text(
                0.5, 0.5,
                f"小时字段恒为占位值 {int(hours[0]):02d}，\n不反映真实发帖时间",
                ha="center", va="center", transform=axes[idx].transAxes,
                fontsize=12, color="#8c2d2d")
            continue
        axes[idx].hist(dates.hour, bins=24, alpha=0.7, color="steelblue")

    plt.tight_layout()
    plt.savefig("time_distribution.png", dpi=150)
    print("\nGenerated: time_distribution.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (f, dates) in enumerate(zip(files, file_dates)):
        axes[idx].set_title(f.replace(".jsonl", "") + " Monthly Post Distribution")
        axes[idx].set_xlabel("Month")
        axes[idx].set_ylabel("Count")
        if dates is None:
            axes[idx].text(0.5, 0.5, "无时间字段", ha="center", va="center",
                           transform=axes[idx].transAxes, fontsize=12)
            continue

        monthly = dates.to_period("M").value_counts().sort_index()
        axes[idx].plot(monthly.index.to_timestamp(), monthly.values, "o-",
                       color="coral")
        axes[idx].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for label in axes[idx].get_xticklabels():
            label.set_rotation(45)

    plt.tight_layout()
    plt.savefig("month_distribution.png", dpi=150)
    print("Generated: month_distribution.png")

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
