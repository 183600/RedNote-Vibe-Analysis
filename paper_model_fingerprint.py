# -*- coding: utf-8 -*-
"""第三步：17 个 LLM 的风格指纹对比，找出最接近人类风格的模型。

输出：
- paper_figs/model_fingerprint_heatmap.png  模型 × 特征 z-score 热力图
- paper_figs/model_human_distance.png       与人类质心的风格距离排名
- paper_model_results.json                  数值结果
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_utils import BASE, setup_chinese_fonts

FIGS = BASE + "paper_figs/"
os.makedirs(FIGS, exist_ok=True)
setup_chinese_fonts()

META = ["dataset", "domain", "month", "model_family", "model",
        "likes", "collections", "comments"]

# 热力图与距离计算使用的核心风格特征（中文标签）
CORE_FEATS = {
    "content_len": "正文长度",
    "excl100": "感叹号/百字",
    "ques100": "问号/百字",
    "tilde100": "波浪号/百字",
    "ellip100": "省略号/百字",
    "emoji100": "emoji/百字",
    "xhsemo100": "小红书表情/百字",
    "n_hashtags": "话题标签数",
    "list_line_ratio": "列表行占比",
    "colon_line_ratio": "冒号行占比",
    "p_的": "「的」/百字",
    "p_了": "「了」/百字",
    "p_我": "「我」/百字",
    "p_是": "「是」/百字",
    "ai_lex100": "AI味词/百字",
    "hum_lex100": "连接词/百字",
    "uniq_char_ratio": "独特字比率",
    "title_len": "标题长度",
    "title_excl": "标题含感叹号",
    "title_emoji": "标题含emoji",
}

FAMILY_NAME = {"gpt": "GPT", "gemini": "Gemini", "deepseek": "DeepSeek",
               "anthropic": "Claude", "glm": "GLM", "qwen": "Qwen"}


def main():
    human = pd.read_csv(BASE + "paper_features_human.csv.gz")
    aigc = pd.read_csv(BASE + "paper_features_aigc.csv.gz")
    feats = list(CORE_FEATS.keys())

    h_feats = human[feats]  # 只切一次列，均值/标准差复用
    h_mean = h_feats.mean()
    h_std = h_feats.std().replace(0, 1)

    # ---- 每个模型的质心 z-score（以人类分布为参照）----
    rows, dists = {}, {}
    for model, g in aigc.groupby("model"):
        z = (g[feats].mean() - h_mean) / h_std
        rows[model] = z
        dists[model] = float(np.sqrt((z ** 2).mean()))  # RMS z-distance
    zdf = pd.DataFrame(rows).T

    # 家族级
    fam_dists = {}
    for fam, g in aigc.groupby("model_family"):
        z = (g[feats].mean() - h_mean) / h_std
        fam_dists[FAMILY_NAME.get(fam, fam)] = float(np.sqrt((z ** 2).mean()))

    order = sorted(dists, key=dists.get)

    # ---- 热力图 ----
    fig, ax = plt.subplots(figsize=(13, 8))
    data = zdf.loc[order, feats]
    im = ax.imshow(data.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([CORE_FEATS[f] for f in feats], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_title("各 LLM 风格特征相对人类基线的偏移（z 分数，红=高于人类，蓝=低于人类）", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="z 分数")
    fig.tight_layout()
    fig.savefig(FIGS + "model_fingerprint_heatmap.png", dpi=150)
    plt.close(fig)

    # ---- 距离条形图 ----
    fam_of = aigc.drop_duplicates("model").set_index("model")["model_family"].to_dict()
    colors = {"gpt": "#4C72B0", "gemini": "#DD8452", "deepseek": "#55A868",
              "anthropic": "#C44E52", "glm": "#8172B3", "qwen": "#937860"}
    fig, ax = plt.subplots(figsize=(9, 7))
    vals = [dists[m] for m in order]
    ax.barh(range(len(order)), vals,
            color=[colors[fam_of[m]] for m in order])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("与人类风格质心的 RMS z 距离（越小越像人）")
    ax.set_title("17 个 LLM 的“仿人程度”排名")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, [FAMILY_NAME[k] for k in colors], loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS + "model_human_distance.png", dpi=150)
    plt.close(fig)

    # ---- 数值输出 ----
    model_counts = aigc["model"].value_counts().to_dict()  # 一次分组计数
    out = {
        "model_distance_rank": [
            {"model": m, "family": FAMILY_NAME.get(fam_of[m], fam_of[m]),
             "rms_z": round(dists[m], 3), "n": int(model_counts.get(m, 0))}
            for m in order
        ],
        "family_distance": {k: round(v, 3) for k, v in
                            sorted(fam_dists.items(), key=lambda x: x[1])},
        "model_feature_z": {m: {f: round(float(zdf.loc[m, f]), 2) for f in feats}
                            for m in order},
        "human_baseline_mean": {f: round(float(h_mean[f]), 3) for f in feats},
        "aigc_overall_mean": {f: round(float(aigc[f].mean()), 3) for f in feats},
    }
    with open(BASE + "paper_model_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("=== 模型仿人程度排名（RMS z 距离，越小越像人）===")
    for item in out["model_distance_rank"]:
        print(f"{item['model']:22s} {item['family']:9s} {item['rms_z']:.3f}  n={item['n']}")
    print("\n=== 家族排名 ===")
    for k, v in out["family_distance"].items():
        print(f"{k:9s} {v:.3f}")
    print("\n=== 人类基线 vs AIGC 总体（核心特征均值）===")
    for f in feats:
        print(f"{CORE_FEATS[f]:12s} human={out['human_baseline_mean'][f]:9.3f}  aigc={out['aigc_overall_mean'][f]:9.3f}")


if __name__ == "__main__":
    main()
