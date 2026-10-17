# -*- coding: utf-8 -*-
"""第四步：可解释风格检测器 + 统计检验。

1. 仅用风格特征的逻辑回归（5 折分层交叉验证）：AUC / F1 / 准确率
2. 标准化系数 -> 特征重要性图
3. 每个模型家族的检测难度（human vs 单一家族的 5 折 AUC）
4. 关键特征 Mann-Whitney U 检验 + Cliff's delta 效应量
5. 训练全量最终模型并保存系数，供漂移分析使用
"""
import json
import os

# 单线程 BLAS：此类小矩阵下多线程同步开销大于收益（4 进程并行时避免超订）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from analysis_utils import (BASE, setup_chinese_fonts, cliffs_delta_from_u,
                            make_pool, close_pool)

FIGS = BASE + "paper_figs/"
os.makedirs(FIGS, exist_ok=True)
setup_chinese_fonts()

META = ["dataset", "domain", "month", "model_family", "model",
        "likes", "collections", "comments"]

# 家族 5 折 CV 的共享只读上下文（fork 前填充，worker 直接复用，避免重复传参）
_FAM_CTX: dict = {}


def _family_fit(fam: str) -> tuple[str, float]:
    """单个家族 vs 人类的 5 折 CV AUC（口径与串行主循环完全一致）。"""
    pipe = _FAM_CTX["pipe"]
    cv = _FAM_CTX["cv"]
    X_human = _FAM_CTX["X_human"]
    # 家族行号与 X_aigc 都是 fork 前算好的只读切片，避免每个 worker 重复
    # 做整表布尔筛选 + fillna。
    Xs = np.vstack([X_human, _FAM_CTX["X_aigc"][_FAM_CTX["fam_rows"][fam]]])
    ys = np.zeros(Xs.shape[0], dtype=int)
    ys[X_human.shape[0]:] = 1
    ps = cross_val_predict(pipe, Xs, ys, cv=cv, method="predict_proba")[:, 1]
    return fam, float(roc_auc_score(ys, ps))

LABELS = {
    "content_len": "正文长度", "n_lines": "行数", "avg_line_len": "平均行长",
    "n_hashtags": "话题标签数", "colon_line_ratio": "冒号行占比",
    "list_line_ratio": "列表行占比", "excl100": "感叹号/百字",
    "ques100": "问号/百字", "ellip100": "省略号/百字", "tilde100": "波浪号/百字",
    "comma100": "逗号/百字", "pause100": "顿号/百字", "period100": "句号/百字",
    "emoji100": "emoji/百字", "emoji_distinct": "emoji种类数",
    "xhsemo100": "小红书表情/百字",
    "p_的": "「的」", "p_了": "「了」", "p_是": "「是」", "p_我": "「我」",
    "p_你": "「你」", "p_就": "「就」", "p_都": "「都」", "p_很": "「很」",
    "p_有": "「有」", "p_不": "「不」",
    "ai_lex100": "AI味词/百字", "hum_lex100": "逻辑连接词/百字",
    "uniq_char_ratio": "独特字比率", "uniq_bigram_ratio": "独特二元组比率",
    "title_len": "标题长度", "title_excl": "标题含感叹号", "title_ques": "标题含问号",
    "title_emoji": "标题含emoji", "title_pipe": "标题含分隔符｜",
    "title_num": "标题含数字", "title_bracket": "标题含【】",
}

FAMILY_NAME = {"gpt": "GPT", "gemini": "Gemini", "deepseek": "DeepSeek",
               "anthropic": "Claude", "glm": "GLM", "qwen": "Qwen"}


def main():
    human = pd.read_csv(BASE + "paper_features_human.csv.gz")
    aigc = pd.read_csv(BASE + "paper_features_aigc.csv.gz")
    feats = [c for c in human.columns if c not in META]

    df = pd.concat([human, aigc], ignore_index=True)
    X = df[feats].fillna(0).values
    y = (df["dataset"] == "aigc").astype(int).values

    # ---- 1. 交叉验证 ----
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y, proba)
    f1 = f1_score(y, pred)
    acc = accuracy_score(y, pred)
    print(f"全特征 LR 5折CV: AUC={auc:.4f} F1={f1:.4f} ACC={acc:.4f}")

    # 消融：去掉与爬取格式可能相关的特征（标签数/标题类），检验稳健性
    robust_feats = [f for f in feats if f != "n_hashtags" and not f.startswith("title_")]
    Xr = df[robust_feats].fillna(0).values
    proba_r = cross_val_predict(pipe, Xr, y, cv=cv, method="predict_proba")[:, 1]
    auc_r = roc_auc_score(y, proba_r)
    f1_r = f1_score(y, (proba_r >= 0.5).astype(int))
    print(f"稳健特征 LR 5折CV: AUC={auc_r:.4f} F1={f1_r:.4f} (n_feats={len(robust_feats)})")

    # ---- 2. 全量模型系数 ----
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(scaler.transform(X), y)
    coefs = pd.Series(lr.coef_[0], index=feats).sort_values()

    fig, ax = plt.subplots(figsize=(9, 9))
    show = pd.concat([coefs.head(12), coefs.tail(12)])
    colors = ["#4C72B0" if v < 0 else "#C44E52" for v in show.values]
    ax.barh([LABELS.get(f, f) for f in show.index], show.values, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("标准化逻辑回归系数（正=AI 信号，负=人类信号）")
    ax.set_title(f"风格特征检测器的特征权重（5折CV AUC={auc:.3f}）")
    fig.tight_layout()
    fig.savefig(FIGS + "detector_coefficients.png", dpi=150)
    plt.close(fig)

    # ---- 3. 各家族检测难度（相互独立，多进程并行，结果与串行一致）----
    fam_auc = {}
    families = sorted(aigc["model_family"].unique())
    _FAM_CTX.update(X_aigc=X[len(human):],
                    fam_rows=aigc.groupby("model_family").indices,
                    pipe=pipe, cv=cv,
                    X_human=X[:len(human)])  # df 前段即人类集，直接切片复用
    pool = make_pool()
    try:
        rows = (pool.map(_family_fit, families) if pool is not None
                else [_family_fit(f) for f in families])
    finally:
        close_pool(pool)
        _FAM_CTX.clear()
    for fam, fam_val in rows:
        fam_auc[FAMILY_NAME[fam]] = round(fam_val, 4)
    print("各家族 vs 人类 检测AUC:", fam_auc)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    items = sorted(fam_auc.items(), key=lambda x: x[1])
    ax.bar([k for k, _ in items], [v for _, v in items], color="#55A868")
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("human vs 该家族 5折CV AUC")
    ax.set_title("各模型家族的风格检测难度（AUC 越低越难检测=越像人）")
    for i, (k, v) in enumerate(items):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS + "family_detection_auc.png", dpi=150)
    plt.close(fig)

    # ---- 4. 统计检验 ----
    key_feats = ["excl100", "emoji100", "xhsemo100", "ai_lex100", "hum_lex100",
                 "content_len", "p_的", "p_是", "p_我", "uniq_char_ratio",
                 "title_excl", "title_emoji", "n_hashtags", "tilde100",
                 "list_line_ratio", "colon_line_ratio"]
    tests = []
    h_idx = (df["dataset"] == "human").to_numpy()
    a_idx = ~h_idx
    for f in key_feats:
        col = df[f].to_numpy()
        a = col[a_idx]
        h = col[h_idx]
        u, p = mannwhitneyu(a, h, alternative="two-sided")
        delta = cliffs_delta_from_u(u, len(a), len(h))
        tests.append({
            "feature": f, "label": LABELS.get(f, f),
            "human_mean": round(float(h.mean()), 4),
            "aigc_mean": round(float(a.mean()), 4),
            "human_median": round(float(np.median(h)), 4),
            "aigc_median": round(float(np.median(a)), 4),
            "p_value": float(p), "cliffs_delta": round(float(delta), 3),
        })
        print(f"{LABELS.get(f,f):14s} H={h.mean():8.3f} A={a.mean():8.3f} "
              f"delta={delta:+.3f} p={p:.2e}")

    # ---- 5. 保存最终模型参数（供漂移分析复用）----
    model_dump = {
        "features": feats,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "robust_features": robust_feats,
    }
    # 稳健模型也保存
    scaler_r = StandardScaler().fit(Xr)
    lr_r = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr_r.fit(scaler_r.transform(Xr), y)
    model_dump["robust_scaler_mean"] = scaler_r.mean_.tolist()
    model_dump["robust_scaler_scale"] = scaler_r.scale_.tolist()
    model_dump["robust_coef"] = lr_r.coef_[0].tolist()
    model_dump["robust_intercept"] = float(lr_r.intercept_[0])

    out = {
        "cv_metrics": {"auc": round(auc, 4), "f1": round(f1, 4), "acc": round(acc, 4)},
        "cv_metrics_robust": {"auc": round(auc_r, 4), "f1": round(f1_r, 4)},
        "family_auc": fam_auc,
        "coefficients": {f: round(float(c), 4) for f, c in coefs.items()},
        "stat_tests": tests,
    }
    with open(BASE + "paper_detector_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(BASE + "paper_detector_model.json", "w", encoding="utf-8") as f:
        json.dump(model_dump, f)
    print("已保存 paper_detector_results.json / paper_detector_model.json")


if __name__ == "__main__":
    main()
