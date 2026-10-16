# -*- coding: utf-8 -*-
"""拓展深潜分析：在既有论文管线之外，从「发布节奏、词表扩散、句子节奏、重复内容」
四个尚未覆盖的维度对 RedNote-Vibe 数据集做一次全量（非抽样）刻画。

分析方法与论文管线保持一致的口径：
- 正文清洗同样删除 `[话题]` 字面标记、剥离 `#话题` 标签（与 paper_features.py 一致）；
- 词表直接复用 paper_features.AI_WORDS / HUMAN_WORDS（log-odds 得到的风格词表）；
- 本地时间统一解析 'YYYYMMDDHH'（探索集有真实小时；人类集 local_time 的小时字段
  恒为占位值 08，因此小时节奏仅对探索集成立，星期与月份对两集均可用）；
- 差异检验统一使用 Mann-Whitney U + Cliff's delta（analysis_utils）。

数据完整性说明（由本脚本实测）：
- 探索集存在大量同一 note_id 的重复记录：其中绝大多数是一条带完整元数据、
  另一条缺 user_id/互动字段的「元数据变体副本」；发布节奏 / 词表扩散 / 句子节奏
  三项按 note_id 去重后统计（每篇笔记只计一次），重复内容一节保留原始逐记录口径。

输出：
  insights/publish_rhythm.png     发布节奏（探索集小时分布、星期对照、分小时互动）
  insights/lexicon_diffusion.png  风格词表的月度扩散轨迹
  insights/sentence_rhythm.png    句子节奏（句长分布与长度分层对照）
  insights/duplicate_content.png  重复笔记/重复内容结构
  deep_dive_results.json          全部数值结果
  deep_dive_results.md            结果摘要
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

from analysis_utils import (BASE, INSIGHTS, PALETTE, init_plot_style, save_fig,
                            cliffs_delta_from_u, engagement_of, iter_jsonl,
                            parse_local_time)
from paper_features import AI_WORDS, HUMAN_WORDS, HUATI, HASHTAG

DATASETS = {
    "human": BASE + "training_set_human_fixed.jsonl",
    "aigc": BASE + "training_set_aigc.jsonl",
    "exploring": BASE + "exploring_set_fixed.jsonl",
}
LABELS = {"human": "人类(前LLM)", "aigc": "AIGC(17模型)", "exploring": "探索集(后LLM)"}
SENT_SPLIT = re.compile(r"[。！？!?；;…]+|\n+")
WS = re.compile(r"\s+")
LENGTH_BUCKETS = [(0, 150, "<150字"), (150, 300, "150–300字"),
                  (300, 500, "300–500字"), (500, 800, "500–800字"),
                  (800, 10 ** 9, "≥800字")]
MIN_MONTH_CHARS = 5000


def clean_body(content: str) -> str:
    """与 paper_features.py 相同的正文清洗：去 [话题]、剥离 #标签、去首尾空白。"""
    body = HUATI.sub("", content or "")
    body = HASHTAG.sub(" ", body)
    return body.strip()


def content_of(rec: dict) -> str:
    """统一取正文：兼容 note_content / desc 两种字段（任一非空即可）。"""
    return rec.get("note_content") or rec.get("desc") or ""


def body_stats(body: str) -> tuple[int, int, int, float | None, int]:
    """返回（正文长度, AI词次数, 人类连接词次数, 平均句长, 句数）。"""
    sents = [s for s in (p.strip() for p in SENT_SPLIT.split(body)) if s]
    ai = sum(body.count(w) for w in AI_WORDS)
    hum = sum(body.count(w) for w in HUMAN_WORDS)
    if sents:
        lengths = [len(s) for s in sents]
        return len(body), ai, hum, sum(lengths) / len(lengths), len(sents)
    return len(body), ai, hum, None, 0


def collect(path: str) -> dict:
    """单次流式扫描，收集四个分析维度所需的全部统计量。

    重复内容按「原始逐记录」口径统计；发布节奏/词表/句子在扫描结束后按
    note_id 去重（无 note_id 的数据集如 AIGC 则每个记录视为一篇独立笔记）。
    """
    hours = Counter()
    weekdays = Counter()
    likes_by_hour = defaultdict(list)
    monthly = defaultdict(lambda: [0, 0, 0])  # [正文字符数, AI词次数, 人类连接词次数]
    sent_mean, sent_counts, sent_body_len = [], [], []
    content_seen, title_seen, id_seen = Counter(), Counter(), Counter()
    notes: dict = {}
    n = 0
    has_user_field = False
    missing_meta = 0

    def add_to_aggregates(rep: dict) -> None:
        if rep["hour"] is not None:
            hours[rep["hour"]] += 1
            weekdays[rep["weekday"]] += 1
            likes_by_hour[rep["hour"]].append(rep["likes"])
            slot = monthly[rep["month"]]
            slot[0] += rep["body_len"]
            slot[1] += rep["ai"]
            slot[2] += rep["hum"]
        if rep["sent_m"] is not None:
            sent_mean.append(rep["sent_m"])
            sent_counts.append(rep["sent_n"])
            sent_body_len.append(rep["body_len"])

    for rec in iter_jsonl(path):
        n += 1
        content = content_of(rec)
        body = clean_body(content)
        body_len, ai, hum, sent_m, sent_n = body_stats(body)
        raw = WS.sub("", content)
        if raw:
            content_seen[raw] += 1
        nt = WS.sub("", rec.get("note_title") or "")
        if nt:
            title_seen[nt] += 1

        likes, colls, comms = engagement_of(rec)
        dt = parse_local_time(rec.get("local_time"))
        if "user_id" in rec:
            has_user_field = True
            if not rec.get("user_id") or rec.get("domain") in (None, "", "未知"):
                missing_meta += 1
        has_meta = (bool(rec.get("user_id"))
                    and rec.get("domain") not in (None, "", "未知"))
        score = (int(has_meta) + int(bool(likes)) + int(bool(colls))
                 + int(bool(comms)) + int(dt is not None) + int(bool(raw)))
        rep = {
            "raw": raw, "body_len": body_len, "ai": ai, "hum": hum,
            "sent_m": sent_m, "sent_n": sent_n, "likes": likes, "score": score,
            "hour": dt.hour if dt else None,
            "weekday": dt.weekday() if dt else None,
            "month": dt.strftime("%Y%m") if dt else None,
        }
        nid = rec.get("note_id")
        if nid is None:
            add_to_aggregates(rep)
            continue
        id_seen[nid] += 1
        entry = notes.get(nid)
        if entry is None:
            rep.update({"copies": 1, "seen_meta": has_meta,
                        "seen_nometa": not has_meta, "raw_same": True})
            notes[nid] = rep
        else:
            entry["copies"] += 1
            entry["seen_meta"] = entry["seen_meta"] or has_meta
            entry["seen_nometa"] = entry["seen_nometa"] or (not has_meta)
            if raw != entry["raw"]:
                entry["raw_same"] = False
            if score > entry["score"]:
                keep = dict(rep)
                keep.update({"copies": entry["copies"],
                             "seen_meta": entry["seen_meta"],
                             "seen_nometa": entry["seen_nometa"],
                             "raw_same": entry["raw_same"]})
                notes[nid] = keep

    meta_variant_records = 0
    for nid, rep in notes.items():
        if (rep["copies"] > 1 and rep["raw_same"]
                and rep["seen_meta"] and rep["seen_nometa"]):
            meta_variant_records += rep["copies"]
        add_to_aggregates(rep)

    def dup_share(counter: Counter) -> tuple[int, int]:
        groups = sum(1 for c in counter.values() if c > 1)
        records = sum(c for c in counter.values() if c > 1)
        return records, groups

    content_dup, content_dup_groups = dup_share(content_seen)
    title_dup, title_dup_groups = dup_share(title_seen)
    note_content_seen = Counter(rep["raw"] for rep in notes.values() if rep["raw"])
    note_dup, note_dup_groups = dup_share(note_content_seen)
    return {
        "n": n,
        "unique_ids": len(id_seen),
        "dup_id_records": sum(c for c in id_seen.values() if c > 1),
        "meta_variant_records": meta_variant_records,
        "has_user_field": has_user_field,
        "missing_meta_records": missing_meta if has_user_field else None,
        "hours": dict(hours),
        "weekdays": dict(weekdays),
        "likes_by_hour": {int(h): v for h, v in likes_by_hour.items()},
        "monthly": {m: v for m, v in monthly.items()},
        "sent_mean": sent_mean,
        "sent_counts": sent_counts,
        "sent_body_len": sent_body_len,
        "content_dup_records": content_dup,
        "content_dup_groups": content_dup_groups,
        "content_unique": len(content_seen),
        "title_dup_records": title_dup,
        "title_dup_groups": title_dup_groups,
        "title_unique": len(title_seen),
        "title_top": title_seen.most_common(5),
        "note_content_dup_records": note_dup,
        "note_content_dup_groups": note_dup_groups,
        "n_notes": len(notes) if notes else n,
    }


def share_series(counter: Counter, total: int) -> np.ndarray:
    return np.array([counter.get(i, 0) / max(total, 1) * 100 for i in range(24)])


def median_by_hour(likes_by_hour: dict) -> np.ndarray:
    med = np.full(24, np.nan)
    for h in range(24):
        vals = likes_by_hour.get(h, [])
        if vals:
            med[h] = float(np.median(vals))
    return med


def plot_publish_rhythm(stats: dict, out: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    hours_labels = [f"{h:02d}" for h in range(24)]
    exp = stats["exploring"]
    n_exp = exp["n_notes"]

    ax = axes[0]
    hrs = share_series(Counter(exp["hours"]), n_exp)
    ax.bar(range(24), hrs, color=PALETTE["exploring"], alpha=0.85)
    peak = int(np.argmax(hrs))
    ax.annotate(f"峰值 {peak:02d}时\n{hrs[peak]:.1f}%", xy=(peak, hrs[peak]),
                xytext=(peak + 2.4, hrs[peak] * 0.98), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels(hours_labels[::3])
    ax.set_xlabel("发布小时（本地时间）")
    ax.set_ylabel("笔记占比 (%)")
    ax.set_title("探索集：发布小时分布")

    ax = axes[1]
    wd_labels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    x = np.arange(7)
    for i, name in enumerate(("human", "exploring")):
        s = stats[name]
        total = s["n_notes"]
        vals = [s["weekdays"].get(d, 0) / max(total, 1) * 100 for d in range(7)]
        ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=PALETTE[name],
               label=LABELS[name], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(wd_labels)
    ax.set_ylabel("笔记占比 (%)")
    ax.set_title("星期分布（人类集仅日期信息可用）")
    ax.legend()

    ax = axes[2]
    med = median_by_hour(exp["likes_by_hour"])
    ax.bar(range(24), med, color=PALETTE["exploring"], alpha=0.85)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels(hours_labels[::3])
    ax.set_xlabel("发布小时")
    ax.set_ylabel("点赞中位数")
    ax.set_title("探索集：分小时点赞中位数")
    if not np.all(np.isnan(med)):
        best = int(np.nanargmax(med))
        worst = int(np.nanargmin(med))
        ax.annotate(f"最高 {med[best]:.0f}", xy=(best, med[best]),
                    xytext=(best + 2.0, med[best] * 1.02), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.annotate(f"最低 {med[worst]:.0f}", xy=(worst, med[worst]),
                    xytext=(worst + 1.5, med[worst] + med[best] * 0.16),
                    fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8))

    fig.suptitle("发布节奏：小时 × 星期 × 互动（按 note_id 去重）", fontsize=13, y=1.02)
    save_fig(fig, INSIGHTS + "publish_rhythm.png")

    out["publish_rhythm"] = {}
    for name in ("human", "exploring"):
        s = stats[name]
        total = s["n_notes"]
        hrs = share_series(Counter(s["hours"]), total)
        weekend = float(sum(s["weekdays"].get(d, 0) for d in (5, 6))
                        / max(total, 1) * 100)
        entry = {
            "n_notes": total,
            "weekday_share_pct": [round(s["weekdays"].get(d, 0) / max(total, 1) * 100, 3)
                                  for d in range(7)],
            "weekend_share_pct": round(weekend, 3),
        }
        if name == "exploring":
            med = median_by_hour(s["likes_by_hour"])
            entry.update({
                "hour_share_pct": [round(float(v), 3) for v in hrs],
                "peak_hour": int(np.argmax(hrs)),
                "peak_hour_share_pct": round(float(hrs.max()), 3),
                "evening_18_23_share_pct": round(float(hrs[18:24].sum()), 3),
                "likes_median_by_hour": {str(h): round(float(med[h]), 1)
                                         for h in range(24) if not np.isnan(med[h])},
                "posts_by_hour": {str(h): s["hours"].get(h, 0) for h in range(24)},
            })
        else:
            entry["hour_field_note"] = "local_time 小时恒为占位值 08，小时分布不可用"
        out["publish_rhythm"][name] = entry
    rho_wd, p_wd = spearmanr(
        [stats["human"]["weekdays"].get(d, 0) for d in range(7)],
        [stats["exploring"]["weekdays"].get(d, 0) for d in range(7)])
    out["publish_rhythm"]["weekday_profile_spearman"] = {
        "rho": round(float(rho_wd), 3), "p": float(p_wd)}
    print(f"[发布节奏] exploring 峰值 {out['publish_rhythm']['exploring']['peak_hour']}时"
          f"（占 {out['publish_rhythm']['exploring']['peak_hour_share_pct']}%），"
          f"星期形态相关 rho={rho_wd:.2f}")


def plot_lexicon_diffusion(stats: dict, out: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6),
                             gridspec_kw={"width_ratios": [0.9, 1.25]})
    out["lexicon_diffusion"] = {}

    aigc_body = 0
    aigc_ai, aigc_hum = 0, 0
    for rec in iter_jsonl(DATASETS["aigc"]):
        body = clean_body(rec.get("note_content") or "")
        aigc_body += len(body)
        aigc_ai += sum(body.count(w) for w in AI_WORDS)
        aigc_hum += sum(body.count(w) for w in HUMAN_WORDS)
    ai_base = aigc_ai / max(aigc_body, 1) * 10000
    hum_base = aigc_hum / max(aigc_body, 1) * 10000
    out["lexicon_diffusion"]["aigc_reference_per10k"] = {
        "ai_lex": round(ai_base, 2), "human_lex": round(hum_base, 2),
        "chars": aigc_body}

    for i, name in enumerate(("human", "exploring")):
        s = stats[name]
        months = sorted(m for m, v in s["monthly"].items() if v[0] >= MIN_MONTH_CHARS)
        ai_rates = [s["monthly"][m][1] / s["monthly"][m][0] * 10000 for m in months]
        hum_rates = [s["monthly"][m][2] / s["monthly"][m][0] * 10000 for m in months]
        ax = axes[i]
        x = np.arange(len(months))
        ax.plot(x, ai_rates, "o-", ms=3.5, lw=1.5, color="#C44E52",
                label="AI 味词/万字符")
        ax.plot(x, hum_rates, "s-", ms=3.5, lw=1.5, color="#4C72B0",
                label="逻辑连接词/万字符")
        ax.axhline(ai_base, color="#C44E52", ls=":", lw=1, alpha=0.8)
        ax.axhline(hum_base, color="#4C72B0", ls=":", lw=1, alpha=0.8)
        if name == "human" and "202207" in months:
            split = months.index("202207")
            ax.axvspan(split - 0.5, len(months) - 0.5, color="#DD8452", alpha=0.10)
            ax.text(split + 0.2, ax.get_ylim()[1] * 0.95, "实时爬取窗口",
                    fontsize=8, color="#7f3b08")
        step = max(1, len(months) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([months[j] for j in range(0, len(months), step)],
                           rotation=45, fontsize=7.5)
        ax.set_ylabel("词频（每万字符）")
        ax.set_title(f"{LABELS[name]}：月度词表使用率")
        ax.legend(fontsize=8.5)

        if name == "human":
            trend_months = [m for m in months if m <= "202206"]
            key = "human_clean_window"
        else:
            trend_months = months
            key = "exploring"
        trend_rates = [s["monthly"][m][1] / s["monthly"][m][0] * 10000
                       for m in trend_months]
        rho, p = spearmanr(range(len(trend_months)), trend_rates)
        label = "干净窗口趋势" if name == "human" else "全窗口趋势"
        ax.text(0.02, 0.02, f"{label}: ρ={rho:.2f} (p={p:.2g})",
                transform=ax.transAxes, fontsize=9, color="#8c2d2d")
        out["lexicon_diffusion"][key] = {
            "months": trend_months,
            "ai_lex_per10k": [round(v, 2) for v in trend_rates],
            "human_lex_per10k": [
                round(s["monthly"][m][2] / s["monthly"][m][0] * 10000, 2)
                for m in trend_months],
            "ai_lex_trend": {"rho": round(float(rho), 3), "p": float(p)},
        }
        if name == "exploring":
            h1 = [v for m, v in zip(months, ai_rates) if m <= "202306"]
            h2 = [v for m, v in zip(months, ai_rates) if m >= "202501"]
            out["lexicon_diffusion"]["exploring_phase_means_ai_lex"] = {
                "2023H1": round(float(np.mean(h1)), 2),
                "2025H1": round(float(np.mean(h2)), 2)}
        print(f"[词表扩散] {name}: 月度AI味词趋势 rho={rho:.2f} (p={p:.2g}), "
              f"均值 {np.mean(trend_rates):.2f}/万字符")

    fig.suptitle("风格词表的月度扩散轨迹（虚线=受控 AIGC 参考水平；按 note_id 去重）",
                 fontsize=12.5, y=1.02)
    save_fig(fig, INSIGHTS + "lexicon_diffusion.png")


def plot_sentence_rhythm(stats: dict, out: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    ax = axes[0]
    bins = np.arange(0, 81, 3)
    for name in ("human", "aigc", "exploring"):
        vals = np.array(stats[name]["sent_mean"])
        ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.7,
                color=PALETTE[name], label=f"{LABELS[name]} (中位 {np.median(vals):.0f}字)")
    ax.set_xlabel("笔记平均句长（字）")
    ax.set_ylabel("密度")
    ax.set_title("平均句长分布")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    x = np.arange(len(LENGTH_BUCKETS))
    width = 0.26
    out["sentence_rhythm"] = {"by_dataset": {}}
    for i, name in enumerate(("human", "aigc", "exploring")):
        s = stats[name]
        meds = []
        for lo, hi, _ in LENGTH_BUCKETS:
            vals = [m for m, bl in zip(s["sent_mean"], s["sent_body_len"])
                    if lo <= bl < hi]
            meds.append(float(np.median(vals)) if vals else np.nan)
        ax.bar(x + (i - 1) * width, meds, width, color=PALETTE[name],
               label=LABELS[name], alpha=0.9)
        out["sentence_rhythm"]["by_dataset"][name] = {
            "n_notes_with_sentences": len(s["sent_mean"]),
            "mean_sentence_len_median": round(float(np.median(s["sent_mean"])), 2),
            "mean_sentence_len_mean": round(float(np.mean(s["sent_mean"])), 2),
            "sentences_per_note_median": round(float(np.median(s["sent_counts"])), 1),
            "median_mean_sentence_len_by_body_bucket": {
                b: (round(mv, 2) if not np.isnan(mv) else None)
                for (_, _, b), mv in zip(LENGTH_BUCKETS, meds)},
        }
    ax.set_xticks(x)
    ax.set_xticklabels([b for _, _, b in LENGTH_BUCKETS], fontsize=8.5)
    ax.set_xlabel("笔记正文长度分层")
    ax.set_ylabel("平均句长中位数（字）")
    ax.set_title("控制笔记长度后的句子节奏")
    ax.legend(fontsize=8.5)

    fig.suptitle("句子节奏：句长分布与长度分层对照（按 note_id 去重）", fontsize=13, y=1.02)
    save_fig(fig, INSIGHTS + "sentence_rhythm.png")

    for other in ("aigc", "exploring"):
        a = np.array(stats[other]["sent_mean"])
        h = np.array(stats["human"]["sent_mean"])
        u, p = mannwhitneyu(a, h, alternative="two-sided")
        out["sentence_rhythm"][f"human_vs_{other}"] = {
            "cliffs_delta": round(float(cliffs_delta_from_u(u, len(a), len(h))), 3),
            "p_value": float(p)}
    print("[句子节奏] 平均句长中位数: " + ", ".join(
        f"{name}={np.median(stats[name]['sent_mean']):.1f}"
        for name in ("human", "aigc", "exploring")))


def plot_duplicate_content(stats: dict, out: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    names = ("human", "aigc", "exploring")
    x = np.arange(len(names))
    content_share, title_share = [], []
    out["duplicates"] = {}
    for name in names:
        s = stats[name]
        cs = s["content_dup_records"] / s["n"] * 100
        ts = s["title_dup_records"] / s["n"] * 100
        content_share.append(cs)
        title_share.append(ts)
        out["duplicates"][name] = {
            "n": s["n"],
            "content_dup_records": s["content_dup_records"],
            "content_dup_groups": s["content_dup_groups"],
            "content_dup_share_pct": round(cs, 3),
            "content_unique_ratio": round(s["content_unique"] / s["n"], 4),
            "title_dup_records": s["title_dup_records"],
            "title_dup_groups": s["title_dup_groups"],
            "title_dup_share_pct": round(ts, 3),
            "title_unique_ratio": round(s["title_unique"] / s["n"], 4),
            "top_repeated_titles": [
                {"title": t[:60], "count": c} for t, c in s["title_top"]],
        }
        if s["unique_ids"]:
            out["duplicates"][name].update({
                "unique_note_ids": s["unique_ids"],
                "dup_note_id_records": s["dup_id_records"],
                "dup_note_id_share_pct": round(s["dup_id_records"] / s["n"] * 100, 3),
                "missing_metadata_records": s["missing_meta_records"],
                "missing_metadata_share_pct": (
                    round(s["missing_meta_records"] / s["n"] * 100, 3)
                    if s["missing_meta_records"] is not None else None),
                "note_level_content_dup_records": s["note_content_dup_records"],
                "note_level_content_dup_share_pct":
                    round(s["note_content_dup_records"] / s["n_notes"] * 100, 3),
            })

    ax = axes[0]
    ax.bar(x - 0.19, content_share, 0.38, color="#8172B3", label="正文重复记录占比")
    ax.bar(x + 0.19, title_share, 0.38, color="#CCB974", label="标题重复记录占比")
    for xi, (a, b) in enumerate(zip(content_share, title_share)):
        ax.text(xi - 0.19, a, f"{a:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.text(xi + 0.19, b, f"{b:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in names])
    ax.set_ylabel("占该子集记录比例 (%)")
    ax.set_title("完全重复内容/标题比例（逐记录，空白归一化）")
    ax.legend(fontsize=8.5)
    ax.set_ylim(0, max(content_share) * 1.18)

    ax = axes[1]
    s = stats["exploring"]
    uniq_first = s["n"] - s["dup_id_records"]
    other_dup = s["dup_id_records"] - s["meta_variant_records"]
    parts = [uniq_first, s["meta_variant_records"], other_dup]
    part_labels = ["唯一 note_id 记录", "元数据变体副本", "其余同 id 重复记录"]
    part_colors = ["#55A868", "#DD8452", "#C44E52"]
    left = 0.0
    for val, lab, col in zip(parts, part_labels, part_colors):
        ax.barh([0], [val / s["n"] * 100], left=left / s["n"] * 100,
                color=col, label=f"{lab}（{val:,}）")
        left += val
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xlabel("探索集记录构成 (%)")
    ax.set_title("探索集重复结构分解")
    ax.legend(fontsize=8.5, loc="lower right")

    save_fig(fig, INSIGHTS + "duplicate_content.png")
    print("[重复内容] 正文重复占比: " + ", ".join(
        f"{n}={v:.2f}%" for n, v in zip(names, content_share)))


def write_markdown(out: dict) -> None:
    pr = out["publish_rhythm"]
    ld = out["lexicon_diffusion"]
    sr = out["sentence_rhythm"]
    dup = out["duplicates"]
    lines = [
        "# 深潜分析结果（Deep Dive）",
        "",
        "> 由 `deep_dive_analysis.py` 全量运行生成；口径与论文管线一致，",
        "> 正文清洗/词表定义见 `paper_features.py`。发布节奏、词表扩散、句子节奏",
        "> 按 note_id 去重（每篇笔记一次）；重复内容一节保留逐记录口径。",
        "",
        "## 1 发布节奏",
        "",
    ]
    ex = pr["exploring"]
    lines += [
        f"- 探索集峰值发布时段 {ex['peak_hour']:02d} 时（占 {ex['peak_hour_share_pct']}%），"
        f"晚间 18–23 时占 {ex['evening_18_23_share_pct']}%，"
        f"周末占 {ex['weekend_share_pct']}%。",
        f"- 人类集：{pr['human']['hour_field_note']}；"
        f"周末占 {pr['human']['weekend_share_pct']}%。",
        f"- 两集星期分布形态相关 rho={pr['weekday_profile_spearman']['rho']}"
        f"（p={pr['weekday_profile_spearman']['p']:.2g}）。",
        "",
        "## 2 风格词表随时间的使用率",
        "",
        f"- 受控 AIGC 参考：AI 味词 {ld['aigc_reference_per10k']['ai_lex']}/万字符，"
        f"逻辑连接词 {ld['aigc_reference_per10k']['human_lex']}/万字符。",
        f"- 人类集干净窗口（≤2022.06）AI 味词趋势 "
        f"rho={ld['human_clean_window']['ai_lex_trend']['rho']}"
        f"（p={ld['human_clean_window']['ai_lex_trend']['p']:.2g}）。",
        f"- 探索集 AI 味词趋势 rho={ld['exploring']['ai_lex_trend']['rho']}"
        f"（p={ld['exploring']['ai_lex_trend']['p']:.2g}）；"
        f"2023H1 均值 {ld['exploring_phase_means_ai_lex']['2023H1']} → "
        f"2025H1 均值 {ld['exploring_phase_means_ai_lex']['2025H1']}（/万字符）。",
        "",
        "## 3 句子节奏",
        "",
        "| 子集 | 平均句长中位数 | 平均句长均值 | 每帖句数中位数 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("human", "aigc", "exploring"):
        d = sr["by_dataset"][name]
        lines.append(f"| {LABELS[name]} | {d['mean_sentence_len_median']} "
                     f"| {d['mean_sentence_len_mean']} "
                     f"| {d['sentences_per_note_median']} |")
    lines += [
        "",
        f"- human vs aigc：Cliff's δ={sr['human_vs_aigc']['cliffs_delta']}"
        f"（p={sr['human_vs_aigc']['p_value']:.2g}）；"
        f"human vs exploring：δ={sr['human_vs_exploring']['cliffs_delta']}"
        f"（p={sr['human_vs_exploring']['p_value']:.2g}）。",
        "",
        "## 4 重复内容与数据完整性",
        "",
        "| 子集 | 正文重复占比 | 标题重复占比 | 重复 note_id 占比 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("human", "aigc", "exploring"):
        d = dup[name]
        note_share = (f"{d['dup_note_id_share_pct']}%"
                      if d.get("dup_note_id_share_pct") is not None
                      else "—（无 note_id）")
        lines.append(f"| {LABELS[name]} | {d['content_dup_share_pct']}% "
                     f"| {d['title_dup_share_pct']}% | {note_share} |")
    exd = dup["exploring"]
    lines += [
        "",
        f"- 探索集 {exd['n']:,} 条记录对应 {exd['unique_note_ids']:,} 个唯一 note_id；"
        f"{exd['dup_note_id_records']:,} 条记录（{exd['dup_note_id_share_pct']}%）"
        f"的 note_id 出现多次，其中绝大多数是「同一笔记的一条完整元数据副本 + "
        f"一条缺 user_id/互动字段的副本」；去重到 note_id 后，正文完全重复的记录占比降至 "
        f"{exd['note_level_content_dup_share_pct']}%。",
        f"- 探索集 {exd['missing_metadata_share_pct']}% 的记录缺少 user_id "
        f"或 domain 为「未知」；该结构在 2024–2025 各月均存在。",
        "",
        "图表见 `insights/`：`publish_rhythm.png`、`lexicon_diffusion.png`、",
        "`sentence_rhythm.png`、`duplicate_content.png`。",
        "",
    ]
    with open(BASE + "deep_dive_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    init_plot_style()
    stats = {}
    for name, path in DATASETS.items():
        print(f"[扫描] {name}: {path}")
        stats[name] = collect(path)
        s = stats[name]
        print(f"        记录 {s['n']} 条 / 唯一 note_id {s['unique_ids'] or '—'}")

    out = {}
    plot_publish_rhythm(stats, out)
    plot_lexicon_diffusion(stats, out)
    plot_sentence_rhythm(stats, out)
    plot_duplicate_content(stats, out)

    with open(BASE + "deep_dive_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    write_markdown(out)
    print("已保存 insights/*.png, deep_dive_results.json, deep_dive_results.md")


if __name__ == "__main__":
    main()
