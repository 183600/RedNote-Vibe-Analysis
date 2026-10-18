#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
import re
import codecs
from collections import Counter

import jieba

from analysis_utils import (iter_jsonl_codecs, load_records_with_total,
                            make_pool, close_pool, split_sentences)


SAMPLE_LIMIT = 5000

jieba.setLogLevel(60)
TOKEN_KEEP = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def words_of(text):
    return [t for t in (tok.strip().lower() for tok in jieba.lcut(str(text)))
            if t and TOKEN_KEEP.search(t)]

RE_PRICE = re.compile(
    r"[¥￥$]\s*\d+|\d+\.?\d*\s*(?:元|块|万元|美元|美金|欧元|日元)"
)
RE_QUANTITY = re.compile(
    r"\d+\s*(?:个|件|次|只|天|年|月|周|小时|分钟|杯|份|款|张|条|篇|本|套|位|人|岁|遍|组|双|对|袋|盒|瓶|支|斤|公斤|克|米|公里)"
)
RE_LIST_NUM = re.compile(r"^\d+[.\s]", re.MULTILINE)
RE_BULLET = re.compile(r"[-+*]\s")
RE_LIST_ITEMS = re.compile(r"^\d+[.\s]|[-+*]\s", re.MULTILINE)


def load_jsonl(filepath, limit=SAMPLE_LIMIT, seed=42):
    return load_records_with_total(filepath, sample=limit, seed=seed,
                                   iterator=iter_jsonl_codecs)


def analyze_structure(text):
    if not text:
        return {}
    lines = text.split("\n")
    paragraphs = [p.strip() for p in lines if p.strip()]
    text_str = str(text)
    has_list = bool(RE_LIST_NUM.search(text_str))
    has_bullet = bool(RE_BULLET.search(text_str))
    list_items = RE_LIST_ITEMS.findall(text_str)
    return {
        "num_paragraphs": len(paragraphs),
        "has_list": has_list,
        "has_bullet": has_bullet,
        "num_list_items": len(list_items),
    }


def analyze_urls(text):
    if not text:
        return {}
    text = str(text)
    urls = re.findall(r"http[s]?\://[^\s]+", text)
    domains = []
    for url in urls:
        match = re.search(r"://([^/]+)", url)
        if match:
            domains.append(match.group(1).lower())
    return {"num_urls": len(urls), "domains": Counter(domains)}


def analyze_numbers(text):
    if not text:
        return {}
    text = str(text)
    numbers = re.findall(r"\d+\.?\d*", text)
    has_price = bool(RE_PRICE.search(text))
    has_count = bool(RE_QUANTITY.search(text))
    return {"num_numbers": len(numbers), "has_price": has_price, "has_count": has_count}


def analyze_sentences(text):
    if not text:
        return {}
    sentences = split_sentences(text)
    if not sentences:
        return {}
    words_list = [words_of(s) for s in sentences]
    total_words = sum(len(w) for w in words_list)
    unique_words = set()
    all_words = []
    for words in words_list:
        unique_words.update(words)
        all_words.extend(words)
    vocab_richness = len(unique_words) / total_words if total_words else 0
    return {
        "num_sentences": len(sentences),
        "total_words": total_words,
        "avg_words": round(total_words / len(sentences), 2),
        "vocab_richness": round(vocab_richness, 4),
        "words": all_words,
    }


def _repeat_ratio(words):
    if len(words) < 3:
        return 0
    repeat_count = sum(
        1
        for i in range(len(words) - 1)
        if words[i] == words[i + 1]
        or (i + 2 < len(words) and words[i] == words[i + 2])
    )
    return round(repeat_count / (len(words) - 1), 4)


def analyze_repetition(text):
    if not text:
        return {}
    return {"repeat_ratio": _repeat_ratio(words_of(text))}


def _aggregate_texts(texts):
    """逐条文本累计 URL/数字/结构/句子/重复度统计（一次遍历完成全部维度）。

    与原先每条文本顺序调用 analyze_* 的口径完全一致；返回可跨进程传递的
    汇总元组，供串行与分块并行共用。
    """
    url_total = 0
    url_domains = Counter()
    num_total = price = count = 0
    paras = list_n = bullet_n = 0
    sent_stats = []
    repeat_ratios = []
    for text in texts:
        if not text:
            continue

        u = analyze_urls(text)
        url_total += u["num_urls"]
        url_domains.update(u["domains"])

        n = analyze_numbers(text)
        num_total += n["num_numbers"]
        if n["has_price"]:
            price += 1
        if n["has_count"]:
            count += 1

        s = analyze_structure(text)
        paras += s["num_paragraphs"]
        if s["has_list"]:
            list_n += 1
        if s["has_bullet"]:
            bullet_n += 1

        sent = analyze_sentences(text)
        if sent:
            # 整段文本的分词结果 = 各句子分词结果的拼接（jieba 以句子边界标点
            # 为跳过分隔），因此重复度直接复用该结果，避免第二次分词。
            words = sent.pop("words")
            sent_stats.append(sent)
            repeat_ratios.append(_repeat_ratio(words))
        else:
            rep = analyze_repetition(text)
            if rep:
                repeat_ratios.append(rep["repeat_ratio"])
    return (url_total, url_domains, num_total, price, count,
            paras, list_n, bullet_n, sent_stats, repeat_ratios)


def _merge_aggregates(parts):
    """按块序合并 _aggregate_texts 的结果（列表拼接顺序 = 原记录顺序）。"""
    url_total = 0
    url_domains = Counter()
    num_total = price = count = 0
    paras = list_n = bullet_n = 0
    sent_stats = []
    repeat_ratios = []
    for (ut, ud, nt, pr, ct, pa, li, bu, ss, rr) in parts:
        url_total += ut
        url_domains.update(ud)
        num_total += nt
        price += pr
        count += ct
        paras += pa
        list_n += li
        bullet_n += bu
        sent_stats.extend(ss)
        repeat_ratios.extend(rr)
    return (url_total, url_domains, num_total, price, count,
            paras, list_n, bullet_n, sent_stats, repeat_ratios)


def analyze_texts(texts, pool=None, chunk_size=64):
    """对文本列表做全维度累计；pool 非空时分块并行，结果与串行完全一致。"""
    if pool is None or len(texts) <= chunk_size:
        agg = _aggregate_texts(texts)
    else:
        chunks = [texts[i:i + chunk_size]
                  for i in range(0, len(texts), chunk_size)]
        agg = _merge_aggregates(pool.map(_aggregate_texts, chunks))
    (url_total, url_domains, num_total, price, count,
     paras, list_n, bullet_n, sent_stats, repeat_ratios) = agg
    return {
        "url_total": url_total,
        "url_domains": url_domains,
        "num_total": num_total,
        "num_price": price,
        "num_count": count,
        "paras": paras,
        "list": list_n,
        "bullet": bullet_n,
        "sent_stats": sent_stats,
        "repeat_ratios": repeat_ratios,
    }


def main():
    files = [
        "training_set_human.jsonl",
        "training_set_aigc.jsonl",
        "exploring_set.jsonl",
    ]
    print("=" * 70)
    print("NEW ANALYSIS: Structure, URLs, Numbers, Sentence Patterns")
    print("=" * 70)

    # 逐条文本的分析彼此独立（jieba 分词占主要开销），按块并行；合并顺序与
    # 串行逐条累计完全一致（拼接列表按块序、计数按块序相加），统计结果不变。
    pool = make_pool(preload=jieba.initialize)
    try:
        for f in files:
            data, n_records = load_jsonl(f)
            print("\n" + "=" * 50)
            print("Dataset: " + f + " (" + str(n_records) + " records)")
            print("=" * 50)

            sample = data

            content = analyze_texts(
                [item.get("desc", "") or item.get("note_content", "")
                 for item in sample], pool)
            url_stats = {"total": content["url_total"],
                         "domains": content["url_domains"]}
            num_stats = {"total": content["num_total"],
                         "price": content["num_price"],
                         "count": content["num_count"]}
            struct_stats = {"paras": content["paras"], "list": content["list"],
                            "bullet": content["bullet"]}
            sent_stats = content["sent_stats"]
            repeat_ratios = content["repeat_ratios"]

            print("\n" + "=" * 50)
            print("Dataset: " + f + " - Content Analysis")
            print("=" * 50)

            print("\n1. URL/LINK ANALYSIS (Content):")
            print("   Total URLs: " + str(url_stats["total"]))
            top_domains = url_stats["domains"].most_common(5)
            if top_domains:
                print("   Top domains:")
                for d, c in top_domains:
                    print("      " + d + ": " + str(c))

            print("\n2. NUMBER USAGE (Content):")
            print("   Numbers found: " + str(num_stats["total"]))
            print("   With price info: " + str(num_stats["price"]))
            print("   With quantities: " + str(num_stats["count"]))

            print("\n3. STRUCTURE (Content):")
            print(
                "   Avg paragraphs: " + str(round(struct_stats["paras"] / len(sample), 2))
            )
            print(
                "   Has numbered lists: "
                + str(struct_stats["list"])
                + " ("
                + str(round(100 * struct_stats["list"] / len(sample), 1))
                + "%)"
            )
            print(
                "   Has bullet points: "
                + str(struct_stats["bullet"])
                + " ("
                + str(round(100 * struct_stats["bullet"] / len(sample), 1))
                + "%)"
            )

            print("\n4. SENTENCE PATTERNS (Content):")
            if sent_stats:
                total_sents = sum(s["num_sentences"] for s in sent_stats)
                total_words = sum(s["total_words"] for s in sent_stats)
                avg_words = total_words / total_sents if total_sents else 0.0
                avg_vocab = sum(s["vocab_richness"] for s in sent_stats) / len(sent_stats)
                print("   Avg words/sentence (pooled): " + str(round(avg_words, 2)))
                print("   Vocab richness (mean per doc): " + str(round(avg_vocab, 4)))

            if repeat_ratios:
                avg_repeat = sum(repeat_ratios) / len(repeat_ratios)
                print("\n5. WORD REPETITION (Content):")
                print("   Avg repeat ratio: " + str(avg_repeat))

            title = analyze_texts(
                [item.get("note_title", "") for item in sample], pool)
            url_stats_title = {"total": title["url_total"],
                               "domains": title["url_domains"]}
            num_stats_title = {"total": title["num_total"],
                               "price": title["num_price"],
                               "count": title["num_count"]}
            struct_stats_title = {"paras": title["paras"],
                                  "list": title["list"],
                                  "bullet": title["bullet"]}
            sent_stats_title = title["sent_stats"]
            repeat_ratios_title = title["repeat_ratios"]

            print("\n" + "=" * 50)
            print("Dataset: " + f + " - Title Analysis")
            print("=" * 50)

            print("\n1. URL/LINK ANALYSIS (Title):")
            print("   Total URLs: " + str(url_stats_title["total"]))
            top_domains_title = url_stats_title["domains"].most_common(5)
            if top_domains_title:
                print("   Top domains:")
                for d, c in top_domains_title:
                    print("      " + d + ": " + str(c))

            print("\n2. NUMBER USAGE (Title):")
            print("   Numbers found: " + str(num_stats_title["total"]))
            print("   With price info: " + str(num_stats_title["price"]))
            print("   With quantities: " + str(num_stats_title["count"]))

            print("\n3. STRUCTURE (Title):")
            print(
                "   Avg paragraphs: "
                + str(round(struct_stats_title["paras"] / len(sample), 2))
            )
            print(
                "   Has numbered lists: "
                + str(struct_stats_title["list"])
                + " ("
                + str(round(100 * struct_stats_title["list"] / len(sample), 1))
                + "%)"
            )
            print(
                "   Has bullet points: "
                + str(struct_stats_title["bullet"])
                + " ("
                + str(round(100 * struct_stats_title["bullet"] / len(sample), 1))
                + "%)"
            )

            print("\n4. SENTENCE PATTERNS (Title):")
            if sent_stats_title:
                total_sents = sum(s["num_sentences"] for s in sent_stats_title)
                total_words = sum(s["total_words"] for s in sent_stats_title)
                avg_words = total_words / total_sents if total_sents else 0.0
                avg_vocab = sum(s["vocab_richness"] for s in sent_stats_title) / len(
                    sent_stats_title
                )
                print("   Avg words/sentence (pooled): " + str(round(avg_words, 2)))
                print("   Vocab richness (mean per doc): " + str(round(avg_vocab, 4)))

            if repeat_ratios_title:
                avg_repeat = sum(repeat_ratios_title) / len(repeat_ratios_title)
                print("\n5. WORD REPETITION (Title):")
                print("   Avg repeat ratio: " + str(avg_repeat))
    finally:
        close_pool(pool)

    print("\n" + "=" * 70)
    print("Analysis Complete")


if __name__ == "__main__":
    main()
