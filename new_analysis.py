#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
import re
import codecs
from collections import Counter

from analysis_utils import iter_jsonl_codecs, load_records_with_total


SAMPLE_LIMIT = 5000

RE_PRICE = re.compile(
    r"[¥￥$]\s*\d+|\d+\.?\d*\s*(?:元|块|万元|美元|美金|欧元|日元)"
)
RE_QUANTITY = re.compile(
    r"\d+\s*(?:个|件|次|只|天|年|月|周|小时|分钟|杯|份|款|张|条|篇|本|套|位|人|岁|遍|组|双|对|袋|盒|瓶|支|斤|公斤|克|米|公里)"
)
RE_SENTENCE = re.compile(r"[。！？!?.]+")
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
    text = str(text)
    sentences = RE_SENTENCE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return {}
    words_list = [s.split() for s in sentences]
    total_words = sum(len(w) for w in words_list)
    unique_words = set()
    for words in words_list:
        unique_words.update(words)
    vocab_richness = len(unique_words) / total_words if total_words else 0
    return {
        "num_sentences": len(sentences),
        "avg_words": round(total_words / len(sentences), 2),
        "vocab_richness": round(vocab_richness, 4),
    }


def analyze_repetition(text):
    if not text:
        return {}
    words = text.split()
    if len(words) < 3:
        return {"repeat_ratio": 0}
    repeat_count = sum(
        1
        for i in range(len(words) - 1)
        if words[i] == words[i + 1]
        or (i + 2 < len(words) and words[i] == words[i + 2])
    )
    repeat_ratio = repeat_count / (len(words) - 1)
    return {"repeat_ratio": round(repeat_ratio, 4)}


def main():
    files = [
        "training_set_human.jsonl",
        "training_set_aigc.jsonl",
        "exploring_set.jsonl",
    ]
    print("=" * 70)
    print("NEW ANALYSIS: Structure, URLs, Numbers, Sentence Patterns")
    print("=" * 70)

    for f in files:
        data, n_records = load_jsonl(f)
        print("\n" + "=" * 50)
        print("Dataset: " + f + " (" + str(n_records) + " records)")
        print("=" * 50)

        url_stats = {"total": 0, "domains": Counter()}
        num_stats = {"total": 0, "price": 0, "count": 0}
        struct_stats = {"paras": 0, "list": 0, "bullet": 0}
        sent_stats = []
        repeat_ratios = []

        sample = data
        for item in sample:
            text = item.get("desc", "") or item.get("note_content", "")
            if not text:
                continue

            u = analyze_urls(text)
            url_stats["total"] += u["num_urls"]
            url_stats["domains"].update(u["domains"])

            n = analyze_numbers(text)
            num_stats["total"] += n["num_numbers"]
            if n["has_price"]:
                num_stats["price"] += 1
            if n["has_count"]:
                num_stats["count"] += 1

            s = analyze_structure(text)
            struct_stats["paras"] += s["num_paragraphs"]
            if s["has_list"]:
                struct_stats["list"] += 1
            if s["has_bullet"]:
                struct_stats["bullet"] += 1

            sent = analyze_sentences(text)
            if sent:
                sent_stats.append(sent)

            rep = analyze_repetition(text)
            if rep.get("repeat_ratio", 0) > 0:
                repeat_ratios.append(rep["repeat_ratio"])

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
            avg_words = sum(s["avg_words"] for s in sent_stats) / len(sent_stats)
            avg_vocab = sum(s["vocab_richness"] for s in sent_stats) / len(sent_stats)
            print("   Avg words/sentence: " + str(avg_words))
            print("   Vocab richness: " + str(avg_vocab))

        if repeat_ratios:
            avg_repeat = sum(repeat_ratios) / len(repeat_ratios)
            print("\n5. WORD REPETITION (Content):")
            print("   Avg repeat ratio: " + str(avg_repeat))

        url_stats_title = {"total": 0, "domains": Counter()}
        num_stats_title = {"total": 0, "price": 0, "count": 0}
        struct_stats_title = {"paras": 0, "list": 0, "bullet": 0}
        sent_stats_title = []
        repeat_ratios_title = []

        for item in sample:
            title = item.get("note_title", "")
            if not title:
                continue

            u = analyze_urls(title)
            url_stats_title["total"] += u["num_urls"]
            url_stats_title["domains"].update(u["domains"])

            n = analyze_numbers(title)
            num_stats_title["total"] += n["num_numbers"]
            if n["has_price"]:
                num_stats_title["price"] += 1
            if n["has_count"]:
                num_stats_title["count"] += 1

            s = analyze_structure(title)
            struct_stats_title["paras"] += s["num_paragraphs"]
            if s["has_list"]:
                struct_stats_title["list"] += 1
            if s["has_bullet"]:
                struct_stats_title["bullet"] += 1

            sent = analyze_sentences(title)
            if sent:
                sent_stats_title.append(sent)

            rep = analyze_repetition(title)
            if rep.get("repeat_ratio", 0) > 0:
                repeat_ratios_title.append(rep["repeat_ratio"])

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
            avg_words = sum(s["avg_words"] for s in sent_stats_title) / len(
                sent_stats_title
            )
            avg_vocab = sum(s["vocab_richness"] for s in sent_stats_title) / len(
                sent_stats_title
            )
            print("   Avg words/sentence: " + str(avg_words))
            print("   Vocab richness: " + str(avg_vocab))

        if repeat_ratios_title:
            avg_repeat = sum(repeat_ratios_title) / len(repeat_ratios_title)
            print("\n5. WORD REPETITION (Title):")
            print("   Avg repeat ratio: " + str(avg_repeat))

    print("\n" + "=" * 70)
    print("Analysis Complete")


if __name__ == "__main__":
    main()
